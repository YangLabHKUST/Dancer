"""Interleaved reference/spatial optimization with isolated reference parameters."""
import copy
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from .model import DancerModel
from .data import read_model_data, subset, split_indices, equal_cell_log_mu, model_hparams, neighbors, dense_block
from .utils import seed_all, random_stream, save_json, log, atomic_torch_save, cpu_store, restore_store, require_empty_stage


class ReferenceIterator:
    def __init__(self, n_cells, batch_size, rng):
        self.n_cells, self.batch_size, self.rng = n_cells, batch_size, rng
        self.order, self.position = None, 0

    def __next__(self):
        if self.order is None or self.position >= self.n_cells:
            self.order = self.rng.permutation(self.n_cells)
            self.position = 0
        rows = self.order[self.position:self.position+self.batch_size]
        self.position += len(rows)
        return rows


def spatial_args(data, indices, distances, rows, device):
    local = indices[rows]
    arrays = (dense_block(data.counts, rows), data.labels[rows], rows, dense_block(data.counts, local),
              data.labels[local], distances[rows], np.zeros(len(rows), dtype=np.int64))
    return tuple(torch.as_tensor(a, device=device) for a in arrays)


def train(cfg, selected, device="cuda:0", resume=False):
    root = Path(cfg["paths"]["results"])/"training"
    require_empty_stage(root, resume)
    seed_all(cfg["seed"]); pyro.clear_param_store(); torch.set_num_threads(cfg["threads"])
    r, x, v = cfg["reference"], cfg["spatial"], cfg["validation"]
    reference, _, _ = read_model_data(cfg["paths"]["reference"], dense=True)
    spatial, _, _ = read_model_data(cfg["paths"]["spatial"], types=reference.cell_types)
    if reference.genes != spatial.genes:
        raise ValueError("Reference and spatial genes must have identical order; run preprocessing first")
    if set(selected) != set(reference.cell_types):
        raise ValueError("Selected program cell types do not match the input annotation")
    rt, rv = split_indices(reference.labels, v["fraction"], cfg["seed"])
    xt, xv = split_indices(spatial.labels, v["fraction"], cfg["seed"])
    R, V = subset(reference, rt), subset(reference, rv)
    payload = torch.load(root/"last.pt", map_location="cpu", weights_only=False) if resume and (root/"last.pt").exists() else None
    if payload and (payload["config"] != cfg or payload["cell_types"] != reference.cell_types or
                    payload["genes"] != reference.genes or not np.array_equal(payload["spatial_ids"], spatial.cell_ids) or
                    not np.array_equal(payload["reference_ids"], reference.cell_ids)):
        raise ValueError("Training inputs/config differ from the interrupted run; use another results directory")
    log("building_neighbors", cells=spatial.n_cells, neighbors=x["n_neighbors"])
    idx, dist, median = neighbors(spatial, x["n_neighbors"], cfg["seed"])
    np.savez_compressed(root/"neighbors.npz", indices=idx, distances=dist, median_distance=median, cell_ids=spatial.cell_ids)
    np.savez_compressed(root/"split_indices.npz", reference_train=rt, reference_validation=rv, spatial_train=xt, spatial_validation=xv)
    hp = model_hparams(cfg, reference, spatial, subset(spatial, xt), median, [selected[t] for t in reference.cell_types])
    model = DancerModel(torch.from_numpy(equal_cell_log_mu(R)), **hp).to(device)
    params = model.reference_parameters()
    optimizer = torch.optim.Adam(params, lr=r["learning_rate"], eps=x["adam_eps"])

    def settings(module_name, param_name):
        return dict(lr=x["extra_learning_rate"] if param_name=="per_neighbor_diffusion_map" else x["learning_rate"], eps=x["adam_eps"], weight_decay=0.)

    svi = SVI(model.model, model.guide, pyro.optim.Adam(settings), Trace_ELBO(num_particles=x["training_particles"], vectorize_particles=True, retain_graph=True))
    val_elbo = Trace_ELBO(num_particles=v["particles"], vectorize_particles=True)
    rng = np.random.default_rng(cfg["seed"])
    iterator = ReferenceIterator(R.n_cells, r["batch_size"], rng)
    ref_steps = math.ceil(R.n_cells/r["batch_size"])
    spatial_steps = math.ceil(len(xt)/x["batch_size"])
    ratio = ref_steps*r["epochs"]/(spatial_steps*x["epochs"])
    state = dict(epoch=0, reference_steps=0, reference_budget=0., best_loss=float("inf"), best_epoch=None, elapsed_seconds=0.)
    history = []
    if payload:
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["reference_optimizer"])
        restore_store(model, payload["pyro_param_store"], device)
        svi.optim.set_state(payload["spatial_optimizer"])
        state = payload["training_state"]
        iterator.order, iterator.position = payload["iterator"]["order"], payload["iterator"]["position"]
        rng.bit_generator.state = payload["rng"]["minibatches"]
        random.setstate(payload["rng"]["python"]); np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch_cpu"])
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state_all(payload["rng"]["torch_cuda"])
        history = payload["history"]
    # Reference counts are small and reused for every update.
    rx, rc = torch.from_numpy(R.counts).to(device), torch.from_numpy(R.labels).to(device)
    vx, vc = torch.from_numpy(V.counts).to(device), torch.from_numpy(V.labels).to(device)
    save_json(root/"config.json", cfg)
    start = time.monotonic()-state["elapsed_seconds"]
    log("training_started", spatial_cells=spatial.n_cells, programs=selected, device=device, reference_updates_per_spatial=ratio)
    for epoch in range(state["epoch"]+1, x["epochs"]+1):
        model.train(); epoch_start = time.monotonic()
        kl = 1. if x["kl_warmup_epochs"]==0 else min(1., (1.-1e-6)*(epoch-1)/x["kl_warmup_epochs"]+1e-6)
        order = rng.permutation(xt)
        spatial_loss, seen, ref_loss, nref = 0., 0, 0., 0
        for step, start_row in enumerate(range(0, len(order), x["batch_size"]), 1):
            rows = order[start_row:start_row+x["batch_size"]]
            state["reference_budget"] += ratio
            while state["reference_budget"] >= 1.:
                rw = min(1., state["reference_steps"]/max(1, ref_steps*r["kl_warmup_epochs"]))
                rr = next(iterator)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = model.reference_loss(rx[rr], rc[rr], kl_weight=rw, n_reference=R.n_cells)
                loss.backward()
                if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                    raise FloatingPointError("Nonfinite reference loss/gradient")
                optimizer.step(); state["reference_steps"] += 1; state["reference_budget"] -= 1.
                ref_loss += float(loss.detach()); nref += 1
            optimizer.zero_grad(set_to_none=True)
            value = svi.step(*spatial_args(spatial, idx, dist, rows, device), n_obs=spatial.n_cells, kl_weight=kl)
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite spatial loss")
            spatial_loss += value/spatial.n_cells*len(rows); seen += len(rows)
            if step==1 or step%100==0:
                log("training_batch", epoch=epoch, batch=step, batches=spatial_steps)
        model.eval(); ref_validation = 0.; spatial_validation = 0.
        with torch.inference_mode(), random_stream(v["seed"], device):
            for start_row in range(0, V.n_cells, r["batch_size"]):
                rows = slice(start_row, min(start_row+r["batch_size"], V.n_cells))
                for _ in range(v["particles"]):
                    loss, _ = model.reference_loss(vx[rows], vc[rows], kl_weight=1., n_reference=R.n_cells)
                    ref_validation += float(loss)*len(vx[rows])/v["particles"]
        with torch.no_grad(), random_stream(v["seed"], device):
            for start_row in range(0, len(xv), x["batch_size"]):
                rows = xv[start_row:start_row+x["batch_size"]]
                value = val_elbo.loss(model.model, model.guide, *spatial_args(spatial, idx, dist, rows, device), n_obs=spatial.n_cells, kl_weight=1.)
                spatial_validation += value/spatial.n_cells*len(rows)
        validation_loss = spatial_validation/len(xv)
        if not np.isfinite(validation_loss):
            raise FloatingPointError("Nonfinite validation loss")
        eligible = kl>=1. and state["reference_steps"]>=ref_steps*r["kl_warmup_epochs"]
        improved = eligible and validation_loss<state["best_loss"]
        if improved:
            state.update(best_loss=validation_loss, best_epoch=epoch)
        state.update(epoch=epoch, elapsed_seconds=time.monotonic()-start)
        history.append(dict(epoch=epoch, spatial_loss=spatial_loss/seen, reference_loss=ref_loss/max(nref,1),
            validation_reference_loss=ref_validation/V.n_cells, validation_spatial_loss=validation_loss,
            eligible_for_best=eligible, best_epoch=state["best_epoch"], epoch_seconds=time.monotonic()-epoch_start))
        payload = dict(config=cfg, model_hparams=hp, state_dict={k:t.detach().cpu().clone() for k,t in model.state_dict().items()},
            pyro_param_store=cpu_store(), training_state=copy.deepcopy(state), reference_optimizer=optimizer.state_dict(),
            spatial_optimizer=svi.optim.get_state(), cell_types=reference.cell_types, genes=reference.genes,
            spatial_ids=spatial.cell_ids, reference_ids=reference.cell_ids, history=history,
            iterator=dict(order=iterator.order, position=iterator.position),
            rng=dict(python=random.getstate(), numpy=np.random.get_state(), minibatches=copy.deepcopy(rng.bit_generator.state),
                     torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
        atomic_torch_save(payload, root/"last.pt")
        if improved:
            atomic_torch_save(payload, root/"best.pt")
            save_json(root/"best.json", dict(epoch=epoch, validation_spatial_loss=validation_loss))
        pd.DataFrame(history).to_csv(root/"training_history.csv", index=False)
        save_json(root/"status.json", dict(status="training", **state))
        log("epoch_complete", **history[-1])
    save_json(root/"status.json", dict(status="complete", **state))
    if not (root/"best.pt").exists():
        raise RuntimeError("Training finished before both KL warmups completed. Increase epochs, or explicitly export last.pt.")
    return root/"best.pt"


def load_model(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    pyro.clear_param_store()
    model = DancerModel(payload["state_dict"]["log_mu"], **payload["model_hparams"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    restore_store(model, payload["pyro_param_store"], device)
    return model, payload
