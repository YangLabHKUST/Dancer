"""Reference-only greedy program selection with GPU-scheduled repeated fits."""
import os
import sys
import json
import copy
import subprocess
from pathlib import Path
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.distributions import kl_divergence
from .model import DancerModel
from .data import read_model_data, subset, split_indices, model_hparams
from .utils import seed_all, random_stream, save_json, log, atomic_torch_save, require_empty_stage


def masked_baseline(counts,labels,mask,n_types):
    """Equal-cell reference baseline estimated without held-out gene values."""
    sums=np.zeros((n_types,counts.shape[1]),np.float64)
    numbers=np.zeros_like(sums)
    libraries=counts.sum(1,dtype=np.float64)
    for start in range(0,len(counts),1024):
        stop=min(start+1024,len(counts));m=mask[start:stop]
        prop=counts[start:stop]/libraries[start:stop,None]
        for t in np.unique(labels[start:stop]):
            rows=labels[start:stop]==t
            sums[t]+=(prop[rows]*m[rows]).sum(0)
            numbers[t]+=m[rows].sum(0)
    mu=np.maximum(np.divide(sums,numbers,out=np.zeros_like(sums),where=numbers>0),1e-8)
    mu/=mu.sum(1,keepdims=True)
    return np.log(mu).astype(np.float32)

def batch(data,mask,rows,device):
    y=torch.from_numpy(data.counts[rows]).to(device)
    c=torch.from_numpy(data.labels[rows]).to(device)
    visible=torch.from_numpy(mask[rows]).to(device)
    rowmean=(y*visible).sum(1,keepdim=True)/visible.sum(1,keepdim=True).clamp_min(1)
    encoder_input=torch.where(visible,y,rowmean)
    return y,c,visible,encoder_input

@torch.inference_mode()
def score(model,data,mask,slot,active,particles,seed,device):
    model.eval();delta=np.zeros(model.n_types);base=np.zeros(model.n_types);n=np.zeros(model.n_types,dtype=np.int64)
    enabled=model.program_mask.clone()
    with random_stream(seed,torch.device(device)):
        for start in range(0,data.n_cells,512):
            rows=np.arange(start,min(start+512,data.n_cells))
            y,c,visible,enc=batch(data,mask,rows,device)
            q,_=model.posterior(enc,c,reference=True)
            for _ in range(particles):
                z=q.rsample()
                rate_on=(y.sum(1,keepdim=True)*model.composition(z,c,detach_loadings=False)).clamp_min(1e-8)
                model.program_mask[:,slot]=False
                rate_off=(y.sum(1,keepdim=True)*model.composition(z,c,detach_loadings=False)).clamp_min(1e-8)
                model.program_mask.copy_(enabled)
                off=(torch.xlogy(y,rate_off)-rate_off-torch.lgamma(y+1))*(~visible)
                gain=(torch.xlogy(y,rate_on)-rate_on-torch.xlogy(y,rate_off)+rate_off)*(~visible)
                for t in torch.unique(c).tolist():
                    if active[t]:
                        delta[t]+=float(gain[c==t].sum())/particles
                        base[t]+=float(off[c==t].sum())/particles
            for t in torch.unique(c).tolist():n[t]+=int((~visible[c==t]).sum())
    return delta,base,n,model.loadings[:,:,slot].detach().cpu().numpy().copy()


def decision(gains, vectors, settings, program):
    """Positive marginal gain, one-sided t test and mean absolute cosine stability."""
    gains = np.asarray(gains, dtype=np.float64)
    mean = float(gains.mean())
    se = float(gains.std(ddof=1)/np.sqrt(len(gains))) if len(gains)>1 else 0.
    p = float(stats.t.sf(mean/se, len(gains)-1)) if se else (0. if mean>0 else 1.)
    vectors = np.asarray(vectors, dtype=np.float64)
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norm < 1e-12, 1., norm)
    stability = float(np.abs(unit@unit.T)[np.triu_indices(len(unit), 1)].mean()) if len(unit)>1 else 1.
    pcut = settings["p_cutoff"]
    scut = settings["stability_cutoff"]
    return dict(mean_gain=mean, se_gain=se, p=p, stability=stability, p_cutoff=pcut, stability_cutoff=scut,
                accepted=bool(program==1 or (mean>0 and p<pcut and stability>=scut)))


def replica(task_path):
    task_path = Path(task_path)
    task = json.loads(task_path.read_text())
    cfg, root = task["config"], Path(task["root"])
    s, ref = cfg["program_selection"], cfg["reference"]
    seed, slot, device = task["seed"], task["program"]-1, task["device"]
    torch.set_num_threads(cfg["threads"])
    data = SimpleNamespace(counts=np.load(root/"cache/counts.npy", mmap_mode="r"), labels=np.load(root/"cache/labels.npy"))
    data.n_cells, data.n_genes = data.counts.shape
    hp = json.loads((root/"cache/hparams.json").read_text())
    nt = len(hp["program_counts"])
    rng = np.random.default_rng(seed)
    mask = np.empty(data.counts.shape, dtype=bool)
    for start in range(0, len(mask), 1024):
        mask[start:start+1024] = rng.random((min(1024, len(mask)-start), data.n_genes)) >= s["holdout_fraction"]
    seed_all(seed)
    previous = torch.load(task["source"], map_location="cpu", weights_only=False) if task["source"] else None
    log_mu = previous["state_dict"]["log_mu"] if previous else torch.from_numpy(masked_baseline(data.counts, data.labels, mask, nt))
    model = DancerModel(log_mu, **hp).to(device)
    if previous:
        model.load_state_dict(previous["state_dict"])
        rng.bit_generator.state = previous["minibatch_rng_state"]
        model.program_mask.copy_(torch.arange(model.n_latent, device=device)[None, :] < torch.tensor(task["selected"], device=device)[:, None])
    seed_all(seed+100*slot)
    active = np.asarray(task["active"], dtype=bool)
    model.program_mask[:, slot] = torch.as_tensor(active, device=device)
    trainable = torch.zeros_like(model.raw_loadings)
    trainable[torch.as_tensor(active, device=device), :, slot] = 1
    hook = model.raw_loadings.register_hook(lambda grad: grad*trainable)
    optimizer = torch.optim.Adam(model.reference_parameters(), lr=ref["learning_rate"], eps=cfg["spatial"]["adam_eps"])
    epochs = s["initial_reference_epochs"] if slot==0 else s["incremental_reference_epochs"]
    usable = np.flatnonzero(active[data.labels])
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(usable)
        weight = min(1., (epoch+slot*s["initial_reference_epochs"])/max(1, ref["kl_warmup_epochs"]))
        total = 0.
        for start in range(0, len(order), ref["batch_size"]):
            rows = order[start:start+ref["batch_size"]]
            y, c, visible, enc = batch(data, mask, rows, device)
            q, _ = model.posterior(enc, c, reference=True)
            z = q.rsample()
            rate = (y.sum(1, keepdim=True)*model.composition(z, c, detach_loadings=False)).clamp_min(1e-8)
            recon = ((rate-torch.xlogy(y, rate)+torch.lgamma(y+1))*visible).sum(1).mean()
            loss = recon+weight*kl_divergence(q, model.latent_prior(y)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.reference_parameters()):
                raise FloatingPointError("Nonfinite selector loss/gradient")
            optimizer.step()
            total += float(loss.detach())*len(rows)
        log("selection_epoch", program=slot+1, seed=seed, epoch=epoch+1, epochs=epochs, loss=total/len(order))
    hook.remove()
    output = score(model, data, mask, slot, active, s["score_particles"], seed+9000, device)
    atomic_torch_save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
        minibatch_rng_state=copy.deepcopy(rng.bit_generator.state), delta=output[0], baseline_ll=output[1],
        heldout_entries=output[2], loadings=output[3]), task_path.parent/"result.pt")


def select(cfg, devices, resume=False):
    root = Path(cfg["paths"]["results"])/"program_selection"
    require_empty_stage(root, resume)
    stored = root/"config.json"
    if stored.exists() and json.loads(stored.read_text()) != cfg:
        raise ValueError("Selection config changed. Use a new results directory.")
    save_json(stored, cfg)
    final = root/"selected_programs.json"
    if final.exists():
        log("selection_reused", file=str(final))
        return json.loads(final.read_text())["program_counts"]
    s = cfg["program_selection"]
    if len(s["repeat_seeds"]) < 2 or len(set(s["repeat_seeds"])) != len(s["repeat_seeds"]):
        raise ValueError("The t test and stability estimate require at least two distinct repeat seeds")
    seed_all(cfg["seed"])
    data, _, _ = read_model_data(cfg["paths"]["reference"], dense=True)
    train, validation = split_indices(data.labels, cfg["validation"]["fraction"], cfg["seed"])
    data = subset(data, train)
    cache = root/"cache"; cache.mkdir(exist_ok=True)
    np.save(cache/"counts.npy", data.counts); np.save(cache/"labels.npy", data.labels)
    np.savez_compressed(cache/"reference_split.npz", train=train, validation=validation, cell_ids=data.cell_ids)
    hp = model_hparams(cfg, data, SimpleNamespace(n_cells=1), data, 1., [1]*len(data.cell_types))
    save_json(cache/"hparams.json", hp)
    active = np.ones(len(data.cell_types), dtype=bool)
    selected = np.ones(len(active), dtype=int)
    curves, scores = [], []
    queue = Queue()
    for device in devices:
        queue.put(device)

    def dispatch(seed, program, active_snapshot, selected_snapshot):
        dest = root/"replicas"/f"K{program:02d}"/f"seed{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        output = dest/"result.pt"
        if not output.exists():
            device = queue.get()
            try:
                source = root/"replicas"/f"K{program-1:02d}"/f"seed{seed}"/"result.pt" if program>1 else None
                save_json(dest/"task.json", dict(config=cfg, root=str(root), seed=seed, program=program, device=device,
                    active=active_snapshot.tolist(), selected=selected_snapshot.tolist(), source=str(source) if source else None))
                env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])+os.pathsep+os.environ.get("PYTHONPATH", ""),
                           OMP_NUM_THREADS=str(cfg["threads"]), MKL_NUM_THREADS=str(cfg["threads"]))
                log("selection_dispatch", program=program, seed=seed, device=device, log=str(dest/"worker.log"))
                with (dest/"worker.log").open("w") as stream:
                    process = subprocess.Popen([sys.executable, "-u", "-m", "dancer.selection", str(dest/"task.json")], env=env, stdout=stream, stderr=subprocess.STDOUT)
                    try:
                        code = process.wait()
                    except BaseException:
                        process.terminate(); process.wait()
                        raise
                if code:
                    raise RuntimeError(f"Selector failed. See {dest/'worker.log'}")
            finally:
                queue.put(device)
        return torch.load(output, map_location="cpu", weights_only=False)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for program in range(1, s["max_programs"]+1):
            if not active.any():
                break
            futures = [pool.submit(dispatch, seed, program, active.copy(), selected.copy()) for seed in s["repeat_seeds"]]
            outputs = [future.result() for future in futures]
            for k, name in enumerate(data.cell_types):
                if not active[k]:
                    continue
                result = decision([o["delta"][k] for o in outputs], [o["loadings"][k] for o in outputs], s, program)
                curves.append(dict(cell_type=name, programs=program, **result))
                for seed, o in zip(s["repeat_seeds"], outputs):
                    scores.append(dict(cell_type=name, programs=program, seed=seed, gain=o["delta"][k], baseline_ll=o["baseline_ll"][k], heldout_entries=o["heldout_entries"][k]))
                if result["accepted"]:
                    selected[k] = program
                else:
                    active[k] = False
                log("selection_decision", cell_type=name, programs=program, **result)
            pd.DataFrame(curves).to_csv(root/"selection_curve.csv", index=False)
            pd.DataFrame(scores).to_csv(root/"replicate_scores.csv", index=False)
    counts = dict(zip(data.cell_types, selected.tolist()))
    save_json(final, dict(program_counts=counts, settings=s))
    pd.DataFrame({"cell_type":data.cell_types,"programs":selected}).to_csv(root/"selected_programs.csv", index=False)
    return counts


if __name__ == "__main__":
    replica(sys.argv[1])
