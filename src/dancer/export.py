"""Predictive-median corrected counts and labeled program activities."""
from pathlib import Path
import anndata as ad
import numpy as np
import pyro
import torch
from scipy import sparse
from .data import read_model_data, dense_block
from .training import load_model
from .utils import seed_all, random_stream, save_json, log


def write_h5ad_atomic(data, path):
    path = Path(path)
    tmp = path.with_suffix(".tmp.h5ad")
    data.write_h5ad(tmp, compression="gzip")
    tmp.replace(path)


@torch.inference_mode()
def predictive_median(model, counts, labels, samples):
    q, _ = model.posterior(counts, labels)
    kappa = pyro.param("px_r_mle").detach().reciprocal()
    library = counts.sum(-1, keepdim=True)
    draws = []
    for _ in range(samples):
        rate = library*model.composition(q.rsample(), labels, detach_loadings=True)
        intensity = torch.distributions.Gamma(kappa, kappa/(rate+1e-8)).sample()
        draws.append(torch.poisson(intensity+1e-9))
    ordered = torch.sort(torch.stack(draws), dim=0, stable=True).values
    if not torch.isfinite(ordered).all():
        raise FloatingPointError("Nonfinite predictive samples")
    return ordered[samples//2] if samples%2 else (ordered[samples//2-1]+ordered[samples//2])/2


@torch.inference_mode()
def export(cfg, device="cuda:0", checkpoint=None):
    torch.set_num_threads(cfg["threads"])
    seed_all(cfg["seed"])
    root = Path(cfg["paths"]["results"])
    checkpoint = Path(checkpoint) if checkpoint else root/"training"/(cfg["export"]["checkpoint"]+".pt")
    model, payload = load_model(checkpoint, device)
    x, raw, source_rows = read_model_data(cfg["paths"]["spatial"], types=payload["cell_types"])
    r, ref, reference_rows = read_model_data(cfg["paths"]["reference"], types=payload["cell_types"])
    if x.genes != payload["genes"] or r.genes != payload["genes"]:
        raise ValueError("Checkpoint and input gene order differ")
    if not np.array_equal(x.cell_ids, payload["spatial_ids"]) or not np.array_equal(r.cell_ids, payload["reference_ids"]):
        raise ValueError("Checkpoint and input cell order differ")
    dest = root/"outputs"; dest.mkdir(parents=True, exist_ok=True)
    batch_size, samples = cfg["export"]["batch_size"], cfg["export"]["predictive_samples"]
    if samples<1:
        raise ValueError("predictive_samples must be positive")
    parts = []
    # Do not interleave activity/alpha sampling with the predictive RNG stream.
    with random_stream(cfg["seed"], device):
        for start in range(0, x.n_cells, batch_size):
            rows = np.arange(start, min(start+batch_size, x.n_cells))
            y = torch.from_numpy(dense_block(x.counts, rows)).to(device)
            labels = torch.from_numpy(x.labels[rows]).to(device)
            median = predictive_median(model, y, labels, samples)
            parts.append(sparse.csr_matrix(median.cpu().numpy()))
            log("export_counts", completed=int(rows[-1])+1, total=x.n_cells)
    generated = sparse.vstack(parts, format="csr")
    mapping = sparse.csr_matrix((np.ones(len(source_rows)), (source_rows, np.arange(len(source_rows)))), shape=(raw.n_obs, len(source_rows)))
    corrected = ad.AnnData((mapping@generated).tocsr().astype(np.float32), obs=raw.obs.copy(), var=raw.var.copy())
    corrected.layers["raw_counts"] = sparse.csr_matrix(raw.X)
    if "spatial" in raw.obsm:
        corrected.obsm["spatial"] = np.asarray(raw.obsm["spatial"])
    for data, full, rows_full, key, reference in [(x, corrected, source_rows, "h", False), (r, ref, reference_rows, "u", True)]:
        means = np.zeros((full.n_obs, model.n_latent), np.float32)
        alpha = np.zeros((full.n_obs, 3), np.float32)
        with random_stream(cfg["seed"], device):
            for start in range(0, data.n_cells, batch_size):
                rows = np.arange(start, min(start+batch_size, data.n_cells))
                y = torch.from_numpy(dense_block(data.counts, rows)).to(device)
                labels = torch.from_numpy(data.labels[rows]).to(device)
                q, _ = model.posterior(y, labels, reference=reference)
                activity = sum(model.state(q.rsample()) for _ in range(cfg["export"]["activity_samples"]))/cfg["export"]["activity_samples"]
                activity *= model.program_mask[labels]
                means[rows_full[rows]] = activity.cpu().numpy()
                if not reference:
                    alpha[rows_full[rows]] = model.alpha_from_counts(y).cpu().numpy()
        full.obsm[key] = means
        full.obs["model_input_nonzero"] = np.asarray(full.layers["raw_counts"].sum(1)).ravel()>0 if not reference else np.asarray(full.X.sum(1)).ravel()>0
        if not reference:
            full.obsm["source_proportions"] = alpha
            for i, name in enumerate(["alpha_self", "alpha_neighbor", "alpha_background"]):
                full.obs[name] = alpha[:, i]
    counts = model.program_mask.sum(1).cpu().numpy()
    metadata = dict(cell_types=np.asarray(payload["cell_types"]), program_counts=counts,
                    activity_definition="posterior mean softplus(latent); inactive columns zero", checkpoint_epoch=payload["training_state"]["epoch"])
    corrected.uns["dancer"] = metadata
    ref.uns["dancer"] = metadata
    write_h5ad_atomic(corrected, dest/"corrected.h5ad")
    write_h5ad_atomic(ref, dest/"reference_activities.h5ad")
    np.savez_compressed(dest/"programs.npz", L=model.loadings.cpu().numpy(), log_mu=model.log_mu.cpu().numpy(),
        genes=np.asarray(payload["genes"]), cell_types=np.asarray(payload["cell_types"]), program_counts=counts,
        program_mask=model.program_mask.cpu().numpy())
    save_json(dest/"export_summary.json", dict(checkpoint=str(checkpoint), checkpoint_epoch=payload["training_state"]["epoch"],
        predictive_samples=samples, seed=cfg["seed"], cells=raw.n_obs, genes=raw.n_vars,
        activity_samples=cfg["export"]["activity_samples"], export="median of posterior Gamma-Poisson intrinsic counts"))
    log("export_complete", directory=str(dest))
