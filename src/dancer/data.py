"""Count-matrix preprocessing and model data preparation."""
from dataclasses import dataclass
from pathlib import Path
import re
import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from .utils import log, save_json


def read_10x(path):
    """Read Gene Expression features; sum duplicate symbols preserving order."""
    with h5py.File(path) as f:
        m = f["matrix"]
        x = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:])).T.tocsr()
        ids = m["barcodes"].asstr()[:]
        features = m["features"]
        genes = features["name"].asstr()[:]
        keep = features["feature_type"].asstr()[:] == "Gene Expression" if "feature_type" in features else np.ones(len(genes), bool)
        x, genes = x[:, keep], genes[keep]
    codes, unique = pd.factorize(genes, sort=False)
    if len(unique) != len(genes):
        mapping = sparse.csr_matrix((np.ones(len(genes)), (np.arange(len(genes)), codes)), shape=(len(genes), len(unique)))
        x = x @ mapping
    return ad.AnnData(sparse.csr_matrix(x, dtype=np.float32), obs=pd.DataFrame(index=pd.Index(ids.astype(str), name="cell_id")), var=pd.DataFrame(index=pd.Index(unique.astype(str), name="gene")))


def normalize_id(value):
    return re.sub(r"\.0$", "", str(value).strip())


def annotated(matrix, workbook, sheet, excluded):
    table = pd.read_excel(workbook, sheet_name=sheet)
    label_col = "Cluster" if "Cluster" in table else "Annotation"
    table["Barcode"] = table["Barcode"].map(normalize_id)
    if table["Barcode"].duplicated().any():
        raise ValueError(f"Duplicate annotation barcodes in {sheet}")
    labels = table.set_index("Barcode")[label_col].astype("string").str.replace("_", " ", regex=False).str.strip()
    matrix.obs_names = [normalize_id(x) for x in matrix.obs_names]
    matrix.obs["cell_type"] = labels.reindex(matrix.obs_names).to_numpy()
    keep = matrix.obs.cell_type.notna() & ~matrix.obs.cell_type.isin(excluded) & (matrix.obs.cell_type != "")
    return matrix[keep].copy()


def preprocess(cfg):
    root, dest = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["processed"])
    p = cfg["preprocessing"]
    workbook = root / p["annotations"]
    log("preprocess_read")
    x = annotated(read_10x(root/p["spatial_matrix"]), workbook, p["spatial_sheet"], p["excluded_labels"])
    r = annotated(read_10x(root/p["reference_matrix"]), workbook, p["reference_sheet"], p["excluded_labels"])
    cells = pd.read_parquet(root/p["spatial_cells"])
    cells["cell_id"] = cells.cell_id.map(normalize_id)
    coords = cells.set_index("cell_id").reindex(x.obs_names)[["x_centroid", "y_centroid"]].to_numpy(np.float32)
    if not np.isfinite(coords).all():
        raise ValueError("Spatial cell coordinates do not match expression barcodes")
    x.obsm["spatial"] = coords
    # Only labels represented in both modalities can be modeled together.
    types = sorted(set(x.obs.cell_type) & set(r.obs.cell_type))
    x, r = x[x.obs.cell_type.isin(types)].copy(), r[r.obs.cell_type.isin(types)].copy()
    shared = [g for g in x.var_names if g in r.var_names]
    x, r = x[:, shared].copy(), r[:, shared].copy()
    nonzero = (np.asarray(x.X.sum(0)).ravel() > 0) & (np.asarray(r.X.sum(0)).ravel() > 0)
    x, r = x[:, nonzero].copy(), r[:, nonzero].copy()
    for a in (x, r):
        a.obs["cell_type"] = pd.Categorical(a.obs.cell_type, categories=types)
    dest.mkdir(parents=True, exist_ok=True)
    for a, key in [(x, "spatial"), (r, "reference")]:
        path = Path(cfg["paths"][key])
        path.parent.mkdir(parents=True, exist_ok=True)
        a.write_h5ad(path, compression="gzip")
    summary = dict(spatial_cells=x.n_obs, reference_cells=r.n_obs, shared_genes=x.n_vars, cell_types=types)
    save_json(dest/"preprocessing_summary.json", summary)
    pd.Series(x.var_names).to_csv(dest/"shared_genes.txt", index=False, header=False)
    log("preprocess_complete", **summary)


@dataclass
class Data:
    counts: object
    labels: np.ndarray
    libraries: np.ndarray
    cell_ids: np.ndarray
    genes: list
    cell_types: list
    spatial: object = None

    @property
    def n_cells(self):
        return self.counts.shape[0]

    @property
    def n_genes(self):
        return self.counts.shape[1]


def dense_block(matrix, rows):
    shape = np.asarray(rows).shape
    selected = matrix[np.asarray(rows).reshape(-1)]
    result = selected.toarray() if sparse.issparse(selected) else np.asarray(selected)
    return np.asarray(result, dtype=np.float32).reshape(*shape, matrix.shape[1])


def read_model_data(path, types=None, dense=False):
    a = ad.read_h5ad(path)
    if not a.obs_names.is_unique or not a.var_names.is_unique:
        raise ValueError("Cell IDs and gene names must be unique")
    x = sparse.csr_matrix(a.X, dtype=np.float32)
    if not np.isfinite(x.data).all() or (x.data < 0).any():
        raise ValueError("Expression must be finite nonnegative counts")
    names = a.obs.cell_type.astype(str)
    types = sorted(names.unique()) if types is None else types
    mapped = names.map({t: k for k, t in enumerate(types)})
    if mapped.isna().any():
        raise ValueError("A spatial cell type is absent from the reference")
    libraries = np.asarray(x.sum(1)).ravel()
    keep = libraries > 0
    data = Data(x[keep].toarray() if dense else x[keep], mapped.to_numpy(np.int64)[keep], libraries[keep],
                a.obs_names.to_numpy(dtype=str)[keep], a.var_names.tolist(), types,
                np.asarray(a.obsm["spatial"])[keep] if "spatial" in a.obsm else None)
    return data, a, np.flatnonzero(keep)


def subset(data, rows):
    return Data(data.counts[rows], data.labels[rows], data.libraries[rows], data.cell_ids[rows], data.genes, data.cell_types,
                data.spatial[rows] if data.spatial is not None else None)


def split_indices(labels, fraction, seed):
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in np.unique(labels):
        local = rng.permutation(np.flatnonzero(labels == label))
        if len(local) < 2:
            raise ValueError("At least two positive-count reference/spatial cells per type are needed for train/validation splitting")
        n = max(1, min(len(local)-1, round(len(local)*fraction)))
        validation.extend(local[:n]); train.extend(local[n:])
    return np.sort(np.array(train, dtype=np.int64)), np.sort(np.array(validation, dtype=np.int64))


def equal_cell_log_mu(data):
    normalized = data.counts / data.libraries[:, None]
    mu = np.stack([normalized[data.labels == k].mean(0) for k in range(len(data.cell_types))]).astype(np.float32)
    mu = np.maximum(mu, 1e-8)
    return np.log(mu/mu.sum(1, keepdims=True)).astype(np.float32)


def background_ratio(data):
    m = max(1, data.n_genes // 50)
    smallest = np.argsort(np.asarray(data.counts.sum(0)).ravel())[:m]
    local = np.asarray(data.counts[:, smallest].mean(1)).ravel()
    return float(np.mean(local / (data.libraries / data.n_genes)))


def model_hparams(cfg, reference, spatial, spatial_train, median, counts):
    s, r, g = cfg["spatial"], cfg["reference"], cfg["gamma"]
    libraries = np.log1p(spatial_train.libraries)
    return dict(program_counts=list(map(int, counts)), n_cells=spatial.n_cells, n_batches=1,
        n_neighbors=s["n_neighbors"], median_distance=median, n_latent=cfg["program_selection"]["max_programs"],
        n_hidden_reference=r["n_hidden"], n_hidden_expression=s["n_hidden_expression"], n_hidden_diffusion=s["n_hidden_diffusion"],
        n_layers=s["n_layers"], dropout=s["dropout"], dispersion_floor=s["dispersion_floor"], diffusion_epsilon=s["diffusion_epsilon"],
        alpha_prior_scale=s["alpha_prior_scale"], prior_proportions_rate=s["prior_proportions_rate"], prior_true_amount=s["prior_true_amount"],
        prior_diffusion_amount=s["prior_diffusion_amount"], sparsity_diffusion_prior_mean=s["sparsity_diffusion_prior_mean"],
        background_ratio=background_ratio(spatial_train), downsample_log_location=float(np.median(libraries)),
        downsample_log_scale=float(libraries.std()), downsample_counts=s["downsample_counts"],
        gamma_sigma_initial=g["sigma_initial"], gamma_sigma_rate=g["sigma_rate"], observation_likelihood_weight=s["observation_likelihood_weight"],
        memory_efficient_loadings=True)


def neighbors(data, count, seed):
    import scanpy as sc
    from sklearn.neighbors._base import _kneighbors_from_graph
    if data.spatial is None:
        raise ValueError("Spatial inputs require obsm['spatial'] coordinates")
    if data.n_cells <= count + 5:
        raise ValueError("Not enough cells for the configured spatial-neighbor graph; lower n_neighbors")
    a = ad.AnnData(np.zeros((data.n_cells, 1), np.float32))
    a.obsm["X_spatial"] = data.spatial
    sc.pp.neighbors(a, n_neighbors=count+5, use_rep="X_spatial", random_state=seed)
    distances, indices = _kneighbors_from_graph(a.obsp["distances"] ** 2, count, return_distance=True)
    median = float(np.median(np.partition(distances, min(5, count-1), axis=1)[:, min(5, count-1)]))
    if median <= 0:
        raise ValueError("Spatial coordinates give a nonpositive distance scale")
    return indices.astype(np.int64), distances.astype(np.float32), median
