"""Load saved model outputs for inspection or interactive analysis."""

import _common
import argparse
from pathlib import Path

import anndata as ad
import numpy as np

from dancer.utils import load_config


def load_outputs(config):
    output_dir = Path(config["paths"]["results"]) / "outputs"
    spatial = ad.read_h5ad(output_dir / "corrected.h5ad")
    reference = ad.read_h5ad(output_dir / "reference_activities.h5ad")
    with np.load(output_dir / "programs.npz") as programs:
        loadings = programs["L"]
    return spatial, reference, loadings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load corrected counts and program outputs; use python -i for interactive analysis."
    )
    parser.add_argument("--config", required=True, help="Path to the model configuration YAML")
    args = parser.parse_args()
    config = load_config(args.config)

    x, r, L = load_outputs(config)
    corrected_counts, h, u = x.X, x.obsm["h"], r.obsm["u"]

    print(f"Loaded model outputs from {Path(config['paths']['results']) / 'outputs'}")
    print(f"corrected_counts: {corrected_counts.shape} (spatial cells, genes)")
    print(f"h: {h.shape} (spatial cells, programs)")
    print(f"u: {u.shape} (reference cells, programs)")
    print(f"L: {L.shape} (cell types, genes, programs)")
