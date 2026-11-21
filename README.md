# Dancer

Code accompanying **Dancer: DisentANgling within-Cell-type hetERogeneity from transcript contamination in imaging-based spatial transcriptomics**.

Dancer uses a labeled single-cell reference to learn cell-type-specific expression programs and separate intrinsic spatial expression from neighboring-cell and background contamination. This repository provides a **human breast cancer example**, from public data download to corrected counts, program loadings/activities and basic evaluation.

## Installation

Use a fresh Python 3.12 environment, preferably with a CUDA-capable GPU. Program selection can distribute five repeated fits across multiple GPUs, while model training uses a single GPU.

```bash
conda create -n dancer python=3.12 -y
conda activate dancer
cd Dancer
```

Install **one** of the following PyTorch builds:

**GPU (CUDA 12.4 example; choose a build compatible with your GPU driver):**

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

**CPU:**

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining dependencies and Dancer:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

## Breast cancer example: four commands

Run the commands from the `Dancer` directory. In step 3, set `--gpus` to the number of GPUs to use, or use `--gpus 0` to run program selection, training and export on the CPU.

```bash
# 1. Download public data and extract the required matrix/cell table.
python scripts/download_data.py --config configs/breast_cancer.yaml

# 2. Prepare labeled reference.h5ad and spatial.h5ad on shared genes.
python scripts/preprocess.py --config configs/breast_cancer.yaml

# 3. Select program numbers, train and export.
python scripts/run_model.py --config configs/breast_cancer.yaml --gpus 5

# 4. Select evaluation markers, calculate metrics and generate plots/report.
python scripts/evaluate.py --config configs/breast_cancer.yaml
```

In step 3, `--gpus 2` distributes the five repeated fits across two visible GPUs. Use `--devices 0,2` instead of `--gpus` to select specific visible GPUs.

This example uses the [10x Genomics breast cancer Xenium dataset](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast), together with the matched single-cell reference and annotations from the [accompanying data deposit](https://zenodo.org/records/10076046), described in [Janesick et al. (2023)](https://doi.org/10.1038/s41467-023-43458-x). The first command downloads the data and extracts the files needed for this example; the Xenium archive is approximately 10 GB. Rerun the same command to resume an interrupted download.

The second command keeps genes shared by the reference and spatial data, attaches the published cell-type labels, and saves `reference.h5ad` and `spatial.h5ad` as model inputs. Spatial cell coordinates are included in `spatial.h5ad`.

## Configuration and execution

Edit **`configs/breast_cancer.yaml`** to change paths, selection thresholds, training parameters, export settings or evaluation options. Relative paths are resolved against `project_root` in the configuration.

- `program_selection`: program cap, repeat seeds, held-out fraction, initial/incremental epochs and selection thresholds. Defaults use five repeats and retain up to 10 programs per cell type. Additional programs require positive mean likelihood gain, P < 0.10 and stability >= 0.50. The first program is retained.
- `reference` and `spatial`: network dimensions, learning rates, batch sizes and training budgets. Training defaults to 300 spatial epochs. The checkpoint with the lowest spatial validation loss after warmup is selected automatically and used for export.
- `export`: checkpoint, predictive samples and batch size. Defaults use `best` and 25 predictive samples for corrected counts.
- `evaluation`: marker thresholds, ASW calculation and UMAP settings. ASW can be time-consuming; set `evaluation.asw.enabled: false` to skip it.

For a new experiment, change `paths.results` to a new directory.

## Repository structure

```text
Dancer/
├── configs/breast_cancer.yaml  # Example configuration
├── scripts/
│   ├── download_data.py       # Download data
│   ├── preprocess.py          # Prepare model inputs
│   ├── run_model.py           # Select programs, train and export
│   ├── evaluate.py            # Calculate metrics and generate plots
│   └── load_outputs.py        # Load saved matrices and activities
├── src/dancer/                # Model and analysis code
├── data/                      # Downloaded and processed data
└── results/                   # Model outputs and evaluation results
```

## Results

All directories are created automatically. Downloads go into `data/raw/breast_cancer/`, processed inputs into `data/processed/breast_cancer/`, and results into `results/breast_cancer/`.

After evaluation, open `results/breast_cancer/report.html` to view the results. You can also load model outputs in an interactive Python session:

```bash
python -i scripts/load_outputs.py --config configs/breast_cancer.yaml
```

The script reads the output paths from the configuration and makes `corrected_counts`, `h`, `u` and `L` available for analysis.

The table below summarizes the all files saved under `results/breast_cancer/`.

| File or directory                           | Contents                                                                                                                                   |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `program_selection/selected_programs.csv` | Selected program number per cell type                                                                                                      |
| `program_selection/selection_curve.csv`   | Likelihood gains, P values, stability and decisions                                                                                        |
| `training/best.pt`, `last.pt`           | Model/optimizer states and random states                                                                                                   |
| `training/training_history.csv`           | Training and validation history                                                                                                            |
| `outputs/corrected.h5ad`                  | Corrected counts in `X`, raw counts in `layers['raw_counts']`, spatial activities in `obsm['h']`, coordinates and source proportions |
| `outputs/reference_activities.h5ad`       | Measured reference counts and reference activities in `obsm['u']`                                                                        |
| `outputs/programs.npz`                    | `L` (type × gene × program), `log_mu`, program mask/counts, ordered cell types and genes                                             |
| `evaluation/`                             | Marker lists, metric tables, ASW cell identifiers and UMAP coordinates                                                                     |
| `figures/`                                | PDF/PNG metric plots, Raw–Dancer UMAP and Reference–Raw–Dancer marker dot plot                                                          |
| `report.html`                             | Evaluation summary and plots; open in a browser                                                                                            |
