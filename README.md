# Dancer

Code accompanying **Dancer: DisentANgling within-Cell-type hetERogeneity from transcript contamination in imaging-based spatial transcriptomics**.

Dancer uses a labeled single-cell reference to learn cell-type-specific expression programs and separate intrinsic spatial expression from neighboring-cell and background contamination. This repository provides a **human breast cancer example**, from public data download to corrected counts, program loadings/activities and basic evaluation.

This example uses the [10x Genomics breast cancer Xenium dataset](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast), together with the matched single-cell reference and annotations from the [accompanying data deposit](https://zenodo.org/records/10076046), described in [Janesick et al. (2023)](https://doi.org/10.1038/s41467-023-43458-x).

## Installation

Use a fresh Python 3.12 environment, preferably with a CUDA-capable GPU.

```bash
git clone https://github.com/jerryliu357/Dancer.git
cd Dancer
conda create -n dancer python=3.12 -y
conda activate dancer
```

Install **one** of the following PyTorch builds:

**GPU:**

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

Run the following steps from the `Dancer` directory.

```bash
# 1. Download public data and extract the required matrix/cell table.
python scripts/download_data.py --config configs/breast_cancer.yaml

# 2. Prepare labeled reference.h5ad and spatial.h5ad on shared genes.
python scripts/preprocess.py --config configs/breast_cancer.yaml

# 3. Select program numbers, train and export (default: up to 5 idle GPUs for selection).
python scripts/run_model.py --config configs/breast_cancer.yaml
```

For step 3, you can use **one of these alternatives instead**:

```bash
# Use two automatically selected idle GPUs.
python scripts/run_model.py --config configs/breast_cancer.yaml --gpus 2

# OR: use specific visible GPUs (training and export use the first listed GPU).
python scripts/run_model.py --config configs/breast_cancer.yaml --devices 0,2

# OR: run selection, training and export on the CPU.
python scripts/run_model.py --config configs/breast_cancer.yaml --cpu
```

Then evaluate the exported results:

```bash
# 4. Select evaluation markers, calculate metrics and generate plots/report.
python scripts/evaluate.py --config configs/breast_cancer.yaml
```

## Configuration and execution

Edit **`configs/breast_cancer.yaml`** to change paths, selection thresholds, training parameters, export settings or evaluation options.

Relative paths are resolved against `project_root` in the configuration. For a new experiment, change `paths.results` to a new directory.

## Results

All directories are created automatically. Downloads go into `data/raw/breast_cancer/`, processed inputs into `data/processed/breast_cancer/`, and results into `results/breast_cancer/`.

The corrected expression matrix is saved in `results/breast_cancer/outputs/corrected.h5ad`.

After evaluation, please open `results/breast_cancer/report.html` to view the results. You can also load model outputs in an interactive Python session:

```bash
# Inspect corrected_counts, h, u and L interactively.
python -i scripts/load_outputs.py --config configs/breast_cancer.yaml
```

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
