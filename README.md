# CAAE Time-Series Anomaly Detection

This repository provides the training and evaluation pipeline for time-series anomaly detection on MSL, SMAP, SMD, SWaT, and WADI.

## Structure

```text
.
+-- train.py              # model training
+-- eval.py               # checkpoint evaluation
+-- configs/              # dataset-specific configs
+-- model/                # model and loss definitions
+-- utils/                # config, data loading, and preprocessing
+-- requirements.txt
```

## Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

PyTorch should match your CUDA environment. See the official PyTorch installation instructions if you need a CUDA-specific wheel.

## Data

Place datasets under the paths specified in the config files. The default paths are:

```text
data/MSL/
data/SMD/
data/swat/
data/wadi/
```

SMAP uses the same root as MSL by default:

```yaml
data_path: ./data/MSL/
```

Each dataset loader expects the standard preprocessed files used by the corresponding config.

## Training

Run training from this directory:

```bash
python train.py --config configs/config_SMD.yaml
```

Available configs:

```text
configs/config_MSL.yaml
configs/config_SMAP.yaml
configs/config_SMD.yaml
configs/config_SWAT.yaml
configs/config_WADI.yaml
```

Training checkpoints are saved to:

```text
experiments/<experiment.name>/
```

## Evaluation

Evaluate all matching checkpoints for a dataset:

```bash
python eval.py --config configs/config_SMD.yaml
```

Evaluate a specific checkpoint directory:

```bash
python eval.py --config configs/config_SMD.yaml --ckpt_dir experiments/smd
```

The evaluation summary is written to:

```text
experiments/<experiment.name>/best_ckpts_summary.csv
```

## Configuration

The config files define:

- dataset path and window settings
- model dimensions
- training hyperparameters
- evaluation score smoothing and post-processing

Update `data.data_path` and `experiment.name` as needed before running a new experiment.
