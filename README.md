# Radar Human Activity Classification

This repository implements a three-stream CNN for human activity classification from radar micro-Doppler signals. It includes radar pre-processing, dataset loading, model training, evaluation, and result logging.

## Project structure

- `config.py` — central experiment settings and dataset/output paths
- `preprocessing.py` — radar signal pre-processing functions (clean, reusable)
- `dataset.py` — dataset discovery, file parsing, and PyTorch DataLoader creation
- `cnn_model.py` — three-stream CNN architecture with late fusion
- `train.py` — model training loop, checkpointing, logging
- `evaluate.py` — validation/test evaluation and confusion matrix generation
- `utils.py` — helpers for checkpoints, metrics, plotting, logging

## Deliverables

- `ONE_PAGER.md` — concise project overview and implementation summary
- `MODEL_CARD.md` — model architecture, input/output, and training details
- `requirements.txt` — reproducible dependency list
- `preprocessing.py` — radar pre-processing functions in runnable form

## How to run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set the dataset path in `config.py` or set the environment variable `DATASET_ROOT`. Also chose the desired path for storing the trained models via setting the variable `OUTPUT_DIR`. The checkpoints of the trained models will be saved in a folder `checkpoints` inside that folder. 
3. Run training:

```bash
python train.py
```

4. After training, evaluate the best checkpoint:

```bash
python evaluate.py
```

## Notes

- The current implementation expects radar `.dat` files organized under the dataset root in the seven dataset folders defined in `config.py`.
- Output checkpoints, logs, and results are saved under `outputs/` by default.
- `preprocessing.py` contains the radar pre-processing pipeline and can be used independently for data inspection and debugging. It is not directly necessary for the whole training/testing pipeline. 
`dataset.py` contains a sanity main call for understanding the constuction of the dataset/dataloaders.  
