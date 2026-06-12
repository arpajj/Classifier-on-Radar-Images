"""
evaluate.py
===========
Evaluation script for the Three-Stream CNN radar activity classifier.

Can be used in two ways:
  1. Called by train.py during training:
        from evaluate import run_evaluation
        metrics = run_evaluation(model, loader, device, ...)

  2. Run standalone after training to evaluate a saved checkpoint: 
        python evaluate.py

What it computes:
  - Loss, Accuracy
  - Precision, Recall, F1  (per class + macro average)
  - Confusion matrix (saved as PNG)
  - Full classification report
"""

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import config
from cnn_model import ThreeStreamCNN
from utils import (load_checkpoint, compute_metrics, plot_confusion_matrix, get_logger)

# ══════════════════════════════════════════════════════════════════════
# CORE EVALUATION FUNCTION  (called by train.py and standalone)
# ══════════════════════════════════════════════════════════════════════
def run_evaluation(model, loader, device, split_name = "Val", logger = None, save_cm = False, cm_name = "confusion_matrix.png"):
    """
    Run inference on a DataLoader and return metrics.
    Parameters
    ----------
    model      : nn.Module  (already on device)
    loader     : DataLoader
    device     : torch.device
    split_name : str  label for logging ('Val' or 'Test')
    logger     : logger instance (if None, prints to console)
    save_cm    : bool  whether to save confusion matrix PNG
    cm_name    : str   filename for the confusion matrix PNG

    Returns
    -------
    metrics : dict  (see utils.compute_metrics for keys)
              also includes 'loss' key
    """
    if logger is None:
        logger = get_logger()

    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"  {split_name}", leave=False, unit="batch", colour="green")
        for x, y in pbar:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)

            preds = logits.argmax(dim=1)
            total_loss    += loss.item() * x.size(0)
            total_samples += x.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{(preds==y).float().mean().item()*100:.1f}%'})

    avg_loss = total_loss / total_samples
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = avg_loss

    # ── Per-class metrics table ───────────────────────────────────────
    logger.info(f"  [{split_name}] loss={avg_loss:.4f} | " f"acc={metrics['accuracy']:.2f}% | " f"F1={metrics['f1_macro']:.2f}%")
    logger.info(f"  Per-class results ({split_name}):")
    for i, name in enumerate(config.CLASS_NAMES):
        logger.info(f"  {name:<20} " f"P={metrics['precision_per_class'][i]:.1f}%  "
                    f"R={metrics['recall_per_class'][i]:.1f}%  "  f"F1={metrics['f1_per_class'][i]:.1f}%")

    # ── Confusion matrix ──────────────────────────────────────────────
    if save_cm:
        cm_path = plot_confusion_matrix(all_labels, all_preds, save_name=cm_name)
        logger.info(f"  Confusion matrix saved → {cm_path}")

    return metrics

# ══════════════════════════════════════════════════════════════════════
# STANDALONE EVALUATION  (loads best checkpoint and runs on test set)
# ══════════════════════════════════════════════════════════════════════
def main():
    logger = get_logger(log_file="evaluate.log")
    logger.info("="*60)
    logger.info("Radar Human Activity Classification — Evaluation")
    logger.info("="*60)

    # ── Device ────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # ── DataLoader (test set only) ─────────────────────────────────────
    logger.info("Building test DataLoader …")
    from dataset import get_dataloaders
    _, _, test_loader = get_dataloaders()

    # ── Model + checkpoint ────────────────────────────────────────────
    logger.info("Loading model from checkpoint …")
    model = ThreeStreamCNN(num_classes=config.NUM_CLASSES, feature_dim=config.STREAM_FEATURE_DIM, dropout=config.DROPOUT, fusion_type=config.FUSION_TYPE)
  
    # Handle DataParallel checkpoint if saved with multi-GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    epoch, val_acc = load_checkpoint(model, filename="best_model.pth")
    model = model.to(device)
    logger.info(f"  Loaded checkpoint from epoch {epoch} (val_acc={val_acc:.2f}%)")

    # ── Run evaluation ────────────────────────────────────────────────
    logger.info("Running evaluation on TEST set …")
    metrics = run_evaluation(model = model, loader = test_loader, device = device, split_name = "Test",
                            logger = logger, save_cm = True, cm_name = "confusion_matrix_test_standalone.png")

    # ── Final summary ─────────────────────────────────────────────────
    logger.info("="*60)
    logger.info("FINAL TEST RESULTS")
    logger.info("="*60)
    logger.info(f"  Accuracy : {metrics['accuracy']:.2f}%")
    logger.info(f"  Precision : {metrics['precision_macro']:.2f}%")
    logger.info(f"  Recall : {metrics['recall_macro']:.2f}%")
    logger.info(f"  F1 macro : {metrics['f1_macro']:.2f}%")
    logger.info(f"\n{metrics['report']}")
    logger.info("Evaluation complete.")


if __name__ == '__main__':
    main()
