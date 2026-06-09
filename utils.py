"""
utils.py
========
Helper functions used by train.py and evaluate.py:
  1. Checkpoint save / load
  2. Metric computation (accuracy, precision, recall, F1)
  3. Confusion matrix plotting
  4. Training curve plotting
  5. Class weight computation (for imbalanced classes)
  6. Logger setup (console + file)
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report)

import config


# ══════════════════════════════════════════════════════════════════════
# 1.  CHECKPOINT SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════
def save_checkpoint(model, optimizer, epoch, val_acc, filename="best_model.pth"):
    """Save model + optimizer state to CHECKPOINT_DIR."""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(config.CHECKPOINT_DIR, filename)
    torch.save({'epoch': epoch, 'val_acc': val_acc, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, path)
    return path


def load_checkpoint(model, optimizer=None, filename="best_model.pth"):
    """
    Load model weights (and optionally optimizer state) from CHECKPOINT_DIR.
    Returns the epoch and val_acc stored in the checkpoint.
    """
    path = os.path.join(config.CHECKPOINT_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint loaded from {path}")

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', -1)
    val_acc = checkpoint.get('val_acc', 0.0)
    return epoch, val_acc


# ══════════════════════════════════════════════════════════════════════
# 2.  METRIC COMPUTATION
# ══════════════════════════════════════════════════════════════════════
def compute_metrics(all_labels, all_preds):
    """
    Compute classification metrics given ground-truth and predicted labels.

    Parameters
    ----------
    all_labels : list or np.ndarray of int
    all_preds  : list or np.ndarray of int

    Returns
    -------
    metrics : dict with keys:
        accuracy, precision_macro, recall_macro, f1_macro,
        precision_per_class, recall_per_class, f1_per_class,
        report (full sklearn classification report string)
    """
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    accuracy = (all_labels == all_preds).mean()*100.0

    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)*100
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)*100
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)*100

    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)*100
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)*100
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)*100

    report = classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES, zero_division=0)

    return {
        'accuracy':           accuracy,
        'precision_macro':    precision_macro,
        'recall_macro':       recall_macro,
        'f1_macro':           f1_macro,
        'precision_per_class': precision_per_class,
        'recall_per_class':   recall_per_class,
        'f1_per_class':       f1_per_class,
        'report':             report,
    }


# ══════════════════════════════════════════════════════════════════════
# 3.  CONFUSION MATRIX PLOT
# ══════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(all_labels, all_preds, save_name="confusion_matrix.png"):
    """
    Plot and save a normalised confusion matrix.

    Parameters
    ----------
    all_labels : list or np.ndarray
    all_preds  : list or np.ndarray
    save_name  : filename (saved to RESULTS_DIR)
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalise

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues',
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(config.NUM_CLASSES)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(config.CLASS_NAMES, fontsize=10)

    # Annotate cells
    thresh = 0.5
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            color = 'white' if cm_norm[i, j] > thresh else 'black'
            ax.text(j, i, f'{cm_norm[i,j]:.2f}\n({cm[i,j]})',
                    ha='center', va='center', fontsize=9, color=color)

    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_title('Confusion Matrix (normalised)', fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(config.RESULTS_DIR, save_name)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


# ══════════════════════════════════════════════════════════════════════
# 4.  TRAINING CURVE PLOT
# ══════════════════════════════════════════════════════════════════════
def plot_training_curves(train_losses, val_losses, train_accs, val_accs, eval_every=1, save_name="training_curves.png"):
    """
    Plot and save loss + accuracy curves over epochs.

    Parameters
    ----------
    train_losses, val_losses: list of float (one per epoch)
    train_accs,  val_accs: list of float (one per epoch, in %)
    save_name: filename (saved to RESULTS_DIR)
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    epochs = np.arange(1, len(train_losses)+1)
    val_epochs = np.arange(eval_every, len(train_losses)+1, eval_every)  # add this

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, train_losses, 'b-o', markersize=3, label='Train')
    ax1.plot(val_epochs, val_losses, 'r-o', markersize=3, label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss'); ax1.legend(); ax1.grid(True)

    # Accuracy
    ax2.plot(epochs, train_accs, 'b-o', markersize=3, label='Train')
    ax2.plot(val_epochs, val_accs, 'r-o', markersize=3, label='Val')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(True)

    plt.suptitle('Training Curves', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(config.RESULTS_DIR, save_name)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


# ══════════════════════════════════════════════════════════════════════
# 5.  CLASS WEIGHT COMPUTATION  (for imbalanced classes)
# ══════════════════════════════════════════════════════════════════════

def compute_class_weights(samples, device):
    """
    Compute inverse-frequency class weights for nn.CrossEntropyLoss.

    Parameters
    ----------
    samples : list of (filepath, label) tuples
    device  : torch.device

    Returns
    -------
    weights : FloatTensor of shape [NUM_CLASSES]
    """
    counts = np.zeros(config.NUM_CLASSES, dtype=float)
    for _, label in samples:
        counts[label] += 1

    # Avoid division by zero for missing classes
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * config.NUM_CLASSES  # normalise

    print("  Class weights:")
    for i, (name, w) in enumerate(zip(config.CLASS_NAMES, weights)):
        print(f"    [{i}] {name:<18} count={int(counts[i]):4d}  weight={w:.4f}")

    return torch.tensor(weights, dtype=torch.float32).to(device)


# ══════════════════════════════════════════════════════════════════════
# 6.  LOGGER SETUP
# ══════════════════════════════════════════════════════════════════════
def get_logger(name="radar_classifier", log_file="train.log"):
    """
    Create a logger that writes to both console and a log file in LOG_DIR.
    """
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_path = os.path.join(config.LOG_DIR, log_file)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
