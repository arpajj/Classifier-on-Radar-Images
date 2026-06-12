"""
train.py
========
Main training script for the Three-Stream CNN radar activity classifier.

What it does:
  1. Builds DataLoaders  (dataset.py)
  2. Instantiates the model  (model.py)
  3. Sets up loss, optimizer, scheduler
  4. Runs the training loop with:
       - per-epoch train loss & accuracy
       - validation every EVAL_EVERY epochs  (evaluate.py)
       - saves best checkpoint  (utils.py)
       - logs everything to console + file  (utils.py)
  5. Saves training curves at the end  (utils.py)

Usage:
    python train.py
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import config
from dataset import get_dataloaders, find_all_dat_files
from cnn_model import ThreeStreamCNN, count_parameters
from utils  import (save_checkpoint, load_checkpoint, compute_class_weights, plot_training_curves, get_logger)
from evaluate import run_evaluation   # imported here to call every EVAL_EVERY epochs
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════
# DEVICE SETUP  (supports multi-GPU with DataParallel)
# ══════════════════════════════════════════════════════════════════════
def get_device():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"  Found {n} GPU(s):")
        for i in range(n):
            print(f"    [{i}] {torch.cuda.get_device_name(i)}")
        return torch.device('cuda')
    print("  No GPU found, using CPU.")
    return torch.device('cpu')

# ══════════════════════════════════════════════════════════════════════
# TRAINING LOOP  (one epoch)
# ══════════════════════════════════════════════════════════════════════

# def train_one_epoch(model, loader, criterion, optimizer, device, logger):
#     model.train()
#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0

#     for batch_idx, (x, y) in enumerate(loader):
#         x, y = x.to(device), y.to(device)

#         optimizer.zero_grad()
#         logits = model(x)
#         loss = criterion(logits, y)
#         loss.backward()
#         optimizer.step()

#         preds = logits.argmax(dim=1)
#         total_correct += (preds == y).sum().item()
#         total_loss += loss.item()*x.size(0)
#         total_samples += x.size(0)

#         if (batch_idx + 1) % 20 == 0:
#             batch_acc = (preds == y).float().mean().item()*100
#             logger.info(f"   Batch [{batch_idx+1}/{len(loader)}] " f"loss={loss.item():.4f}  acc={batch_acc:.1f}%")

#     epoch_loss = total_loss / total_samples
#     epoch_acc  = total_correct / total_samples*100
#     return epoch_loss, epoch_acc


def train_one_epoch(model, loader, criterion, optimizer, device, logger):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="  Training", leave=False, unit="batch", colour="blue")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_loss += loss.item()*x.size(0)
        total_samples += x.size(0)

        # Live update of the progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc':  f'{(preds==y).float().mean().item()*100:.1f}%'})

    epoch_loss = total_loss/total_samples
    epoch_acc = (total_correct/total_samples)*100
    return epoch_loss, epoch_acc 


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    # ── Logger ────────────────────────────────────────────────────────
    logger = get_logger(log_file="train.log")
    logger.info("=" * 60)
    logger.info("Radar Human Activity Classification — Training")
    logger.info("=" * 60)

    # ── Reproducibility ───────────────────────────────────────────────
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # ── Device ────────────────────────────────────────────────────────
    device = get_device()

    # ── DataLoaders ───────────────────────────────────────────────────
    logger.info("Building DataLoaders …")
    train_loader, val_loader, test_loader = get_dataloaders()

    # ── Class weights (for imbalanced fall class) ─────────────────────
    logger.info("Computing class weights …")
    all_samples = find_all_dat_files(config.DATASET_ROOT, config.DATASET_FOLDERS)
    class_weights = compute_class_weights(all_samples, device)

    # ── Model ─────────────────────────────────────────────────────────
    logger.info("Instantiating model …")
    model = ThreeStreamCNN(num_classes = config.NUM_CLASSES, feature_dim = config.STREAM_FEATURE_DIM, dropout = config.DROPOUT, fusion_type = config.FUSION_TYPE)
  
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        logger.info(f" Wrapping model in DataParallel ({torch.cuda.device_count()} GPUs)")
        model = nn.DataParallel(model)

    model = model.to(device)
    logger.info(f"  Trainable parameters: {count_parameters(model):,}")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr = config.LEARNING_RATE, weight_decay = config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = config.LR_STEP_SIZE, gamma = config.LR_GAMMA)

    # ── Training state ────────────────────────────────────────────────
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR,    exist_ok=True)

    # ── Epoch loop ────────────────────────────────────────────────────
    logger.info(f"Starting training for {config.NUM_EPOCHS} epochs …")
    logger.info(f"  Batch size: {config.BATCH_SIZE}")
    logger.info(f"  Learning rate: {config.LEARNING_RATE}")
    logger.info(f"  Eval every: {config.EVAL_EVERY} epochs")
    logger.info("-" * 60)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        logger.info(f"Epoch [{epoch}/{config.NUM_EPOCHS}]  " f"LR={scheduler.get_last_lr()[0]:.6f}")

        # ── Train ─────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, logger)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        logger.info(f" → Train loss={train_loss:.4f}  acc={train_acc:.2f}%")
        scheduler.step()

        # ── Validation (every EVAL_EVERY epochs) ──────────────────────
        if epoch % config.EVAL_EVERY == 0 or epoch == config.NUM_EPOCHS:
            logger.info(f"  Running validation …")
            val_metrics = run_evaluation(model = model, loader = val_loader, device = device, split_name = "Val",
                                        logger = logger, save_cm = False) # save confusion matrix only at end
            val_loss = val_metrics['loss']
            val_acc  = val_metrics['accuracy']

            # Pad val lists to align with train lists
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            logger.info(f" → Val  loss={val_loss:.4f}  acc={val_acc:.2f}%  " f"F1={val_metrics['f1_macro']:.2f}%")

            # ── Save best checkpoint ───────────────────────────────────
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                path = save_checkpoint(model, optimizer, epoch, val_acc, filename="best_model.pth")
                logger.info(f"  ★ New best val acc={best_val_acc:.2f}% — saved → {path}")
        else:
            # Append NaN so lists stay same length as train lists
            val_losses.append(float('nan'))
            val_accs.append(float('nan'))

        logger.info("-"*60)

    # ── Final evaluation on test set ──────────────────────────────────
    logger.info("Loading best checkpoint for final test evaluation …")
    load_checkpoint(model, filename="best_model.pth")
    model = model.to(device)

    logger.info("Running final evaluation on TEST set …")
    test_metrics = run_evaluation(model = model, loader = test_loader, device = device, split_name = "Test",
                                 logger = logger, save_cm = True, cm_name = "confusion_matrix_test.png")
    logger.info(f"  Test accuracy  : {test_metrics['accuracy']:.2f}%")
    logger.info(f"  Test F1 macro  : {test_metrics['f1_macro']:.2f}%")
    logger.info(f"  Test Precision : {test_metrics['precision_macro']:.2f}%")
    logger.info(f"  Test Recall    : {test_metrics['recall_macro']:.2f}%")
    logger.info(f"\n{test_metrics['report']}")

    # ── Save training curves ──────────────────────────────────────────
    # Filter NaN validation values for plotting
    valid_epochs   = [i+1 for i, v in enumerate(val_accs) if not np.isnan(v)]
    valid_val_loss = [v for v in val_losses if not np.isnan(v)]
    valid_val_acc  = [v for v in val_accs if not np.isnan(v)]

    curve_path = plot_training_curves(train_losses = train_losses, val_losses = valid_val_loss,
                                    train_accs = train_accs, val_accs = valid_val_acc, eval_every = config.EVAL_EVERY)
    logger.info(f"Training curves saved → {curve_path}")
    logger.info("Training complete.")


if __name__ == '__main__':
    main()
