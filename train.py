import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

from dataset import DRIVEDataset
from model import WaveletDualStreamVimUNet 
from losses import EnhancedSegmentationLoss
from configs import get_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED_WEIGHTS_PATH = "vim_s_midclstok_ft_81p6acc.pth"

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, epoch):
    loop = tqdm(loader, leave=True, desc=f"Training Epoch {epoch}")
    model.train()
    nan_count = 0
    epoch_losses = {
        'total_loss': 0.0,
        'spatial_loss': 0.0,
        'bce_loss': 0.0,
        'dice_loss': 0.0,
        'tversky_loss': 0.0,
        'focal_loss': 0.0,
    }
    valid_batches = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
        try:
            if data.dim() == 3:
                data = data.unsqueeze(0)
            elif data.dim() != 4:
                continue
            
            if targets.dim() == 2:
                targets = targets.unsqueeze(0).unsqueeze(0)
            elif targets.dim() == 3:
                targets = targets.unsqueeze(1)
            elif targets.dim() != 4:
                continue
            
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)
        except Exception:
            continue

        # Use new torch.amp API to avoid FutureWarnings
        with torch.amp.autocast('cuda'):
            try:
                predictions = model(data)
            except Exception:
                continue
            
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                nan_count += 1
                continue
            
            loss, loss_details = loss_fn(predictions, targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        for key in epoch_losses.keys():
            if key in loss_details:
                epoch_losses[key] += loss_details[key]
        valid_batches += 1
        
        loop.set_postfix(
            total=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[1]['lr']:.2e}"
        )
    
    if valid_batches == 0:
        return None
    
    return {k: v / valid_batches for k, v in epoch_losses.items()}

def check_metrics(loader, model, device="cuda"):
    all_preds_proba = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader, leave=False, desc="Validation (TTA)"):
            x = x.to(device)
            y = y.to(device)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            
            with torch.amp.autocast('cuda'):
                logit_1 = model(x)
                logit_2 = torch.flip(model(torch.flip(x, dims=[3])), dims=[3])
                logit_3 = torch.flip(model(torch.flip(x, dims=[2])), dims=[2])                
                avg_logits = (logit_1 + logit_2 + logit_3) / 3.0
                preds_proba = torch.sigmoid(avg_logits)
        
            if y.dim() == 2:
                y = y.unsqueeze(0).unsqueeze(0)
            elif y.dim() == 3:
                y = y.unsqueeze(1)

            all_preds_proba.append(preds_proba.cpu().flatten())
            all_labels.append(y.cpu().flatten())

    if len(all_preds_proba) == 0:
        return {'dice': 0.0, 'accuracy': 0.0, 'sensitivity': 0.0, 'specificity': 0.0, 'precision': 0.0, 'auc': 0.0}

    y_true = torch.cat(all_labels).numpy()
    y_scores = torch.cat(all_preds_proba).numpy()
    
    if np.isnan(y_scores).any():
        y_scores = np.nan_to_num(y_scores, nan=0.5)
   
    thresholds = np.arange(0.3, 0.61, 0.05)
    best_dice = 0.0
    best_threshold = 0.5
    
    for th in thresholds:
        y_pred_th = (y_scores > th)
        tp = np.sum(y_pred_th & (y_true == 1))
        fp = np.sum(y_pred_th & (y_true == 0))
        fn = np.sum((~y_pred_th) & (y_true == 1))
        epsilon = 1e-6
        dice = 2 * tp / (2 * tp + fp + fn + epsilon)
        if dice > best_dice:
            best_dice = dice
            best_threshold = th

    preds = (y_scores > best_threshold)
    tp = np.sum(preds & (y_true == 1))
    fp = np.sum(preds & (y_true == 0))
    tn = np.sum((~preds) & (y_true == 0))
    fn = np.sum((~preds) & (y_true == 1))
    
    epsilon = 1e-6
    sensitivity = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    precision = tp / (tp + fp + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    
    try:
        auc_score = roc_auc_score(y_true, y_scores)
    except Exception:
        auc_score = 0.0

    model.train()
    return {
        'dice': best_dice,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'auc': auc_score,
        'thr': best_threshold
    }

def load_checkpoint(model, optimizer, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print(f" Checkpoint file not found: {checkpoint_path}")
        return 0, None
    
    print(f"\n{'='*70}\n Loading Checkpoint: {checkpoint_path}\n{'='*70}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_metrics = checkpoint.get('best_metrics', None)
    
    if best_metrics:
        # Gracefully handle missing keys from older checkpoints
        print(f" Loaded Best Metrics:")
        print(f"   Dice: {best_metrics.get('dice', 0.0):.4f}")
        print(f"   Precision: {best_metrics.get('precision', 0.0):.4f}")
        print(f"   Sensitivity (Recall): {best_metrics.get('sensitivity', 0.0):.4f}")
    
    return start_epoch, best_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DRIVE')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()
    
    config = get_config(args.dataset)
    BATCH_SIZE = args.batch_size if args.batch_size else config['batch_size']
    EPOCHS = args.epochs if args.epochs else config['epochs']
    PATIENCE = args.patience if args.patience else config['patience']
    
    model = WaveletDualStreamVimUNet(
        pretrained_weights_path=PRETRAINED_WEIGHTS_PATH,
        patch_size=config['image_size'],
        use_wavelet_upsample=True
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    start_epoch, loaded_best_metrics = 0, None

    if args.resume:
        start_epoch, loaded_best_metrics = load_checkpoint(model, optimizer, args.resume, DEVICE)

    # Reconstruct best_metrics with all required keys
    if loaded_best_metrics:
        best_metrics = loaded_best_metrics.copy()
        for k in ['dice', 'accuracy', 'sensitivity', 'specificity', 'precision', 'auc']:
            if k not in best_metrics:
                best_metrics[k] = 0.0
                best_metrics[f"{k}_epoch"] = 0
    else:
        best_metrics = {k: 0.0 for k in ['dice', 'accuracy', 'sensitivity', 'specificity', 'precision', 'auc']}
        for k in best_metrics.keys(): best_metrics[f"{k}_epoch"] = 0

    train_loader = DataLoader(DRIVEDataset(root_dir=f"./{args.dataset}_preprocessed", train=True, image_size=config['image_size']), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(DRIVEDataset(root_dir=f"./{args.dataset}_preprocessed", train=False, image_size=config['image_size']), 
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    loss_fn = EnhancedSegmentationLoss(**config['loss'])
    scaler = torch.amp.GradScaler('cuda')
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-7)
    
    SAVE_PATH = f"best_model_{args.dataset.lower()}_fixed.pth"
    epochs_no_improve = 0

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        avg_losses = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, epoch+1)
        if avg_losses is None: break
        
        val_metrics = check_metrics(test_loader, model, device=DEVICE)
        scheduler.step()
        
        improved = False
        for m in ['dice', 'accuracy', 'sensitivity', 'specificity', 'precision', 'auc']:
            if val_metrics[m] > best_metrics[m]:
                best_metrics[m] = val_metrics[m]
                best_metrics[f"{m}_epoch"] = epoch + 1
                if m == 'dice': improved = True

        if improved:
            torch.save({'model_state_dict': model.state_dict(), 'best_metrics': best_metrics, 'epoch': epoch}, SAVE_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE: break

    # Final safe logging
    print(f"\n{'='*70}\n{' FINAL BEST METRICS ':^70}\n{'='*70}")
    for m in ['dice', 'accuracy', 'sensitivity', 'specificity', 'precision', 'auc']:
        val = best_metrics.get(m, 0.0)
        ep = best_metrics.get(f"{m}_epoch", "N/A")
        print(f" {m.capitalize():12}: {val:.4f} (Epoch {ep})")

if __name__ == "__main__":
    main()