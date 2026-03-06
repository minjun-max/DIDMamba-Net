import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import argparse
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve

from model import WaveletDualStreamVimUNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATCH_SIZE = 48 
STRIDE = 16 


DATASET_CONFIG = {
    'DRIVE': {
        'base_path': './data/DRIVE/test',
        'image_dir': 'images',
        'mask_dir': '1st_manual',
        'image_extensions': ['.tif'],
        'mask_pattern': lambda img_name: img_name.split('_')[0] + '_manual1.gif',
        'default_model': 'best_model_drive_wavelet.pth'
    },
    'STARE': {
        'base_path': './data/STARE',
        'image_dir': 'images',
        'mask_dir': '1st_manual',
        'image_extensions': ['.ppm', '.png'],
        'mask_pattern': lambda img_name: os.path.splitext(img_name)[0] + '.ah.ppm',
        'default_model': 'best_model_stare_wavelet.pth'
    },
    'HRF': {
        'base_path': './data/HRF',
        'image_dir': 'images',
        'mask_dir': '1st_manual',
        'image_extensions': ['.jpg', '.JPG', '.jpeg'],
        'mask_pattern': lambda img_name: os.path.splitext(img_name)[0] + '.tif',
        'default_model': 'best_model_hrf_wavelet.pth'
    },
    'CHASE': {
        'base_path': './data/CHASE/test',
        'image_dir': 'images',
        'mask_dir': '1st_manual',
        'image_extensions': ['.jpg', '.png'],
        'mask_pattern': lambda img_name: os.path.splitext(img_name)[0] + '.png',
        'default_model': 'best_model_chase_wavelet.pth'
    }
}


def preprocess_full_image(image_np):
    
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)
    gamma = 1.2
    gamma_corrected = np.power(clahe_image / 255.0, gamma) * 255.0
    gamma_corrected = np.uint8(gamma_corrected)
    return gamma_corrected


class PatchesDataset(Dataset):
    
    def __init__(self, patches):
        self.patches = patches
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        patch = np.expand_dims(patch, axis=0)
        patch = patch / 255.0 * 2.0 - 1.0
        patch = torch.from_numpy(patch).float()
        return patch.repeat(3, 1, 1)


def find_mask_path(dataset_name, config, image_name):
    
    mask_dir = os.path.join(config['base_path'], config['mask_dir'])
    mask_filename = config['mask_pattern'](image_name)
    return os.path.join(mask_dir, mask_filename)


def get_all_images(dataset_name, config):
 
    image_dir = os.path.join(config['base_path'], config['image_dir'])
    
    if not os.path.exists(image_dir):
        return []
    
    all_images = []
    for ext in config['image_extensions']:
        all_images.extend([f for f in os.listdir(image_dir) if f.endswith(ext)])
    
    return sorted(all_images)


def load_model_with_filtering(model, model_path, device):
    
    state_dict = torch.load(model_path, map_location=device)
    
    filtered_state_dict = {
        k: v for k, v in state_dict.items() 
        if not any(pattern in k for pattern in [
            '.dwt.h0_col', '.dwt.h1_col', 
            '.dwt.h0_row', '.dwt.h1_row',
            '.idwt.h0_col', '.idwt.h1_col',
            '.idwt.h0_row', '.idwt.h1_row'
        ])
    }
    
    model.load_state_dict(filtered_state_dict, strict=False)
    return model


def predict_single_image(model, image_path, mask_path, output_path):
    
    original_image = Image.open(image_path).convert("RGB")
    
    try:
        original_mask = Image.open(mask_path).convert("L")
    except:
        original_mask_cv = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if original_mask_cv is not None:
            original_mask = Image.fromarray(original_mask_cv)
        else:
            raise ValueError(f"Failed to load mask file: {mask_path}")
    
    image_np = np.array(original_image)
    processed_image_np = preprocess_full_image(image_np)
    H, W = processed_image_np.shape
    
    
    patches = []
    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            patch = processed_image_np[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patches.append(patch)
    
    patch_dataset = PatchesDataset(patches)
    patch_loader = DataLoader(patch_dataset, batch_size=32, shuffle=False)
    
    
    preds_list = []
    model.eval()
    with torch.no_grad():
        for batch_patches in patch_loader:
            batch_patches = batch_patches.to(DEVICE)
            with torch.cuda.amp.autocast():
                logits = model(batch_patches)
            probs = torch.sigmoid(logits)
            preds_list.append(probs.cpu())
    
    all_preds = torch.cat(preds_list, dim=0)
    
  
    full_pred_prob = torch.zeros((H, W), dtype=torch.float32)
    count_map = torch.zeros((H, W), dtype=torch.float32)
    
    idx = 0
    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            full_pred_prob[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += all_preds[idx].squeeze()
            count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1
            idx += 1
    
    count_map = torch.clamp(count_map, min=1.0)
    full_pred_prob = full_pred_prob / count_map
    full_pred_prob_np = full_pred_prob.numpy().astype(np.float32)
    final_mask = (full_pred_prob_np > 0.5).astype(np.uint8)
    
  
    gt_mask_np = np.array(original_mask)
    if gt_mask_np.max() > 1:
        gt_mask_np = gt_mask_np / 255.0
    
    gt_mask = gt_mask_np
    pred_mask = final_mask
    pred_prob = full_pred_prob_np
    
    if gt_mask.shape != pred_mask.shape:
        from skimage.transform import resize
        pred_mask_resized = resize(pred_mask, gt_mask.shape, order=0, 
                                   preserve_range=True, anti_aliasing=False)
        pred_mask_resized = (pred_mask_resized > 0.5).astype(np.uint8)
        pred_prob_resized = resize(pred_prob, gt_mask.shape, order=1, 
                                   preserve_range=True, anti_aliasing=True)
        pred_prob_resized = np.clip(pred_prob_resized, 0.0, 1.0)
    else:
        pred_mask_resized = pred_mask
        pred_prob_resized = pred_prob
    
 
    gt_binary = (gt_mask > 0.5).astype(np.uint8)
    tp = np.sum((pred_mask_resized == 1) & (gt_binary == 1))
    fp = np.sum((pred_mask_resized == 1) & (gt_binary == 0))
    fn = np.sum((pred_mask_resized == 0) & (gt_binary == 1))
    tn = np.sum((pred_mask_resized == 0) & (gt_binary == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
   
    y_true = gt_binary.flatten()
    y_pred_proba = pred_prob_resized.flatten()
    
    auc_score = 0.0
    try:
        if len(np.unique(y_true)) >= 2 and y_pred_proba.min() != y_pred_proba.max():
            auc_score = roc_auc_score(y_true, y_pred_proba)
    except:
        pass
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(original_mask, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(final_mask, cmap='gray')
    axes[2].set_title(f'Prediction\nDice: {dice:.4f} | AUC: {auc_score:.4f}', 
                     fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'dice': dice,
        'auc': auc_score,
    }


def predict_multiple_images(model_path, dataset_name, config, output_dir, image_list=None):
    
    print(f"\n{'='*70}")
    print(f"Batch Prediction - {dataset_name} Dataset")
    print(f"{'='*70}\n")
    
    model = WaveletDualStreamVimUNet(pretrained_weights_path=None).to(DEVICE)
    model = load_model_with_filtering(model, model_path, DEVICE)
    model.eval()
   
    if image_list is None:
        image_list = get_all_images(dataset_name, config)
    
    if not image_list:
        print(f"No image files found.")
        return
    
    print(f"Found {len(image_list)} images\n")
    os.makedirs(output_dir, exist_ok=True)
    
    # Batch processing
    all_metrics = []
    failed_images = []
    
    for idx, image_name in enumerate(image_list, 1):
        print(f"[{idx}/{len(image_list)}] {image_name}", end=" ")
        
        image_path = os.path.join(config['base_path'], config['image_dir'], image_name)
        mask_path = find_mask_path(dataset_name, config, image_name)
        
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print("✗ File not found")
            failed_images.append(image_name)
            continue
        
        image_basename = os.path.splitext(image_name)[0]
        output_path = os.path.join(output_dir, f"result_{image_basename}.png")
        
        try:
            metrics = predict_single_image(model, image_path, mask_path, output_path)
            metrics['image_name'] = image_name
            all_metrics.append(metrics)
            
            print(f"✓ Dice: {metrics['dice']:.4f} | AUC: {metrics['auc']:.4f}")
            
        except Exception as e:
            print(f"✗ {str(e)}")
            failed_images.append(image_name)
    
    if all_metrics:
        print(f"{'='*70}\n")
        
        valid_auc = [m for m in all_metrics if m['auc'] > 0]
        
        avg_acc = np.mean([m['accuracy'] for m in all_metrics])
        avg_sen = np.mean([m['sensitivity'] for m in all_metrics])
        avg_spe = np.mean([m['specificity'] for m in all_metrics])
        avg_dice = np.mean([m['dice'] for m in all_metrics])
        avg_auc = np.mean([m['auc'] for m in valid_auc]) if valid_auc else 0.0
        
        print(f"Succeeded: {len(all_metrics)}/{len(image_list)}")
        print(f"Accuracy:    {avg_acc:.4f}")
        print(f"Sensitivity: {avg_sen:.4f}")
        print(f"Specificity: {avg_spe:.4f}")
        print(f"Dice Score:  {avg_dice:.4f}")
        print(f"AUC (ROC):   {avg_auc:.4f}")
        
        # Save CSV
        csv_path = os.path.join(output_dir, 'results.csv')
        with open(csv_path, 'w') as f:
            f.write("image,dice,accuracy,sensitivity,specificity,auc\n")
            for m in all_metrics:
                f.write(f"{m['image_name']},{m['dice']:.4f},{m['accuracy']:.4f},"
                       f"{m['sensitivity']:.4f},{m['specificity']:.4f},{m['auc']:.4f}\n")
        print(f"\nResults saved to: {csv_path}")
        
        generate_summary_plot(all_metrics, output_dir, dataset_name)
    
    if failed_images:
        print(f"\nFailed: {len(failed_images)} image(s)")
    
    print(f"\n{'='*70}\n")


def generate_summary_plot(all_metrics, output_dir, dataset_name):
    """Generate summary plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{dataset_name} Dataset - Evaluation Summary', 
                 fontsize=16, fontweight='bold')
    
    metrics_names = ['dice', 'accuracy', 'sensitivity', 'specificity', 'auc']
    titles = ['Dice Score', 'Accuracy', 'Sensitivity', 'Specificity', 'AUC (ROC)']
    
    for idx, (metric_name, title) in enumerate(zip(metrics_names, titles)):
        ax = axes[idx // 3, idx % 3]
        values = [m[metric_name] for m in all_metrics]
        
        bars = ax.bar(range(len(values)), values, alpha=0.7)
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='r', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.4f}')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Image Index', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if max(values) > 0:
            ax.set_ylim([max(0, min(values) - 0.05), min(1, max(values) + 0.05)])
            max_idx = values.index(max(values))
            min_idx = values.index(min(values))
            bars[max_idx].set_color('green')
            bars[min_idx].set_color('red')
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Retinal Vessel Segmentation Prediction')
    
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['DRIVE', 'STARE', 'HRF', 'CHASE'])
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--batch', action='store_true')
    
    args = parser.parse_args()
    
    dataset_name = args.dataset
    config = DATASET_CONFIG[dataset_name]
    model_path = args.model if args.model else config['default_model']
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('./predictions', dataset_name, timestamp)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if args.batch or args.image is None:
        predict_multiple_images(model_path, dataset_name, config, output_dir)
    else:
        predict_multiple_images(model_path, dataset_name, config, output_dir, [args.image])


if __name__ == "__main__":
    main()