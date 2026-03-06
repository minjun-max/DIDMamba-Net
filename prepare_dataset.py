import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import argparse


BASE_INPUT_PATH = "./data"
BASE_OUTPUT_PATH = "./" 
PATCH_SIZE = 48
NUM_PATCHES_PER_IMAGE = 800  
MIN_FOREGROUND_RATIO = 0.01

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)
    gamma = 1.2
    gamma_corrected = np.power(clahe_image / 255.0, gamma) * 255.0
    gamma_corrected = np.uint8(gamma_corrected)
    normalized_image = gamma_corrected / 255.0
    return normalized_image

def create_patches_for_dataset(dataset_name, mode='training'):
    print(f"\n{'='*20} Processing dataset: {dataset_name} ({mode}) {'='*20}")
    
    if dataset_name == 'DRIVE':
        img_ext = '.tif'
        mask_suffix = '_manual1.gif'
        train_test_split_needed = False
        img_folder_name = 'images'
        mask_folder_name = '1st_manual'
    elif dataset_name == 'STARE':
        img_ext = '.ppm'
        mask_suffix = '.ah.ppm' 
        train_test_split_needed = True
        img_folder_name = 'images'
        mask_folder_name = '1st_manual' 
    elif dataset_name == 'HRF':
        img_ext = '.jpg'
        mask_suffix = '.tif' 
        train_test_split_needed = True
        img_folder_name = 'images'
        mask_folder_name = '1st_manual' 
    elif dataset_name == 'CHASE':
        img_ext = '.jpg'
        mask_suffix = '.png'
        train_test_split_needed = False
        img_folder_name = 'images'
        mask_folder_name = '1st_manual'
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    original_path = os.path.join(BASE_INPUT_PATH, dataset_name)
    processed_path = os.path.join(BASE_OUTPUT_PATH, f"{dataset_name}_preprocessed")

    if train_test_split_needed:
        image_dir = os.path.join(original_path, img_folder_name)
        mask_dir = os.path.join(original_path, mask_folder_name)
        all_image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(img_ext)])
        # Split into train/test at an 8:2 ratio
        split_idx = int(len(all_image_files) * 0.8)
        if mode == 'training':
            image_files = all_image_files[:split_idx]
        else: 
            image_files = all_image_files[split_idx:]
    else: 
        image_dir = os.path.join(original_path, mode, img_folder_name)
        mask_dir = os.path.join(original_path, mode, mask_folder_name)
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(img_ext)])

    output_img_dir = os.path.join(processed_path, mode, 'patches_img')
    output_mask_dir = os.path.join(processed_path, mode, 'patches_mask')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    patch_counter = 0
    for img_name in tqdm(image_files, desc=f"Processing {dataset_name} {mode} images"):
        base_name = img_name.split('.')[0]
        img_path = os.path.join(image_dir, img_name)
        
        if dataset_name == 'DRIVE':
            mask_name = f"{img_name.split('_')[0]}{mask_suffix}"
        elif dataset_name == 'STARE':
            mask_name = f"{base_name}{mask_suffix}" 
        else: 
            mask_name = f"{base_name}{mask_suffix}"

        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            print(f"Warning: mask not found at {mask_path}, skipping image {img_name}")
            continue

        original_image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 128).astype(np.uint8)

        processed_image = preprocess_image(original_image)
        
        if dataset_name in ['DRIVE', 'CHASE']:
            num_patches_to_generate = 600 if mode == 'training' else 150
        elif dataset_name == 'STARE':
            num_patches_to_generate = 800 if mode == 'training' else 200
        elif dataset_name == 'HRF':
            num_patches_to_generate = 400 if mode == 'training' else 100
        else:
            num_patches_to_generate = NUM_PATCHES_PER_IMAGE if mode == 'training' else NUM_PATCHES_PER_IMAGE // 10
        
        for i in range(num_patches_to_generate):
            h, w = processed_image.shape
            if h < PATCH_SIZE or w < PATCH_SIZE: 
                continue
            
            x = random.randint(0, w - PATCH_SIZE)
            y = random.randint(0, h - PATCH_SIZE)
            img_patch = processed_image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            foreground_ratio = np.sum(mask_patch) / (PATCH_SIZE ** 2)
            if mode == 'training' and foreground_ratio < MIN_FOREGROUND_RATIO:
                if random.random() > 0.3:
                    continue
            
            patch_base_name = img_name.split('.')[0]
            patch_name = f"{patch_base_name}_patch_{i:05d}.png"
            Image.fromarray((img_patch * 255).astype(np.uint8)).save(
                os.path.join(output_img_dir, patch_name)
            )
            Image.fromarray(mask_patch * 255).save(
                os.path.join(output_mask_dir, patch_name)
            )
            patch_counter += 1
            
    print(f"Generated {patch_counter} patches in total for {dataset_name} {mode} set.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all', 
                        help='DRIVE, STARE, HRF, CHASE or all')
    args = parser.parse_args()
    
    all_datasets = ['DRIVE', 'STARE', 'HRF', 'CHASE']
    
    if args.dataset == 'all':
        datasets_to_process = all_datasets
    elif args.dataset in all_datasets:
        datasets_to_process = [args.dataset]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    for name in datasets_to_process:
        create_patches_for_dataset(name, mode='training')
        create_patches_for_dataset(name, mode='test')
        
    print("\n--- Patch generation complete for all datasets ---")