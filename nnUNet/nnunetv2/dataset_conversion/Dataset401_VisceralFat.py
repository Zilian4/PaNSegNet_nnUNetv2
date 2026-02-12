from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import os
import random


def convert_visceral_fat_dataset(input_dir, nnunet_dataset_id: int = 401):
    """
    Convert visceral fat segmentation dataset to nnU-Net format.
    
    Args:
        input_dir: Path to the dataset directory containing 'Images' and 'Mask' subdirectories
        nnunet_dataset_id: nnU-Net Dataset ID, default: 401
    """
    task_name = "VisceralFat"
    foldername = f"Dataset{nnunet_dataset_id:03.0f}_{task_name}"
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    imagests = join(out_base, "imagesTs")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(imagests)

    # Get image and mask directories
    image_dir = join(input_dir, 'Images')
    mask_dir = join(input_dir, 'Mask')
    
    if not isdir(image_dir) or not isdir(mask_dir):
        raise ValueError(f"Input directory must contain 'Images' and 'Mask' subdirectories. Found: {listdir(input_dir)}")
    
    # Get all image files
    image_files = subfiles(image_dir, suffix='.nii.gz', join=False)
    print(f"Found {len(image_files)} image files")
    
    # Verify corresponding mask files exist
    missing_masks = []
    for img_file in image_files:
        mask_file = img_file  # Same filename
        if not isfile(join(mask_dir, mask_file)):
            missing_masks.append(mask_file)
    
    if missing_masks:
        raise ValueError(f"Missing mask files: {missing_masks}")
    
    # Shuffle and split 8:1:1 for train:val:test
    random.seed(42)
    random.shuffle(image_files)
    n_total = len(image_files)
    n_test = max(1, int(round(0.1 * n_total)))
    n_val = max(1, int(round(0.1 * n_total)))
    n_train = n_total - n_test - n_val
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    print(f"Total: {n_total}, Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Copy files and build training/validation/test lists
    training = []
    validation = []
    test = []
    
    # Process training files
    for i, img_file in enumerate(train_files):
        case_id = img_file.replace('.nii.gz', '')
        src_img = join(image_dir, img_file)
        src_mask = join(mask_dir, img_file)
        out_img = join(imagestr, f'{case_id}_0000.nii.gz')
        out_mask = join(labelstr, f'{case_id}.nii.gz')
        
        shutil.copy(src_img, out_img)
        shutil.copy(src_mask, out_mask)
        
        training.append({
            "image": f"./imagesTr/{case_id}_0000.nii.gz",
            "label": f"./labelsTr/{case_id}.nii.gz"
        })
    
    # Process validation files
    for i, img_file in enumerate(val_files):
        case_id = img_file.replace('.nii.gz', '')
        src_img = join(image_dir, img_file)
        src_mask = join(mask_dir, img_file)
        out_img = join(imagestr, f'{case_id}_0000.nii.gz')
        out_mask = join(labelstr, f'{case_id}.nii.gz')
        
        shutil.copy(src_img, out_img)
        shutil.copy(src_mask, out_mask)
        
        validation.append({
            "image": f"./imagesTr/{case_id}_0000.nii.gz",
            "label": f"./labelsTr/{case_id}.nii.gz"
        })
    
    # Process test files
    for i, img_file in enumerate(test_files):
        case_id = img_file.replace('.nii.gz', '')
        src_img = join(image_dir, img_file)
        out_img = join(imagests, f'{case_id}_0000.nii.gz')
        
        shutil.copy(src_img, out_img)
        test.append(f"./imagesTs/{case_id}_0000.nii.gz")
    
    # Define labels for visceral fat segmentation
    labels = {0: "background", 1: "visceral_fat"}
    
    # Write dataset.json
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {str(k): v for k, v in labels.items()},
        "numTraining": len(training),
        "file_ending": ".nii.gz",
        "training": training,
        "validation": validation,
        "test": test,
        "dataset_name": task_name,
        "reference": "n/a",
        "release": "1.0",
        "description": "Visceral Fat Segmentation Dataset: Single-channel CT images with visceral fat masks. Background=0, Visceral Fat=1.",
        "overwrite_image_reader_writer": "NibabelIOWithReorient"
    }
    
    save_json(dataset_json, join(out_base, "dataset.json"))
    print(f"Dataset converted successfully to {out_base}")
    print(f"Training cases: {len(training)}")
    print(f"Validation cases: {len(validation)}")
    print(f"Test cases: {len(test)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str,
                        help="Path to the dataset directory containing 'Images' and 'Mask' subdirectories")
    parser.add_argument('-d', required=False, type=int, default=401, 
                        help='nnU-Net Dataset ID, default: 401')
    args = parser.parse_args()
    convert_visceral_fat_dataset(args.input_dir, args.d) 