from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import os


def convert_pancreas_ct_dataset(input_dir, mask_dir=None, nnunet_dataset_id: int = 109):
    """
    Convert Pancreas CT segmentation dataset to nnU-Net format.
    
    Args:
        input_dir: Path to the dataset directory. Can be:
                   - Directory containing 'images' and 'masks' subdirectories
                   - Directory directly containing image files
        mask_dir: Optional path to mask directory. If None, will try to find masks automatically.
        nnunet_dataset_id: nnU-Net Dataset ID, default: 109
    """
    task_name = "PancreasCT"
    foldername = f"Dataset{nnunet_dataset_id:03.0f}_{task_name}"
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    # Try to find image and mask directories
    image_dir = None
    mask_dir_found = None
    
    # First, check if input_dir has 'images' and 'masks' subdirectories
    if isdir(join(input_dir, 'images')) and isdir(join(input_dir, 'masks')):
        image_dir = join(input_dir, 'images')
        mask_dir_found = join(input_dir, 'masks')
        print(f"Found 'images' and 'masks' subdirectories in {input_dir}")
    elif isdir(join(input_dir, 'Images')) and isdir(join(input_dir, 'Mask')):
        image_dir = join(input_dir, 'Images')
        mask_dir_found = join(input_dir, 'Mask')
        print(f"Found 'Images' and 'Mask' subdirectories in {input_dir}")
    # If mask_dir is explicitly provided, use it
    elif mask_dir and isdir(mask_dir):
        image_dir = input_dir
        mask_dir_found = mask_dir
        print(f"Using provided mask directory: {mask_dir}")
    # Otherwise, assume input_dir directly contains image files
    # and masks are in the same directory or need to be found
    else:
        image_dir = input_dir
        # Try to find mask directory in parent directory
        parent_dir = os.path.dirname(input_dir)
        if isdir(join(parent_dir, 'masks')):
            mask_dir_found = join(parent_dir, 'masks')
            print(f"Found 'masks' directory in parent: {mask_dir_found}")
        elif isdir(join(parent_dir, 'Mask')):
            mask_dir_found = join(parent_dir, 'Mask')
            print(f"Found 'Mask' directory in parent: {mask_dir_found}")
        else:
            # Assume masks are in the same directory as images
            mask_dir_found = input_dir
            print(f"Assuming masks are in the same directory as images: {mask_dir_found}")
    
    if not isdir(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")
    
    print(f"Image directory: {image_dir}")
    print(f"Mask directory: {mask_dir_found}")
    
    # Get all image files
    image_files = subfiles(image_dir, suffix='.nii.gz', join=False)
    print(f"Found {len(image_files)} image files in {image_dir}")
    
    if len(image_files) == 0:
        raise ValueError(f"No .nii.gz files found in {image_dir}")
    
    # Verify corresponding mask files exist
    missing_masks = []
    for img_file in image_files:
        # Try to find corresponding mask file
        # First, try same filename
        mask_file = img_file
        mask_path = join(mask_dir_found, mask_file)
        
        # If not found and image has _0000 suffix, try without it
        if not isfile(mask_path) and img_file.endswith('_0000.nii.gz'):
            mask_file = img_file.replace('_0000.nii.gz', '.nii.gz')
            mask_path = join(mask_dir_found, mask_file)
        
        # If still not found, try with different naming pattern
        if not isfile(mask_path):
            # Try removing _0000 and adding _mask or _label
            base_name = img_file.replace('_0000.nii.gz', '').replace('.nii.gz', '')
            for suffix in ['_mask.nii.gz', '_label.nii.gz', '_seg.nii.gz', '.nii.gz']:
                mask_file = base_name + suffix
                mask_path = join(mask_dir_found, mask_file)
                if isfile(mask_path):
                    break
        
        if not isfile(mask_path):
            missing_masks.append(img_file)
    
    if missing_masks:
        print(f"Warning: Missing mask files for {len(missing_masks)} images (showing first 5): {missing_masks[:5]}")
        # Remove files without masks from processing
        image_files = [f for f in image_files if f not in missing_masks]
        print(f"Processing {len(image_files)} files with corresponding masks")
    
    if len(image_files) == 0:
        raise ValueError("No image files with corresponding masks found!")
    
    print(f"Total: {len(image_files)} files to process")
    
    # Copy files and build training list
    training = []
    
    def find_mask_file(img_file):
        """Helper function to find corresponding mask file"""
        # Try same filename
        mask_file = img_file
        mask_path = join(mask_dir_found, mask_file)
        
        # If not found and image has _0000 suffix, try without it
        if not isfile(mask_path) and img_file.endswith('_0000.nii.gz'):
            mask_file = img_file.replace('_0000.nii.gz', '.nii.gz')
            mask_path = join(mask_dir_found, mask_file)
        
        # If still not found, try with different naming patterns
        if not isfile(mask_path):
            base_name = img_file.replace('_0000.nii.gz', '').replace('.nii.gz', '')
            for suffix in ['_mask.nii.gz', '_label.nii.gz', '_seg.nii.gz', '.nii.gz']:
                mask_file = base_name + suffix
                mask_path = join(mask_dir_found, mask_file)
                if isfile(mask_path):
                    break
        
        return mask_path if isfile(mask_path) else None
    
    # Process all files as training data
    for img_file in image_files:
        case_id = img_file.replace('.nii.gz', '')
        # Remove channel suffix if present (e.g., _0000)
        if case_id.endswith('_0000'):
            case_id = case_id[:-5]
        src_img = join(image_dir, img_file)
        src_mask = find_mask_file(img_file)
        if src_mask is None:
            print(f"Warning: Could not find mask for {img_file}, skipping")
            continue
        out_img = join(imagestr, f'{case_id}_0000.nii.gz')
        out_mask = join(labelstr, f'{case_id}.nii.gz')
        
        shutil.copy(src_img, out_img)
        shutil.copy(src_mask, out_mask)
        
        training.append({
            "image": f"./imagesTr/{case_id}_0000.nii.gz",
            "label": f"./labelsTr/{case_id}.nii.gz"
        })
    
    # Define labels for pancreas segmentation
    labels = {0: "background", 1: "pancreas"}
    
    # Write dataset.json
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {str(k): v for k, v in labels.items()},
        "numTraining": len(training),
        "file_ending": ".nii.gz",
        "training": training,
        "dataset_name": task_name,
        "reference": "n/a",
        "release": "1.0",
        "description": "Pancreas CT Segmentation Dataset: Single-channel CT images with pancreas masks. Background=0, Pancreas=1.",
        "overwrite_image_reader_writer": "NibabelIOWithReorient"
    }
    
    save_json(dataset_json, join(out_base, "dataset.json"))
    print(f"Dataset converted successfully to {out_base}")
    print(f"Training cases: {len(training)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='/data2/pyq6817/Pansegnet_data_raw/CT',
                        help="Path to the dataset directory. Can contain 'images'/'masks' subdirectories or directly contain image files. Default: /data2/pyq6817/Pansegnet_data_raw/CT")
    parser.add_argument('-m', '--mask_dir', type=str, default=None,
                        help="Optional path to mask directory. If not provided, will try to find masks automatically.")
    parser.add_argument('-d', required=False, type=int, default=109, 
                        help='nnU-Net Dataset ID, default: 109')
    args = parser.parse_args()
    convert_pancreas_ct_dataset(args.input_dir, args.mask_dir, args.d)

