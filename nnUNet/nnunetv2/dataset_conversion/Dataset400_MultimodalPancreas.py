from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import os
import re
import random


def convert_foundationmodel_panc(input_dirs, nnunet_dataset_id: int = 400):
    task_name = "MultimodalPancreas"
    foldername = f"Dataset{nnunet_dataset_id:03.0f}_{task_name}"
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    # Define canonical modality order and mapping
    canonical_modalities = ["CT", "T1", "T2", "OOP"]
    modality_to_channel = {m.lower(): i for i, m in enumerate(canonical_modalities)}
    channel_names = {i: m for i, m in enumerate(canonical_modalities)}
    tr_ctr = 0
    case_to_modalities = {}
    all_cases = {}  # {case_id: {modality: (src_img, src_mask)}}
    dataset_case_ids = []  # List of lists, each for one dataset

    # Build a list of all image entries (with dataset, case_id, modality)
    all_image_entries = []
    for dataset_idx, dataset_dir in enumerate(input_dirs):
        found = False
        for m in canonical_modalities:
            if m.lower() in dataset_dir.lower():
                modality_lower = m.lower()
                modality_str = m
                channel_idx = modality_to_channel[modality_lower]
                found = True
                break
        if not found:
            print(f"Warning: Could not determine canonical modality for {dataset_dir}, skipping.")
            continue
        image_dir = join(dataset_dir, 'images')
        mask_dir = join(dataset_dir, 'masks')
        image_files = subfiles(image_dir, suffix='.nii.gz', join=False)
        dataset_prefix = os.path.basename(dataset_dir.rstrip('/'))
        for img_file in image_files:
            case_id = img_file.replace('.nii.gz', '')
            # Unique case_id by prefixing with dataset
            # unique_case_id = f"{dataset_prefix}_{case_id}"
            unique_case_id = f"{case_id}"
            src_img = join(image_dir, img_file)
            src_mask = join(mask_dir, img_file)
            out_img = join(imagestr, f'{unique_case_id}_{modality_str}_{"0000"}.nii.gz')
            out_mask = join(labelstr, f'{unique_case_id}_{modality_str}.nii.gz')
            all_image_entries.append({
                'unique_case_id': unique_case_id,
                'modality_str': modality_str,
                'channel_idx': channel_idx,
                'src_img': src_img,
                'src_mask': src_mask,
                'out_img': out_img,
                'out_mask': out_mask
            })

    # Shuffle and split 8:1:1 for train:val:test
    random.seed(42)
    random.shuffle(all_image_entries)
    n_total = len(all_image_entries)
    n_test = max(1, int(round(0.1 * n_total)))
    n_val = max(1, int(round(0.1 * n_total)))
    n_train = n_total - n_test - n_val
    test_entries = all_image_entries[:n_test]
    val_entries = all_image_entries[n_test:n_test + n_val]
    train_entries = all_image_entries[n_test + n_val:]
    print(f"Total: {n_total}, Train: {len(train_entries)}, Val: {len(val_entries)}, Test: {len(test_entries)}")
    # Copy files and build training/validation/test lists
    training = []
    validation = []
    test = []
    for entry in train_entries:
        shutil.copy(entry['src_img'], entry['out_img'])
        shutil.copy(entry['src_mask'], entry['out_mask'])
        training.append({
            "image": f"./imagesTr/{entry['unique_case_id']}_{entry['modality_str']}_{entry['channel_idx']:04d}.nii.gz",
            "label": f"./labelsTr/{entry['unique_case_id']}_{entry['modality_str']}.nii.gz"
        })
    for entry in val_entries:
        shutil.copy(entry['src_img'], entry['out_img'])
        shutil.copy(entry['src_mask'], entry['out_mask'])
        validation.append({
            "image": f"./imagesTr/{entry['unique_case_id']}_{entry['modality_str']}_{entry['channel_idx']:04d}.nii.gz",
            "label": f"./labelsTr/{entry['unique_case_id']}_{entry['modality_str']}.nii.gz"
        })
    for entry in test_entries:
        shutil.copy(entry['src_img'], entry['out_img'])
        shutil.copy(entry['src_mask'], entry['out_mask'])
        test.append(f"./imagesTs/{entry['unique_case_id']}_{entry['modality_str']}_{entry['channel_idx']:04d}.nii.gz")

    # Simple label scheme: background=0, pancreas=1
    labels = {0: "background", 1: "pancreas"}
    # Write dataset.json
    dataset_json = {
        "channel_names": {str(i): channel_names[i] for i in channel_names},
        "labels": {str(k): v for k, v in labels.items()},
        "numTraining": len(training),
        "file_ending": ".nii.gz",
        "training": training,
        "validation": validation,
        "test": test,
        "dataset_name": task_name,
        "reference": "n/a",
        "release": "1.0",
        "description": "MultimodalPancreas: merged datasets with image/mask pairs. Each input dir must have 'image' and 'mask' subdirs with matching filenames. Modality is inferred from input dir name. Output filenames use nnU-Net channel index convention.",
        "overwrite_image_reader_writer": "NibabelIOWithReorient"
    }
    save_json(dataset_json, join(out_base, "dataset.json"))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dirs', type=str, nargs='+',
                        help="List of dataset directories, each with 'image' and 'mask' subdirs containing .nii.gz files with matching names. Each input dir must end with _MODALITY (e.g., _t1, _t2, _CT)")
    parser.add_argument('-d', required=False, type=int, default=400, help='nnU-Net Dataset ID, default: 400')
    args = parser.parse_args()
    convert_foundationmodel_panc(args.input_dirs, args.d) 