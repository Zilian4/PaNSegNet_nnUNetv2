# PaNSegNet Quick Start Guide

This is a simplified guide for using PaNSegNet. Just copy and paste the commands below.

**Prerequisites:**
- nnUNetv2 installed (see [INSTALL.md](INSTALL.md))
- Environment variables set:
  ```bash
  export nnUNet_raw="/path/to/nnUNet_raw"
  export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
  export nnUNet_results="/path/to/nnUNet_results"
  ```
- Dataset prepared in nnUNet format (see [dataset format documentation](nnUNet/documentation/dataset_format.md))

---

## Section 1: Preprocessing

Preprocess your dataset. Replace `DATASET_ID` with your dataset ID (e.g., `001`, `002`, etc.).

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

**Example:**
```bash
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
```

This will:
- Extract dataset fingerprint
- Create experiment plans
- Preprocess the data for training

---

## Section 2: Training

Train PaNSegNet using the `PaNSegNet_Trainer`. Replace:
- `DATASET_ID` with your dataset ID
- `CONFIGURATION` with `2d`, `3d_fullres`, `3d_lowres`, or `3d_cascade_fullres`
- `FOLD` with `0`, `1`, `2`, `3`, or `4` (for 5-fold cross-validation)

### Basic Training Command

```bash
nnUNetv2_train DATASET_ID CONFIGURATION FOLD -tr PaNSegNet_Trainer --npz
```

**Examples:**

#### Train 2D U-Net (Fold 0):
```bash
nnUNetv2_train 001 2d 0 -tr PaNSegNet_Trainer --npz
```

#### Train all 5 folds of 3D full resolution U-Net:
```bash
nnUNetv2_train 001 3d_fullres 0 -tr PaNSegNet_Trainer --npz
nnUNetv2_train 001 3d_fullres 1 -tr PaNSegNet_Trainer --npz
nnUNetv2_train 001 3d_fullres 2 -tr PaNSegNet_Trainer --npz
nnUNetv2_train 001 3d_fullres 3 -tr PaNSegNet_Trainer --npz
nnUNetv2_train 001 3d_fullres 4 -tr PaNSegNet_Trainer --npz
```

#### Train on multiple GPUs (parallel):
```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 001 3d_fullres 0 -tr PaNSegNet_Trainer --npz &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 001 3d_fullres 1 -tr PaNSegNet_Trainer --npz &
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 001 3d_fullres 2 -tr PaNSegNet_Trainer --npz &
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 001 3d_fullres 3 -tr PaNSegNet_Trainer --npz &
CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 001 3d_fullres 4 -tr PaNSegNet_Trainer --npz &
wait
```

**Important Notes:**
- The `--npz` flag saves softmax predictions needed for finding the best configuration
- Wait for the first fold to start using GPU before starting other folds
- Training checkpoints are saved every 50 epochs
- To continue training: add `--c` flag
- To run validation only: add `--val` flag

---

## Section 3: Inference

Run inference on new data. Replace:
- `INPUT_FOLDER` with path to folder containing test images
- `OUTPUT_FOLDER` with path where predictions will be saved
- `DATASET_ID` with your dataset ID
- `CONFIGURATION` with the configuration you trained (e.g., `3d_fullres`)

### Basic Inference Command

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -tr PaNSegNet_Trainer --save_probabilities
```

**Example:**
```bash
nnUNetv2_predict -i /path/to/test/images -o /path/to/predictions -d 001 -c 3d_fullres -tr PaNSegNet_Trainer --save_probabilities
```

**Notes:**
- By default, inference uses all 5 folds as an ensemble (recommended)
- `--save_probabilities` saves probability maps (needed for ensembling)
- The `-tr PaNSegNet_Trainer` flag ensures the correct architecture is used (can be auto-detected from checkpoint, but recommended to specify)
- Test images must follow nnUNet naming convention (see [dataset format](nnUNet/documentation/dataset_format_inference.md))

### Optional: Find Best Configuration

After training all folds, find the best configuration:

```bash
nnUNetv2_find_best_configuration DATASET_ID -c CONFIGURATION
```

**Example:**
```bash
nnUNetv2_find_best_configuration 001 -c 3d_fullres
```

This will print the exact inference commands you should use.

### Optional: Ensemble Multiple Configurations

If you trained multiple configurations and want to ensemble them:

```bash
nnUNetv2_ensemble -i FOLDER1 FOLDER2 ... -o OUTPUT_FOLDER -np NUM_PROCESSES
```

**Example:**
```bash
nnUNetv2_ensemble -i /path/to/predictions_2d /path/to/predictions_3d_fullres -o /path/to/ensemble_output -np 4
```

### Optional: Apply Postprocessing

Apply postprocessing to remove small connected components:

```bash
nnUNetv2_apply_postprocessing -i FOLDER_WITH_PREDICTIONS -o OUTPUT_FOLDER --pp_pkl_file POSTPROCESSING_FILE -plans_json PLANS_FILE -dataset_json DATASET_JSON_FILE
```

The postprocessing file location will be shown by `nnUNetv2_find_best_configuration`.

---

## Quick Reference

| Step | Command |
|------|---------|
| **Preprocess** | `nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity` |
| **Train** | `nnUNetv2_train DATASET_ID CONFIGURATION FOLD -tr PaNSegNet_Trainer --npz` |
| **Inference** | `nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -tr PaNSegNet_Trainer --save_probabilities` |

**Key Flag:**
- Always use `-tr PaNSegNet_Trainer` to use the PaNSegNet architecture

---

## Troubleshooting

- **"No labels found" error**: Make sure your mask files have integer labels (0, 1, 2, ...). Use the [fix_mask_labels.py](../Pancreatitis_Classification/Radiomics/fix_mask_labels.py) script if needed.
- **CUDA out of memory**: Reduce batch size in plans or use a smaller configuration (2d instead of 3d_fullres)
- **Trainer not found**: Make sure you're in the correct environment and nnUNetv2 is installed with `pip install -e .`

For more details, see the [full nnUNet documentation](nnUNet/documentation/how_to_use_nnunet.md).

