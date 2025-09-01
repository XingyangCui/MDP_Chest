# MDP_Chest


A pipeline that after segmentation major anatomical structures by CT images, how to solver the mislabeling problems.  

---
#STEPS
### 🔧 Step 1: Retrain the New Model


---

### 📁 Step 2: Predict the results by test cases



### ⚙️ Step 3: Relabeling the segmentation results

Use the nnU-Net preprocessing tool:
```bash
nnUNetv2_plan_and_preprocess -d <your_dataset_id> -pl ExperimentPlanner -c 3d_fullres -np 2
```
[Preprocess Code](https://github.com/XingyangCui/UMTRI_3D_Segmentation/blob/main/Code/Preprocess.ipynb)

To run data preprocessing with nnUNetv2, you need to set up the environment paths and execute the planning and preprocessing pipeline for your dataset.

✅ 1. Set Environment Variables
```bash
import os

os.environ["nnUNet_raw"] = "your/path/to/raw_data"                # Folder containing your DatasetXXX folder
os.environ["nnUNet_preprocessed"] = "your/path/to/preprocessed"  # Where preprocessed data will be stored
os.environ["nnUNet_results"] = "your/path/to/results"            # Directory for trained models and logs
```

✅ 2. Preprocess code
```bash
# Optional environment settings to avoid locale errors
env_vars = {
    "LC_ALL": "C.UTF-8",
    "LANG": "C.UTF-8"
}

# Build the nnU-Net planning command
command = [
    "nnUNetv2_plan_and_preprocess",
    "-d", "001",      # Replace with your Dataset ID (e.g., 001, 002, etc.)
    "-np", "6"        # Number of preprocessing threads (adjust to your CPU)
]
```
### Run the command with both system and custom environment variables
subprocess.run(command, env={**env_vars, **dict(os.environ)}, text=True)

### 🧠 Step 4: Cropped Data(not mandatory)
Sometimes the images is too big that will cause system shut down due to lack of space. So it's critical to cut the whole body images into the part you need only.
Forexample for the Foot&Ankle data, we only need the second half of the body, but the CT Scans is look like this:
<img src="Images/2.png" alt="Example Image" width="600"/>

We need to tur it into and then taining:
<img src="Images/3.png" alt="Example Image" width="600"/>


[Cropped Code](https://github.com/XingyangCui/UMTRI_3D_Segmentation/blob/main/Code/Cropped.ipynb)
🛠️ Features
Batch processing: Automatically loops over patient IDs.
Orientation-aware cropping: Determines cropping direction based on the affine matrix.
Safe I/O: Skips missing files and handles exceptions.
Output management: Saves cropped files to a subdirectory named test/.

**** Suppose your dataset includes full-foot CT volumes like:
```bash
LM7100_resampled.nii.gz
LM7101_resampled.nii.gz
```

This script will:
- Load each file
- Crop it along the Z-axis (keeping either top or bottom half)
- Save the result as:
```bash
LM7100_cropped.nii.gz → in /test/ subfolder
```


### 🧠 Step 5: Train the Model

Use the `nnUNetTrainerNoMirroring` trainer to start training:

```bash
nnUNetv2_train <your_dataset_id> 3d_fullres 0 -tr nnUNetTrainerNoMirroring
```
<your_dataset_id>: Replace with your dataset ID (e.g., 002)

* 3d_fullres: Configuration for full-resolution 3D training

* 0: Fold number (typically 0 for default)

* -tr: Specifies the trainer to use

⏱️ Training may take several days depending on your hardware setup.

### 🔍 Step 6: Predict on the Test Set

Use the trained nnU-Net model to predict segmentations on the test set:

```bash
nnUNetv2_predict -i path/to/imagesTs \
                 -o path/to/labelsTs_predicted \
                 -d <your_dataset_id> \
                 -c 3d_fullres \
                 -tr nnUNetTrainerNoMirroring \
                 --disable_tta -f 0
```
* -i: Input directory of test images (e.g., imagesTs)

* -o: Output directory for predicted labels

* -d: Dataset ID (e.g., 002)

* -c: Configuration (e.g., 3d_fullres)

* -tr: Trainer used (e.g., nnUNetTrainerNoMirroring)

* --disable_tta: Disables test-time augmentation

* -f 0: Predict with fold 0


### 📊 Step 7: Evaluate Predictions

To evaluate the predicted segmentation results against ground truth labels, follow the steps below.

#### 1. Install Required Dependencies

```bash
pip install git+https://github.com/google-deepmind/surface-distance.git
pip install p_tqdm
```
#### 2. Run Evaluation Script
```bash
python resources/evaluate.py path/to/labelsTs path/to/labelsTs_predicted
```
* path/to/labelsTs: Directory containing the ground-truth labels.
* path/to/labelsTs_predicted: Directory containing the predicted segmentations.
This script will compute surface-based Dice scores and other metrics.
📄 Results can be compared with the baseline in resources/evaluate_results.txt
🎯 Note: Due to non-deterministic training, average Dice scores may vary by approximately ±1 point.

### 📊 Step 8: Done!!!

> Note: This will not give you the same results as 3D sgementation for two reasons:
1. 3D segmentation uses a bigger dataset which is not completely public
2. The origional model parameters use for training is different.
