# MDP_Chest

MDP_Chest is a modular pipeline designed to post-process segmentation results of major anatomical structures (especially the rib cage) from chest CT images. This pipeline focuses on resolving mislabeling, inconsistent rib ordering, and segmentation noise, ensuring anatomical consistency and enabling accurate downstream evaluation (e.g., Dice scores).

📌 Key Features
Noise removal based on connected components
Anatomical relabeling of ribs (1–24, right to left, superior to inferior)
Re-mapping between predicted labels and ground truth
Built-in evaluation & visualization for Dice scores per case and per rib


#STEPS
### 🔧 Step 1: Preprocess data & Retrain Model.ipynb
- Preprocess original CT images (orientation, resampling, normalization)
- Organize datasets into nnUNet format with consistent label naming
- Fine-tune a segmentation model (e.g., TotalSegmentator) with custom ground truth
- Export model checkpoints for inference

[1.Preprocess data & Retrain Model.ipynb](https://github.com/XingyangCui/MDP_Chest/blob/main/Retraining/1.Preprocess%20data%20%26%20Retrain%20Model.ipynb)

---

### 📁 Step 2: Predict the results by test cases
- Apply the trained model to test CT scans
- Export predicted segmentation maps
- Filter out irrelevant structures (e.g., sternum, cartilage)

[2.Predict_Test_Cases](https://github.com/XingyangCui/MDP_Chest/blob/main/Retraining/2.Predict_Test_Cases.ipynb)


### ⚙️ Step 3: Merge_Ground_Truth_file
- Merge individual rib ground truth masks into a single 3D volume
- Optionally filter out noisy labels or non-rib structures
- Output: a clean multi-label ground truth file for each case
  
[3.Merge_Ground_Truth_file](https://github.com/XingyangCui/MDP_Chest/blob/main/Retraining/3.Merge_Ground_Truth_file.ipynb)




### 🧠 Step 4: Sort Label and exclude noises
- Apply 3D connected component analysis to predicted rib masks
- Retain the top 24 largest components (i.e., 24 ribs)
- Sort them anatomically by:
- Side (left vs right, based on x-axis center)
- Vertical position (superior to inferior)
- Assign consistent labels:
- rib_1 to rib_12 → right ribs,
- rib_13 to rib_24 → left ribs,
  
- Then add other 2 labels:
- 25: sternum
- 26: costal_cartilages
  
[4.Sort Label and exclude noises](https://github.com/XingyangCui/MDP_Chest/blob/main/Retraining/4.Sort%20Label%20and%20exclude%20noises.ipynb)


### 🧠 Step 5: Relabel(not mandatory)




### 🔍 Step 6: Evaluate the Performance
- Compare predicted rib labels against ground truth masks
- Compute Dice scores per rib, per case, and overall
  
[6.Test_new_Dice_Scores.ipynb](https://github.com/XingyangCui/MDP_Chest/blob/main/Retraining/6.Test_new_Dice_Scores.ipynb)
