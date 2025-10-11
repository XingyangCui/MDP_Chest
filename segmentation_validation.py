import os
import re
import numpy as np
import nibabel as nib
import pandas as pd
from datetime import datetime

def dice_score(pred, gt):
    """Compute Dice Similarity Coefficient for binary masks."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    if pred.sum() + gt.sum() == 0:
        return 1.0  # both empty, define as perfect
    return 2.0 * intersection / (pred.sum() + gt.sum())

def process_case(case_id, gt_root, pred_root):
    """
    Compute Dice scores for one test case (all ribs + sternum).
    """
    gt_case_folder = os.path.join(gt_root, case_id, "segmentations")
    pred_case_folder = os.path.join(pred_root, case_id)
    
    if not os.path.isdir(gt_case_folder) or not os.path.isdir(pred_case_folder):
        raise FileNotFoundError(f"Missing folders for case {case_id}")
    
    # Match components by filename
    components = sorted(os.listdir(pred_case_folder))  # e.g., ['sternum.nii.gz', 'rib_right_1.nii.gz', ...]
    print(components)

    results = {}
    overall_gt = None
    overall_pred = None
    
    for comp in components:
        gt_path = os.path.join(gt_case_folder, comp)
        pred_path = os.path.join(pred_case_folder, comp)
        
        if not os.path.exists(gt_path):
            print(f"⚠️ Warning: Missing predicted file {pred_path}, skipping")
            continue
        
        gt_img = nib.load(gt_path).get_fdata()
        pred_img = nib.load(pred_path).get_fdata()
        
        comp_name = re.sub(r'\.nii\.gz$', '', comp)
        score = dice_score(pred_img, gt_img)
        results[comp_name] = round(score, 3)  # round to 3 decimals
        
        # Build overall masks
        overall_gt = gt_img.astype(bool) if overall_gt is None else np.logical_or(overall_gt, gt_img)
        overall_pred = pred_img.astype(bool) if overall_pred is None else np.logical_or(overall_pred, pred_img)
    
    # Add overall ribcage Dice
    if overall_gt is not None and overall_pred is not None:
        results["Ribcage_Overall"] = round(dice_score(overall_pred, overall_gt), 3)
    else:
        results["Ribcage_Overall"] = np.nan
    
    print(results)
    return results


def main(gt_root, pred_root, output_csv="segmentation_validation_log.csv"):
    all_components = set()
    
    # First pass: collect all possible component names across cases
    for case_id in sorted(os.listdir(pred_root)):
        case_folder = os.path.join(pred_root, case_id)
        if not os.path.isdir(case_folder):
            continue
        for comp in sorted(os.listdir(case_folder)):
            # Use regex to remove .nii.gz extension
            comp_name = re.sub(r'\.nii\.gz$', '', comp)
            all_components.add(comp_name)
    print("All components:", all_components)
    all_components.add("Ribcage_Overall")
    
    # Define fixed order
    fixed_order = [
        "Ribcage_Overall",
        "sternum",
    ] + [f"rib_left_{i}" for i in range(1, 13)] \
    + [f"rib_right_{i}" for i in range(1, 13)]

    # Now filter to keep only components that actually exist in all_components
    component_cols = [c for c in fixed_order if c in all_components]

    # Final column order
    cols = ["Timestamp", "Test_Case"] + component_cols

    # Final summary table
    summary = []
    
    # Create CSV with header if not already present
    if not os.path.exists(output_csv):
        pd.DataFrame(columns=cols).to_csv(output_csv, index=False)
    
    # Process each case and append immediately
    for case_id in sorted(os.listdir(pred_root)):

        try:
            results = process_case(case_id, gt_root, pred_root)
            results["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results["Test_Case"] = case_id
            
            # Ensure all columns exist, fill missing ones with NaN
            row = {col: results.get(col, np.nan) for col in cols}
            
            # Append row immediately
            pd.DataFrame([row]).to_csv(output_csv, mode="a", header=False, index=False)
            summary.append((case_id, row["Ribcage_Overall"]))
            
            print(f"✅ Logged {case_id}")
        except Exception as e:
            print(f"Skipping {case_id}, error: {e}")
    
    # Print summary verdict
    print("\n========== Overall Ribcage Dice Score Summary ==========")
    for case_id, score in summary:
        verdict = "PASS ✅" if (isinstance(score, float) and score >= 0.9) else "FAIL ❌"
        print(f"{case_id:10s} | {score:.3f} | {verdict}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Dice scores for ribcage segmentation cases.")
    parser.add_argument("gt_root", type=str, help="Path to Ground Truth folder")
    parser.add_argument("pred_root", type=str, help="Path to Predicted folder")
    parser.add_argument("--output", type=str, default="segmentation_validation_log.csv", help="CSV log file name")
    args = parser.parse_args()
    
    main(args.gt_root, args.pred_root, args.output)
