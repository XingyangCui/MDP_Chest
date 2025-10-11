import SimpleITK as sitk
import numpy as np
import os
import csv
from datetime import datetime

def calculate_hausdorff_distance(volume1_path, volume2_path):
    """
    Calculates the Hausdorff Distance between two segmentation volumes.
    """
    image1 = sitk.ReadImage(volume1_path)
    image2 = sitk.ReadImage(volume2_path)
    
    image2 = sitk.Resample(image2, image1)
    
    hd_filter = sitk.HausdorffDistanceImageFilter()
    hd_filter.Execute(image1, image2)
    return hd_filter.GetHausdorffDistance()

def log_results_to_csv(log_file, timestamp, test_id, gt_file, model_file, hd_model, hd_ts, hd_variation, pass_fail):
    """
    Appends results to a CSV log file.
    """
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        if not file_exists:
            writer.writerow(['Timestamp', 'Test_ID', 'Ground_Truth_File', 'Model_Result_File', 
                           'HD_Model_mm', 'HD_TS_mm', 'HD_Variation_Percent', 'Pass_Fail'])
        
        # Write the data row
        writer.writerow([timestamp, test_id, gt_file, model_file, 
                        f"{hd_model:.4f}", f"{hd_ts:.4f}", f"{hd_variation:.2f}", pass_fail])

def main():
    """
    Main function to execute the smoothness validation test.
    """
    print("--- Ribcage Smoothness Validation ---")
    
    # Update these paths with your real paths
    ground_truth_path = "/path/to/ground/truth/data"
    model_result_path = "/path/to/retrained/model/data"
    ts_result_path = "/path/to/totalsegmentator/data"
    
    # CSV log file path
    log_file_path = "/path/to/smoothness_validation_log.csv"
    
    # Test identification. Update this for each test run (optional)
    test_id = "T-001"
    
    # Calculate the key distances
    print("Calculating Hausdorff Distances...")
    hd_model = calculate_hausdorff_distance(ground_truth_path, model_result_path)
    hd_ts = calculate_hausdorff_distance(ground_truth_path, ts_result_path)
    
    print(f"Hausdorff Distance (Model vs. Ground Truth): {hd_model:.4f} mm")
    print(f"Hausdorff Distance (TotalSegmentator vs. Ground Truth): {hd_ts:.4f} mm")
    
    # Calculate the variation (improvement)
    if hd_ts != 0:
        hd_variation = ((hd_ts - hd_model) / hd_ts) * 100
    else:
        hd_variation = 0

    print(f"Hausdorff Distance Variation (Improvement): {hd_variation:.2f}%")
    
    # Determine Pass/Fail
    if hd_variation >= 30:
        result = "PASS"
        print(f"\n>>> RESULT: {result}")
    else:
        result = "FAIL"
        print(f"\n>>> RESULT: {result}")
    
    # Log results to CSV
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_results_to_csv(log_file_path, timestamp, test_id, 
                      os.path.basename(ground_truth_path), 
                      os.path.basename(model_result_path),
                      hd_model, hd_ts, hd_variation, result)
    
    print(f"Results logged to: {log_file_path}")
    
    return hd_model, hd_ts, hd_variation, result

if __name__ == "__main__":
    hd_model, hd_ts, hd_variation, result = main()
