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
    
    # Convert both images to the same type (8-bit unsigned integer)
    image1 = sitk.Cast(image1, sitk.sitkUInt8)
    image2 = sitk.Cast(image2, sitk.sitkUInt8)
    
    image2 = sitk.Resample(image2, image1)
    
    hd_filter = sitk.HausdorffDistanceImageFilter()
    hd_filter.Execute(image1, image2)
    return hd_filter.GetHausdorffDistance()

def classify_failure_severity(hd_variation):
    """
    Classifies failures by severity to guide manual intervention
    """
    if hd_variation >= 30:
        return "PASS", "No intervention needed"
    elif 20 <= hd_variation < 30:
        return "FAIL", "MILD - Optional smoothing"
    elif 10 <= hd_variation < 20:
        return "FAIL", "MODERATE - Recommended smoothing"
    else:
        return "FAIL", "SEVERE - Manual correction required"

def process_single_patient(patient_id, base_directory, output_directory):
    """
    Process a single patient and return results
    """
    print(f"Processing patient: {patient_id}")
    
    # Construct file paths with the correct naming structure
    # UPDATE THESE NAMES BASED ON FILES
    ground_truth_path = f"{base_directory}/{patient_id}/{patient_id}_ribcage_segmentationGT.nii.gz"
    model_result_path = f"{base_directory}/{patient_id}/labeled_combined.nii.gz"
    ts_result_path = f"{base_directory}/{patient_id}/{patient_id}_ribcage_segmentation.nii.gz"
    
    # Check if all required files exist
    missing_files = []
    for path, desc in [(ground_truth_path, "Ground Truth"), 
                       (model_result_path, "Model Result"), 
                       (ts_result_path, "TotalSegmentator")]:
        if not os.path.exists(path):
            missing_files.append(f"{desc}: {os.path.basename(path)}")
    
    if missing_files:
        print(f"  ❌ Missing files for {patient_id}:")
        for missing in missing_files:
            print(f"     - {missing}")
        
        # Show what files are actually in the directory
        patient_dir = f"{base_directory}/{patient_id}"
        if os.path.exists(patient_dir):
            actual_files = os.listdir(patient_dir)
            print(f"     Files in directory: {actual_files}")
        return None
    
    try:
        # Calculate Hausdorff distances
        hd_model = calculate_hausdorff_distance(ground_truth_path, model_result_path)
        hd_ts = calculate_hausdorff_distance(ground_truth_path, ts_result_path)
        
        # Calculate improvement
        if hd_ts != 0:
            hd_variation = ((hd_ts - hd_model) / hd_ts) * 100
        else:
            hd_variation = 0
        
        # Determine result
        result, recommendation = classify_failure_severity(hd_variation)
        
        print(f"  ✅ {patient_id}: {result} ({hd_variation:.1f}% improvement)")
        print(f"     Model HD: {hd_model:.2f}mm, TS HD: {hd_ts:.2f}mm")
        
        return {
            'patient_id': patient_id,
            'hd_model': hd_model,
            'hd_ts': hd_ts,
            'hd_variation': hd_variation,
            'result': result,
            'recommendation': recommendation,
            'ground_truth_file': os.path.basename(ground_truth_path),
            'model_file': os.path.basename(model_result_path),
            'ts_file': os.path.basename(ts_result_path)
        }
        
    except Exception as e:
        print(f"  ❌ Error processing {patient_id}: {str(e)}")
        return None

def generate_batch_report(results, output_directory):
    """
    Generate a comprehensive batch validation report
    """
    report_path = f"{output_directory}/smoothness_batch_validation_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("BATCH SMOOTHNESS VALIDATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Patients Processed: {len(results)}\n\n")
        
        # Calculate statistics
        passed = [r for r in results if r['result'] == 'PASS']
        failed = [r for r in results if r['result'] == 'FAIL']
        pass_rate = (len(passed) / len(results)) * 100 if results else 0
        avg_improvement = sum(r['hd_variation'] for r in results) / len(results) if results else 0
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Pass Rate: {pass_rate:.1f}% ({len(passed)}/{len(results)} patients)\n")
        f.write(f"Average Improvement: {avg_improvement:.1f}%\n")
        f.write(f"Failed Cases: {len(failed)}\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Patient ID':<12} {'Result':<8} {'Improvement':<12} {'HD Model':<10} {'HD TS':<10} {'Recommendation'}\n")
        f.write("-" * 80 + "\n")
        
        for result in sorted(results, key=lambda x: x['patient_id']):
            f.write(f"{result['patient_id']:<12} {result['result']:<8} {result['hd_variation']:>11.1f}% {result['hd_model']:>9.2f} {result['hd_ts']:>9.2f} {result['recommendation']}\n")
        
        # Failure analysis
        if failed:
            f.write(f"\nFAILURE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            mild_failures = [r for r in failed if "MILD" in r['recommendation']]
            moderate_failures = [r for r in failed if "MODERATE" in r['recommendation']]
            severe_failures = [r for r in failed if "SEVERE" in r['recommendation']]
            
            f.write(f"MILD failures (optional smoothing): {len(mild_failures)}\n")
            f.write(f"MODERATE failures (recommended smoothing): {len(moderate_failures)}\n")
            f.write(f"SEVERE failures (manual correction needed): {len(severe_failures)}\n")
            
            if severe_failures:
                f.write(f"\nPatients requiring manual correction:\n")
                for patient in severe_failures:
                    f.write(f"  - {patient['patient_id']} ({patient['hd_variation']:.1f}% improvement)\n")
        
        # File information
        f.write(f"\nFILE INFORMATION:\n")
        f.write("-" * 40 + "\n")
        for result in results:
            f.write(f"\n{result['patient_id']}:\n")
            f.write(f"  Ground Truth: {result['ground_truth_file']}\n")
            f.write(f"  Model Result: {result['model_file']}\n")
            f.write(f"  TotalSegmentator: {result['ts_file']}\n")

def save_batch_csv(results, output_directory):
    """
    Save batch results to CSV file
    """
    csv_path = f"{output_directory}/smoothness_batch_validation.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Patient_ID', 'HD_Model_mm', 'HD_TS_mm', 'HD_Variation_Percent', 
            'Pass_Fail', 'Recommendation', 'Ground_Truth_File', 'Model_File', 'TS_File'
        ])
        
        # Write data rows
        for result in sorted(results, key=lambda x: x['patient_id']):
            writer.writerow([
                result['patient_id'],
                f"{result['hd_model']:.4f}",
                f"{result['hd_ts']:.4f}",
                f"{result['hd_variation']:.2f}",
                result['result'],
                result['recommendation'],
                result['ground_truth_file'],
                result['model_file'],
                result['ts_file']
            ])
    
    return csv_path

def main():
    """
    Main function for batch processing multiple patients
    """
    print("=== BATCH RIB CAGE SMOOTHNESS VALIDATION ===\n")
    
    # Configuration - UPDATE THESE PATHS
    base_directory = "/path/to/base/directory/"
    output_directory = "/path/to/output/directory/batch_validation_results"
    
    # List of patient IDs to process - UPDATE THIS LIST, ADD ALL PATIENTS FOR PROCESSING
    patient_ids = [
        "CF2002"
    ]
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"Base directory: {base_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Patients to process: {len(patient_ids)}\n")
    print("Expected file structure per patient:")
    # UPDATE THESE NAMES BASED ON FILES
    print(f"  {base_directory}/PATIENT_ID/")
    print(f"    ├── PATIENT_ID_ribcage_segmentationGT.nii.gz")
    print(f"    ├── labeled_combined.nii.gz")
    print(f"    └── PATIENT_ID_ribcage_segmentation.nii.gz")
    print()
    
    # Process all patients
    results = []
    for patient_id in patient_ids:
        result = process_single_patient(patient_id, base_directory, output_directory)
        if result:
            results.append(result)
    
    print(f"\n" + "="*50)
    print(f"PROCESSING COMPLETE")
    print(f"="*50)
    
    if results:
        # Generate reports
        generate_batch_report(results, output_directory)
        csv_path = save_batch_csv(results, output_directory)
        
        # Print summary
        passed = [r for r in results if r['result'] == 'PASS']
        failed = [r for r in results if r['result'] == 'FAIL']
        pass_rate = (len(passed) / len(results)) * 100
        
        print(f"Successfully processed: {len(results)}/{len(patient_ids)} patients")
        print(f"Pass rate: {pass_rate:.1f}% ({len(passed)} passed, {len(failed)} failed)")
        print(f"\nReports generated:")
        print(f"  - {output_directory}/smoothness_batch_validation_report.txt")
        print(f"  - {csv_path}")
        
        # Show failure breakdown
        if failed:
            mild = len([r for r in failed if "MILD" in r['recommendation']])
            moderate = len([r for r in failed if "MODERATE" in r['recommendation']])
            severe = len([r for r in failed if "SEVERE" in r['recommendation']])
            print(f"\nFailure severity breakdown:")
            print(f"  - MILD (optional): {mild} patients")
            print(f"  - MODERATE (recommended): {moderate} patients") 
            print(f"  - SEVERE (required): {severe} patients")
    else:
        print("No patients were successfully processed. Please check file paths and patient IDs.")

if __name__ == "__main__":
    main()