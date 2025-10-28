#!/usr/bin/env python3
"""
Chest Post-Processing Pipeline

This script combines sternum noise reduction and first rib tubercle recovery
into a single pipeline for efficient post-processing of chest segmentations.
"""

import os
import argparse
import nibabel as nib
import SimpleITK as sitk

# Import functions from existing modules
from sternum_noise_reduction import keep_largest_component
from tubercle_recovery import repair_first_rib

def process_sternum(input_path, output_path):
    """
    Process sternum segmentation to keep only the largest component.
    
    Args:
        input_path: Path to the input sternum segmentation file
        output_path: Path to save the processed sternum segmentation
    """
    print(f"Processing sternum: {input_path}")
    
    # Load the sternum segmentation
    img = nib.load(input_path)
    data = img.get_fdata()
    
    # Keep only largest connected sternum component
    cleaned = keep_largest_component(data)
    
    # Save cleaned mask
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_img = nib.Nifti1Image(cleaned, affine=img.affine, header=img.header)
    nib.save(new_img, output_path)
    
    print(f"Saved cleaned sternum: {output_path}")
    return output_path

def process_first_rib(ct_path, rib_path, output_path):
    """
    Process first rib to recover tubercles.
    
    Args:
        ct_path: Path to the CT scan
        rib_path: Path to the first rib segmentation
        output_path: Path to save the processed first rib segmentation
    """
    print(f"Processing first rib: {rib_path}")
    
    # Call the repair_first_rib function
    repair_first_rib(sitk.ReadImage(ct_path), rib_path, output_path)
    
    print(f"Saved repaired first rib: {output_path}")
    return output_path

def run_pipeline_single_subject(ct_path, segmentation_dir, output_dir):
    """
    Run the complete chest post-processing pipeline for a single subject.
    
    Args:
        ct_path: Path to the CT scan
        segmentation_dir: Directory containing segmentation files
        output_dir: Directory to save processed segmentations
    """
    import shutil
    os.makedirs(output_dir, exist_ok=True)
    
    # Process sternum
    sternum_path = os.path.join(segmentation_dir, "sternum.nii.gz")
    if os.path.exists(sternum_path):
        sternum_output = os.path.join(output_dir, "sternum.nii.gz")
        process_sternum(sternum_path, sternum_output)
    else:
        print(f"Warning: Sternum file not found at {sternum_path}")
    
    # Process first ribs (left and right) and copy all other ribs
    for side in ["left", "right"]:
        # Process all ribs (1-12) for each side
        for rib_num in range(1, 13):
            rib_filename = f"rib_{side}_{rib_num}.nii.gz"
            rib_path = os.path.join(segmentation_dir, rib_filename)
            
            if not os.path.exists(rib_path):
                print(f"Warning: Rib file not found at {rib_path}")
                continue
                
            rib_output = os.path.join(output_dir, rib_filename)
            
            # Special processing for first ribs
            if rib_num == 1:
                try:
                    process_first_rib(ct_path, rib_path, rib_output)
                    print(f"Processed first rib: {rib_filename}")
                except Exception as e:
                    print(f"Error processing first rib {rib_filename}: {str(e)}")
                    # If processing fails, still copy the original file
                    shutil.copy2(rib_path, rib_output)
                    print(f"Copied original first rib: {rib_filename}")
            else:
                # For ribs 2-12, just copy the files
                try:
                    shutil.copy2(rib_path, rib_output)
                    print(f"Copied rib: {rib_filename}")
                except Exception as e:
                    print(f"Error copying rib {rib_filename}: {str(e)}")
    
    # Copy any other segmentation files that might exist
    for filename in os.listdir(segmentation_dir):
        if filename.endswith(".nii.gz") and not (filename.startswith("rib_") or filename == "sternum.nii.gz"):
            src_path = os.path.join(segmentation_dir, filename)
            dst_path = os.path.join(output_dir, filename)
            if not os.path.exists(dst_path):  # Skip if already processed
                try:
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied additional file: {filename}")
                except Exception as e:
                    print(f"Error copying file {filename}: {str(e)}")
    
    return True

def run_pipeline(ct_root, segmentation_root, output_root):
    """
    Run the complete chest post-processing pipeline for all subjects.
    
    Args:
        ct_root: Root directory containing CT scans for all subjects
        segmentation_root: Root directory containing segmentation files for all subjects
        output_root: Root directory to save processed segmentations for all subjects
    """
    os.makedirs(output_root, exist_ok=True)
    
    # Get list of all subjects
    subjects = []
    for item in os.listdir(segmentation_root):
        subject_dir = os.path.join(segmentation_root, item)
        if os.path.isdir(subject_dir):
            subjects.append(item)
    
    if not subjects:
        print("No subject directories found in the segmentation root directory.")
        return
    
    print(f"Found {len(subjects)} subjects to process.")
    
    # Process each subject
    successful = 0
    failed = 0
    
    for i, subject in enumerate(subjects):
        print(f"\n[{i+1}/{len(subjects)}] Processing subject: {subject}")
        
        # Find CT scan for this subject
        # ct_path = None
        # for ct_file in os.listdir(os.path.join(ct_root, subject + "_resampled.nii.gz")):
        #     if ct_file.endswith(".nii.gz") and not any(x in ct_file for x in ["seg", "label", "mask"]):
        #         ct_path = os.path.join(ct_root, subject, ct_file)
        #         break
        ct_path = os.path.join(ct_root, subject + "_resampled.nii.gz")
        
        if not ct_path:
            print(f"Warning: No CT scan found for subject {subject}, skipping...")
            failed += 1
            continue
        
        # Set up paths
        segmentation_dir = os.path.join(segmentation_root, subject)
        output_dir = os.path.join(output_root, subject)
        
        try:
            # Process this subject
            success = run_pipeline_single_subject(ct_path, segmentation_dir, output_dir)
            if success:
                successful += 1
                print(f"✅ Successfully processed subject {subject}")
            else:
                failed += 1
                print(f"❌ Failed to process subject {subject}")
        except Exception as e:
            failed += 1
            print(f"❌ Error processing subject {subject}: {str(e)}")
    
    print(f"\nPipeline completed: {successful} subjects processed successfully, {failed} failed.")

def main():
    parser = argparse.ArgumentParser(description="Chest Post-Processing Pipeline")
    parser.add_argument("--ct_root", required=True, help="Root directory containing CT scans for all subjects")
    parser.add_argument("--segmentation_root", required=True, help="Root directory containing segmentation files for all subjects")
    parser.add_argument("--output_root", required=True, help="Root directory to save processed segmentations for all subjects")
    parser.add_argument("--single_subject", action="store_true", help="Process a single subject instead of batch processing")
    parser.add_argument("--subject_id", help="Subject ID to process (required if --single_subject is used)")
    
    args = parser.parse_args()
    
    if args.single_subject:
        if not args.subject_id:
            print("Error: --subject_id is required when using --single_subject")
            return
        
        # Process single subject
        ct_path = None
        for ct_file in os.listdir(os.path.join(args.ct_root, args.subject_id)):
            if ct_file.endswith(".nii.gz") and not any(x in ct_file for x in ["seg", "label", "mask"]):
                ct_path = os.path.join(args.ct_root, args.subject_id, ct_file)
                break
        
        if not ct_path:
            print(f"Error: No CT scan found for subject {args.subject_id}")
            return
        
        segmentation_dir = os.path.join(args.segmentation_root, args.subject_id)
        output_dir = os.path.join(args.output_root, args.subject_id)
        
        success = run_pipeline_single_subject(ct_path, segmentation_dir, output_dir)
        if success:
            print(f"✅ Successfully processed subject {args.subject_id}")
        else:
            print(f"❌ Failed to process subject {args.subject_id}")
    else:
        # Process all subjects
        run_pipeline(args.ct_root, args.segmentation_root, args.output_root)

if __name__ == "__main__":
    main()