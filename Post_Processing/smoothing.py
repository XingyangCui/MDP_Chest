import os
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_closing

# Morphological smoothing for all patients
# Update to patient data directory
base_dir = "/path/to/patient/data"

# List of all patient IDs
# Example: 'CF5008', 'CF6031', etc.
# Folder named 'patientID' (e.g. 'CF5008') should contain labeled.nii.gz
patient_ids = []

for patient_id in patient_ids:
    patient_dir = os.path.join(base_dir, patient_id)
    
    # Check if patient directory exists
    if not os.path.exists(patient_dir):
        print(f"Directory not found: {patient_dir}, skipping...")
        continue
        
    # File paths
    # Path to segmentation file. Here, the retrained result
    seg_path = os.path.join(patient_dir, "labeled.nii.gz")
    
    # Check if segmentation file exists
    if not os.path.exists(seg_path):
        print(f"Segmentation file not found: {seg_path}, skipping...")
        continue
    
    try:
        # Load segmentation
        seg = nib.load(seg_path)
        seg_data = seg.get_fdata().astype(np.int16)
        affine = seg.affine
        
        print(f"Processing {patient_id}...")
        
        # Apply morphological closing (smooths surface without thickening)
        smoothed_morph = binary_closing(seg_data, iterations=1).astype(np.int16)
        output_path = os.path.join(patient_dir, f"{patient_id}_smoothed.nii.gz")
        nib.save(nib.Nifti1Image(smoothed_morph, affine), output_path)
        print(f"Smoothing complete: {output_path}")
        
    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        continue

print("Finished processing all patients!")