import os
import nibabel as nib
import numpy as np
from scipy import ndimage

def keep_largest_component(data):
    """
    Keep only the largest connected component in a binary segmentation.
    data: 3D numpy array (non-zero = foreground)
    """
    # Convert to binary mask
    mask = (data > 0).astype(np.uint8)

    # Label connected components
    labeled, num_features = ndimage.label(mask)

    if num_features == 0:
        return np.zeros_like(data)

    # Get sizes of components
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))

    # Largest component index
    largest_idx = (np.argmax(sizes) + 1)

    # Keep only the largest
    largest_component = (labeled == largest_idx).astype(np.uint8)

    return largest_component


# Input/output folders
input_folder = "/Users/anish/UMTRI-TotalSegmentator/data/totalsegmentator_results/relabeled"
output_folder = "/Users/anish/UMTRI-TotalSegmentator/post-processing/sternum/sternum_noisereduction/results"
os.makedirs(output_folder, exist_ok=True)

for subject in os.listdir(input_folder):
    file = f"{subject.split('_')[0]}_sternum.nii.gz"
    if file.endswith(".nii.gz"):
        filepath = os.path.join(input_folder, subject, file)

        if not os.path.exists(filepath):
            print(f"🚨 Missing file {file}, skipping...")
            continue

        img = nib.load(filepath)
        data = img.get_fdata()

        # Keep only largest connected sternum component
        cleaned = keep_largest_component(data)

        # Save cleaned mask
        new_img = nib.Nifti1Image(cleaned, affine=img.affine, header=img.header)
        outpath = os.path.join(output_folder, f"{file.replace('.nii.gz','')}_processed.nii.gz")
        nib.save(new_img, outpath)

        print(f"Saved cleaned mask for {file} -> {outpath}")
