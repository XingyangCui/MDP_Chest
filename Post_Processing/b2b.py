#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This method is intended for repairing large holes only and should be used selectively.

import os
import numpy as np
import nibabel as nib
import argparse


def bezier_curve(p0, p1, p2, n_points=50):
    t = np.linspace(0, 1, n_points)[:, None]
    curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    return curve.astype(int)

def create_curved_box_mask(shape, p1, p2,
                           half_size_x=5, half_size_y=2, half_size_z=1,
                           curve_strength=0, rotate_deg=0):
    """
    Generate a flat rectangular cuboid rod with curvature and rotation in IJK index space.

    :param shape: Volume shape (z, y, x)
    :param p1, p2: IJK voxel coordinates (z, y, x)
    :param half_size_x: Half-width along the x direction (voxels, longer horizontally)
    :param half_size_y: Half-width along the y direction (voxels, flatter)
    :param half_size_z: Half-width along the z direction (voxels, thickness)
    :param curve_strength: Curvature strength; 0 = straight, >0 = curved inward, <0 = curved outward
    :param rotate_deg: Rotation angle in the xy plane (degrees, positive = counterclockwise, negative = clockwise)
    """
    mask = np.zeros(shape, dtype=np.uint8)

    p1 = np.array(p1, dtype=int)
    p2 = np.array(p2, dtype=int)

    if curve_strength == 0:
        num_points = int(np.linalg.norm(p2 - p1)) + 1
        curve_points = np.linspace(p1, p2, num_points).astype(int)
    else:
        mid = (p1 + p2) / 2
        offset = np.array([0, curve_strength, 0])  # y 方向偏移
        p_mid = mid + offset
        curve_points = bezier_curve(p1, p_mid, p2, n_points=100)

    theta = np.deg2rad(rotate_deg)
    rot = np.array([[ np.cos(theta), -np.sin(theta)],
                    [ np.sin(theta),  np.cos(theta)]])

    for z, y, x in curve_points:
        if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
            continue

        for dz in range(-half_size_z, half_size_z+1):
            zz = z + dz
            if not (0 <= zz < shape[0]):
                continue
            for dy in range(-half_size_y, half_size_y+1):
                for dx in range(-half_size_x, half_size_x+1):
                    vec = np.dot(rot, np.array([dx, dy]))
                    yy = int(y + vec[1])
                    xx = int(x + vec[0])
                    if (0 <= yy < shape[1]) and (0 <= xx < shape[2]):
                        mask[zz, yy, xx] = 1

    return mask

def repair_rib(input_path, output_dir, p1, p2,
               half_size_x=5, half_size_y=2, half_size_z=1,
               curve_strength=0, rotate_deg=0):
    """
    repair segmentation and output result+mask
    """
    os.makedirs(output_dir, exist_ok=True)

    nii = nib.load(input_path)
    data = nii.get_fdata().astype(np.uint8)

    box_mask = create_curved_box_mask(data.shape, p1, p2,
                                      half_size_x, half_size_y, half_size_z,
                                      curve_strength, rotate_deg)

    repaired = np.maximum(data, box_mask)

    repaired_path = os.path.join(output_dir, "repaired_rib.nii.gz")
    nib.save(nib.Nifti1Image(repaired, nii.affine, nii.header), repaired_path)

    box_path = os.path.join(output_dir, "box_mask.nii.gz")
    nib.save(nib.Nifti1Image(box_mask, nii.affine, nii.header), box_path)

    print(f"✅ finishde repair\n  Rib: {repaired_path}\n  Box: {box_path}")

def main():
    parser = argparse.ArgumentParser(description="Repair rib segmentation with seed growing.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .nii.gz file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--p1", type=int, nargs=3, required=True, help="First probe point (x y z)")
    parser.add_argument("--p2", type=int, nargs=3, required=True, help="Second probe point (x y z)")
    parser.add_argument("--half_size_x", type=int, default=5, help="Half size in x direction")
    parser.add_argument("--half_size_y", type=int, default=1, help="Half size in y direction")
    parser.add_argument("--half_size_z", type=int, default=1, help="Half size in z direction")
    parser.add_argument("--curve_strength", type=int, default=8, help="Curve strength")
    parser.add_argument("--rotate_deg", type=int, default=0, help="Rotation degree")

    args = parser.parse_args()

    repair_rib(
        args.input,
        args.output_dir,
        tuple(args.p1),
        tuple(args.p2),
        half_size_x=args.half_size_x,
        half_size_y=args.half_size_y,
        half_size_z=args.half_size_z,
        curve_strength=args.curve_strength,
        rotate_deg=args.rotate_deg
    )

if __name__ == "__main__":
    main()


# sample usage:

# python3 b2b.py \
#   --input ../data/CM7109/rib_left_8.nii.gz \
#   --output_dir ../b2b_output/CM7109/ \
#   --p1 327 365 238 \
#   --p2 361 343 228 \
#   --half_size_x 5 --half_size_y 1 --half_size_z 1 \
#   --curve_strength 8 --rotate_deg 0

    # CF5008_left_rib_3
    # p1_probe = (300, 97, 290)
    # p2_probe = (278, 88, 281)

    # CF7109_left_rib_8
    # p1_probe = (327, 365, 238)
    # p2_probe = (361, 343, 228)