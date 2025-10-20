#!/usr/bin/env python3
# Only extend FIRST ribs (L/R) and write one file per rib to:
#   postprocessed_output/<subj>_<ribfile>_postprocessed.nii.gz
#
# Goal: Apply Robust Growth ONLY to High-Confidence Missing Tubercle Ribs.
#       Uses ULTRA-STRICT thresholds and HU floors for the right rib to ensure NO PROCESSING.

import os
import re
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import convolve

# -------------------- knobs (tuned for robust recovery / NO OVERSEGMENTATION) --------------------
UPPER_HU = 1200.0

# Classify posterior endpoints vs anterior (target spine-side endpoints)
NBHD_RADIUS_MM       = 12.0     # Inspection distance for a robust check
NBHD_BONE_HU         = 600.0    # HU counted as "other dense bone" (vertebrae, joints)

# CRITICAL: SIDE-SPECIFIC HARD CUTOFF THRESHOLDS
# Cutoff for the problematic side (Left) - Allows recovery
NBHD_BONE_FRAC_LEFT_CUTOFF  = 0.05 
# Ultra-Stricter Cutoff for the non-problematic side (Right) - Designed to SKIP
NBHD_BONE_FRAC_RIGHT_CUTOFF = 0.005   # ULTIMATE RESTRICTION (0.5% bone contamination max)

# Strict anatomical filtering to isolate posterior tip
POSTERIOR_Y_IDX_MIN_FRAC = 0.55 
MIDLINE_X_TOL_MM         = 50.0 

# Single Robust Growth Parameters (applied ONLY to filtered endpoints)
PCTL_ROBUST          = 40       # More permissive HU percentile for robust recovery
LOWER_PCTL_FLOOR_R   = 200.0    # Used for Left/Problematic Side
R_RIB_MM_R           = 10.0     # Wider corridor for robust recovery
R_TIP_MM_R           = 14.0

# NEW: Stricter HU Floor for the Right Rib, overriding adaptive calculation
LOWER_HU_RIGHT_MIN   = 400.0   # Enforce a much higher minimum HU for the right rib growth

# Auto spine exclusion (mask vertebrae) + extra corridor clearance
SPINE_HU             = 480.0
SPINE_MID_TOL_MM     = 20.0     
SPINE_Z_EXT_MM       = 60.0     
SPINE_CLEAR_MM       = 8.0      

# Tiny smoothing on repaired rib only
FINAL_CLOSING        = (8, 0, 8)  # voxel radii (x,y,z)

# -------------------- helpers --------------------
def read_img(p): return sitk.ReadImage(p)
def write_img(img, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    sitk.WriteImage(img, p)

def to_u8(img): return sitk.Cast(img>0, sitk.sitkUInt8)
def to_f32(img): return sitk.Cast(img, sitk.sitkFloat32)
def arr(img): return sitk.GetArrayFromImage(img)

def resample_like(moving, fixed, interp=sitk.sitkNearestNeighbor):
    R = sitk.ResampleImageFilter()
    R.SetReferenceImage(fixed)
    R.SetInterpolator(interp)
    R.SetTransform(sitk.Transform())
    R.SetDefaultPixelValue(0)
    return R.Execute(moving)

def dist_map_mm(bin_u8):
    return sitk.SignedMaurerDistanceMap(bin_u8, insideIsPositive=False,
                                        squaredDistance=False, useImageSpacing=True)

def skeleton(bin_u8): return sitk.BinaryThinning(bin_u8)

def endpoints_from_skeleton(skel_u8):
    """Return endpoints as physical points."""
    sk = arr(skel_u8).astype(np.uint8)           # (z,y,x)
    nb = convolve(sk, np.ones((3,3,3), dtype=np.uint8), mode="constant", cval=0)
    ep = (sk == 1) & (nb == 2)                   # voxel + exactly one neighbor
    zyx = np.argwhere(ep)
    to_phys = lambda z,y,x: skel_u8.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
    return [to_phys(*p) for p in zyx]

def stats_percentiles_in_mask(ct, mask_u8, pcts=(20, 40)):
    m = arr(mask_u8)>0
    hu = arr(ct)[m]
    if hu.size == 0:
        return [300.0 for _ in pcts]
    return [float(np.percentile(hu, p)) for p in pcts]

def build_spine_mask(ct):
    """Detect spine: high-HU components near midline with large z-extent."""
    sp = ct.GetSpacing(); origin = ct.GetOrigin(); size = ct.GetSize()
    mid_x_phys = origin[0] + sp[0]*(size[0]/2.0)
    high = to_u8(ct >= SPINE_HU)
    if not arr(high).any():
        out = sitk.Image(ct.GetSize(), sitk.sitkUInt8); out.CopyInformation(ct); return out
    cc = sitk.ConnectedComponent(high)
    ls = sitk.LabelShapeStatisticsImageFilter(); ls.Execute(cc)
    out = sitk.Image(ct.GetSize(), sitk.sitkUInt8); out.CopyInformation(ct)
    for L in ls.GetLabels():
        cx, cy, cz = ls.GetCentroid(L)
        x,y,z,sx,sy,sz = ls.GetBoundingBox(L)
        z_extent_mm = sz*sp[2]
        if abs(cx - mid_x_phys) <= SPINE_MID_TOL_MM and z_extent_mm >= SPINE_Z_EXT_MM:
            out = sitk.Or(out, to_u8(sitk.BinaryThreshold(cc, L, L, 1, 0)))
    return sitk.BinaryDilate(out, (1,1,1)) if arr(out).any() else out

# Now accepts anatomical parameters
def endpoint_is_posterior_missing_tubercle(ct, ribs_u8, ep_phys, nbhd_mm, bone_hu, y_frac_min, x_tol_mm):
    """
    Returns the fraction of 'other bone' (dense non-rib bone) in the neighborhood.
    Returns 1.0 if not anatomically posterior/midline (no growth needed).
    """
    sp = ct.GetSpacing()
    idx = ct.TransformPhysicalPointToIndex(ep_phys)
    X, Y, Z = ct.GetSize()
    
    # --- 1. Anatomical Position Check (Strict Posterior) ---
    mid_y_idx = Y * y_frac_min
    is_posterior_y = idx[1] >= mid_y_idx

    mid_x_phys = ct.GetOrigin()[0] + sp[0] * (X / 2.0)
    is_near_midline_x = abs(ep_phys[0] - mid_x_phys) <= x_tol_mm
    
    if not (is_posterior_y and is_near_midline_x):
        return 1.0

    # --- 2. Neighborhood Bone Fraction Check ---
    rx = max(1, int(round(nbhd_mm/sp[0])))
    ry = max(1, int(round(nbhd_mm/sp[1])))
    rz = max(1, int(round(nbhd_mm/sp[2])))
    x0,x1 = max(0, idx[0]-rx), min(X-1, idx[0]+rx)
    y0,y1 = max(0, idx[1]-ry), min(Y-1, idx[1]+ry)
    z0,z1 = max(0, idx[2]-rz), min(Z-1, idx[2]+rz)
    ct_a  = arr(ct); rib_a = arr(ribs_u8)
    roi_ct  = ct_a[z0:z1+1, y0:y1+1, x0:x1+1]
    roi_rib = rib_a[z0:z1+1, y0:y1+1, x0:x1+1]
    
    other_bone = (roi_ct >= bone_hu) & (roi_rib == 0)

    return other_bone.mean()

def corridor_for_tip(ribs_u8, spine_u8, ep_phys, r_rib_mm, r_tip_mm, ct=None):
    """Corridor = near-rib band ∩ endpoint sphere ∩ spine-clearance."""
    d_rib = dist_map_mm(ribs_u8)
    near_rib = to_u8(d_rib <= r_rib_mm)

    center_idx = ribs_u8.TransformPhysicalPointToIndex(ep_phys)
    sphere = sitk.Image(ribs_u8.GetSize(), sitk.sitkUInt8); sphere.CopyInformation(ribs_u8)
    if all(0 <= i < s for i,s in zip(center_idx, ribs_u8.GetSize())):
        sphere[center_idx] = 1
    sp = ribs_u8.GetSpacing()
    rad = (max(1,int(round(r_tip_mm/sp[0]))),
           max(1,int(round(r_tip_mm/sp[1]))),
           max(1,int(round(r_tip_mm/sp[2]))))
    sphere = sitk.BinaryDilate(sphere, rad)

    cor = sitk.And(near_rib, sphere)

    if arr(spine_u8).any():
        d_spine = dist_map_mm(spine_u8)
        spine_clear = to_u8(d_spine >= SPINE_CLEAR_MM)
        cor = sitk.And(cor, spine_clear)

    return cor

def region_grow_local(ct, seeds_phys, lo, hi, corridor_u8):
    """ConnectedThreshold inside corridor only."""
    if not seeds_phys:
        empty = sitk.Image(ct.GetSize(), sitk.sitkUInt8); empty.CopyInformation(ct); return empty
    ct_masked = sitk.Mask(ct, corridor_u8, outsideValue=-2000.0)
    seed_idx = [ct.TransformPhysicalPointToIndex(p) for p in seeds_phys]
    rg = sitk.ConnectedThreshold(ct_masked, seedList=seed_idx, lower=lo, upper=hi)
    return to_u8(rg)

def keep_growth_touching_rib(grow_u8, rib_u8):
    """Keep only grown components that touch the rib (1-voxel dilated)."""
    if not arr(grow_u8).any():
        return grow_u8
    rib_d = sitk.BinaryDilate(rib_u8, (1,1,1))
    cc = sitk.ConnectedComponent(grow_u8)
    ls = sitk.LabelShapeStatisticsImageFilter(); ls.Execute(cc)
    out = sitk.Image(grow_u8.GetSize(), sitk.sitkUInt8); out.CopyInformation(grow_u8)
    for L in ls.GetLabels():
        comp = to_u8(sitk.BinaryThreshold(cc, L, L, 1, 0))
        if arr(sitk.And(comp, rib_d)).any():
            out = sitk.Or(out, comp)
    return out

# MODIFIED: Implements side-specific hard-cutoff and stricter HU floor strategy
def repair_first_rib(ct, rib_path, out_path):
    rib_raw = read_img(rib_path)
    rib = to_u8(resample_like(to_u8(rib_raw), ct))
    spine = build_spine_mask(ct)

    rib_filename = os.path.basename(rib_path)
    
    # -------------------- SIDE-SPECIFIC CUTOFF AND HU SELECTION --------------------
    if "rib_right_1" in rib_filename:
        # Ultra-Stricter constraints for the non-problematic side (Right)
        active_cutoff = NBHD_BONE_FRAC_RIGHT_CUTOFF
        side_tag = "Right"
        # Use a high fixed floor to prevent aggressive HU growth
        target_hu_floor = LOWER_HU_RIGHT_MIN 
    else:
        # Standard constraints for the problematic side (Left)
        active_cutoff = NBHD_BONE_FRAC_LEFT_CUTOFF
        side_tag = "Left"
        # Use the standard adaptive floor
        target_hu_floor = LOWER_PCTL_FLOOR_R 
    # -------------------------------------------------------------------------------

    skel = skeleton(rib)
    endpoints = endpoints_from_skeleton(skel)
    
    growth_candidates = []
    for ep in endpoints:
        # Pass required anatomical parameters
        frac = endpoint_is_posterior_missing_tubercle(
            ct, rib, ep, NBHD_RADIUS_MM, NBHD_BONE_HU, 
            POSTERIOR_Y_IDX_MIN_FRAC, MIDLINE_X_TOL_MM
        )
        
        # HARD CUTOFF: Only process if the dense bone fraction is extremely low
        if frac <= active_cutoff:
             growth_candidates.append((ep, frac))

    print(f"[INFO] {rib_filename} ({side_tag}) → endpoints: {len(endpoints)}, Candidates for Robust Growth (frac<={active_cutoff:.3f}): {len(growth_candidates)}")

    # -------------------- CRITICAL BLOCK: SKIP IF NO HIGH-CONFIDENCE NEED --------------------
    if not growth_candidates:
        # Write the ORIGINAL SEGMENTATION and exit. This prevents any oversegmentation.
        print(f"[SKIP] No high-confidence missing tubercles found. Writing original segmentation.")
        # Ensure the original image (before resampling/casting) is written
        write_img(rib_raw, out_path) 
        return

    grow_union = sitk.Image(ct.GetSize(), sitk.sitkUInt8); grow_union.CopyInformation(ct)
    
    # Apply single, robust growth parameters
    pctl_val, _ = stats_percentiles_in_mask(ct, rib, (PCTL_ROBUST, 0))
    
    # Use the stricter HU floor determined above
    lower_hu = max(pctl_val, target_hu_floor) 
    r_rib, r_tip = R_RIB_MM_R, R_TIP_MM_R
    print(f"[INFO] Applying Robust Growth (HU: {lower_hu:.1f}) to {len(growth_candidates)} endpoints.")

    for ep, frac in growth_candidates:
        # Execute Robust Growth
        corridor = corridor_for_tip(rib, spine, ep, r_rib, r_tip, ct)
        growth   = region_grow_local(ct, [ep], lower_hu, UPPER_HU, corridor)
        growth   = keep_growth_touching_rib(growth, rib)

        grow_union = sitk.Or(grow_union, growth)

    pre = sitk.Or(rib, grow_union)

    # Refine (small, anisotropic closing) and write single output
    refined = sitk.BinaryMorphologicalClosing(pre, FINAL_CLOSING)
    # Resample back to original rib space
    refined_out = resample_like(refined, rib_raw)
    write_img(refined_out, out_path)
    print(f"[OK] saved → {out_path}")

# -------------------- file matching (only rib_left_1 / rib_right_1) --------------------
STEM_RE = re.compile(r'^rib[-_](left|right)[-_]0?1$', re.I)

def stem_no_ext(filename: str) -> str:
    f = filename.lower()
    if f.endswith(".nii.gz"): return f[:-7]
    if f.endswith(".nii"):    return f[:-4]
    return f

def list_first_ribs(ribs_dir: str):
    hits = []
    for root, _, files in os.walk(ribs_dir):
        for f in files:
            lf = f.lower()
            if not (lf.endswith(".nii") or lf.endswith(".nii.gz")):
                continue
            if STEM_RE.match(stem_no_ext(lf)):
                hits.append(os.path.join(root, f))
    return hits

# -------------------------- Driver ---------------------------
if __name__ == "__main__":
    # NOTE: Update base_dir to your actual working directory
    base_dir = "/Users/chensirong/TotalSegmentator_postprocessing"
    totalSegmentator_data = os.path.join(base_dir, "TotalSegmentator_Data")

    # SINGLE output directory
    out_dir  = os.path.join(base_dir, "postprocessed_output")
    os.makedirs(out_dir, exist_ok=True)

    SUBJECT_PREFIXES = ("CF", "CM")
    subjects = sorted([
    d for d in os.listdir(totalSegmentator_data)
    if os.path.isdir(os.path.join(totalSegmentator_data, d))
    and d.upper().startswith(SUBJECT_PREFIXES)
    ])

    for subj in subjects:
        print(f"\n===== Processing subject: {subj} =====")
        subj_dir = os.path.join(totalSegmentator_data, subj)
        ct_path  = os.path.join(subj_dir, f"{subj}_resampled.nii.gz")
        ribs_dir = os.path.join(subj_dir, f"{subj}_test")

        if not os.path.exists(ct_path):
            print(f"[WARNING] Missing CT for {subj}, skipping.")
            continue
        if not os.path.isdir(ribs_dir):
            print(f"[WARNING] {ribs_dir} not found; skipping {subj}.")
            continue

        rib_files = list_first_ribs(ribs_dir)
        if not rib_files:
            print(f"[WARNING] No rib_left_1 / rib_right_1 in {ribs_dir}; skipping {subj}.")
            continue
        else:
            print(f"[INFO] Found ribs: {[os.path.basename(p) for p in rib_files]}")

        # read CT once
        try:
            ct = to_f32(read_img(ct_path))
        except Exception as e:
            print(f"[ERROR] Could not read CT for {subj}: {e}")
            continue

        for rib_path in sorted(rib_files):
            base = os.path.basename(rib_path).replace(".nii.gz", "").replace(".nii", "")
            out_path = os.path.join(out_dir, f"{subj}_{base}_postprocessed.nii.gz")
            try:
                repair_first_rib(ct, rib_path, out_path)
            except Exception as e:
                # Capture any errors during processing
                print(f"[ERROR] {subj}/{base}: {e}")

    print("\n🎉 All subjects processed!")