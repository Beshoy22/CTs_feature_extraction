"""
Feature extraction matching the I3LUNG / MIRP pipeline exactly.

Every function in this module is annotated with the original source it replicates.
The extraction path is:

  read_itk_image / read_itk_segmentation          (imageReaders.py)
  → crop_image(boundary=50, z_only=True)           (imageProcess.py)
  → interpolate_image (Gaussian AA + cubic spline) (ImageClass.interpolate)
  → interpolate_roi   (Gaussian AA + order-5 spline + binarise) per label
                                                    (RoiClass.interpolate/register/binarise)
  → combine_all_rois                                (imageProcess.py)
  → sitk.GetImageFromArray (NO spacing)             (I3LUNG_march_2025_multi_thread.py:28)
  → sitk.WriteImage → NRRD on disk                  (I3LUNG_march_2025_multi_thread.py:48)
  → featureextractor.execute(path, path)            (i3lungRadiomics.py:62)
"""

import argparse
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import SimpleITK as sitk
from radiomics import featureextractor


# ──────────────────────────────────────────────────────────────────────────────
# Settings — importSettings.py  (default values, hard-coded)
# ──────────────────────────────────────────────────────────────────────────────
TARGET_SPACING       = np.array([1.25, 1.25, 1.25])  # isotropic, zyx
IMG_SPLINE_ORDER     = 3          # cubic
ROI_SPLINE_ORDER     = 5          # quintic
ANTI_ALIASING        = True
SMOOTHING_BETA       = 0.95
CROP_BOUNDARY_MM     = 50.0
CROP_Z_ONLY          = True
INCL_THRESHOLD       = 0.5       # RoiClass.__init__ default


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — imageClass.py
# ──────────────────────────────────────────────────────────────────────────────

def _build_affine(spacing, orientation):
    """
    Replicate ImageClass.set_spacing / m_affine construction.
    spacing, orientation in (z,y,x) order.
    """
    m = np.zeros((3, 3), dtype=np.float64)
    m[:, 0] = spacing[0] * np.array(orientation[0:3], dtype=np.float64)
    m[:, 1] = spacing[1] * np.array(orientation[3:6], dtype=np.float64)
    m[:, 2] = spacing[2] * np.array(orientation[6:9], dtype=np.float64)
    return m


# ──────────────────────────────────────────────────────────────────────────────
# Reading — imageReaders.py
# ──────────────────────────────────────────────────────────────────────────────

def read_itk_image(path):
    """Replicate imageReaders.read_itk_image."""
    sitk_img = sitk.ReadImage(path)
    # np.float = float64 in the original code (numpy < 1.24)
    voxel_grid = sitk.GetArrayFromImage(sitk_img).astype(np.float64)
    origin      = np.array(sitk_img.GetOrigin())[::-1]       # xyz → zyx
    spacing     = np.array(sitk_img.GetSpacing())[::-1]
    orientation = np.array(sitk_img.GetDirection())[::-1]
    return voxel_grid, origin, spacing, orientation


def read_itk_segmentation(path):
    """
    Replicate imageReaders.read_itk_segmentation.
    Returns a list of (boolean_mask, label_value) per non-zero label.
    """
    sitk_img = sitk.ReadImage(path)
    int_mask = sitk.GetArrayFromImage(sitk_img).astype(np.int64)
    origin      = np.array(sitk_img.GetOrigin())[::-1]
    spacing     = np.array(sitk_img.GetSpacing())[::-1]
    orientation = np.array(sitk_img.GetDirection())[::-1]

    if len(np.unique(int_mask)) > 256:
        raise RuntimeError("More than 256 unique values in mask")

    labels = []
    for lv in np.unique(int_mask):
        if lv == 0:
            continue
        mask = (int_mask == lv)
        if mask.sum() < 5:
            continue
        labels.append((mask, int(lv)))

    if not labels:
        raise ValueError(f"Empty segmentation: {path}")

    return labels, origin, spacing, orientation


# ──────────────────────────────────────────────────────────────────────────────
# Crop — imageProcess.crop_image (z_only=True, boundary=50.0)
# ──────────────────────────────────────────────────────────────────────────────

def crop_z_only(img_grid, roi_masks, img_spacing, img_origin, img_orientation,
                mask_spacing, mask_origin, mask_orientation, boundary_mm=50.0):
    """
    Replicate imageProcess.crop_image with z_only=True.

    The bounding box is found across ALL ROI labels in the MASK's voxel space.
    The boundary is expressed in mm and converted to voxels using IMAGE spacing.
    Both image and masks are cropped with the SAME z-indices.
    Origins are updated using each object's own affine matrix.
    """
    # Collect z-extents across all labels  (roi_ext_z in original)
    z_min_all, z_max_all = [], []
    for mask, _lv in roi_masks:
        z_ind = np.where(mask)[0]
        if len(z_ind) == 0:
            continue
        z_min_all.append(z_ind.min())
        z_max_all.append(z_ind.max())

    if not z_min_all:
        raise ValueError("All ROIs are empty")

    # boundary in voxels — np.ceil(50.0 / spacing).astype(int)   (original uses img_obj.spacing)
    boundary_vox = np.ceil(boundary_mm / img_spacing).astype(int)

    # ind_ext_z = [global_z_min - boundary, global_z_max + boundary]
    z_lo_raw = min(z_min_all) - boundary_vox[0]
    z_hi_raw = max(z_max_all) + boundary_vox[0]

    # Original: min_ind = floor(...), max_ind = ceil(...)
    # Since they're already int, floor/ceil are identity.
    # Clamp to grid bounds:
    #   min_bound = max(z_lo_raw, 0)
    #   max_bound = min(z_hi_raw, size[0])     ← NOTE: clamped to size, not size-1
    #   slice is  [min_bound : max_bound + 1]
    nz = img_grid.shape[0]
    z_lo = max(int(z_lo_raw), 0)
    z_hi = min(int(z_hi_raw), nz)  # clamped to size (not size-1), then +1 in slice

    # Crop image  — voxel_grid[z_lo:z_hi+1, :, :]  (z_only → x/y untouched)
    img_cropped = img_grid[z_lo: z_hi + 1, :, :].copy()

    # Update image origin:  origin += dot(affine, [z_lo, 0, 0])
    img_affine = _build_affine(img_spacing, img_orientation)
    img_origin_new = img_origin + np.dot(img_affine, np.array([z_lo, 0, 0], dtype=np.float64))

    # Crop each mask identically and update its origin
    masks_cropped = []
    mask_affine = _build_affine(mask_spacing, mask_orientation)
    mask_origin_new = mask_origin + np.dot(mask_affine, np.array([z_lo, 0, 0], dtype=np.float64))
    for mask, lv in roi_masks:
        masks_cropped.append((mask[z_lo: z_hi + 1, :, :].copy(), lv))

    return (img_cropped, img_origin_new,
            masks_cropped, mask_origin_new)


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian anti-aliasing — imageProcess.gaussian_preprocess_filter
# ──────────────────────────────────────────────────────────────────────────────

def gaussian_prefilter(voxel_grid, orig_spacing, sample_spacing, beta=0.95):
    """
    Replicate imageProcess.gaussian_preprocess_filter (by_slice=None/False).
    """
    orig_spacing   = orig_spacing.astype(np.float64)
    sample_spacing = sample_spacing.astype(np.float64)
    map_spacing = sample_spacing / orig_spacing
    sigma = np.sqrt(-8.0 * np.power(map_spacing, 2.0) * np.log(beta))
    return ndi.gaussian_filter(
        input=voxel_grid.astype(np.float32),
        sigma=sigma,
        order=0,
        mode="nearest",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Grid interpolation — imageProcess.interpolate_to_new_grid (processor="scipy")
# ──────────────────────────────────────────────────────────────────────────────

def interpolate_to_new_grid(orig_vox, orig_dim, orig_spacing,
                            sample_spacing,
                            sample_dim=None,
                            grid_origin=None,
                            translation=np.array([0.0, 0.0, 0.0]),
                            order=3, mode="nearest",
                            align_to_center=True):
    """
    Replicate imageProcess.interpolate_to_new_grid (processor="scipy").
    Returns (sample_dim, sample_spacing, resampled_vox, grid_origin).
    """
    # Cast to float64 — matches np.float in original
    sample_spacing = sample_spacing.astype(np.float64)
    orig_spacing   = orig_spacing.astype(np.float64)

    if sample_dim is None:
        sample_dim = np.ceil(np.multiply(orig_dim.astype(np.float64),
                                         orig_spacing / sample_spacing))

    grid_spacing = sample_spacing / orig_spacing

    if grid_origin is None:
        if align_to_center:
            grid_origin = (
                0.5 * (np.array(orig_dim, dtype=np.float64) - 1.0)
                - 0.5 * (np.array(sample_dim, dtype=np.float64) - 1.0) * grid_spacing
            )
        else:
            grid_origin = np.array([0.0, 0.0, 0.0])
        grid_origin += translation * grid_spacing

    sd = sample_dim.astype(int)
    map_z, map_y, map_x = np.mgrid[:sd[0], :sd[1], :sd[2]]
    map_z = (map_z * grid_spacing[0] + grid_origin[0]).astype(np.float32)
    map_y = (map_y * grid_spacing[1] + grid_origin[1]).astype(np.float32)
    map_x = (map_x * grid_spacing[2] + grid_origin[2]).astype(np.float32)

    resampled = ndi.map_coordinates(
        input=orig_vox.astype(np.float32),
        coordinates=np.array([map_z, map_y, map_x], dtype=np.float32),
        order=order,
        mode=mode,
    )

    return sample_dim, sample_spacing, resampled, grid_origin


# ──────────────────────────────────────────────────────────────────────────────
# Image interpolation — ImageClass.interpolate
# ──────────────────────────────────────────────────────────────────────────────

def interpolate_image(img_grid, img_spacing, img_origin, img_orientation,
                      new_spacing=TARGET_SPACING):
    """
    Replicate ImageClass.interpolate with Settings defaults.
    by_slice=None, translate=[0,0,0], modality=CT, spat_transform=base.

    Returns (interp_grid, new_origin, new_spacing, grid_origin_for_rois).
    """
    # 1. Gaussian anti-aliasing
    if ANTI_ALIASING:
        img_grid = gaussian_prefilter(img_grid, img_spacing, new_spacing, beta=SMOOTHING_BETA)

    # 2. Interpolate
    orig_dim = np.array(img_grid.shape, dtype=np.float64)
    sample_dim, sample_sp, interp_grid, grid_origin = interpolate_to_new_grid(
        orig_vox=img_grid,
        orig_dim=orig_dim,
        orig_spacing=img_spacing,
        sample_spacing=new_spacing,
        order=IMG_SPLINE_ORDER,
        mode="nearest",
        align_to_center=True,
    )

    # 3. Update origin:  origin += dot(affine, grid_origin)
    affine = _build_affine(img_spacing, img_orientation)
    new_origin = img_origin + np.dot(affine, grid_origin)

    # 4. Round CT intensities (modality == "CT" and spat_transform == "base")
    interp_grid = np.round(interp_grid)

    return interp_grid, new_origin, sample_sp, grid_origin


# ──────────────────────────────────────────────────────────────────────────────
# ROI interpolation — RoiClass.interpolate + .register + .binarise_mask
# ──────────────────────────────────────────────────────────────────────────────

def interpolate_single_roi(mask_bool, mask_spacing, mask_origin, mask_orientation,
                           img_interp_grid, img_new_origin, img_new_spacing):
    """
    Replicate RoiClass.interpolate → register → binarise_mask for ONE label.

    mask_bool: boolean array (one label) in mask's voxel space
    Returns: boolean array in the interpolated image's voxel space
    """
    # ---- Step 1: Gaussian anti-aliasing on boolean mask ----
    # RoiClass.interpolate → gaussian_preprocess_filter
    # orig_spacing = self.roi.spacing (mask spacing)
    # sample_spacing = img_obj.spacing (INTERPOLATED image spacing)
    if ANTI_ALIASING:
        mask_float = gaussian_prefilter(
            mask_bool.astype(np.float64),
            mask_spacing,
            img_new_spacing,
            beta=SMOOTHING_BETA,
        )
    else:
        mask_float = mask_bool.astype(np.float32)

    # ---- Step 2: Register (resample) to image grid ----
    # RoiClass.register:
    #   grid_origin = dot(inv(roi_affine), img_new_origin - roi_old_origin)
    #   interpolate_to_new_grid(sample_dim=img_size, sample_spacing=img_spacing,
    #                           grid_origin=computed, order=5, align_to_center=False)
    roi_affine     = _build_affine(mask_spacing, mask_orientation)
    roi_affine_inv = np.linalg.inv(roi_affine)
    reg_grid_origin = np.dot(roi_affine_inv, img_new_origin - mask_origin)

    img_size = np.array(img_interp_grid.shape, dtype=np.float64)

    _, _, mask_interp, _ = interpolate_to_new_grid(
        orig_vox=mask_float,
        orig_dim=np.array(mask_float.shape, dtype=np.float64),
        orig_spacing=mask_spacing,
        sample_spacing=img_new_spacing,
        sample_dim=img_size,
        grid_origin=reg_grid_origin,
        order=ROI_SPLINE_ORDER,
        mode="nearest",
        align_to_center=False,   # origin-aligned for registration
    )

    # ---- Step 3: Binarise — RoiClass.binarise_mask ----
    # np.around(grid, 6) >= np.around(0.5, 6)
    mask_bin = np.around(mask_interp, 6) >= np.around(INCL_THRESHOLD, 6)

    return mask_bin


# ──────────────────────────────────────────────────────────────────────────────
# Combine labels — imageProcess.combine_all_rois
# ──────────────────────────────────────────────────────────────────────────────

def combine_all_rois(interpolated_masks):
    """
    Combine masks using the same strategy as combine_pertubation_rois
    (imageProcess.py:733) which is what gets saved to NRRD in the original.

    Assigns actual label_value (not boolean True=1).
    Iterates highest-label-first → lowest label wins in overlap.
    PyRadiomics default label=1 → extracts features for label-1 voxels.
    """
    ref_shape = interpolated_masks[0][0].shape
    combined = np.zeros(ref_shape, dtype=np.uint8)

    label_grid_map = {lv: mask for mask, lv in interpolated_masks}

    for lv in sorted(label_grid_map.keys(), reverse=True):
        combined[np.where(label_grid_map[lv] != 0)] = lv

    return combined


# ──────────────────────────────────────────────────────────────────────────────
# Full preprocessing pipeline
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(image_path, seg_path, tmp_dir):
    """
    Run the full MIRP/I3LUNG preprocessing pipeline and save intermediate
    NRRD files to tmp_dir (matching the original save format exactly).

    Returns (volume_nrrd_path, mask_nrrd_path).
    """
    print("=" * 60)
    print("PREPROCESSING (matching I3LUNG / MIRP pipeline)")
    print("=" * 60)

    # ---- 1. Read ----
    print(f"\n[1] Reading image: {image_path}")
    img_grid, img_origin, img_spacing, img_orientation = read_itk_image(image_path)
    print(f"    shape={img_grid.shape}  spacing(zyx)={img_spacing}")

    print(f"[1] Reading segmentation: {seg_path}")
    roi_masks, mask_origin, mask_spacing, mask_orientation = read_itk_segmentation(seg_path)
    print(f"    labels={[lv for _,lv in roi_masks]}  mask_spacing(zyx)={mask_spacing}")

    # ---- 2. Crop z-only ----
    print(f"\n[2] Cropping z-only (boundary={CROP_BOUNDARY_MM}mm)")
    (img_cropped, img_origin_c,
     masks_cropped, mask_origin_c) = crop_z_only(
        img_grid, roi_masks, img_spacing, img_origin, img_orientation,
        mask_spacing, mask_origin, mask_orientation,
        boundary_mm=CROP_BOUNDARY_MM,
    )
    print(f"    cropped shape={img_cropped.shape}")

    # ---- 3. Interpolate image ----
    print(f"\n[3] Interpolating image → {TARGET_SPACING.tolist()} mm  (order={IMG_SPLINE_ORDER})")
    img_interp, img_new_origin, img_new_spacing, _go = interpolate_image(
        img_cropped, img_spacing, img_origin_c, img_orientation,
        new_spacing=TARGET_SPACING,
    )
    print(f"    interpolated shape={img_interp.shape}")

    # ---- 4. Interpolate each ROI label ----
    print(f"\n[4] Interpolating {len(masks_cropped)} ROI label(s) (order={ROI_SPLINE_ORDER})")
    interp_masks = []
    for mask_bool, lv in masks_cropped:
        mask_bin = interpolate_single_roi(
            mask_bool, mask_spacing, mask_origin_c, mask_orientation,
            img_interp, img_new_origin, img_new_spacing,
        )
        n_vox = mask_bin.sum()
        print(f"    label {lv}: {n_vox} voxels after interpolation + binarise")
        interp_masks.append((mask_bin, lv))

    # ---- 5. Combine ----
    print(f"\n[5] Combining labels")
    combined_mask = combine_all_rois(interp_masks)
    n_combined = (combined_mask > 0).sum()
    print(f"    combined mask: {n_combined} nonzero voxels  (unique values: {np.unique(combined_mask).tolist()})")

    if n_combined == 0:
        raise ValueError("Combined mask is empty after preprocessing — check your inputs")

    # ---- 6. Save to NRRD (matching original exactly) ----
    # Original:  sitk.GetImageFromArray(ctscan.voxel_grid)   ← NO SetSpacing
    #            sitk.WriteImage(img, path)
    # Image dtype: float32 (from map_coordinates output + np.round)
    # Mask dtype:  uint8   (from combine_all_rois)
    print(f"\n[6] Saving to NRRD (spacing defaults to [1,1,1])")

    os.makedirs(tmp_dir, exist_ok=True)
    vol_path  = os.path.join(tmp_dir, "volume.nrrd")
    mask_path = os.path.join(tmp_dir, "mask.nrrd")

    sitk_vol = sitk.GetImageFromArray(img_interp.astype(np.float32))
    sitk.WriteImage(sitk_vol, vol_path)

    sitk_mask = sitk.GetImageFromArray(combined_mask)  # already uint8
    sitk.WriteImage(sitk_mask, mask_path)

    print(f"    volume: {vol_path}  (dtype=float32)")
    print(f"    mask:   {mask_path}  (dtype=uint8)")

    return vol_path, mask_path


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction — i3lungRadiomics.extract_features
# ──────────────────────────────────────────────────────────────────────────────

def extract_features_from_nrrd(volume_path, mask_path, patient_id):
    """
    Replicate i3lungRadiomics.extract_features.
    Reads from FILE PATHS (not in-memory objects) — exactly like the original.
    """
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures()
    features = extractor.execute(volume_path, mask_path)
    features["patient_id"] = patient_id
    return features


# ──────────────────────────────────────────────────────────────────────────────
# The 128 features
# ──────────────────────────────────────────────────────────────────────────────

DESIRED_FEATURES = [
    'logarithm_firstorder_Energy',
    'logarithm_firstorder_TotalEnergy',
    'logarithm_glcm_ClusterProminence',
    'wavelet-LLL_firstorder_Energy',
    'wavelet-LLL_firstorder_TotalEnergy',
    'exponential_glszm_ZoneVariance',
    'gradient_firstorder_Energy',
    'wavelet-HLL_glszm_LargeAreaHighGrayLevelEmphasis',
    'exponential_glszm_LargeAreaHighGrayLevelEmphasis',
    'wavelet-LHL_firstorder_Energy',
    'wavelet-LLH_firstorder_TotalEnergy',
    'wavelet-HHL_firstorder_Energy',
    'gradient_firstorder_TotalEnergy',
    'wavelet-LHL_firstorder_TotalEnergy',
    'wavelet-LLH_glszm_LargeAreaHighGrayLevelEmphasis',
    'wavelet-LLH_firstorder_Energy',
    'wavelet-LHH_firstorder_TotalEnergy',
    'wavelet-LHH_firstorder_Energy',
    'square_glszm_LargeAreaLowGrayLevelEmphasis',
    'wavelet-HHH_glszm_ZoneVariance',
    'logarithm_glcm_ClusterShade',
    'squareroot_glszm_LargeAreaHighGrayLevelEmphasis',
    'wavelet-HHL_glszm_LargeAreaLowGrayLevelEmphasis',
    'logarithm_glszm_LargeAreaHighGrayLevelEmphasis',
    'logarithm_firstorder_Variance',
    'square_firstorder_TotalEnergy',
    'logarithm_ngtdm_Complexity',
    'wavelet-LHL_glszm_LargeAreaEmphasis',
    'square_firstorder_Energy',
    'gradient_glszm_LargeAreaHighGrayLevelEmphasis',
    'wavelet-LLL_glcm_ClusterProminence',
    'logarithm_glcm_Autocorrelation',
    'logarithm_glszm_HighGrayLevelZoneEmphasis',
    'wavelet-LLH_glcm_ClusterProminence',
    'logarithm_glcm_ClusterTendency',
    'logarithm_gldm_SmallDependenceHighGrayLevelEmphasis',
    'wavelet-LLH_gldm_LargeDependenceHighGrayLevelEmphasis',
    'wavelet-HHH_gldm_GrayLevelNonUniformity',
    'wavelet-LHL_glcm_ClusterProminence',
    'wavelet-LHL_gldm_LargeDependenceHighGrayLevelEmphasis',
    'wavelet-HHL_glrlm_RunLengthNonUniformity',
    'wavelet-HLL_glcm_ClusterProminence',
    'logarithm_glszm_GrayLevelVariance',
    'wavelet-HLL_gldm_LargeDependenceHighGrayLevelEmphasis',
    'wavelet-LLL_firstorder_Variance',
    'gradient_glrlm_GrayLevelNonUniformity',
    'gradient_glcm_ClusterProminence',
    'wavelet-LLL_gldm_LargeDependenceHighGrayLevelEmphasis',
    'logarithm_glcm_Contrast',
    'original_glcm_ClusterProminence',
    'wavelet-HLH_glcm_ClusterProminence',
    'wavelet-LHL_ngtdm_Complexity',
    'wavelet-HHL_gldm_LargeDependenceHighGrayLevelEmphasis',
    'wavelet-HHH_glszm_SizeZoneNonUniformity',
    'wavelet-HLH_gldm_LargeDependenceHighGrayLevelEmphasis',
    'wavelet-HHL_glcm_ClusterProminence',
    'wavelet-HLL_ngtdm_Complexity',
    'wavelet-LLH_ngtdm_Complexity',
    'gradient_firstorder_Variance',
    'wavelet-LHH_gldm_LargeDependenceHighGrayLevelEmphasis',
    'wavelet-HHH_ngtdm_Busyness',
    'wavelet-LLL_ngtdm_Complexity',
    'wavelet-LLL_glcm_ClusterShade',
    'wavelet-LHH_glcm_ClusterProminence',
    'wavelet-LLH_glszm_SizeZoneNonUniformity',
    'diagnostics_Image-original_Minimum',
    'square_glrlm_RunLengthNonUniformity',
    'original_firstorder_Variance',
    'wavelet-LLH_glrlm_LongRunHighGrayLevelEmphasis',
    'wavelet-HHL_ngtdm_Complexity',
    'wavelet-LHL_glrlm_LongRunHighGrayLevelEmphasis',
    'logarithm_firstorder_Range',
    'diagnostics_Image-original_Maximum',
    'wavelet-HHH_firstorder_Variance',
    'logarithm_firstorder_InterquartileRange',
    'logarithm_firstorder_Median',
    'wavelet-HHH_gldm_LargeDependenceHighGrayLevelEmphasis',
    'squareroot_glszm_HighGrayLevelZoneEmphasis',
    'wavelet-LLH_glcm_ClusterShade',
    'gradient_ngtdm_Complexity',
    'squareroot_gldm_SmallDependenceHighGrayLevelEmphasis',
    'wavelet-LHH_ngtdm_Complexity',
    'logarithm_firstorder_10Percentile',
    'wavelet-HLH_ngtdm_Complexity',
    'wavelet-HLL_glrlm_LongRunHighGrayLevelEmphasis',
    'wavelet-LHL_glcm_Autocorrelation',
    'logarithm_firstorder_Mean',
    'logarithm_firstorder_Minimum',
    'wavelet-HHL_glszm_GrayLevelNonUniformity',
    'wavelet-HHH_glcm_ClusterProminence',
    'logarithm_firstorder_90Percentile',
    'wavelet-LLH_glcm_Autocorrelation',
    'wavelet-HHL_glrlm_LongRunHighGrayLevelEmphasis',
    'logarithm_firstorder_RootMeanSquared',
    'wavelet-HLL_glrlm_HighGrayLevelRunEmphasis',
    'logarithm_glszm_ZoneVariance',
    'exponential_glrlm_RunLengthNonUniformity',
    'square_glszm_GrayLevelNonUniformity',
    'lbp-2D_glrlm_LongRunEmphasis',
    'wavelet-LLH_gldm_SmallDependenceHighGrayLevelEmphasis',
    'wavelet-HHH_ngtdm_Complexity',
    'wavelet-HLL_gldm_SmallDependenceHighGrayLevelEmphasis',
    'wavelet-LLL_firstorder_Median',
    'wavelet-LLL_glrlm_LongRunHighGrayLevelEmphasis',
    'wavelet-HLH_glrlm_LongRunHighGrayLevelEmphasis',
    'squareroot_glszm_GrayLevelVariance',
    'squareroot_glcm_SumSquares',
    'diagnostics_Image-original_Mean',
    'wavelet-LLL_firstorder_10Percentile',
    'wavelet-LLL_firstorder_90Percentile',
    'wavelet-LLH_glszm_LargeAreaLowGrayLevelEmphasis',
    'wavelet-LLL_glcm_Autocorrelation',
    'wavelet-LLL_firstorder_InterquartileRange',
    'original_glcm_ClusterShade',
    'wavelet-LLL_gldm_SmallDependenceHighGrayLevelEmphasis',
    'wavelet-LHL_firstorder_Maximum',
    'wavelet-LHH_glrlm_LongRunHighGrayLevelEmphasis',
    'wavelet-LLL_firstorder_Maximum',
    'wavelet-HLL_firstorder_Maximum',
    'wavelet-LLH_firstorder_Range',
    'wavelet-LLL_glszm_SmallAreaHighGrayLevelEmphasis',
    'wavelet-HHH_glrlm_LongRunHighGrayLevelEmphasis',
    'square_gldm_LargeDependenceLowGrayLevelEmphasis',
    'gradient_firstorder_Range',
    'wavelet-LLL_firstorder_Range',
    'wavelet-HHH_glrlm_HighGrayLevelRunEmphasis',
    'original_ngtdm_Complexity',
    'squareroot_firstorder_Maximum',
]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract PyRadiomics features matching the I3LUNG/MIRP preprocessing pipeline exactly."
    )
    parser.add_argument("--input_dir", default=".")
    parser.add_argument("--image", default="image.nrrd",
                        help="Image filename (relative to --input_dir)")
    parser.add_argument("--seg", default="mask.seg.nrrd",
                        help="Segmentation filename (relative to --input_dir)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: same as --input_dir)")
    parser.add_argument("--patient_id", default="patient_0",
                        help="Patient identifier for the output CSV")
    parser.add_argument("--filter_features", action="store_true",
                        help="Only output the 128 selected features")
    parser.add_argument("--troubleshoot", action="store_true",
                        help="Keep intermediate NRRD files for inspection")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    image_path = os.path.join(args.input_dir, args.image)
    seg_path   = os.path.join(args.input_dir, args.seg)

    # Decide where intermediate NRRDs go
    if args.troubleshoot:
        tmp_dir = os.path.join(output_dir, "_preprocessed")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="radiomics_")

    try:
        # ---- Preprocess ----
        vol_nrrd, mask_nrrd = preprocess(image_path, seg_path, tmp_dir)

        # ---- Extract ----
        print("\n" + "=" * 60)
        print("FEATURE EXTRACTION (PyRadiomics)")
        print("=" * 60)
        all_features = extract_features_from_nrrd(vol_nrrd, mask_nrrd, args.patient_id)

        n_total = sum(1 for k in all_features if not k.startswith("diagnostics_") and k != "patient_id")
        print(f"\nExtracted {n_total} non-diagnostic features")

        # ---- Filter ----
        if args.filter_features:
            output_features = {k: v for k, v in all_features.items() if k in DESIRED_FEATURES}
            missing = [f for f in DESIRED_FEATURES if f not in all_features]
            if missing:
                print(f"\nWARNING: {len(missing)} desired feature(s) not found:")
                for m in missing:
                    print(f"  - {m}")
            print(f"Filtered to {len(output_features)} / {len(DESIRED_FEATURES)} desired features")
        else:
            output_features = all_features

        # ---- Save ----
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "radiomics_features.csv")
        pd.DataFrame([output_features]).to_csv(output_path, index=False)
        print(f"\n✓ Saved to: {output_path}")

    finally:
        # Clean up temp NRRDs unless troubleshooting
        if not args.troubleshoot and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print("✓ Cleaned up temporary NRRD files")
        elif args.troubleshoot:
            print(f"\n✓ Intermediate NRRDs kept at: {tmp_dir}")
            print(f"  volume: {os.path.join(tmp_dir, 'volume.nrrd')}")
            print(f"  mask:   {os.path.join(tmp_dir, 'mask.nrrd')}")


if __name__ == "__main__":
    main()
