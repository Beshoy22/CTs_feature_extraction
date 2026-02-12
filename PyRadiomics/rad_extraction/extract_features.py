import os
import argparse
import SimpleITK as sitk
import pandas as pd
import numpy as np
from radiomics import featureextractor
import scipy.ndimage as ndi


# =============================================================================
# Preprocessing functions — matching the MIRP / I3LUNG codebase exactly
# =============================================================================

def gaussian_preprocess_filter(orig_vox, orig_spacing, sample_spacing, param_beta=0.95, mode="nearest"):
    """
    Gaussian anti-aliasing before resampling.
    Matches: imageProcess.gaussian_preprocess_filter (by_slice=False)
    """
    sample_spacing = sample_spacing.astype(np.float64)
    orig_spacing = orig_spacing.astype(np.float64)
    map_spacing = sample_spacing / orig_spacing
    sigma = np.sqrt(-8 * np.power(map_spacing, 2.0) * np.log(param_beta))
    return ndi.gaussian_filter(
        input=orig_vox.astype(np.float32),
        sigma=sigma,
        order=0,
        mode=mode,
    )


def interpolate_to_new_grid(orig_vox, orig_spacing, sample_spacing,
                            order=3, mode="nearest",
                            sample_dim=None, grid_origin=None,
                            align_to_center=True):
    """
    Interpolate to new grid using scipy.ndimage.map_coordinates.
    Matches: imageProcess.interpolate_to_new_grid (processor="scipy")

    Returns: (resampled_vox, sample_dim, sample_spacing, grid_origin)
    """
    orig_dim = np.array(orig_vox.shape, dtype=np.float64)
    sample_spacing = sample_spacing.astype(np.float64)
    orig_spacing = orig_spacing.astype(np.float64)

    if sample_dim is None:
        sample_dim = np.ceil(orig_dim * orig_spacing / sample_spacing)

    grid_spacing = sample_spacing / orig_spacing

    if grid_origin is None:
        if align_to_center:
            grid_origin = (
                0.5 * (orig_dim - 1.0)
                - 0.5 * (sample_dim - 1.0) * grid_spacing
            )
        else:
            grid_origin = np.array([0.0, 0.0, 0.0])

    sd = sample_dim.astype(int)
    map_z, map_y, map_x = np.mgrid[:sd[0], :sd[1], :sd[2]]
    map_z = (map_z * grid_spacing[0] + grid_origin[0]).astype(np.float32)
    map_y = (map_y * grid_spacing[1] + grid_origin[1]).astype(np.float32)
    map_x = (map_x * grid_spacing[2] + grid_origin[2]).astype(np.float32)

    resampled_vox = ndi.map_coordinates(
        input=orig_vox.astype(np.float32),
        coordinates=np.array([map_z, map_y, map_x], dtype=np.float32),
        order=order,
        mode=mode,
    )

    return resampled_vox, sample_dim, sample_spacing, grid_origin


def crop_to_mask_z_only(image_array, mask_array, spacing_zyx, boundary_mm=50.0):
    """
    Crop image and mask along z-axis only with a boundary in mm.
    Matches: imageProcess.crop_image with z_only=True, boundary=50.0

    Args:
        image_array: numpy array (z, y, x)
        mask_array:  numpy array (z, y, x), integer labels
        spacing_zyx: spacing in (z, y, x) order
        boundary_mm: boundary in mm

    Returns: (cropped_image, cropped_mask, z_lo) — z_lo for origin update
    """
    nonzero = np.where(mask_array > 0)
    if len(nonzero[0]) == 0:
        raise ValueError("Mask is empty")

    # Boundary in voxels (per axis, but only z is used)
    boundary_vox = np.ceil(boundary_mm / spacing_zyx).astype(int)

    z_min = nonzero[0].min()
    z_max = nonzero[0].max()

    z_lo = max(0, z_min - boundary_vox[0])
    z_hi = min(image_array.shape[0], z_max + boundary_vox[0] + 1)

    return image_array[z_lo:z_hi, :, :], mask_array[z_lo:z_hi, :, :], z_lo


# =============================================================================
# Full preprocessing pipeline
# =============================================================================

def preprocess_image_and_mask(image_sitk, mask_sitk,
                              target_spacing=[1.25, 1.25, 1.25],
                              crop_boundary=50.0, smoothing_beta=0.95,
                              img_spline_order=3, roi_spline_order=5,
                              incl_threshold=0.5):
    """
    Complete preprocessing pipeline matching the I3LUNG codebase exactly.

    Pipeline:
        1. Crop z-only (50mm boundary)
        2. Gaussian anti-aliasing on image (beta=0.95)
        3. Interpolate image (cubic spline, order=3, center-aligned)
        4. Round CT intensities to nearest integer
        5. Gaussian anti-aliasing on mask
        6. Interpolate mask (spline order=5, registered to image grid)
        7. Binarise mask at >= 0.5

    IMPORTANT: Output SimpleITK images have DEFAULT spacing [1,1,1] and
    origin [0,0,0]. This matches the original code which does:
        sitk.GetImageFromArray(array)  # no SetSpacing
    """
    print("\n=== PREPROCESSING PIPELINE ===")

    # Read arrays and spatial info
    img_array = sitk.GetArrayFromImage(image_sitk).astype(np.float64)   # (z,y,x)
    mask_array = sitk.GetArrayFromImage(mask_sitk).astype(np.int32)     # (z,y,x)

    # Spacing: SimpleITK is (x,y,z) → reverse to (z,y,x)
    orig_spacing_zyx = np.array(image_sitk.GetSpacing())[::-1]
    orig_origin_zyx = np.array(image_sitk.GetOrigin())[::-1]
    mask_spacing_zyx = np.array(mask_sitk.GetSpacing())[::-1]
    mask_origin_zyx = np.array(mask_sitk.GetOrigin())[::-1]

    target_spacing_zyx = np.array(target_spacing)  # already [z, y, x] = [1.25, 1.25, 1.25]

    print(f"  Image shape: {img_array.shape}, spacing(zyx): {orig_spacing_zyx}")
    print(f"  Mask  shape: {mask_array.shape}, spacing(zyx): {mask_spacing_zyx}")

    # --- Step 1: Crop z-only ---
    print(f"1. Cropping z-only (boundary={crop_boundary}mm)...")
    img_cropped, mask_cropped, z_lo = crop_to_mask_z_only(
        img_array, mask_array, orig_spacing_zyx, boundary_mm=crop_boundary
    )
    print(f"   Cropped shape: {img_cropped.shape}")

    # Update origins after crop (needed for ROI registration)
    # origin_new = origin + affine * [z_lo, 0, 0]
    # For standard orientation, this simplifies to shifting z by z_lo * spacing_z
    img_origin_cropped = orig_origin_zyx.copy()
    img_origin_cropped[0] += z_lo * orig_spacing_zyx[0]

    mask_origin_cropped = mask_origin_zyx.copy()
    mask_origin_cropped[0] += z_lo * mask_spacing_zyx[0]

    # --- Step 2: Gaussian anti-aliasing on IMAGE ---
    print(f"2. Anti-aliasing image (beta={smoothing_beta})...")
    img_smoothed = gaussian_preprocess_filter(
        img_cropped, orig_spacing_zyx, target_spacing_zyx,
        param_beta=smoothing_beta, mode="nearest"
    )

    # --- Step 3: Interpolate IMAGE (order=3, center-aligned) ---
    print(f"3. Interpolating image (order={img_spline_order}) to {target_spacing}mm...")
    img_resampled, new_dim, new_spacing, img_grid_origin = interpolate_to_new_grid(
        img_smoothed, orig_spacing_zyx, target_spacing_zyx,
        order=img_spline_order, mode="nearest",
        align_to_center=True
    )

    # --- Step 4: Round CT intensities ---
    print(f"4. Rounding CT intensities...")
    img_resampled = np.round(img_resampled)

    print(f"   Resampled image shape: {img_resampled.shape}")

    # Compute the interpolated image's new origin (needed for ROI registration)
    # In the original code: img_obj.origin = img_obj.origin + dot(affine, grid_origin)
    # For standard axis-aligned images this simplifies to:
    img_origin_interp = img_origin_cropped + orig_spacing_zyx * img_grid_origin

    # --- Step 5: Anti-aliasing on MASK ---
    print(f"5. Anti-aliasing mask (beta={smoothing_beta})...")
    mask_float = mask_cropped.astype(np.float64)
    mask_smoothed = gaussian_preprocess_filter(
        mask_float, mask_spacing_zyx, target_spacing_zyx,
        param_beta=smoothing_beta, mode="nearest"
    )

    # --- Step 6: Interpolate MASK (order=5, registered to image grid) ---
    print(f"6. Interpolating mask (order={roi_spline_order}, registered to image grid)...")

    # Compute grid_origin for registration:
    # grid_origin = inv(roi_affine) * (img_new_origin - roi_old_origin)
    # For axis-aligned grids: grid_origin = (img_origin_interp - mask_origin_cropped) / mask_spacing
    roi_grid_origin = (img_origin_interp - mask_origin_cropped) / mask_spacing_zyx

    mask_resampled, _, _, _ = interpolate_to_new_grid(
        mask_smoothed, mask_spacing_zyx, new_spacing,
        order=roi_spline_order, mode="nearest",
        sample_dim=np.array(img_resampled.shape, dtype=np.float64),
        grid_origin=roi_grid_origin,
        align_to_center=False  # origin-aligned for registration
    )

    # --- Step 7: Binarise mask at >= 0.5 ---
    print(f"7. Binarising mask (threshold >= {incl_threshold})...")
    mask_resampled = (np.around(mask_resampled, 6) >= np.around(incl_threshold, 6)).astype(np.uint8)

    n_voxels = mask_resampled.sum()
    print(f"   Resampled mask shape: {mask_resampled.shape}, nonzero voxels: {n_voxels}")

    if n_voxels == 0:
        raise ValueError("Mask is empty after preprocessing — check your inputs")

    # --- Convert to SimpleITK ---
    # CRITICAL: Do NOT set spacing/origin. The original code does:
    #   sitk.GetImageFromArray(array)   → defaults to spacing=[1,1,1], origin=[0,0,0]
    # Setting spacing would change TotalEnergy, shape features, etc.
    img_sitk_out = sitk.GetImageFromArray(img_resampled.astype(np.float32))
    mask_sitk_out = sitk.GetImageFromArray(mask_resampled.astype(np.int16))

    print("   Output SimpleITK spacing: [1,1,1] (matches original pipeline)")

    return img_sitk_out, mask_sitk_out


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract PyRadiomics features with I3LUNG preprocessing pipeline"
    )
    parser.add_argument("--input_dir", default=".")
    parser.add_argument("--image", default="image.nrrd")
    parser.add_argument("--seg", default="mask.seg.nrrd")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--filter_features", action="store_true",
                        help="Only output the 128 selected features")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    image_path = os.path.join(args.input_dir, args.image)
    seg_path = os.path.join(args.input_dir, args.seg)

    print(f"Reading: {image_path}")
    print(f"Reading: {seg_path}")

    # Read images
    image = sitk.ReadImage(image_path)
    segmentation = sitk.ReadImage(seg_path)

    # Preprocess — matches the I3LUNG codebase exactly
    img_preprocessed, mask_preprocessed = preprocess_image_and_mask(
        image, segmentation,
        target_spacing=[1.25, 1.25, 1.25],
        crop_boundary=50.0,
        smoothing_beta=0.95,
        img_spline_order=3,
        roi_spline_order=5,
        incl_threshold=0.5,
    )

    # Extract features
    print("\n=== FEATURE EXTRACTION ===")
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures()

    all_features = extractor.execute(img_preprocessed, mask_preprocessed)
    print(f"Extracted {len(all_features)} features")

    # Optional: filter to the 128 selected features
    if args.filter_features:
        desired_features = [
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
        output_features = {k: v for k, v in all_features.items() if k in desired_features}
        missing = [f for f in desired_features if f not in all_features]
        if missing:
            print(f"WARNING: {len(missing)} desired features not found in extraction output:")
            for m in missing[:10]:
                print(f"  - {m}")
            if len(missing) > 10:
                print(f"  ... and {len(missing)-10} more")
        print(f"Filtered to {len(output_features)} / {len(desired_features)} desired features")
    else:
        output_features = all_features

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "radiomics_features.csv")
    pd.DataFrame([output_features]).to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")


if __name__ == "__main__":
    main()
