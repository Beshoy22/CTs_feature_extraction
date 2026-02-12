import os
import argparse
import SimpleITK as sitk
import pandas as pd
import numpy as np
from radiomics import featureextractor
import scipy.ndimage as ndi


def gaussian_preprocess_filter(orig_vox, orig_spacing, sample_spacing, param_beta=0.95, mode="nearest"):
    """
    Apply Gaussian smoothing for anti-aliasing before resampling.
    Matches the MIRP preprocessing in your codebase.
    """
    # Calculate zoom factors
    map_spacing = sample_spacing / orig_spacing
    
    # Calculate sigma for Gaussian filter
    # Formula from your codebase: sigma = sqrt(-8 * map_spacing^2 * log(beta))
    sigma = np.sqrt(-8 * np.power(map_spacing, 2.0) * np.log(param_beta))
    
    # Apply Gaussian filter
    smoothed_vox = ndi.gaussian_filter(
        input=orig_vox.astype(np.float32), 
        sigma=sigma, 
        order=0, 
        mode=mode
    )
    
    return smoothed_vox


def interpolate_to_new_grid(orig_vox, orig_spacing, sample_spacing, order=3, mode="nearest"):
    """
    Interpolate to new grid using scipy.ndimage.map_coordinates.
    Matches the MIRP interpolation in your codebase.
    """
    orig_dim = np.array(orig_vox.shape)
    sample_spacing = sample_spacing.astype(np.float64)
    orig_spacing = orig_spacing.astype(np.float64)
    
    # Calculate new dimensions
    sample_dim = np.ceil(np.multiply(orig_dim, orig_spacing / sample_spacing)).astype(int)
    
    # Grid spacing in voxel units
    grid_spacing = sample_spacing / orig_spacing
    
    # Center alignment (default in your codebase)
    grid_origin = (
        0.5 * (orig_dim - 1.0) - 0.5 * (sample_dim - 1.0) * grid_spacing
    )
    
    # Generate interpolation grid
    map_z, map_y, map_x = np.mgrid[:sample_dim[0], :sample_dim[1], :sample_dim[2]]
    
    # Transform to original space
    map_z = map_z * grid_spacing[0] + grid_origin[0]
    map_y = map_y * grid_spacing[1] + grid_origin[1]
    map_x = map_x * grid_spacing[2] + grid_origin[2]
    
    # Interpolate
    resampled_vox = ndi.map_coordinates(
        input=orig_vox.astype(np.float32),
        coordinates=np.array([map_z, map_y, map_x], dtype=np.float32),
        order=order,
        mode=mode
    )
    
    return resampled_vox, sample_dim


def crop_to_mask_with_boundary(image, mask, boundary_mm=50.0, z_only=True):
    """Crop image and mask to ROI with boundary (z-axis only by default)."""
    mask_array = sitk.GetArrayFromImage(mask)
    nonzero_indices = np.where(mask_array > 0)
    
    if len(nonzero_indices[0]) == 0:
        raise ValueError("Mask is empty")
    
    z_min, z_max = nonzero_indices[0].min(), nonzero_indices[0].max()
    y_min, y_max = nonzero_indices[1].min(), nonzero_indices[1].max()
    x_min, x_max = nonzero_indices[2].min(), nonzero_indices[2].max()
    
    spacing = np.array(image.GetSpacing())  # (x, y, z)
    boundary_voxels = boundary_mm / spacing
    size = np.array(mask_array.shape)  # (z, y, x)
    
    if z_only:
        z_min = max(0, int(z_min - boundary_voxels[2]))
        z_max = min(size[0] - 1, int(z_max + boundary_voxels[2]))
        y_min, y_max = 0, size[1] - 1
        x_min, x_max = 0, size[2] - 1
    else:
        z_min = max(0, int(z_min - boundary_voxels[2]))
        z_max = min(size[0] - 1, int(z_max + boundary_voxels[2]))
        y_min = max(0, int(y_min - boundary_voxels[1]))
        y_max = min(size[1] - 1, int(y_max + boundary_voxels[1]))
        x_min = max(0, int(x_min - boundary_voxels[0]))
        x_max = min(size[2] - 1, int(x_max + boundary_voxels[0]))
    
    crop_size = [int(x_max - x_min + 1), int(y_max - y_min + 1), int(z_max - z_min + 1)]
    crop_index = [int(x_min), int(y_min), int(z_min)]
    
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetSize(crop_size)
    roi_filter.SetIndex(crop_index)
    
    return roi_filter.Execute(image), roi_filter.Execute(mask)


def preprocess_image_and_mask(image_sitk, mask_sitk, target_spacing=[1.25, 1.25, 1.25], 
                               crop_boundary=50.0, smoothing_beta=0.95):
    """
    Complete preprocessing pipeline matching your codebase.
    """
    print("\n=== PREPROCESSING PIPELINE ===")
    
    # Step 1: Crop
    print(f"1. Cropping (boundary={crop_boundary}mm, z-only)...")
    image_cropped, mask_cropped = crop_to_mask_with_boundary(
        image_sitk, mask_sitk, boundary_mm=crop_boundary, z_only=True
    )
    
    # Convert to numpy arrays
    img_array = sitk.GetArrayFromImage(image_cropped)
    mask_array = sitk.GetArrayFromImage(mask_cropped)
    orig_spacing = np.array(image_cropped.GetSpacing())  # (x, y, z)
    # Convert to (z, y, x) for processing
    orig_spacing_zyx = orig_spacing[::-1]
    target_spacing_zyx = np.array(target_spacing[::-1])
    
    print(f"   Original shape: {img_array.shape}, spacing: {orig_spacing}")
    
    # Step 2: Apply anti-aliasing Gaussian filter to IMAGE
    print(f"2. Applying Gaussian anti-aliasing (beta={smoothing_beta})...")
    img_smoothed = gaussian_preprocess_filter(
        img_array, orig_spacing_zyx, target_spacing_zyx, 
        param_beta=smoothing_beta, mode="nearest"
    )
    
    # Step 3: Interpolate IMAGE (order=3, cubic)
    print(f"3. Interpolating image (order=3, cubic) to {target_spacing}mm...")
    img_resampled, new_dim = interpolate_to_new_grid(
        img_smoothed, orig_spacing_zyx, target_spacing_zyx, order=3, mode="nearest"
    )
    
    # Step 4: Apply anti-aliasing to MASK
    print(f"4. Applying Gaussian anti-aliasing to mask...")
    mask_smoothed = gaussian_preprocess_filter(
        mask_array, orig_spacing_zyx, target_spacing_zyx,
        param_beta=smoothing_beta, mode="nearest"
    )
    
    # Step 5: Interpolate MASK (order=5 in codebase, but nearest neighbor for final)
    print(f"5. Interpolating mask (order=5)...")
    mask_resampled, _ = interpolate_to_new_grid(
        mask_smoothed, orig_spacing_zyx, target_spacing_zyx, order=5, mode="nearest"
    )
    
    # Step 6: Binarize mask
    print(f"6. Binarizing mask...")
    mask_resampled = (mask_resampled > 0.5).astype(np.uint8)
    
    print(f"   Final shape: {img_resampled.shape}, spacing: {target_spacing}mm")
    
    # Convert back to SimpleITK
    img_sitk_final = sitk.GetImageFromArray(img_resampled.astype(np.float32))
    img_sitk_final.SetSpacing(target_spacing)
    
    mask_sitk_final = sitk.GetImageFromArray(mask_resampled.astype(np.uint8))
    mask_sitk_final.SetSpacing(target_spacing)
    mask_sitk_final.CopyInformation(img_sitk_final)
    
    return img_sitk_final, mask_sitk_final


def main():
    parser = argparse.ArgumentParser(description="Extract PyRadiomics features matching codebase preprocessing")
    parser.add_argument("--input_dir", default=".")
    parser.add_argument("--image", default="image.nrrd")
    parser.add_argument("--seg", default="mask.seg.nrrd")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--filter_features", action="store_true")
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.input_dir
    image_path = os.path.join(args.input_dir, args.image)
    seg_path = os.path.join(args.input_dir, args.seg)
    
    print(f"Reading: {image_path}")
    print(f"Reading: {seg_path}")
    
    # Read images
    image = sitk.ReadImage(image_path)
    segmentation = sitk.ReadImage(seg_path)
    segmentation.CopyInformation(image)
    
    # Preprocess to match codebase EXACTLY
    img_preprocessed, mask_preprocessed = preprocess_image_and_mask(
        image, segmentation,
        target_spacing=[1.25, 1.25, 1.25],  # Match codebase
        crop_boundary=50.0,                  # Match codebase
        smoothing_beta=0.95                  # Match codebase
    )
    
    # Extract features
    print("\n=== FEATURE EXTRACTION ===")
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures()
    
    all_features = extractor.execute(img_preprocessed, mask_preprocessed)
    print(f"Extracted {len(all_features)} features")
    
    # Optional filtering
    if args.filter_features:
        desired_features = ['logarithm_firstorder_Energy', 'logarithm_firstorder_TotalEnergy', 'logarithm_glcm_ClusterProminence', 'wavelet-LLL_firstorder_Energy', 'wavelet-LLL_firstorder_TotalEnergy', 'exponential_glszm_ZoneVariance', 'gradient_firstorder_Energy', 'wavelet-HLL_glszm_LargeAreaHighGrayLevelEmphasis', 'exponential_glszm_LargeAreaHighGrayLevelEmphasis', 'wavelet-LHL_firstorder_Energy', 'wavelet-LLH_firstorder_TotalEnergy', 'wavelet-HHL_firstorder_Energy', 'gradient_firstorder_TotalEnergy', 'wavelet-LHL_firstorder_TotalEnergy', 'wavelet-LLH_glszm_LargeAreaHighGrayLevelEmphasis', 'wavelet-LLH_firstorder_Energy', 'wavelet-LHH_firstorder_TotalEnergy', 'wavelet-LHH_firstorder_Energy', 'square_glszm_LargeAreaLowGrayLevelEmphasis', 'wavelet-HHH_glszm_ZoneVariance', 'logarithm_glcm_ClusterShade', 'squareroot_glszm_LargeAreaHighGrayLevelEmphasis', 'wavelet-HHL_glszm_LargeAreaLowGrayLevelEmphasis', 'logarithm_glszm_LargeAreaHighGrayLevelEmphasis', 'logarithm_firstorder_Variance', 'square_firstorder_TotalEnergy', 'logarithm_ngtdm_Complexity', 'wavelet-LHL_glszm_LargeAreaEmphasis', 'square_firstorder_Energy', 'gradient_glszm_LargeAreaHighGrayLevelEmphasis', 'wavelet-LLL_glcm_ClusterProminence', 'logarithm_glcm_Autocorrelation', 'logarithm_glszm_HighGrayLevelZoneEmphasis', 'wavelet-LLH_glcm_ClusterProminence', 'logarithm_glcm_ClusterTendency', 'logarithm_gldm_SmallDependenceHighGrayLevelEmphasis', 'wavelet-LLH_gldm_LargeDependenceHighGrayLevelEmphasis', 'wavelet-HHH_gldm_GrayLevelNonUniformity', 'wavelet-LHL_glcm_ClusterProminence', 'wavelet-LHL_gldm_LargeDependenceHighGrayLevelEmphasis', 'wavelet-HHL_glrlm_RunLengthNonUniformity', 'wavelet-HLL_glcm_ClusterProminence', 'logarithm_glszm_GrayLevelVariance', 'wavelet-HLL_gldm_LargeDependenceHighGrayLevelEmphasis', 'wavelet-LLL_firstorder_Variance', 'gradient_glrlm_GrayLevelNonUniformity', 'gradient_glcm_ClusterProminence', 'wavelet-LLL_gldm_LargeDependenceHighGrayLevelEmphasis', 'logarithm_glcm_Contrast', 'original_glcm_ClusterProminence', 'wavelet-HLH_glcm_ClusterProminence', 'wavelet-LHL_ngtdm_Complexity', 'wavelet-HHL_gldm_LargeDependenceHighGrayLevelEmphasis', 'wavelet-HHH_glszm_SizeZoneNonUniformity', 'wavelet-HLH_gldm_LargeDependenceHighGrayLevelEmphasis', 'wavelet-HHL_glcm_ClusterProminence', 'wavelet-HLL_ngtdm_Complexity', 'wavelet-LLH_ngtdm_Complexity', 'gradient_firstorder_Variance', 'wavelet-LHH_gldm_LargeDependenceHighGrayLevelEmphasis', 'wavelet-HHH_ngtdm_Busyness', 'wavelet-LLL_ngtdm_Complexity', 'wavelet-LLL_glcm_ClusterShade', 'wavelet-LHH_glcm_ClusterProminence', 'wavelet-LLH_glszm_SizeZoneNonUniformity', 'diagnostics_Image-original_Minimum', 'square_glrlm_RunLengthNonUniformity', 'original_firstorder_Variance', 'wavelet-LLH_glrlm_LongRunHighGrayLevelEmphasis', 'wavelet-HHL_ngtdm_Complexity', 'wavelet-LHL_glrlm_LongRunHighGrayLevelEmphasis', 'logarithm_firstorder_Range', 'diagnostics_Image-original_Maximum', 'wavelet-HHH_firstorder_Variance', 'logarithm_firstorder_InterquartileRange', 'logarithm_firstorder_Median', 'wavelet-HHH_gldm_LargeDependenceHighGrayLevelEmphasis', 'squareroot_glszm_HighGrayLevelZoneEmphasis', 'wavelet-LLH_glcm_ClusterShade', 'gradient_ngtdm_Complexity', 'squareroot_gldm_SmallDependenceHighGrayLevelEmphasis', 'wavelet-LHH_ngtdm_Complexity', 'logarithm_firstorder_10Percentile', 'wavelet-HLH_ngtdm_Complexity', 'wavelet-HLL_glrlm_LongRunHighGrayLevelEmphasis', 'wavelet-LHL_glcm_Autocorrelation', 'logarithm_firstorder_Mean', 'logarithm_firstorder_Minimum', 'wavelet-HHL_glszm_GrayLevelNonUniformity', 'wavelet-HHH_glcm_ClusterProminence', 'logarithm_firstorder_90Percentile', 'wavelet-LLH_glcm_Autocorrelation', 'wavelet-HHL_glrlm_LongRunHighGrayLevelEmphasis', 'logarithm_firstorder_RootMeanSquared', 'wavelet-HLL_glrlm_HighGrayLevelRunEmphasis', 'logarithm_glszm_ZoneVariance', 'exponential_glrlm_RunLengthNonUniformity', 'square_glszm_GrayLevelNonUniformity', 'lbp-2D_glrlm_LongRunEmphasis', 'wavelet-LLH_gldm_SmallDependenceHighGrayLevelEmphasis', 'wavelet-HHH_ngtdm_Complexity', 'wavelet-HLL_gldm_SmallDependenceHighGrayLevelEmphasis', 'wavelet-LLL_firstorder_Median', 'wavelet-LLL_glrlm_LongRunHighGrayLevelEmphasis', 'wavelet-HLH_glrlm_LongRunHighGrayLevelEmphasis', 'squareroot_glszm_GrayLevelVariance', 'squareroot_glcm_SumSquares', 'diagnostics_Image-original_Mean', 'wavelet-LLL_firstorder_10Percentile', 'wavelet-LLL_firstorder_90Percentile', 'wavelet-LLH_glszm_LargeAreaLowGrayLevelEmphasis', 'wavelet-LLL_glcm_Autocorrelation', 'wavelet-LLL_firstorder_InterquartileRange', 'original_glcm_ClusterShade', 'wavelet-LLL_gldm_SmallDependenceHighGrayLevelEmphasis', 'wavelet-LHL_firstorder_Maximum', 'wavelet-LHH_glrlm_LongRunHighGrayLevelEmphasis', 'wavelet-LLL_firstorder_Maximum', 'wavelet-HLL_firstorder_Maximum', 'wavelet-LLH_firstorder_Range', 'wavelet-LLL_glszm_SmallAreaHighGrayLevelEmphasis', 'wavelet-HHH_glrlm_LongRunHighGrayLevelEmphasis', 'square_gldm_LargeDependenceLowGrayLevelEmphasis', 'gradient_firstorder_Range', 'wavelet-LLL_firstorder_Range', 'wavelet-HHH_glrlm_HighGrayLevelRunEmphasis', 'original_ngtdm_Complexity', 'squareroot_firstorder_Maximum']
        output_features = {k: v for k, v in all_features.items() if k in desired_features}
        print(f"Filtered to {len(output_features)} desired features")
    else:
        output_features = all_features
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "radiomics_features.csv")
    pd.DataFrame([output_features]).to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")


if __name__ == "__main__":
    main()
