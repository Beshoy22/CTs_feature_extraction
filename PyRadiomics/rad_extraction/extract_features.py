import os
import argparse
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor, imageoperations

def main():
    parser = argparse.ArgumentParser(description="Extract PyRadiomics features from CT segmentation")
    parser.add_argument("--input_dir", default=".")
    parser.add_argument("--image", default="image.nrrd")
    parser.add_argument("--seg", default="mask.seg.nrrd")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--filter_features", action="store_true", 
                        help="Filter to specific feature set (default: extract all)")
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.input_dir
    image_path = os.path.join(args.input_dir, args.image)
    seg_path = os.path.join(args.input_dir, args.seg)
    
    # Read image and segmentation
    image = sitk.ReadImage(image_path)
    segmentation = sitk.ReadImage(seg_path)
    
    nda_img = sitk.GetArrayViewFromImage(image)
    nda_seg = sitk.GetArrayViewFromImage(segmentation)
    
    if nda_img.shape != nda_seg.shape:
        raise ValueError(f"Shape mismatch: image {nda_img.shape} vs segmentation {nda_seg.shape}")
    
    # ==================================================================
    # PREPROCESSING - Matching your codebase configuration
    # ==================================================================
    # Copy segmentation information to ensure alignment
    segmentation.CopyInformation(image)
    
    # Resample to 1.25 x 1.25 x 1.25 mm (isotropic)
    # Use BSpline interpolation (order 3 = cubic) for image
    # Use nearest neighbor for mask (label interpolation)
    img_resampled, seg_resampled = imageoperations.resampleImage(
        image, segmentation,
        resampledPixelSpacing=[1.25, 1.25, 1.25],  # Match codebase: 1.25mm isotropic
        interpolator=sitk.sitkBSpline,              # Cubic interpolation (order 3)
        label=1,
        padDistance=0,
    )
    
    # ==================================================================
    # FEATURE EXTRACTION - Matching your codebase configuration
    # ==================================================================
    extractor = featureextractor.RadiomicsFeatureExtractor()
    
    # Enable ALL image types (matches enableAllImageTypes() in codebase)
    extractor.enableAllImageTypes()
    
    # Enable ALL features (matches enableAllFeatures() in codebase)
    extractor.enableAllFeatures()
    
    # Extract features
    all_features = extractor.execute(img_resampled, seg_resampled)
    
    # ==================================================================
    # OPTIONAL: Filter to specific features if needed
    # ==================================================================
    if args.filter_features:
        desired_features = [
            'logarithm_firstorder_Energy', 'logarithm_firstorder_TotalEnergy', 
            'logarithm_glcm_ClusterProminence', 'wavelet-LLL_firstorder_Energy', 
            'wavelet-LLL_firstorder_TotalEnergy', 'exponential_glszm_ZoneVariance', 
            'gradient_firstorder_Energy', 'wavelet-HLL_glszm_LargeAreaHighGrayLevelEmphasis', 
            'exponential_glszm_LargeAreaHighGrayLevelEmphasis', 'wavelet-LHL_firstorder_Energy', 
            'wavelet-LLH_firstorder_TotalEnergy', 'wavelet-HHL_firstorder_Energy', 
            'gradient_firstorder_TotalEnergy', 'wavelet-LHL_firstorder_TotalEnergy', 
            'wavelet-LLH_glszm_LargeAreaHighGrayLevelEmphasis', 'wavelet-LLH_firstorder_Energy', 
            'wavelet-LHH_firstorder_TotalEnergy', 'wavelet-LHH_firstorder_Energy', 
            'square_glszm_LargeAreaLowGrayLevelEmphasis', 'wavelet-HHH_glszm_ZoneVariance', 
            'logarithm_glcm_ClusterShade', 'squareroot_glszm_LargeAreaHighGrayLevelEmphasis', 
            'wavelet-HHL_glszm_LargeAreaLowGrayLevelEmphasis', 'logarithm_glszm_LargeAreaHighGrayLevelEmphasis', 
            'logarithm_firstorder_Variance', 'square_firstorder_TotalEnergy', 
            'logarithm_ngtdm_Complexity', 'wavelet-LHL_glszm_LargeAreaEmphasis', 
            'square_firstorder_Energy', 'gradient_glszm_LargeAreaHighGrayLevelEmphasis', 
            'wavelet-LLL_glcm_ClusterProminence', 'logarithm_glcm_Autocorrelation', 
            'logarithm_glszm_HighGrayLevelZoneEmphasis', 'wavelet-LLH_glcm_ClusterProminence', 
            'logarithm_glcm_ClusterTendency', 'logarithm_gldm_SmallDependenceHighGrayLevelEmphasis', 
            'wavelet-LLH_gldm_LargeDependenceHighGrayLevelEmphasis', 'wavelet-HHH_gldm_GrayLevelNonUniformity', 
            'wavelet-LHL_glcm_ClusterProminence', 'wavelet-LHL_gldm_LargeDependenceHighGrayLevelEmphasis', 
            'wavelet-HHL_glrlm_RunLengthNonUniformity', 'wavelet-HLL_glcm_ClusterProminence', 
            'logarithm_glszm_GrayLevelVariance', 'wavelet-HLL_gldm_LargeDependenceHighGrayLevelEmphasis', 
            'wavelet-LLL_firstorder_Variance', 'gradient_glrlm_GrayLevelNonUniformity', 
            'gradient_glcm_ClusterProminence', 'wavelet-LLL_gldm_LargeDependenceHighGrayLevelEmphasis', 
            'logarithm_glcm_Contrast', 'original_glcm_ClusterProminence', 
            'wavelet-HLH_glcm_ClusterProminence', 'wavelet-LHL_ngtdm_Complexity', 
            'wavelet-HHL_gldm_LargeDependenceHighGrayLevelEmphasis', 'wavelet-HHH_glszm_SizeZoneNonUniformity', 
            'wavelet-HLH_gldm_LargeDependenceHighGrayLevelEmphasis', 'wavelet-HHL_glcm_ClusterProminence', 
            'wavelet-HLL_ngtdm_Complexity', 'wavelet-LLH_ngtdm_Complexity', 
            'gradient_firstorder_Variance', 'wavelet-LHH_gldm_LargeDependenceHighGrayLevelEmphasis', 
            'wavelet-HHH_ngtdm_Busyness', 'wavelet-LLL_ngtdm_Complexity', 
            'wavelet-LLL_glcm_ClusterShade', 'wavelet-LHH_glcm_ClusterProminence', 
            'wavelet-LLH_glszm_SizeZoneNonUniformity', 'diagnostics_Image-original_Minimum', 
            'square_glrlm_RunLengthNonUniformity', 'original_firstorder_Variance', 
            'wavelet-LLH_glrlm_LongRunHighGrayLevelEmphasis', 'wavelet-HHL_ngtdm_Complexity', 
            'wavelet-LHL_glrlm_LongRunHighGrayLevelEmphasis', 'logarithm_firstorder_Range', 
            'diagnostics_Image-original_Maximum', 'wavelet-HHH_firstorder_Variance', 
            'logarithm_firstorder_InterquartileRange', 'logarithm_firstorder_Median', 
            'wavelet-HHH_gldm_LargeDependenceHighGrayLevelEmphasis', 'squareroot_glszm_HighGrayLevelZoneEmphasis', 
            'wavelet-LLH_glcm_ClusterShade', 'gradient_ngtdm_Complexity', 
            'squareroot_gldm_SmallDependenceHighGrayLevelEmphasis', 'wavelet-LHH_ngtdm_Complexity', 
            'logarithm_firstorder_10Percentile', 'wavelet-HLH_ngtdm_Complexity', 
            'wavelet-HLL_glrlm_LongRunHighGrayLevelEmphasis', 'wavelet-LHL_glcm_Autocorrelation', 
            'logarithm_firstorder_Mean', 'logarithm_firstorder_Minimum', 
            'wavelet-HHL_glszm_GrayLevelNonUniformity', 'wavelet-HHH_glcm_ClusterProminence', 
            'logarithm_firstorder_90Percentile', 'wavelet-LLH_glcm_Autocorrelation', 
            'wavelet-HHL_glrlm_LongRunHighGrayLevelEmphasis', 'logarithm_firstorder_RootMeanSquared', 
            'wavelet-HLL_glrlm_HighGrayLevelRunEmphasis', 'logarithm_glszm_ZoneVariance', 
            'exponential_glrlm_RunLengthNonUniformity', 'square_glszm_GrayLevelNonUniformity', 
            'lbp-2D_glrlm_LongRunEmphasis', 'wavelet-LLH_gldm_SmallDependenceHighGrayLevelEmphasis', 
            'wavelet-HHH_ngtdm_Complexity', 'wavelet-HLL_gldm_SmallDependenceHighGrayLevelEmphasis', 
            'wavelet-LLL_firstorder_Median', 'wavelet-LLL_glrlm_LongRunHighGrayLevelEmphasis', 
            'wavelet-HLH_glrlm_LongRunHighGrayLevelEmphasis', 'squareroot_glszm_GrayLevelVariance', 
            'squareroot_glcm_SumSquares', 'diagnostics_Image-original_Mean', 
            'wavelet-LLL_firstorder_10Percentile', 'wavelet-LLL_firstorder_90Percentile', 
            'wavelet-LLH_glszm_LargeAreaLowGrayLevelEmphasis', 'wavelet-LLL_glcm_Autocorrelation', 
            'wavelet-LLL_firstorder_InterquartileRange', 'original_glcm_ClusterShade', 
            'wavelet-LLL_gldm_SmallDependenceHighGrayLevelEmphasis', 'wavelet-LHL_firstorder_Maximum', 
            'wavelet-LHH_glrlm_LongRunHighGrayLevelEmphasis', 'wavelet-LLL_firstorder_Maximum', 
            'wavelet-HLL_firstorder_Maximum', 'wavelet-LLH_firstorder_Range', 
            'wavelet-LLL_glszm_SmallAreaHighGrayLevelEmphasis', 'wavelet-HHH_glrlm_LongRunHighGrayLevelEmphasis', 
            'square_gldm_LargeDependenceLowGrayLevelEmphasis', 'gradient_firstorder_Range', 
            'wavelet-LLL_firstorder_Range', 'wavelet-HHH_glrlm_HighGrayLevelRunEmphasis', 
            'original_ngtdm_Complexity', 'squareroot_firstorder_Maximum'
        ]
        output_features = {k: v for k, v in all_features.items() if k in desired_features}
    else:
        output_features = all_features
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "radiomics_features.csv")
    pd.DataFrame([output_features]).to_csv(output_path, index=False)
    print(f"Extracted {len(output_features)} features")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
