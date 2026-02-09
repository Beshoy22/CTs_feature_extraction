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
    
    # Normalize image and resample
    im_norm = imageoperations.normalizeImage(image, removeOutliers=True)
    segmentation.CopyInformation(im_norm)
    img_resampled, seg_resampled = imageoperations.resampleImage(
        im_norm, segmentation,
        resampledPixelSpacing=[1, 1, 1],
        interpolator=sitk.sitkLinear,
        label=1,
        padDistance=0,
    )
    
    # Configure extractor with specific features
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableImageTypeByName("Original")
    
    # PASTE YOUR FEATURE LIST HERE
    # Remove "original_" prefix and split into feature class and feature name
    desired_features = ['vanilla_logarithm_firstorder_Energy', 'vanilla_logarithm_firstorder_TotalEnergy', 'vanilla_logarithm_glcm_ClusterProminence', 'vanilla_wavelet-LLL_firstorder_Energy', 'vanilla_wavelet-LLL_firstorder_TotalEnergy', 'vanilla_exponential_glszm_ZoneVariance', 'vanilla_gradient_firstorder_Energy', 'vanilla_wavelet-HLL_glszm_LargeAreaHighGrayLevelEmphasis', 'vanilla_exponential_glszm_LargeAreaHighGrayLevelEmphasis', 'vanilla_wavelet-LHL_firstorder_Energy', 'vanilla_wavelet-LLH_firstorder_TotalEnergy', 'vanilla_wavelet-HHL_firstorder_Energy', 'vanilla_gradient_firstorder_TotalEnergy', 'vanilla_wavelet-LHL_firstorder_TotalEnergy', 'vanilla_wavelet-LLH_glszm_LargeAreaHighGrayLevelEmphasis', 'vanilla_wavelet-LLH_firstorder_Energy', 'vanilla_wavelet-LHH_firstorder_TotalEnergy', 'vanilla_wavelet-LHH_firstorder_Energy', 'vanilla_square_glszm_LargeAreaLowGrayLevelEmphasis', 'vanilla_wavelet-HHH_glszm_ZoneVariance', 'vanilla_logarithm_glcm_ClusterShade', 'vanilla_squareroot_glszm_LargeAreaHighGrayLevelEmphasis', 'vanilla_wavelet-HHL_glszm_LargeAreaLowGrayLevelEmphasis', 'vanilla_logarithm_glszm_LargeAreaHighGrayLevelEmphasis', 'vanilla_logarithm_firstorder_Variance', 'vanilla_square_firstorder_TotalEnergy', 'vanilla_logarithm_ngtdm_Complexity', 'vanilla_wavelet-LHL_glszm_LargeAreaEmphasis', 'vanilla_square_firstorder_Energy', 'vanilla_gradient_glszm_LargeAreaHighGrayLevelEmphasis', 'vanilla_wavelet-LLL_glcm_ClusterProminence', 'vanilla_logarithm_glcm_Autocorrelation', 'vanilla_logarithm_glszm_HighGrayLevelZoneEmphasis', 'vanilla_wavelet-LLH_glcm_ClusterProminence', 'vanilla_logarithm_glcm_ClusterTendency', 'vanilla_logarithm_gldm_SmallDependenceHighGrayLevelEmphasis', 'vanilla_wavelet-LLH_gldm_LargeDependenceHighGrayLevelEmphasis', 'vanilla_wavelet-HHH_gldm_GrayLevelNonUniformity', 'vanilla_wavelet-LHL_glcm_ClusterProminence', 'vanilla_wavelet-LHL_gldm_LargeDependenceHighGrayLevelEmphasis', 'vanilla_wavelet-HHL_glrlm_RunLengthNonUniformity', 'vanilla_wavelet-HLL_glcm_ClusterProminence', 'vanilla_logarithm_glszm_GrayLevelVariance', 'vanilla_wavelet-HLL_gldm_LargeDependenceHighGrayLevelEmphasis', 'vanilla_wavelet-LLL_firstorder_Variance', 'vanilla_gradient_glrlm_GrayLevelNonUniformity', 'vanilla_gradient_glcm_ClusterProminence', 'vanilla_wavelet-LLL_gldm_LargeDependenceHighGrayLevelEmphasis', 'vanilla_logarithm_glcm_Contrast', 'vanilla_original_glcm_ClusterProminence', 'vanilla_wavelet-HLH_glcm_ClusterProminence', 'vanilla_wavelet-LHL_ngtdm_Complexity', 'vanilla_wavelet-HHL_gldm_LargeDependenceHighGrayLevelEmphasis', 'vanilla_wavelet-HHH_glszm_SizeZoneNonUniformity', 'vanilla_wavelet-HLH_gldm_LargeDependenceHighGrayLevelEmphasis', 'vanilla_wavelet-HHL_glcm_ClusterProminence', 'vanilla_wavelet-HLL_ngtdm_Complexity', 'vanilla_wavelet-LLH_ngtdm_Complexity', 'vanilla_gradient_firstorder_Variance', 'vanilla_wavelet-LHH_gldm_LargeDependenceHighGrayLevelEmphasis', 'vanilla_wavelet-HHH_ngtdm_Busyness', 'vanilla_wavelet-LLL_ngtdm_Complexity', 'vanilla_wavelet-LLL_glcm_ClusterShade', 'vanilla_wavelet-LHH_glcm_ClusterProminence', 'vanilla_wavelet-LLH_glszm_SizeZoneNonUniformity', 'vanilla_diagnostics_Image-original_Minimum', 'vanilla_square_glrlm_RunLengthNonUniformity', 'vanilla_original_firstorder_Variance', 'vanilla_wavelet-LLH_glrlm_LongRunHighGrayLevelEmphasis', 'vanilla_wavelet-HHL_ngtdm_Complexity', 'vanilla_wavelet-LHL_glrlm_LongRunHighGrayLevelEmphasis', 'vanilla_logarithm_firstorder_Range', 'vanilla_diagnostics_Image-original_Maximum', 'vanilla_wavelet-HHH_firstorder_Variance', 'vanilla_logarithm_firstorder_InterquartileRange', 'vanilla_logarithm_firstorder_Median', 'vanilla_wavelet-HHH_gldm_LargeDependenceHighGrayLevelEmphasis', 'vanilla_squareroot_glszm_HighGrayLevelZoneEmphasis', 'vanilla_wavelet-LLH_glcm_ClusterShade', 'vanilla_gradient_ngtdm_Complexity', 'vanilla_squareroot_gldm_SmallDependenceHighGrayLevelEmphasis', 'vanilla_wavelet-LHH_ngtdm_Complexity', 'vanilla_logarithm_firstorder_10Percentile', 'vanilla_wavelet-HLH_ngtdm_Complexity', 'vanilla_wavelet-HLL_glrlm_LongRunHighGrayLevelEmphasis', 'vanilla_wavelet-LHL_glcm_Autocorrelation', 'vanilla_logarithm_firstorder_Mean', 'vanilla_logarithm_firstorder_Minimum', 'vanilla_wavelet-HHL_glszm_GrayLevelNonUniformity', 'vanilla_wavelet-HHH_glcm_ClusterProminence', 'vanilla_logarithm_firstorder_90Percentile', 'vanilla_wavelet-LLH_glcm_Autocorrelation', 'vanilla_wavelet-HHL_glrlm_LongRunHighGrayLevelEmphasis', 'vanilla_logarithm_firstorder_RootMeanSquared', 'vanilla_wavelet-HLL_glrlm_HighGrayLevelRunEmphasis', 'vanilla_logarithm_glszm_ZoneVariance', 'vanilla_exponential_glrlm_RunLengthNonUniformity', 'vanilla_square_glszm_GrayLevelNonUniformity', 'vanilla_lbp-2D_glrlm_LongRunEmphasis', 'vanilla_wavelet-LLH_gldm_SmallDependenceHighGrayLevelEmphasis', 'vanilla_wavelet-HHH_ngtdm_Complexity', 'vanilla_wavelet-HLL_gldm_SmallDependenceHighGrayLevelEmphasis', 'vanilla_wavelet-LLL_firstorder_Median', 'vanilla_wavelet-LLL_glrlm_LongRunHighGrayLevelEmphasis', 'vanilla_wavelet-HLH_glrlm_LongRunHighGrayLevelEmphasis', 'vanilla_squareroot_glszm_GrayLevelVariance', 'vanilla_squareroot_glcm_SumSquares', 'vanilla_diagnostics_Image-original_Mean', 'vanilla_wavelet-LLL_firstorder_10Percentile', 'vanilla_wavelet-LLL_firstorder_90Percentile', 'vanilla_wavelet-LLH_glszm_LargeAreaLowGrayLevelEmphasis', 'vanilla_wavelet-LLL_glcm_Autocorrelation', 'vanilla_wavelet-LLL_firstorder_InterquartileRange', 'vanilla_original_glcm_ClusterShade', 'vanilla_wavelet-LLL_gldm_SmallDependenceHighGrayLevelEmphasis', 'vanilla_wavelet-LHL_firstorder_Maximum', 'vanilla_wavelet-LHH_glrlm_LongRunHighGrayLevelEmphasis', 'vanilla_wavelet-LLL_firstorder_Maximum', 'vanilla_wavelet-HLL_firstorder_Maximum', 'vanilla_wavelet-LLH_firstorder_Range', 'vanilla_wavelet-LLL_glszm_SmallAreaHighGrayLevelEmphasis', 'vanilla_wavelet-HHH_glrlm_LongRunHighGrayLevelEmphasis', 'vanilla_square_gldm_LargeDependenceLowGrayLevelEmphasis', 'vanilla_gradient_firstorder_Range', 'vanilla_wavelet-LLL_firstorder_Range', 'vanilla_wavelet-HHH_glrlm_HighGrayLevelRunEmphasis', 'vanilla_original_ngtdm_Complexity', 'vanilla_squareroot_firstorder_Maximum']
    
    # Disable all features first
    extractor.disableAllFeatures()
    
    # Enable only the desired features
    for feature_name in desired_features:
        # Remove "original_" prefix
        if feature_name.startswith("original_"):
            feature_name = feature_name.replace("original_", "")
        
        # Split into class and name (e.g., "firstorder_Mean" -> class="firstorder", name="Mean")
        feature_class, feature = feature_name.split("_", 1)
        
        # Enable the specific feature
        extractor.enableFeaturesByName(**{feature_class: [feature]})
    
    # Extract features
    features = extractor.execute(img_resampled, seg_resampled)
    
    # Keep only original_ features
    original_features = {k: v for k, v in features.items() if k.startswith("original_")}
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame([original_features]).to_csv(
        os.path.join(output_dir, "radiomics_features.csv"), index=False
    )

if __name__ == "__main__":
    main()
