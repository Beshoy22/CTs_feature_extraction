import os
import argparse
import tempfile
import SimpleITK as sitk
import numpy as np
import pandas as pd
from radiomics import featureextractor

from utils.imageReaders import read_itk_image, read_itk_segmentation
from utils.imageProcess import (
    crop_image,
    interpolate_image,
    interpolate_roi,
    combine_all_rois,
)
from utils.importSettings import Settings

# ---------------------------------------------------------------------------
# 128 desired radiomic features
# ---------------------------------------------------------------------------
DESIRED_FEATURES = [
    "logarithm_firstorder_Energy",
    "logarithm_firstorder_TotalEnergy",
    "logarithm_glcm_ClusterProminence",
    "wavelet-LLL_firstorder_Energy",
    "wavelet-LLL_firstorder_TotalEnergy",
    "exponential_glszm_ZoneVariance",
    "gradient_firstorder_Energy",
    "wavelet-HLL_glszm_LargeAreaHighGrayLevelEmphasis",
    "exponential_glszm_LargeAreaHighGrayLevelEmphasis",
    "wavelet-LHL_firstorder_Energy",
    "wavelet-LLH_firstorder_TotalEnergy",
    "wavelet-HHL_firstorder_Energy",
    "gradient_firstorder_TotalEnergy",
    "wavelet-LHL_firstorder_TotalEnergy",
    "wavelet-LLH_glszm_LargeAreaHighGrayLevelEmphasis",
    "wavelet-LLH_firstorder_Energy",
    "wavelet-LHH_firstorder_TotalEnergy",
    "wavelet-LHH_firstorder_Energy",
    "square_glszm_LargeAreaLowGrayLevelEmphasis",
    "wavelet-HHH_glszm_ZoneVariance",
    "logarithm_glcm_ClusterShade",
    "squareroot_glszm_LargeAreaHighGrayLevelEmphasis",
    "wavelet-HHL_glszm_LargeAreaLowGrayLevelEmphasis",
    "logarithm_glszm_LargeAreaHighGrayLevelEmphasis",
    "logarithm_firstorder_Variance",
    "square_firstorder_TotalEnergy",
    "logarithm_ngtdm_Complexity",
    "wavelet-LHL_glszm_LargeAreaEmphasis",
    "square_firstorder_Energy",
    "gradient_glszm_LargeAreaHighGrayLevelEmphasis",
    "wavelet-LLL_glcm_ClusterProminence",
    "logarithm_glcm_Autocorrelation",
    "logarithm_glszm_HighGrayLevelZoneEmphasis",
    "wavelet-LLH_glcm_ClusterProminence",
    "logarithm_glcm_ClusterTendency",
    "logarithm_gldm_SmallDependenceHighGrayLevelEmphasis",
    "wavelet-LLH_gldm_LargeDependenceHighGrayLevelEmphasis",
    "wavelet-HHH_gldm_GrayLevelNonUniformity",
    "wavelet-LHL_glcm_ClusterProminence",
    "wavelet-LHL_gldm_LargeDependenceHighGrayLevelEmphasis",
    "wavelet-HHL_glrlm_RunLengthNonUniformity",
    "wavelet-HLL_glcm_ClusterProminence",
    "logarithm_glszm_GrayLevelVariance",
    "wavelet-HLL_gldm_LargeDependenceHighGrayLevelEmphasis",
    "wavelet-LLL_firstorder_Variance",
    "gradient_glrlm_GrayLevelNonUniformity",
    "gradient_glcm_ClusterProminence",
    "wavelet-LLL_gldm_LargeDependenceHighGrayLevelEmphasis",
    "logarithm_glcm_Contrast",
    "original_glcm_ClusterProminence",
    "wavelet-HLH_glcm_ClusterProminence",
    "wavelet-LHL_ngtdm_Complexity",
    "wavelet-HHL_gldm_LargeDependenceHighGrayLevelEmphasis",
    "wavelet-HHH_glszm_SizeZoneNonUniformity",
    "wavelet-HLH_gldm_LargeDependenceHighGrayLevelEmphasis",
    "wavelet-HHL_glcm_ClusterProminence",
    "wavelet-HLL_ngtdm_Complexity",
    "wavelet-LLH_ngtdm_Complexity",
    "gradient_firstorder_Variance",
    "wavelet-LHH_gldm_LargeDependenceHighGrayLevelEmphasis",
    "wavelet-HHH_ngtdm_Busyness",
    "wavelet-LLL_ngtdm_Complexity",
    "wavelet-LLL_glcm_ClusterShade",
    "wavelet-LHH_glcm_ClusterProminence",
    "wavelet-LLH_glszm_SizeZoneNonUniformity",
    "diagnostics_Image-original_Minimum",
    "square_glrlm_RunLengthNonUniformity",
    "original_firstorder_Variance",
    "wavelet-LLH_glrlm_LongRunHighGrayLevelEmphasis",
    "wavelet-HHL_ngtdm_Complexity",
    "wavelet-LHL_glrlm_LongRunHighGrayLevelEmphasis",
    "logarithm_firstorder_Range",
    "diagnostics_Image-original_Maximum",
    "wavelet-HHH_firstorder_Variance",
    "logarithm_firstorder_InterquartileRange",
    "logarithm_firstorder_Median",
    "wavelet-HHH_gldm_LargeDependenceHighGrayLevelEmphasis",
    "squareroot_glszm_HighGrayLevelZoneEmphasis",
    "wavelet-LLH_glcm_ClusterShade",
    "gradient_ngtdm_Complexity",
    "squareroot_gldm_SmallDependenceHighGrayLevelEmphasis",
    "wavelet-LHH_ngtdm_Complexity",
    "logarithm_firstorder_10Percentile",
    "wavelet-HLH_ngtdm_Complexity",
    "wavelet-HLL_glrlm_LongRunHighGrayLevelEmphasis",
    "wavelet-LHL_glcm_Autocorrelation",
    "logarithm_firstorder_Mean",
    "logarithm_firstorder_Minimum",
    "wavelet-HHL_glszm_GrayLevelNonUniformity",
    "wavelet-HHH_glcm_ClusterProminence",
    "logarithm_firstorder_90Percentile",
    "wavelet-LLH_glcm_Autocorrelation",
    "wavelet-HHL_glrlm_LongRunHighGrayLevelEmphasis",
    "logarithm_firstorder_RootMeanSquared",
    "wavelet-HLL_glrlm_HighGrayLevelRunEmphasis",
    "logarithm_glszm_ZoneVariance",
    "exponential_glrlm_RunLengthNonUniformity",
    "square_glszm_GrayLevelNonUniformity",
    "lbp-2D_glrlm_LongRunEmphasis",
    "wavelet-LLH_gldm_SmallDependenceHighGrayLevelEmphasis",
    "wavelet-HHH_ngtdm_Complexity",
    "wavelet-HLL_gldm_SmallDependenceHighGrayLevelEmphasis",
    "wavelet-LLL_firstorder_Median",
    "wavelet-LLL_glrlm_LongRunHighGrayLevelEmphasis",
    "wavelet-HLH_glrlm_LongRunHighGrayLevelEmphasis",
    "squareroot_glszm_GrayLevelVariance",
    "squareroot_glcm_SumSquares",
    "diagnostics_Image-original_Mean",
    "wavelet-LLL_firstorder_10Percentile",
    "wavelet-LLL_firstorder_90Percentile",
    "wavelet-LLH_glszm_LargeAreaLowGrayLevelEmphasis",
    "wavelet-LLL_glcm_Autocorrelation",
    "wavelet-LLL_firstorder_InterquartileRange",
    "original_glcm_ClusterShade",
    "wavelet-LLL_gldm_SmallDependenceHighGrayLevelEmphasis",
    "wavelet-LHL_firstorder_Maximum",
    "wavelet-LHH_glrlm_LongRunHighGrayLevelEmphasis",
    "wavelet-LLL_firstorder_Maximum",
    "wavelet-HLL_firstorder_Maximum",
    "wavelet-LLH_firstorder_Range",
    "wavelet-LLL_glszm_SmallAreaHighGrayLevelEmphasis",
    "wavelet-HHH_glrlm_LongRunHighGrayLevelEmphasis",
    "square_gldm_LargeDependenceLowGrayLevelEmphasis",
    "gradient_firstorder_Range",
    "wavelet-LLL_firstorder_Range",
    "wavelet-HHH_glrlm_HighGrayLevelRunEmphasis",
    "original_ngtdm_Complexity",
    "squareroot_firstorder_Maximum",
]


def extract_features(image, segmentation, pid):
    """
    Identical to utils.i3lungRadiomics.extract_features.
    """
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures()

    features = extractor.execute(image, segmentation)
    features["patient_id"] = pid
    return features


def get_preprocessed_scan(image_path, seg_path):
    """
    Run the EXACT same preprocessing as get_perturbated_scans() in
    i3lungRadiomics.py, but STOP before randomise_roi_contours.

    Pipeline:
        1. read_itk_image  +  read_itk_segmentation
        2. crop_image  (boundary=50.0, z_only=True)
        3. interpolate_image  +  interpolate_roi
        4. combine_all_rois

    Returns:
        image_class_object : ImageClass — preprocessed CT volume
        roi_combined       : RoiClass   — combined (non-perturbed) mask
    """
    settings = Settings()

    # ---- Read ----
    ctscan_obj = read_itk_image(image_path, "CT")
    seg_obj = read_itk_segmentation(seg_path)

    # ---- Crop (boundary=50 mm, z-only) ----
    image_class_object, roi_class_object_list = crop_image(
        img_obj=ctscan_obj,
        roi_list=seg_obj,
        boundary=50.0,
        z_only=True,
    )

    # ---- Interpolate to isotropic spacing (default 1.25 mm) ----
    image_class_object = interpolate_image(
        img_obj=image_class_object, settings=settings
    )
    roi_class_object_list = interpolate_roi(
        img_obj=image_class_object,
        roi_list=roi_class_object_list,
        settings=settings,
    )

    # ---- Combine ROIs ----
    roi_combined = combine_all_rois(
        roi_list=roi_class_object_list, settings=settings
    )

    return image_class_object, roi_combined


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract the 128 selected radiomic features from a CT volume "
            "and segmentation mask.  Follows the exact same preprocessing "
            "pipeline as the I3LUNG perturbation code (crop -> interpolate -> "
            "combine ROIs) but without contour randomisation."
        ),
    )
    parser.add_argument("--input_dir", default=".")
    parser.add_argument("--image", default="image.nrrd")
    parser.add_argument("--seg", default="mask.seg.nrrd")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--troubleshoot",
        action="store_true",
        help=(
            "Save the cropped + interpolated volume and mask to "
            "<output_dir>/troubleshoot/ for visual inspection."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    image_path = os.path.abspath(os.path.join(args.input_dir, args.image))
    seg_path = os.path.abspath(os.path.join(args.input_dir, args.seg))

    # ------------------------------------------------------------------
    # Preprocessing — same pipeline as get_perturbated_scans()
    # ------------------------------------------------------------------
    print(f"Preprocessing:\n  image: {image_path}\n  seg  : {seg_path}")
    image_class_object, roi_combined = get_preprocessed_scan(image_path, seg_path)

    # ------------------------------------------------------------------
    # Convert to SimpleITK the EXACT same way as save_pertrubated_scan_mask:
    #
    #     ctscan_img = sitk.GetImageFromArray(ctscan.voxel_grid)
    #     mask_img   = sitk.GetImageFromArray(mask.roi.voxel_grid)
    #
    # This deliberately drops spacing/origin — pyradiomics will see
    # default (1,1,1) spacing, which is what the perturbed path uses.
    # ------------------------------------------------------------------
    ctscan_img = sitk.GetImageFromArray(image_class_object.voxel_grid)
    mask_img = sitk.GetImageFromArray(roi_combined.roi.voxel_grid)

    # ------------------------------------------------------------------
    # Save to temp files (same as save_pertrubated_scan_mask writes .nrrd)
    # ------------------------------------------------------------------
    tmp_dir = tempfile.mkdtemp()
    volume_path = os.path.join(tmp_dir, "volume.nrrd")
    mask_path = os.path.join(tmp_dir, "mask.nrrd")

    sitk.WriteImage(ctscan_img, volume_path)
    sitk.WriteImage(mask_img, mask_path)

    # ------------------------------------------------------------------
    # Troubleshoot — save copies for inspection
    # ------------------------------------------------------------------
    if args.troubleshoot:
        ts_dir = os.path.join(output_dir, "troubleshoot")
        os.makedirs(ts_dir, exist_ok=True)

        # Files exactly as pyradiomics sees them (no spacing/origin)
        sitk.WriteImage(ctscan_img, os.path.join(ts_dir, "volume_no_metadata.nrrd"))
        sitk.WriteImage(mask_img, os.path.join(ts_dir, "mask_no_metadata.nrrd"))

        # With real spacing/origin so you can overlay in a viewer
        ctscan_img_meta = sitk.GetImageFromArray(image_class_object.voxel_grid)
        ctscan_img_meta.SetSpacing(image_class_object.spacing[::-1].tolist())
        ctscan_img_meta.SetOrigin(image_class_object.origin[::-1].tolist())

        mask_array = roi_combined.roi.voxel_grid.astype(np.uint8)
        mask_img_meta = sitk.GetImageFromArray(mask_array)
        mask_img_meta.SetSpacing(image_class_object.spacing[::-1].tolist())
        mask_img_meta.SetOrigin(image_class_object.origin[::-1].tolist())

        sitk.WriteImage(ctscan_img_meta, os.path.join(ts_dir, "preprocessed_volume.nrrd"))
        sitk.WriteImage(mask_img_meta, os.path.join(ts_dir, "preprocessed_mask.nrrd"))

        print(f"[TROUBLESHOOT] Volume shape  : {image_class_object.voxel_grid.shape}")
        print(f"[TROUBLESHOOT] Volume spacing : {image_class_object.spacing}")
        print(f"[TROUBLESHOOT] Mask sum       : {mask_array.sum()}")
        print(f"[TROUBLESHOOT] Saved to       : {ts_dir}")

    # ------------------------------------------------------------------
    # Feature extraction — same as extract_features in the perturbed path
    # ------------------------------------------------------------------
    print("Extracting features ...")
    all_features = extract_features(volume_path, mask_path, pid="subject")

    # Clean up temp files
    os.remove(volume_path)
    os.remove(mask_path)
    os.rmdir(tmp_dir)

    # ------------------------------------------------------------------
    # Filter to the 128 desired features
    # ------------------------------------------------------------------
    filtered = {"patient_id": all_features["patient_id"]}
    missing = []
    for feat in DESIRED_FEATURES:
        if feat in all_features:
            filtered[feat] = all_features[feat]
        else:
            missing.append(feat)
            filtered[feat] = np.nan

    if missing:
        print(
            f"\n[WARNING] {len(missing)} of 128 features were not found in "
            f"pyradiomics output and were set to NaN:\n  {missing}"
        )

    # Save
    df = pd.DataFrame([filtered])
    out_csv = os.path.join(output_dir, "radiomics_features.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved {len(DESIRED_FEATURES)} features -> {out_csv}")


if __name__ == "__main__":
    main()
