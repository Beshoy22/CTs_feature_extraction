import os
import argparse
import SimpleITK as sitk
import numpy as np
import pandas as pd
from radiomics import featureextractor

# ---------------------------------------------------------------------------
# 128 desired radiomic features (subset of pyradiomics enableAll output)
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
    Extract radiomic features — identical to utils.i3lungRadiomics.extract_features.

    Uses the default RadiomicsFeatureExtractor with all image types
    (original, wavelet, LoG, square, squareroot, logarithm, exponential,
    gradient, lbp-2D) and all feature classes enabled.

    Parameters
    ----------
    image : str or sitk.Image
        Path to the CT volume or a SimpleITK image object.
    segmentation : str or sitk.Image
        Path to the segmentation mask or a SimpleITK image object.
    pid : str
        Patient / subject identifier.

    Returns
    -------
    dict
        Dictionary of all extracted features plus ``patient_id``.
    """
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures()

    features = extractor.execute(image, segmentation)
    features["patient_id"] = pid
    return features


def run_troubleshoot(image_path, seg_path, output_dir):
    """
    Run the same crop → interpolate pipeline used by the perturbation code
    and save the intermediate volumes so they can be visually inspected.

    This replicates the preprocessing inside ``get_perturbated_scans``
    (i3lungRadiomics.py) WITHOUT the contour-randomisation step.
    """
    # Lazy-import the MIRP utilities — they are only needed in troubleshoot
    # mode and the user may not always have them on the PYTHONPATH.
    from utils.imageReaders import read_itk_image, read_itk_segmentation
    from utils.imageProcess import (
        crop_image,
        interpolate_image,
        interpolate_roi,
        combine_all_rois,
    )
    from utils.importSettings import Settings

    settings = Settings()

    # ---- Read ----
    ctscan_obj = read_itk_image(image_path, "CT")
    seg_obj = read_itk_segmentation(seg_path)

    # ---- Crop (boundary=50 mm, z-only) — same as get_perturbated_scans ----
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

    # ---- Combine ROIs (in case of multi-label segmentation) ----
    roi_class_object = combine_all_rois(
        roi_list=roi_class_object_list, settings=settings
    )

    # ---- Save ----
    ts_dir = os.path.join(output_dir, "troubleshoot")
    os.makedirs(ts_dir, exist_ok=True)

    # Save as SimpleITK images with spacing/origin preserved
    cropped_vol = sitk.GetImageFromArray(image_class_object.voxel_grid)
    cropped_vol.SetSpacing(image_class_object.spacing[::-1].tolist())
    cropped_vol.SetOrigin(image_class_object.origin[::-1].tolist())

    mask_array = roi_class_object.roi.get_voxel_grid().astype(np.uint8)
    cropped_mask = sitk.GetImageFromArray(mask_array)
    cropped_mask.SetSpacing(image_class_object.spacing[::-1].tolist())
    cropped_mask.SetOrigin(image_class_object.origin[::-1].tolist())

    vol_out = os.path.join(ts_dir, "preprocessed_volume.nrrd")
    mask_out = os.path.join(ts_dir, "preprocessed_mask.nrrd")

    sitk.WriteImage(cropped_vol, vol_out)
    sitk.WriteImage(cropped_mask, mask_out)

    print(f"[TROUBLESHOOT] Saved preprocessed volume → {vol_out}")
    print(f"[TROUBLESHOOT] Saved preprocessed mask   → {mask_out}")
    print(f"[TROUBLESHOOT] Volume shape : {image_class_object.voxel_grid.shape}")
    print(f"[TROUBLESHOOT] Volume spacing: {image_class_object.spacing}")
    print(f"[TROUBLESHOOT] Mask sum      : {mask_array.sum()}")

    return ts_dir


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract the 128 selected radiomic features from a CT volume "
            "and segmentation mask.  Replicates the vanilla (non-perturbed) "
            "extraction path of I3LUNG_march_2025_multi_thread.py exactly."
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
            "Save the cropped + interpolated volume and mask produced by the "
            "MIRP preprocessing pipeline (the same one used before perturbation) "
            "into <output_dir>/troubleshoot/ for visual inspection."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    image_path = os.path.abspath(os.path.join(args.input_dir, args.image))
    seg_path = os.path.abspath(os.path.join(args.input_dir, args.seg))

    # ------------------------------------------------------------------
    # Troubleshoot: save the preprocessed (crop + interpolate) volumes
    # so the user can inspect them.  This does NOT affect feature values.
    # ------------------------------------------------------------------
    if args.troubleshoot:
        run_troubleshoot(image_path, seg_path, output_dir)

    # ------------------------------------------------------------------
    # Feature extraction — identical to extract_features_non_perturbed()
    # in I3LUNG_march_2025_multi_thread.py.
    #
    # NOTE: the vanilla path passes the ORIGINAL image and segmentation
    # directly to pyradiomics.  No crop, no interpolation, no binary
    # mask thresholding.  This is exactly what the multi-thread script
    # does for the "vanilla_" prefixed columns.
    # ------------------------------------------------------------------
    print(f"Extracting features from:\n  image: {image_path}\n  seg  : {seg_path}")
    all_features = extract_features(image_path, seg_path, pid="subject")

    # Filter to the 128 desired features
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
    print(f"\n[INFO] Saved {len(DESIRED_FEATURES)} features → {out_csv}")


if __name__ == "__main__":
    main()
