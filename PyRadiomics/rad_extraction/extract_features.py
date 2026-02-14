"""
Extract the 128 selected radiomic features from a CT volume and segmentation mask.

Replicates the VANILLA (non-perturbed) extraction path of
I3LUNG_march_2025_multi_thread.py exactly:

    extract_features_non_perturbed(patient):
        volume_path = os.path.normpath(patient['image_path'])
        mask_path   = os.path.normpath(patient['seg_path'])
        radiomics_features = extract_features(volume_path, mask_path, patient['Subject'])

That function passes the ORIGINAL file paths directly to pyradiomics.
No crop, no interpolation, no GetImageFromArray.
"""

import os
import argparse
import SimpleITK as sitk
import numpy as np
import pandas as pd
from radiomics import featureextractor

# ---------------------------------------------------------------------------
# Match the global side-effect from the I3LUNG import chain:
#
#   I3LUNG_march_2025_multi_thread.py
#     -> from utils.i3lungRadiomics import extract_features
#       -> from utils.imageReaders import ...
#         -> sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
#
# This fires BEFORE any pyradiomics call in the I3LUNG code.
# ---------------------------------------------------------------------------
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)

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


def diagnose_inputs(image_path, seg_path):
    """
    Print everything pyradiomics will see when it reads these files.
    This helps identify mismatches vs the I3LUNG extraction.
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTICS — what pyradiomics will see")
    print("=" * 70)

    img = sitk.ReadImage(image_path)
    seg = sitk.ReadImage(seg_path)

    seg_array = sitk.GetArrayFromImage(seg)
    unique_labels = np.unique(seg_array)

    print(f"\n  IMAGE: {image_path}")
    print(f"    Size    : {img.GetSize()}")
    print(f"    Spacing : {img.GetSpacing()}")
    print(f"    Origin  : {img.GetOrigin()}")
    print(f"    PixelID : {img.GetPixelIDTypeAsString()}")
    print(f"    Dtype   : {sitk.GetArrayFromImage(img).dtype}")

    print(f"\n  MASK: {seg_path}")
    print(f"    Size    : {seg.GetSize()}")
    print(f"    Spacing : {seg.GetSpacing()}")
    print(f"    Origin  : {seg.GetOrigin()}")
    print(f"    PixelID : {seg.GetPixelIDTypeAsString()}")
    print(f"    Dtype   : {seg_array.dtype}")
    print(f"    Unique label values : {unique_labels}")
    print(f"    Total non-zero voxels: {np.count_nonzero(seg_array)}")

    # Check what pyradiomics will actually use (default label=1)
    label1_count = np.sum(seg_array == 1)
    print(f"\n    Voxels with label=1 : {label1_count}")

    if label1_count == 0:
        print(f"\n    *** WARNING: NO VOXELS WITH LABEL=1 ***")
        print(f"    *** Pyradiomics default label is 1.")
        print(f"    *** Your mask has labels: {unique_labels[unique_labels != 0]}")
        print(f"    *** This WILL produce wrong features!")
        print(f"    *** You likely need: --label {unique_labels[unique_labels != 0][0]}")

    # Check size/spacing match
    if img.GetSize() != seg.GetSize():
        print(f"\n    *** WARNING: IMAGE AND MASK SIZE MISMATCH ***")
        print(f"    *** Image: {img.GetSize()}, Mask: {seg.GetSize()}")

    img_spacing_rounded = tuple(round(s, 4) for s in img.GetSpacing())
    seg_spacing_rounded = tuple(round(s, 4) for s in seg.GetSpacing())
    if img_spacing_rounded != seg_spacing_rounded:
        print(f"\n    *** WARNING: IMAGE AND MASK SPACING MISMATCH ***")
        print(f"    *** Image: {img.GetSpacing()}, Mask: {seg.GetSpacing()}")

    # Voxel volume — this is what makes TotalEnergy != Energy
    voxel_vol = float(np.prod(img.GetSpacing()))
    print(f"\n    Voxel volume (spacing product): {voxel_vol}")
    print(f"    If this is 1.0, Energy == TotalEnergy")

    print("=" * 70 + "\n")

    return unique_labels


def extract_features(image, segmentation, pid, label=1):
    """
    Identical to utils.i3lungRadiomics.extract_features,
    except allows overriding the mask label.
    """
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures()

    # Match pyradiomics default, but allow override for non-standard masks
    if label != 1:
        extractor.settings["label"] = label

    features = extractor.execute(image, segmentation)
    features["patient_id"] = pid
    return features


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract the 128 selected radiomic features. "
            "Matches I3LUNG vanilla path exactly: passes original files "
            "directly to pyradiomics with no preprocessing."
        ),
    )
    parser.add_argument("--input_dir", default=".")
    parser.add_argument("--image", default="image.nrrd")
    parser.add_argument("--seg", default="mask.seg.nrrd")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--label", type=int, default=None,
        help=(
            "Mask label value to use. Default: auto-detect if mask has "
            "exactly one non-zero label, otherwise 1 (pyradiomics default). "
            "Use this if your mask uses e.g. 255 instead of 1."
        ),
    )
    parser.add_argument(
        "--troubleshoot",
        action="store_true",
        help="Print detailed diagnostics about inputs before extracting.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    image_path = os.path.abspath(os.path.join(args.input_dir, args.image))
    seg_path = os.path.abspath(os.path.join(args.input_dir, args.seg))

    # ------------------------------------------------------------------
    # Always show basic info; --troubleshoot shows full diagnostics
    # ------------------------------------------------------------------
    print(f"Image: {image_path}")
    print(f"Seg  : {seg_path}")

    if args.troubleshoot:
        unique_labels = diagnose_inputs(image_path, seg_path)
    else:
        # Quick label check even without troubleshoot
        seg = sitk.ReadImage(seg_path)
        seg_array = sitk.GetArrayFromImage(seg)
        unique_labels = np.unique(seg_array)

    # ------------------------------------------------------------------
    # Determine mask label
    # ------------------------------------------------------------------
    non_zero_labels = unique_labels[unique_labels != 0]

    if args.label is not None:
        label = args.label
        print(f"Using user-specified label: {label}")
    elif len(non_zero_labels) == 1 and non_zero_labels[0] != 1:
        # Auto-detect: mask has exactly one non-zero label and it's not 1
        label = int(non_zero_labels[0])
        print(f"Auto-detected mask label: {label} (not the default 1!)")
    else:
        label = 1
        if 1 not in non_zero_labels:
            print(
                f"\n*** CRITICAL: mask has no label=1 voxels! "
                f"Labels found: {non_zero_labels}. "
                f"Use --label {int(non_zero_labels[0])} ***\n"
            )

    # ------------------------------------------------------------------
    # Feature extraction — identical to I3LUNG vanilla path:
    #
    #   extract_features_non_perturbed:
    #       volume_path = os.path.normpath(patient['image_path'])
    #       mask_path = os.path.normpath(patient['seg_path'])
    #       extract_features(volume_path, mask_path, patient['Subject'])
    #
    # Passes ORIGINAL file paths directly to pyradiomics.
    # No crop, no interpolation, no GetImageFromArray.
    # ------------------------------------------------------------------
    print(f"Extracting features (label={label}) ...")
    all_features = extract_features(
        os.path.normpath(image_path),
        os.path.normpath(seg_path),
        pid="subject",
        label=label,
    )

    # ------------------------------------------------------------------
    # Quick sanity check
    # ------------------------------------------------------------------
    energy = all_features.get("original_firstorder_Energy")
    total_energy = all_features.get("original_firstorder_TotalEnergy")
    if energy is not None and total_energy is not None:
        if energy == total_energy:
            print("\n*** WARNING: Energy == TotalEnergy — spacing is (1,1,1)!")
            print("*** This means pyradiomics is not seeing real voxel spacing.")
        else:
            ratio = float(total_energy) / float(energy) if float(energy) != 0 else 0
            print(f"\n  Sanity check: TotalEnergy / Energy = {ratio:.6f} (voxel volume)")

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
            f"\n[WARNING] {len(missing)} of 128 features not found in "
            f"pyradiomics output (set to NaN):\n  {missing}"
        )

    # Save
    df = pd.DataFrame([filtered])
    out_csv = os.path.join(output_dir, "radiomics_features.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved {len(DESIRED_FEATURES)} features -> {out_csv}")


if __name__ == "__main__":
    main()
