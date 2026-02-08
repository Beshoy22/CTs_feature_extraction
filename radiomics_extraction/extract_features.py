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

    # Extract features
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableImageTypeByName("Original")
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
