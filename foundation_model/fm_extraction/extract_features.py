import os
import argparse
import tempfile
import SimpleITK as sitk
import numpy as np
import pandas as pd
from scipy import ndimage
from fmcib.run import get_features


def extract_centroid(seg_path):
    seg_image = sitk.ReadImage(seg_path)
    seg_array = sitk.GetArrayFromImage(seg_image)

    labeled_array, num_features = ndimage.label(seg_array > 0)
    largest_region = None
    largest_region_size = 0

    for region_label in range(1, num_features + 1):
        region_size = np.sum(labeled_array == region_label)
        if region_size > largest_region_size:
            largest_region_size = region_size
            largest_region = (labeled_array == region_label)

    if largest_region is None:
        raise ValueError(f"No segmented region found in {seg_path}")

    indices = np.argwhere(largest_region)
    centroid_pixel = np.mean(indices, axis=0)
    centroid_physical = seg_image.TransformContinuousIndexToPhysicalPoint(centroid_pixel[::-1])

    return centroid_physical


def main():
    parser = argparse.ArgumentParser(description="Extract NMI foundation model features from CT segmentation")
    parser.add_argument("--input_dir", default=".")
    parser.add_argument("--image", default="image.nrrd")
    parser.add_argument("--seg", default="mask.seg.nrrd")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    image_path = os.path.abspath(os.path.join(args.input_dir, args.image))
    seg_path = os.path.abspath(os.path.join(args.input_dir, args.seg))
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "model_weights.torch")

    # Extract centroid of largest lesion
    centroid_x, centroid_y, centroid_z = extract_centroid(seg_path)

    # Create temp CSV for fmcib
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_csv = f.name
        pd.DataFrame([{
            "image_path": image_path,
            "coordX": centroid_x,
            "coordY": centroid_y,
            "coordZ": centroid_z,
        }]).to_csv(f, index=False)

    try:
        feature_df = get_features(temp_csv, weights_path=weights_path, spatial_size=(50, 50, 50))
        drop_cols = [c for c in ["image_path", "coordX", "coordY", "coordZ"] if c in feature_df.columns]
        feature_df = feature_df.drop(columns=drop_cols)
        os.makedirs(output_dir, exist_ok=True)
        feature_df.to_csv(os.path.join(output_dir, "FM_features.csv"), index=False)
    finally:
        os.unlink(temp_csv)


if __name__ == "__main__":
    main()
