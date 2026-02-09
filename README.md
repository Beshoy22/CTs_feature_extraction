# CT Segmentation Feature Extraction

Two standalone scripts for extracting features from CT segmentation data in NRRD format.

## Structure

```
foundation_model/   # NMI foundation model features
PyRadiomics/        # PyRadiomics features
```

Each folder has its own uv environment.

## FM Extraction

Finds the centroid of the largest lesion in the segmentation, then extracts features using the [NMI foundation model](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker).

### Setup

```bash
cd foundation_model
uv sync
```

Place model weights at `foundation_model/fm_extraction/model/model_weights.torch`.
```bash
cd foundation_model/fm_extraction/model
wget -O model_weights.torch "https://zenodo.org/records/10528450/files/model_weights.torch?download=1"
```

### Usage

```bash
uv run python -m fm_extraction.extract_features --input_dir /path/to/data
```

## PyRadiomics Extraction

Extracts radiomics features with image normalization (outlier removal) and resampling to 1x1x1mm spacing.

### Setup

```bash
cd PyRadiomics
uv sync
```

### Usage

```bash
uv run python -m rad_extraction.extract_features --input_dir /path/to/data
```

## Common Options

| Argument       | Default          | Description                        |
|----------------|------------------|------------------------------------|
| `--input_dir`  | `.`              | Directory containing input files   |
| `--image`      | `image.nrrd`     | CT image filename                  |
| `--seg`        | `mask.seg.nrrd`  | Segmentation filename              |
| `--output_dir` | same as input    | Directory for output CSV           |

### Output

- `FM_features.csv` — foundation model features
- `radiomics_features.csv` — PyRadiomics `original_*` features
