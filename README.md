# Cine Generated Enhancement

This is the official implementation for **"Predicting Late Gadolinium Enhancement of Acute Myocardial Infarction in Contrast-free Cardiac Cine MRI using Deep Generative Learning"**.

We provide:

- **models/**: Directory containing the model definitions used in our study.
- **models/discriminator.py**: The conditional projection discriminator used during CGE training.
- **data/**: Directory containing the pretrained weights and a sample batch from the ACDC dataset.
- **evaluate.ipynb**: A Jupyter notebook containing the inference code and visualization tools to generate CGE images from cine MRI images in the sample batch.
- **scripts/train.py**: A standalone PyTorch training implementation using a public NPZ interface.

## Citation

You can cite our work using the following BibTeX entry:

```bibtex
@article{
title = {Predicting Late Gadolinium Enhancement of Acute Myocardial Infarction in Contrast-Free Cardiac Cine MRI Using Deep Generative Learning},
author = {Haikun Qi and Pengfang Qian and Langlang Tang and Binghua Chen and Dongaolei An and Lian-Ming Wu},
journal = {Circulation: Cardiovascular Imaging},
volume = {17},
number = {9},
pages = {e016786},
year = {2024},
doi = {10.1161/CIRCIMAGING.124.016786},
url = {https://www.ahajournals.org/doi/abs/10.1161/CIRCIMAGING.124.016786},
eprint = {https://www.ahajournals.org/doi/pdf/10.1161/CIRCIMAGING.124.016786}
}
```

## Usage

We adopted Python 3.9 as the development and evaluation environment. You can install the necessary packages according to the provided `environment.yaml` file using `conda`.

1. **Clone the Repository**

    ```bash
    git clone https://github.com/ShanghaiTechCMR/CineGeneratedEnhancement.git
    cd CineGeneratedEnhancement
    ```

2. **Create a Conda Environment**

    Ensure you have Anaconda or Miniconda installed. Create a new conda environment using the `environment.yaml` file:

    ```bash
    conda env create -n py39cge --file environment.yaml
    ```

    Activate the environment:
    
    ```bash
    conda activate py39cge
    ```

3. **Running Inference**

    You can use `evaluate.ipynb` to perform inference with our pretrained weights on the sample batch. Launch Jupyter Notebook Server and open the notebook `evaluate.ipynb`. Run the cells sequentially to perform inference and visualize the generated CGE images from the cine MRI images in the sample batch.

## Training

The training implementation is intentionally independent of the private clinical-data pipeline. It preserves the model, conditional discriminator, active loss terms, optimizer settings, EMA, and alternating update schedule used for CGE training, while requiring users to prepare their own preprocessed data.

### Public NPZ data interface

Pass `--train-data` and, optionally, `--val-data` as directories of NPZ shards. Each `.npz` file represents one training sample and must contain the following arrays. The cine inputs use the same layout and names as `evaluate.ipynb`, except that the per-shard files do not include a batch dimension.

| Key | Type and shape | Description |
| --- | --- | --- |
| `selected_frame` | `float32 [1, 192, 192]` | Cine frame selected for the target slice. |
| `cine_volumes` | `float32 [1, 20, 3, 192, 192]` | Three-slice cine context: channel, time, depth, height, width. |
| `target_image` | `float32 [1, 192, 192]` | Target CGE/PSIR image for supervised training. |
| `class_id` | `int64` scalar in `0..4` | Conditional contrast class. |
| `valid_mask` | optional `bool [192, 192]` | Pixels included in mutual-information loss; omitted means all pixels are valid. |

All image arrays must be finite and normalized to `[0, 1]`. The public loader deliberately does not include patient discovery, DICOM parsing, registration, ROI localization, data-quality filtering, or partitioning logic. Those preprocessing steps must be completed before creating the shards.

For example, a single preprocessed sample can be written with:

```python
np.savez_compressed(
    "train_shards/sample_0001.npz",
    selected_frame=selected_frame.astype(np.float32),
    cine_volumes=cine_volumes.astype(np.float32),
    target_image=target_image.astype(np.float32),
    class_id=np.asarray(contrast_class, dtype=np.int64),
    valid_mask=valid_mask.astype(bool),
)
```

### Run training

Activate the environment created above, then run either command from the repository root:

```bash
python scripts/train.py \
  --train-data /path/to/train_shards \
  --val-data /path/to/validation_shards \
  --output-dir training_runs/cge
```

```bash
bash scripts/train.sh \
  --train-data /path/to/train_shards \
  --val-data /path/to/validation_shards \
  --output-dir training_runs/cge
```

Defaults reproduce the released training configuration: 400 epochs, batch size 4, CUDA bfloat16 autocast, a D/G/D/G alternating update order, hinge GAN loss, generator/discriminator Adam learning rates of `5e-5` and `2e-4`, and generator EMA with decay `0.999`. The active generator objective is `100 × L1 + 10 × MI + 2 × VGG perceptual/style + 2 × adversarial`.

The VGG perceptual objective always uses torchvision's ImageNet-pretrained VGG16 (`VGG16_Weights.DEFAULT`), matching the original training code. `torchvision` downloads the weights on first use if they are not already cached, so an offline environment must pre-populate that cache. Use `--no-augment` to disable the synchronized flip/affine augmentation for debugging.

### Checkpoints and resuming

The output directory contains:

- `latest.pt`: a full training checkpoint saved every 10 epochs and at the final epoch.
- `training_e<epoch>.pt`: full archive checkpoints saved every 100 epochs.
- `generator_e<epoch>.ckpt`: EMA generator weights saved every 100 epochs and at the final epoch. This is a raw `state_dict` compatible with the existing `evaluate.ipynb` loader.
- `metrics.csv` and `run_config.json`: epoch-level metrics and the launch configuration.

Resume the optimizer, schedulers, EMA model, RNG state, and alternating-update state with:

```bash
python scripts/train.py \
  --train-data /path/to/train_shards \
  --val-data /path/to/validation_shards \
  --output-dir training_runs/cge \
  --resume training_runs/cge/latest.pt
```

Only resume checkpoints created by this script or otherwise trusted by you: full training checkpoints include optimizer and RNG state and therefore require normal PyTorch deserialization.
