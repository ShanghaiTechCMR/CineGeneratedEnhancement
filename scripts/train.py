#!/usr/bin/env python3
"""Train the Cine Generated Enhancement (CGE) model from NPZ shards.

This script deliberately uses a small, public data contract instead of the
private data-loading pipeline used during the study.  See ``README.md`` for
the required fields in each NPZ shard.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
from torchvision.models import VGG16_Weights, vgg16
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.discriminator import CGANSADiscriminator
from models.generator import Generator


IMAGE_SIZE = 192
NUM_CINE_FRAMES = 20
DEPTH_WINDOW_SIZE = 3
NUM_CLASSES = 5
EMA_BETA = 0.999
METRIC_COLUMNS = (
    "epoch",
    "global_step",
    "learning_rate_G",
    "learning_rate_D",
    "seconds",
    "train_loss_G",
    "train_loss_l1",
    "train_loss_mi",
    "train_loss_perceptual",
    "train_loss_adversarial",
    "train_loss_D",
    "train_loss_D_real",
    "train_loss_D_fake",
    "val_loss_G",
    "val_loss_l1",
    "val_loss_mi",
    "val_loss_perceptual",
    "val_loss_adversarial",
    "val_loss_D",
    "val_loss_D_real",
    "val_loss_D_fake",
)


class NPZShardDataset(Dataset):
    """Dataset of independently stored, preprocessed public NPZ samples.

    Each shard is intentionally self-contained.  The dataset neither knows
    about patient identifiers nor implements registration, cropping, DICOM
    parsing, or cohort partitioning.
    """

    REQUIRED_KEYS = ("selected_frame", "cine_volumes", "target_image", "class_id")

    def __init__(
        self,
        directory: Path,
        *,
        augment: bool = False,
        image_size: int = IMAGE_SIZE,
        num_frames: int = NUM_CINE_FRAMES,
        depth_window_size: int = DEPTH_WINDOW_SIZE,
    ) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise ValueError(f"NPZ shard directory does not exist: {self.directory}")
        self.files = sorted(path for path in self.directory.glob("*.npz") if path.is_file())
        if not self.files:
            raise ValueError(f"No .npz shards were found in {self.directory}")

        self.augment = SynchronizedGeometricAugmentation() if augment else None
        self.image_size = image_size
        self.num_frames = num_frames
        self.depth_window_size = depth_window_size

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _require_array(
        arrays: Mapping[str, np.ndarray], key: str, path: Path
    ) -> np.ndarray:
        if key not in arrays:
            raise ValueError(f"{path}: missing required NPZ key '{key}'")
        return np.asarray(arrays[key])

    def _validate_image(
        self, array: np.ndarray, expected_shape: Tuple[int, ...], key: str, path: Path
    ) -> np.ndarray:
        if array.shape != expected_shape:
            raise ValueError(
                f"{path}: '{key}' must have shape {expected_shape}, got {array.shape}"
            )
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{path}: '{key}' must be a floating-point array, got {array.dtype}")
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: '{key}' contains NaN or infinity")
        minimum = float(array.min())
        maximum = float(array.max())
        if minimum < -1e-6 or maximum > 1.0 + 1e-6:
            raise ValueError(
                f"{path}: '{key}' must be normalized to [0, 1], got range "
                f"[{minimum:.5g}, {maximum:.5g}]"
            )
        return np.ascontiguousarray(array, dtype=np.float32)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as arrays:
            selected_frame = self._validate_image(
                self._require_array(arrays, "selected_frame", path),
                (1, self.image_size, self.image_size),
                "selected_frame",
                path,
            )
            cine_volumes = self._validate_image(
                self._require_array(arrays, "cine_volumes", path),
                (
                    1,
                    self.num_frames,
                    self.depth_window_size,
                    self.image_size,
                    self.image_size,
                ),
                "cine_volumes",
                path,
            )
            target_image = self._validate_image(
                self._require_array(arrays, "target_image", path),
                (1, self.image_size, self.image_size),
                "target_image",
                path,
            )

            class_id_array = self._require_array(arrays, "class_id", path)
            if class_id_array.size != 1 or not np.issubdtype(class_id_array.dtype, np.integer):
                raise ValueError(f"{path}: 'class_id' must be one integer scalar")
            class_id = int(class_id_array.reshape(-1)[0])
            if not 0 <= class_id < NUM_CLASSES:
                raise ValueError(
                    f"{path}: 'class_id' must be in [0, {NUM_CLASSES - 1}], got {class_id}"
                )

            if "valid_mask" in arrays:
                valid_mask = np.asarray(arrays["valid_mask"])
                if valid_mask.shape != (self.image_size, self.image_size):
                    raise ValueError(
                        f"{path}: 'valid_mask' must have shape "
                        f"({self.image_size}, {self.image_size}), got {valid_mask.shape}"
                    )
                if valid_mask.dtype != np.bool_:
                    raise ValueError(f"{path}: 'valid_mask' must have dtype bool, got {valid_mask.dtype}")
                valid_mask = np.ascontiguousarray(valid_mask)
            else:
                valid_mask = np.ones((self.image_size, self.image_size), dtype=np.bool_)

        sample: Dict[str, Any] = {
            "selected_frame": torch.from_numpy(selected_frame),
            "cine_volumes": torch.from_numpy(cine_volumes),
            "target_image": torch.from_numpy(target_image),
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "valid_mask": torch.from_numpy(valid_mask),
            "sample_id": path.stem,
        }
        if self.augment is not None:
            sample = self.augment(sample)
        return sample


class SynchronizedGeometricAugmentation:
    """Apply the original training geometry to all image fields together."""

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        selected_frame = sample["selected_frame"]
        cine_volumes = sample["cine_volumes"]
        target_image = sample["target_image"]
        valid_mask = sample["valid_mask"]

        if random.random() < 0.5:
            selected_frame = torch.flip(selected_frame, dims=(-2,))
            cine_volumes = torch.flip(cine_volumes, dims=(-2,))
            target_image = torch.flip(target_image, dims=(-2,))
            valid_mask = torch.flip(valid_mask, dims=(-2,))
        if random.random() < 0.5:
            selected_frame = torch.flip(selected_frame, dims=(-1,))
            cine_volumes = torch.flip(cine_volumes, dims=(-1,))
            target_image = torch.flip(target_image, dims=(-1,))
            valid_mask = torch.flip(valid_mask, dims=(-1,))

        # Mirrors the active ``hvflip_affine`` recipe: a random rotation,
        # x-shear, scale, and small translation.  Images share one transform;
        # the validity mask uses nearest-neighbour interpolation.
        angle = random.uniform(0.0, 360.0)
        shear = [random.uniform(0.0, 10.0), 0.0]
        scale = random.uniform(0.8, 1.0)
        translate = [random.randint(0, 4), random.randint(0, 4)]

        num_cine_images = cine_volumes.shape[0] * cine_volumes.shape[1] * cine_volumes.shape[2]
        stacked_images = torch.cat(
            [
                selected_frame,
                cine_volumes.reshape(num_cine_images, *cine_volumes.shape[-2:]),
                target_image,
            ],
            dim=0,
        )
        stacked_images = tvf.affine(
            stacked_images,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        transformed_mask = tvf.affine(
            valid_mask.unsqueeze(0).to(dtype=torch.float32),
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        ).squeeze(0).to(dtype=torch.bool)

        sample["selected_frame"] = stacked_images[:1]
        sample["cine_volumes"] = stacked_images[1 : 1 + num_cine_images].reshape_as(cine_volumes)
        sample["target_image"] = stacked_images[1 + num_cine_images :]
        sample["valid_mask"] = transformed_mask
        return sample


class MutualInformationLoss(nn.Module):
    """Gaussian-kernel normalized mutual-information loss used in CGE training."""

    def __init__(
        self,
        vmin: float = 0.0,
        vmax: float = 1.0,
        num_bins: int = 64,
        sample_ratio: float = 0.1,
        normalized: bool = True,
        masking_value: float = -1000.0,
    ) -> None:
        super().__init__()
        if not 0.0 < sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be in (0, 1]")
        self.sample_ratio = sample_ratio
        self.normalized = normalized
        self.masking_value = masking_value
        bin_width = (vmax - vmin) / num_bins
        self.sigma = bin_width / (2 * math.sqrt(2 * math.log(2)))
        self.register_buffer("bins", torch.linspace(vmin, vmax, num_bins).unsqueeze(1))

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if valid_mask.ndim == prediction.ndim - 1:
            valid_mask = valid_mask.unsqueeze(1)
        if valid_mask.shape != prediction.shape:
            raise ValueError(
                "valid_mask must have shape [B, 1, H, W] or [B, H, W], got "
                f"{tuple(valid_mask.shape)} for prediction {tuple(prediction.shape)}"
            )

        mask_positive = valid_mask.to(dtype=prediction.dtype)
        mask_negative = 1.0 - mask_positive
        prediction = prediction * mask_positive + self.masking_value * mask_negative
        target = target * mask_positive + self.masking_value * mask_negative

        prediction = prediction.flatten(start_dim=2)
        target = target.flatten(start_dim=2)
        pixel_count = prediction.shape[-1]
        if self.sample_ratio < 1.0:
            selected_count = int(self.sample_ratio * pixel_count)
            # The source implementation samples with the CPU RNG even when
            # the images live on CUDA.  Keeping the index tensor on CPU also
            # preserves that RNG stream under bf16 GPU training.
            selected_indices = torch.randperm(pixel_count)[:selected_count]
            prediction = prediction[:, :, selected_indices]
            target = target[:, :, selected_indices]

        bins = self.bins.to(device=prediction.device, dtype=prediction.dtype)
        normalizer = math.sqrt(2 * math.pi) * self.sigma
        kernel_prediction = torch.exp(-(prediction - bins) ** 2 / (2 * self.sigma ** 2)) / normalizer
        kernel_target = torch.exp(-(target - bins) ** 2 / (2 * self.sigma ** 2)) / normalizer
        joint_histogram = torch.bmm(kernel_prediction, kernel_target.transpose(1, 2))
        joint_probability = joint_histogram / (
            joint_histogram.flatten(start_dim=1).sum(dim=1).view(-1, 1, 1) + 1e-5
        )

        probability_prediction = joint_probability.sum(dim=2)
        probability_target = joint_probability.sum(dim=1)
        entropy_prediction = -(probability_prediction * torch.log(probability_prediction + 1e-5)).sum(dim=1)
        entropy_target = -(probability_target * torch.log(probability_target + 1e-5)).sum(dim=1)
        entropy_joint = -(joint_probability * torch.log(joint_probability + 1e-5)).sum(dim=(1, 2))

        if self.normalized:
            mutual_information = (entropy_prediction + entropy_target) / entropy_joint
        else:
            mutual_information = entropy_prediction + entropy_target - entropy_joint
        return -mutual_information


class VGGPerceptualStyleLoss(nn.Module):
    """Frozen VGG16 feature and style objective used by the original trainer."""

    def __init__(self) -> None:
        super().__init__()
        network = vgg16(weights=VGG16_Weights.DEFAULT).eval()
        for parameter in network.parameters():
            parameter.requires_grad = False

        self.blocks = nn.ModuleList(
            [
                network.features[:4],
                network.features[4:9],
                network.features[9:16],
                network.features[16:23],
                network.features[23:25],
                network.features[25:30],
            ]
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        feature_layers: Sequence[int] = (5,),
        style_layers: Sequence[int] = (1, 2),
    ) -> torch.Tensor:
        if not feature_layers and not style_layers:
            return prediction.new_zeros(prediction.shape[0])
        if prediction.shape[1] != 3:
            prediction = prediction.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        prediction = (prediction - self.mean) / self.std
        target = (target - self.mean) / self.std

        selected_layers = tuple(feature_layers) + tuple(style_layers)
        max_block = max(selected_layers)
        if max_block >= len(self.blocks):
            raise ValueError(f"VGG block {max_block} is not available")

        losses = []
        features_prediction = prediction
        features_target = target
        for block_index, block in enumerate(self.blocks[: max_block + 1]):
            features_prediction = block(features_prediction)
            features_target = block(features_target)
            channels_height_width = features_prediction[0].numel()

            if block_index in feature_layers:
                mse = nnf.mse_loss(features_prediction, features_target, reduction="none")
                losses.append(mse.flatten(start_dim=1).sum(dim=1) / channels_height_width)
            if block_index in style_layers:
                activations_prediction = features_prediction.flatten(start_dim=2) / channels_height_width
                activations_target = features_target.flatten(start_dim=2) / channels_height_width
                gram_prediction = torch.bmm(
                    activations_prediction, activations_prediction.transpose(1, 2)
                )
                gram_target = torch.bmm(activations_target, activations_target.transpose(1, 2))
                mse = nnf.mse_loss(gram_prediction, gram_target, reduction="none")
                losses.append(mse.flatten(start_dim=1).sum(dim=1))
        return torch.stack(losses, dim=-1).mean(dim=-1)


class CGEGeneratorObjective(nn.Module):
    """The active supervised generator objective from the CGE training run."""

    def __init__(
        self,
        *,
        weight_l1: float = 100.0,
        weight_mutual_information: float = 10.0,
        weight_perceptual: float = 2.0,
        weight_adversarial: float = 2.0,
    ) -> None:
        super().__init__()
        self.weight_l1 = weight_l1
        self.weight_mutual_information = weight_mutual_information
        self.weight_perceptual = weight_perceptual
        self.weight_adversarial = weight_adversarial
        self.mutual_information = MutualInformationLoss()
        self.perceptual = VGGPerceptualStyleLoss() if weight_perceptual > 0 else None

    def forward(
        self,
        discriminator: nn.Module,
        prediction: torch.Tensor,
        target: torch.Tensor,
        class_id: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        adversarial = -discriminator(prediction, class_id=class_id).reshape(-1)
        l1 = nnf.l1_loss(prediction, target, reduction="none").mean(dim=(1, 2, 3))
        mutual_information = self.mutual_information(prediction, target, valid_mask)
        if self.perceptual is None:
            perceptual = torch.zeros_like(l1)
        else:
            perceptual = self.perceptual(prediction, target)

        # Match the original seven-slot reduction order.  The inactive LNCC,
        # heatmap-regression, and heatmap-similarity terms remain exact zeros.
        zero = torch.zeros_like(l1.mean())
        losses = torch.stack(
            [
                l1.mean(),
                zero,
                mutual_information.mean(),
                zero,
                zero,
                perceptual.mean(),
                adversarial.mean(),
            ]
        )
        weights = losses.new_tensor(
            [
                self.weight_l1,
                0.0,
                self.weight_mutual_information,
                0.0,
                0.0,
                self.weight_perceptual,
                self.weight_adversarial,
            ]
        )
        total = (losses * weights).sum()
        metrics = {
            "loss_G": total,
            "loss_l1": l1.mean(),
            "loss_mi": mutual_information.mean(),
            "loss_perceptual": perceptual.mean(),
            "loss_adversarial": adversarial.mean(),
        }
        return total, metrics


def discriminator_hinge_loss(
    discriminator: nn.Module,
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_id: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return the source implementation's per-batch hinge discriminator loss."""

    real_loss = nnf.relu(1.0 - discriminator(target, class_id=class_id).reshape(-1))
    fake_loss = nnf.relu(1.0 + discriminator(prediction.detach(), class_id=class_id).reshape(-1))
    total = (real_loss + fake_loss).mean()
    return total, {"loss_D": total, "loss_D_real": real_loss.mean(), "loss_D_fake": fake_loss.mean()}


def _ema_average(
    averaged_parameter: torch.Tensor, model_parameter: torch.Tensor, _: torch.Tensor
) -> torch.Tensor:
    return torch.lerp(model_parameter, averaged_parameter, EMA_BETA)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def move_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def autocast_context(precision: str, device: torch.device) -> Any:
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _mean_metrics(
    totals: Mapping[str, float], counts: Mapping[str, int]
) -> Dict[str, float]:
    return {key: totals[key] / counts[key] for key in totals if counts[key] > 0}


def train_one_epoch(
    loader: Iterable[Mapping[str, Any]],
    *,
    generator: Generator,
    discriminator: CGANSADiscriminator,
    ema_generator: AveragedModel,
    objective: CGEGeneratorObjective,
    optimizer_generator: torch.optim.Optimizer,
    optimizer_discriminator: torch.optim.Optimizer,
    device: torch.device,
    precision: str,
    next_update_is_discriminator: bool,
    global_step: int,
    log_every: int,
) -> Tuple[Dict[str, float], bool, int]:
    """Run the original D, G, D, G ... update order across loader batches."""

    generator.train()
    discriminator.train()
    objective.train()
    if objective.perceptual is not None:
        objective.perceptual.eval()

    totals: MutableMapping[str, float] = defaultdict(float)
    counts: MutableMapping[str, int] = defaultdict(int)
    start = time.monotonic()

    for batch_index, batch in enumerate(loader, start=1):
        batch = move_to_device(batch, device)
        selected_frame = batch["selected_frame"]
        cine_volumes = batch["cine_volumes"]
        target_image = batch["target_image"]
        class_id = batch["class_id"]
        valid_mask = batch["valid_mask"]

        if next_update_is_discriminator:
            optimizer_discriminator.zero_grad(set_to_none=True)
            with autocast_context(precision, device):
                prediction, _ = generator(selected_frame, cine_volumes, class_id=class_id)
                loss, metrics = discriminator_hinge_loss(
                    discriminator, prediction, target_image, class_id
                )
            loss.backward()
            optimizer_discriminator.step()
            phase = "D"
        else:
            optimizer_generator.zero_grad(set_to_none=True)
            with autocast_context(precision, device):
                prediction, _ = generator(selected_frame, cine_volumes, class_id=class_id)
                loss, metrics = objective(
                    discriminator,
                    prediction,
                    target_image,
                    class_id,
                    valid_mask,
                )
            loss.backward()
            optimizer_generator.step()
            phase = "G"

        # The private solver updates EMA after every loader batch, including
        # discriminator iterations; this deliberately keeps the same cadence.
        ema_generator.update_parameters(generator)
        next_update_is_discriminator = not next_update_is_discriminator
        global_step += 1

        for key, value in metrics.items():
            totals[key] += float(value.detach().float().cpu())
            counts[key] += 1

        if log_every > 0 and batch_index % log_every == 0:
            average = _mean_metrics(totals, counts)
            summary = ", ".join(f"{key}={value:.5f}" for key, value in sorted(average.items()))
            elapsed = time.monotonic() - start
            print(
                f"  batch {batch_index}: last phase={phase}, {summary} "
                f"({elapsed / batch_index:.2f}s/batch)",
                flush=True,
            )

    return _mean_metrics(totals, counts), next_update_is_discriminator, global_step


@torch.inference_mode()
def validate(
    loader: Iterable[Mapping[str, Any]],
    *,
    discriminator: CGANSADiscriminator,
    ema_generator: AveragedModel,
    objective: CGEGeneratorObjective,
    device: torch.device,
    precision: str,
) -> Dict[str, float]:
    ema_generator.eval()
    discriminator.eval()
    objective.eval()

    totals: MutableMapping[str, float] = defaultdict(float)
    counts: MutableMapping[str, int] = defaultdict(int)
    for batch in loader:
        batch = move_to_device(batch, device)
        with autocast_context(precision, device):
            prediction, _ = ema_generator.module(
                batch["selected_frame"], batch["cine_volumes"], class_id=batch["class_id"]
            )
            _, generator_metrics = objective(
                discriminator,
                prediction,
                batch["target_image"],
                batch["class_id"],
                batch["valid_mask"],
            )
            _, discriminator_metrics = discriminator_hinge_loss(
                discriminator,
                prediction,
                batch["target_image"],
                batch["class_id"],
            )
        for key, value in {**generator_metrics, **discriminator_metrics}.items():
            totals[key] += float(value.detach().float().cpu())
            counts[key] += 1

    return _mean_metrics(totals, counts)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _serializable_arguments(arguments: argparse.Namespace) -> Dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(arguments).items()
    }


def save_training_checkpoint(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    next_update_is_discriminator: bool,
    generator: Generator,
    discriminator: CGANSADiscriminator,
    ema_generator: AveragedModel,
    optimizer_generator: torch.optim.Optimizer,
    optimizer_discriminator: torch.optim.Optimizer,
    scheduler_generator: torch.optim.lr_scheduler.LRScheduler,
    scheduler_discriminator: torch.optim.lr_scheduler.LRScheduler,
    data_loader_generator: torch.Generator,
    arguments: argparse.Namespace,
) -> None:
    checkpoint = {
        "format_version": 1,
        "epoch": epoch,
        "global_step": global_step,
        "next_update_is_discriminator": next_update_is_discriminator,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "ema_generator": ema_generator.state_dict(),
        "optimizer_generator": optimizer_generator.state_dict(),
        "optimizer_discriminator": optimizer_discriminator.state_dict(),
        "scheduler_generator": scheduler_generator.state_dict(),
        "scheduler_discriminator": scheduler_discriminator.state_dict(),
        "rng_state": capture_rng_state(),
        "data_loader_rng_state": data_loader_generator.get_state(),
        "arguments": _serializable_arguments(arguments),
    }
    _atomic_torch_save(checkpoint, path)


def load_training_checkpoint(
    path: Path,
    *,
    generator: Generator,
    discriminator: CGANSADiscriminator,
    ema_generator: AveragedModel,
    optimizer_generator: torch.optim.Optimizer,
    optimizer_discriminator: torch.optim.Optimizer,
    scheduler_generator: torch.optim.lr_scheduler.LRScheduler,
    scheduler_discriminator: torch.optim.lr_scheduler.LRScheduler,
    data_loader_generator: torch.Generator,
    device: torch.device,
) -> Tuple[int, int, bool]:
    # Full training checkpoints include optimizer and RNG states.  PyTorch
    # 2.6 changed ``torch.load`` to default to weights-only mode, which cannot
    # deserialize these states.  The CLI accepts only checkpoints produced by
    # this script, so opt into full deserialization explicitly.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch releases before the weights_only argument.
        checkpoint = torch.load(path, map_location=device)
    required = {
        "epoch",
        "global_step",
        "generator",
        "discriminator",
        "ema_generator",
        "optimizer_generator",
        "optimizer_discriminator",
        "scheduler_generator",
        "scheduler_discriminator",
        "rng_state",
        "data_loader_rng_state",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(
            f"{path} is not a CGE training checkpoint; missing keys: {', '.join(missing)}"
        )

    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    ema_generator.load_state_dict(checkpoint["ema_generator"])
    optimizer_generator.load_state_dict(checkpoint["optimizer_generator"])
    optimizer_discriminator.load_state_dict(checkpoint["optimizer_discriminator"])
    scheduler_generator.load_state_dict(checkpoint["scheduler_generator"])
    scheduler_discriminator.load_state_dict(checkpoint["scheduler_discriminator"])
    restore_rng_state(checkpoint["rng_state"])
    data_loader_generator.set_state(checkpoint["data_loader_rng_state"])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["global_step"]),
        bool(checkpoint.get("next_update_is_discriminator", True)),
    )


def write_metrics(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in METRIC_COLUMNS})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True, help="Directory of training NPZ shards")
    parser.add_argument("--val-data", type=Path, default=None, help="Optional directory of validation NPZ shards")
    parser.add_argument("--output-dir", type=Path, default=Path("training_runs/cge"))
    parser.add_argument("--resume", type=Path, default=None, help="Full training checkpoint created by this script")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--save-latest-every", type=int, default=10)
    parser.add_argument("--save-archive-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--augment", dest="augment", action="store_true")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.set_defaults(augment=True)
    return parser


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(choice)


def main(arguments: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs and batch-size must be positive; num-workers must be non-negative")
    for option_name in ("validate_every", "save_latest_every", "save_archive_every", "log_every"):
        if getattr(args, option_name) < 0:
            parser.error(f"--{option_name.replace('_', '-')} must be non-negative")

    device = resolve_device(args.device)
    precision = args.precision
    if precision == "bf16" and device.type != "cuda":
        print("CUDA is unavailable; using fp32 instead of requested bf16.", flush=True)
        precision = "fp32"
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(_serializable_arguments(args), file, indent=2, sort_keys=True)

    train_dataset = NPZShardDataset(args.train_data, augment=args.augment)
    validation_dataset = NPZShardDataset(args.val_data, augment=False) if args.val_data else None
    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker if args.num_workers > 0 else None,
        "generator": data_loader_generator,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = (
        DataLoader(validation_dataset, shuffle=False, **loader_kwargs) if validation_dataset is not None else None
    )

    generator = Generator(
        num_classes=NUM_CLASSES,
        latent_dim=256,
        image_size=IMAGE_SIZE,
        base_channels_encoder3d=32,
        base_channels_encoder2d=32,
        base_channels_decoder=64,
        depth_window_size=DEPTH_WINDOW_SIZE,
        level_ch_mult=(1, 2, 4, 4, 4),
        temporal_reduction_ratio=(1, 1, 1, 1),
        attn_fusion_resolutions=(48, 24, 12),
        encoder2d_norm="batch_norm_2d",
        encoder3d_norm="batch_norm_3d",
        decoder2d_norm="cc_instance_norm_2d",
        p_dropout_encoder=0.0,
    ).to(device)
    discriminator = CGANSADiscriminator(
        num_classes=NUM_CLASSES,
        in_channels=1,
        image_size=IMAGE_SIZE,
        base_channels=64,
        level_ch_multi=(1, 2, 4, 8),
        attn_at_resolutions=(64,),
    ).to(device)
    ema_generator = AveragedModel(generator, avg_fn=_ema_average).to(device)

    objective = CGEGeneratorObjective().to(device)
    if objective.perceptual is not None:
        objective.perceptual.eval()

    optimizer_generator = torch.optim.Adam(
        generator.parameters(), lr=5e-5, betas=(0.0, 0.9), eps=1e-6, weight_decay=1e-5
    )
    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(), lr=2e-4, betas=(0.0, 0.9), eps=1e-6, weight_decay=1e-6
    )
    scheduler_generator = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_generator, T_0=400, T_mult=1, eta_min=1e-6
    )
    scheduler_discriminator = torch.optim.lr_scheduler.ConstantLR(
        optimizer_discriminator, factor=1.0, total_iters=5
    )

    start_epoch = 1
    global_step = 0
    next_update_is_discriminator = True
    if args.resume is not None:
        start_epoch, global_step, next_update_is_discriminator = load_training_checkpoint(
            args.resume,
            generator=generator,
            discriminator=discriminator,
            ema_generator=ema_generator,
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            scheduler_generator=scheduler_generator,
            scheduler_discriminator=scheduler_discriminator,
            data_loader_generator=data_loader_generator,
            device=device,
        )
        print(f"Resumed {args.resume} at epoch {start_epoch}, global step {global_step}.", flush=True)

    print(
        f"Training on {device} with {precision}; {len(train_dataset)} training shards"
        + (f", {len(validation_dataset)} validation shards." if validation_dataset is not None else "."),
        flush=True,
    )
    print(
        "Parameters: "
        f"G={sum(parameter.numel() for parameter in generator.parameters()):,}, "
        f"D={sum(parameter.numel() for parameter in discriminator.parameters()):,}",
        flush=True,
    )

    metrics_path = args.output_dir / "metrics.csv"
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.monotonic()
        training_metrics, next_update_is_discriminator, global_step = train_one_epoch(
            train_loader,
            generator=generator,
            discriminator=discriminator,
            ema_generator=ema_generator,
            objective=objective,
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            device=device,
            precision=precision,
            next_update_is_discriminator=next_update_is_discriminator,
            global_step=global_step,
            log_every=args.log_every,
        )
        scheduler_generator.step()
        scheduler_discriminator.step()

        validation_metrics: Dict[str, float] = {}
        if validation_loader is not None and args.validate_every > 0 and epoch % args.validate_every == 0:
            validation_metrics = validate(
                validation_loader,
                discriminator=discriminator,
                ema_generator=ema_generator,
                objective=objective,
                device=device,
                precision=precision,
            )

        row: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate_G": optimizer_generator.param_groups[0]["lr"],
            "learning_rate_D": optimizer_discriminator.param_groups[0]["lr"],
            "seconds": time.monotonic() - epoch_start,
        }
        row.update({f"train_{key}": value for key, value in training_metrics.items()})
        row.update({f"val_{key}": value for key, value in validation_metrics.items()})
        write_metrics(metrics_path, row)
        summary = ", ".join(
            f"{key}={value:.5f}" for key, value in sorted(row.items()) if isinstance(value, float)
        )
        print(f"Epoch {epoch}/{args.epochs}: {summary}", flush=True)

        should_save_latest = args.save_latest_every > 0 and epoch % args.save_latest_every == 0
        should_save_archive = args.save_archive_every > 0 and epoch % args.save_archive_every == 0
        is_final_epoch = epoch == args.epochs
        if should_save_latest or is_final_epoch:
            save_training_checkpoint(
                args.output_dir / "latest.pt",
                epoch=epoch,
                global_step=global_step,
                next_update_is_discriminator=next_update_is_discriminator,
                generator=generator,
                discriminator=discriminator,
                ema_generator=ema_generator,
                optimizer_generator=optimizer_generator,
                optimizer_discriminator=optimizer_discriminator,
                scheduler_generator=scheduler_generator,
                scheduler_discriminator=scheduler_discriminator,
                data_loader_generator=data_loader_generator,
                arguments=args,
            )
        if should_save_archive:
            save_training_checkpoint(
                args.output_dir / f"training_e{epoch}.pt",
                epoch=epoch,
                global_step=global_step,
                next_update_is_discriminator=next_update_is_discriminator,
                generator=generator,
                discriminator=discriminator,
                ema_generator=ema_generator,
                optimizer_generator=optimizer_generator,
                optimizer_discriminator=optimizer_discriminator,
                scheduler_generator=scheduler_generator,
                scheduler_discriminator=scheduler_discriminator,
                data_loader_generator=data_loader_generator,
                arguments=args,
            )
            _atomic_torch_save(ema_generator.module.state_dict(), args.output_dir / f"generator_e{epoch}.ckpt")
        elif is_final_epoch:
            _atomic_torch_save(ema_generator.module.state_dict(), args.output_dir / f"generator_e{epoch}.ckpt")


if __name__ == "__main__":
    main()
