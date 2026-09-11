"""Training and evaluation utilities for static XCA vessel segmentation."""

import argparse
import csv
import inspect
import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument(
        "--encoder",
        choices=["resnet18", "resnet34", "resnet50"],
        default=None,
        help="Default: resnet34 for training; inferred from a baseline checkpoint for evaluation.",
    )
    parser.add_argument(
        "--encoder-weights",
        default=None,
        help="Optional local ResNet weight file, or 'imagenet' to download official weights.",
    )
    parser.add_argument(
        "--data-dir",
        "--dir",
        dest="data_dir",
        default=str(DATA_DIR),
        help="Dataset root for direct single-dataset loading.",
    )
    parser.add_argument(
        "--loader-file",
        default=None,
        help="Dataloader .py file. Defaults to <data-dir>/data_loader_xca_static.py.",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="JSON registry for selecting one or more named XCA datasets.",
    )
    parser.add_argument(
        "--train-datasets",
        nargs="+",
        default=None,
        help="Named datasets to concatenate for training, for example: mosxav ica_nj.",
    )
    parser.add_argument(
        "--val-datasets",
        nargs="+",
        default=None,
        help="Named datasets to concatenate for validation during training.",
    )
    parser.add_argument(
        "--eval-datasets",
        nargs="+",
        default=None,
        help="Named datasets to concatenate in eval mode.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None, help="Required in eval mode.")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "Optional full-model checkpoint used only to initialize a new training run. "
            "Optimizer and scheduler state are intentionally not resumed."
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA automatic mixed precision to reduce memory and accelerate training.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable cuDNN autotuning; recommended when input image size is fixed.",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs when num-workers > 0.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches prefetched per worker when num-workers > 0.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-scan",
        action="store_true",
        help="Scan thresholds during eval and save the best-threshold predictions.",
    )
    parser.add_argument(
        "--tta-flips",
        action="store_true",
        help=(
            "During evaluation, average probabilities from the original image and "
            "horizontal, vertical, and combined flip views."
        ),
    )
    parser.add_argument(
        "--tta-horizontal",
        action="store_true",
        help="During evaluation, average probabilities from the original and horizontal-flip views.",
    )
    parser.add_argument("--threshold-scan-min", type=float, default=0.05)
    parser.add_argument("--threshold-scan-max", type=float, default=0.95)
    parser.add_argument("--threshold-scan-coarse-step", type=float, default=0.01)
    parser.add_argument("--threshold-scan-fine-step", type=float, default=0.001)
    parser.add_argument("--threshold-scan-fine-radius", type=float, default=0.015)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument(
        "--loss",
        choices=("dicebce", "eil", "optimized"),
        default="dicebce",
        help="Loss function: dicebce, eil, or optimized.",
    )
    parser.add_argument(
        "--mse-weight",
        type=float,
        default=1.0,
        help="MSE weight for --loss eil.",
    )
    parser.add_argument(
        "--lambda-eil",
        type=float,
        default=0.05,
        help="EIL weight for --loss eil or optimized.",
    )
    parser.add_argument(
        "--eil-alpha",
        type=float,
        default=0.35,
        help="ElasticInteractionLoss alpha for --loss eil.",
    )
    parser.add_argument(
        "--eil-width",
        type=float,
        default=0.25,
        help="ElasticInteractionLoss width for --loss eil.",
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=0.35,
        help="Soft Dice weight for --loss optimized.",
    )
    parser.add_argument(
        "--lambda-tv",
        type=float,
        default=0.01,
        help="Background TV weight for --loss optimized.",
    )
    parser.add_argument(
        "--lambda-fp",
        type=float,
        default=0.10,
        help="Hard-negative false-positive weight for --loss optimized.",
    )
    parser.add_argument(
        "--lambda-cc",
        type=float,
        default=0.04,
        help="Small-component regularization weight for --loss optimized.",
    )
    parser.add_argument(
        "--lambda-cldice",
        type=float,
        default=0.16,
        help="Soft clDice weight for --loss optimized.",
    )
    parser.add_argument(
        "--dilate-radius",
        type=int,
        default=3,
        help="GT dilation radius for masked EIL.",
    )
    parser.add_argument(
        "--eil-gamma",
        type=float,
        default=1.0,
        help="Frequency radial exponent for masked EIL.",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="Alpha for the hard-negative focal term.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma for the hard-negative focal term.",
    )
    parser.add_argument(
        "--fp-threshold",
        type=float,
        default=0.62,
        help="High-confidence false-positive threshold.",
    )
    parser.add_argument(
        "--fp-hardness",
        type=float,
        default=1.5,
        help="Nonlinear boost for high-confidence false positives.",
    )
    parser.add_argument(
        "--component-kernel-size",
        type=int,
        default=9,
        help="Neighborhood size for small-component regularization.",
    )
    parser.add_argument(
        "--cldice-iters",
        type=int,
        default=8,
        help="Soft skeletonization iterations used by clDice.",
    )
    parser.add_argument(
        "--schedule-warmup-epochs",
        type=int,
        default=15,
        help="Epochs focused on MSE and Dice before topology terms ramp up.",
    )
    parser.add_argument(
        "--schedule-ramp-epochs",
        type=int,
        default=12,
        help="Ramp duration for EIL, clDice, and anti-FP terms.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--cosine-lr",
        action="store_true",
        help="Enable CosineAnnealingLR over the full training run.",
    )
    parser.add_argument(
        "--eta-min",
        type=float,
        default=1e-6,
        help="Minimum learning rate for --cosine-lr.",
    )
    parser.add_argument("--repeat-factor", type=int, default=5)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--branch-crop-prob",
        type=float,
        default=0.0,
        help="Training-only probability for branch-aware crop-resize augmentation. Default: 0.0.",
    )
    parser.add_argument(
        "--branch-crop-ratio",
        type=float,
        default=0.5,
        help="Crop side ratio before resizing back to the training image size. Default: 0.5.",
    )
    parser.add_argument(
        "--branch-crop-min-component-area",
        type=int,
        default=1,
        help="Minimum connected-component area used as branch crop candidate.",
    )
    parser.add_argument(
        "--branch-crop-small-component-area",
        type=int,
        default=50,
        help="Maximum connected-component area treated as a small branch/component candidate.",
    )
    parser.add_argument("--train-image-root", default=None)
    parser.add_argument("--train-mask-root", default=None)
    parser.add_argument("--train-split-file", default=None)
    parser.add_argument("--val-image-root", default=None)
    parser.add_argument("--val-mask-root", default=None)
    parser.add_argument("--val-split-file", default=None)
    parser.add_argument(
        "--no-val-during-train",
        action="store_true",
        help="Do not evaluate a validation set each epoch. By default the eval/test split is reused for validation.",
    )
    parser.add_argument("--eval-image-root", default=None)
    parser.add_argument("--eval-mask-root", default=None)
    parser.add_argument("--eval-split-file", default=None)
    return parser


def resolve_data_paths(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    trainval_dir = data_dir / "trainval-pre"
    train_pre_dir = data_dir / "train-pre"
    train_dir = trainval_dir if trainval_dir.exists() else train_pre_dir
    test_dir = data_dir / "test-pre"
    if args.loader_file is None:
        args.loader_file = str(data_dir / "data_loader_xca_static.py")
    if args.train_image_root is None:
        args.train_image_root = str(train_dir / "JPEGImages_Static")
    if args.train_mask_root is None:
        args.train_mask_root = str(train_dir / "MOSXAV_Static_Dataset")
    if args.train_split_file is None:
        args.train_split_file = str(train_dir / "train.txt")
    if args.eval_image_root is None:
        args.eval_image_root = str(test_dir / "JPEGImages_Static")
    if args.eval_mask_root is None:
        args.eval_mask_root = str(test_dir / "MOSXAV_Static_Dataset")
    if args.eval_split_file is None:
        args.eval_split_file = str(test_dir / "test.txt")


def load_data_module(loader_file: str) -> Any:
    path = Path(loader_file)
    if not path.exists():
        raise FileNotFoundError(f"XCA loader file not found: {path}")
    spec = importlib.util.spec_from_file_location("xca_static_data", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import XCA loader from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    enabled: bool,
    epochs: int,
    eta_min: float,
) -> Optional[torch.optim.lr_scheduler.CosineAnnealingLR]:
    if not enabled:
        return None
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}.")
    if eta_min < 0.0:
        raise ValueError(f"eta_min must be non-negative, got {eta_min}.")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=eta_min,
    )


def make_dataset(
    data_module: Any,
    image_root: Optional[str],
    mask_root: Optional[str],
    split_file: Optional[str],
    args: argparse.Namespace,
    mode: str,
    dataset_names: Optional[list[str]] = None,
    config_split: Optional[str] = None,
) -> Any:
    augment_params = {
        "p_hflip": 0.5,
        "p_vflip": 0.0,
        "max_rotate_degree": 8,
        "max_translate_ratio": 0.04,
        "scale_range": (0.95, 1.05),
        "p_intensity": 0.6,
        "noise_std": 0.01,
        "brightness_limit": 0.06,
        "contrast_limit": 0.06,
    }
    branch_crop_params = {
        "enabled": mode == "train" and args.branch_crop_prob > 0.0,
        "prob": args.branch_crop_prob,
        "ratio": args.branch_crop_ratio,
        "min_component_area": args.branch_crop_min_component_area,
        "small_component_area": args.branch_crop_small_component_area,
    }
    if dataset_names is not None:
        if not args.dataset_config:
            raise ValueError("--dataset-config is required with named datasets.")
        if not hasattr(data_module, "build_multi_dataset_from_config"):
            raise TypeError(
                "The loaded dataloader does not support multi-dataset configuration."
            )
        return data_module.build_multi_dataset_from_config(
            config_path=args.dataset_config,
            dataset_names=dataset_names,
            split=config_split or ("train" if mode == "train" else "test"),
            image_size=(args.width, args.height),
            mode=mode,
            augment=mode == "train" and not args.no_augment,
            augment_params=augment_params,
            branch_crop_params=branch_crop_params,
            require_mask=True,
            repeat_factor=args.repeat_factor if mode == "train" else 1,
        )
    if image_root is None or mask_root is None:
        raise ValueError("image_root and mask_root are required for legacy dataset loading.")
    dataset_kwargs = {
        "image_root": image_root,
        "mask_root": mask_root,
        "split_file": split_file,
        "image_size": (args.width, args.height),
        "mode": mode,
        "augment": mode == "train" and not args.no_augment,
        "augment_params": augment_params,
        "require_mask": True,
        "repeat_factor": args.repeat_factor if mode == "train" else 1,
    }
    dataset_signature = inspect.signature(data_module.XCAStaticDataset)
    if "branch_crop_params" in dataset_signature.parameters:
        dataset_kwargs["branch_crop_params"] = branch_crop_params
    elif branch_crop_params["enabled"]:
        raise TypeError(
            "The loaded XCAStaticDataset does not support branch_crop_params. "
            "Please use the updated static dataloader or set --branch-crop-prob 0.0."
        )
    return data_module.XCAStaticDataset(**dataset_kwargs)


def make_loader(
    dataset: Any,
    data_module: Any,
    args: argparse.Namespace,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    loader_options: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": data_module.xca_static_collate_fn,
    }
    if args.num_workers > 0:
        loader_options["persistent_workers"] = args.persistent_workers
        loader_options["prefetch_factor"] = args.prefetch_factor
    return DataLoader(
        **loader_options,
    )


def prediction_output_path(
    output_root: Path,
    batch: Dict[str, Any],
    index: int,
) -> Path:
    """Keep each dataset's native mask hierarchy while avoiding name collisions."""
    if "dataset_name" not in batch or "output_rel_path" not in batch:
        return output_root / batch["video_id"][index] / batch["save_name"][index]

    dataset_name = str(batch["dataset_name"][index])
    if Path(dataset_name).name != dataset_name or dataset_name in {"", ".", ".."}:
        raise ValueError(f"Unsafe dataset name: {dataset_name!r}")
    relative_path = Path(str(batch["output_rel_path"][index]))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe output relative path: {relative_path}")
    return output_root / dataset_name / relative_path


class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 3.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.last_loss_parts: Dict[str, float] = {}

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # logits shape: (B, 1, H, W), target shape: (B, H, W)
        target = target.float().unsqueeze(1)
        # target shape: (B, 1, H, W)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * target).sum(dim=dims)
        denominator = probs.sum(dim=dims) + target.sum(dim=dims)
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        base_loss = bce + dice_loss
        self.last_loss_parts = {
            "BaseLoss": base_loss.detach().cpu().item(),
        }
        return base_loss


def _grad_xy(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # x shape: (B, 1, H, W)
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :, 1:-1] = 0.5 * (x[..., :, 2:] - x[..., :, :-2])
    gx[..., :, 0] = x[..., :, 1] - x[..., :, 0]
    gx[..., :, -1] = x[..., :, -1] - x[..., :, -2]
    gy[..., 1:-1, :] = 0.5 * (x[..., 2:, :] - x[..., :-2, :])
    gy[..., 0, :] = x[..., 1, :] - x[..., 0, :]
    gy[..., -1, :] = x[..., -1, :] - x[..., -2, :]
    # gx/gy shape: (B, 1, H, W)
    return gx, gy


def _freq_grid(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    fy = torch.fft.fftfreq(height, d=1.0, device=device, dtype=dtype)
    fx = torch.fft.rfftfreq(width, d=1.0, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    freq = torch.sqrt(yy.square() + xx.square() + 1e-12).unsqueeze(0).unsqueeze(0)
    # freq shape: (1, 1, H, W//2 + 1)
    return freq


class ElasticInteractionLoss(nn.Module):
    def __init__(self, alpha: float = 0.35, width: float = 0.25) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.width = float(width)

    def forward(self, pred_prob: torch.Tensor, gt_prob: torch.Tensor) -> torch.Tensor:
        # pred_prob/gt_prob shape: (B, 1, H, W), range: [0, 1]
        phi = pred_prob - 0.5
        level_term = gt_prob + self.alpha * torch.sigmoid(phi / self.width)
        # level_term shape: (B, 1, H, W)
        gx, gy = _grad_xy(level_term)
        fx = torch.fft.rfft2(gx, norm="ortho")
        fy = torch.fft.rfft2(gy, norm="ortho")
        # fx/fy shape: (B, 1, H, W//2 + 1)
        freq = _freq_grid(
            pred_prob.shape[-2],
            pred_prob.shape[-1],
            pred_prob.device,
            pred_prob.dtype,
        )
        energy = freq * (fx.abs().square() + fy.abs().square())
        return energy.mean()


class ConnectivityMSELoss(nn.Module):
    def __init__(
        self,
        mse_weight: float = 1.0,
        lambda_eil: float = 0.1,
        eil_alpha: float = 0.35,
        eil_width: float = 0.25,
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.lambda_eil = float(lambda_eil)
        self.mse = nn.MSELoss()
        self.eil = ElasticInteractionLoss(alpha=eil_alpha, width=eil_width)
        self.last_loss_parts: Dict[str, float] = {}

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # logits shape: (B, 1, H, W), target shape: (B, H, W) or (B, 1, H, W)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        elif target.ndim != 4:
            raise ValueError(f"target must have shape (B,H,W) or (B,1,H,W), got {tuple(target.shape)}")
        target = target.float()
        # target shape: (B, 1, H, W)
        probs = torch.sigmoid(logits).float()
        # probs shape: (B, 1, H, W)
        loss_mse = self.mse(probs, target)
        loss_eil = self.eil(probs.clamp(0.0, 1.0), target.clamp(0.0, 1.0))
        total_loss = self.mse_weight * loss_mse + self.lambda_eil * loss_eil
        self.last_loss_parts = {
            "LossMSE": loss_mse.detach().cpu().item(),
            "LossEIL": loss_eil.detach().cpu().item(),
        }
        return total_loss


class OptimizedVesselLoss(nn.Module):
    """Dynamic vessel loss with region, frequency, anti-FP, and soft clDice terms."""

    def __init__(
        self,
        mse_weight: float = 1.0,
        dice_weight: float = 0.35,
        lambda_eil: float = 0.05,
        lambda_tv: float = 0.01,
        lambda_fp: float = 0.10,
        lambda_cc: float = 0.04,
        lambda_cldice: float = 0.16,
        dilate_radius: int = 3,
        eil_gamma: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        fp_threshold: float = 0.62,
        fp_hardness: float = 1.5,
        component_kernel_size: int = 9,
        cldice_iters: int = 8,
        schedule_warmup_epochs: int = 15,
        schedule_ramp_epochs: int = 12,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.dice_weight = float(dice_weight)
        self.lambda_eil = float(lambda_eil)
        self.lambda_tv = float(lambda_tv)
        self.lambda_fp = float(lambda_fp)
        self.lambda_cc = float(lambda_cc)
        self.lambda_cldice = float(lambda_cldice)
        self.dilate_radius = int(dilate_radius)
        self.eil_gamma = float(eil_gamma)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.fp_threshold = float(fp_threshold)
        self.fp_hardness = float(fp_hardness)
        self.component_kernel_size = int(component_kernel_size)
        self.cldice_iters = int(cldice_iters)
        self.schedule_warmup_epochs = int(schedule_warmup_epochs)
        self.schedule_ramp_epochs = int(schedule_ramp_epochs)
        self.eps = float(eps)
        self.mse = nn.MSELoss()
        self.last_loss_parts: Dict[str, float] = {}

    def _soft_dice_loss(
        self,
        pred_prob: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # pred_prob/target shape: (B, 1, H, W)
        numerator = 2.0 * (pred_prob * target).sum(dim=(-2, -1)) + self.eps
        denominator = (
            pred_prob.sum(dim=(-2, -1))
            + target.sum(dim=(-2, -1))
            + self.eps
        )
        return 1.0 - (numerator / denominator).mean()

    def _dilated_roi(self, target: torch.Tensor) -> torch.Tensor:
        # target shape: (B, 1, H, W)
        target_binary = (target > 0.5).float()
        if self.dilate_radius <= 0:
            return target_binary
        kernel_size = 2 * self.dilate_radius + 1
        return F.max_pool2d(
            target_binary,
            kernel_size=kernel_size,
            stride=1,
            padding=self.dilate_radius,
        )

    def _masked_eil(
        self,
        pred_prob: torch.Tensor,
        target: torch.Tensor,
        roi: torch.Tensor,
    ) -> torch.Tensor:
        # pred_prob/target/roi shape: (B, 1, H, W)
        grad_x = pred_prob[..., :, 1:] - pred_prob[..., :, :-1]
        grad_y = pred_prob[..., 1:, :] - pred_prob[..., :-1, :]
        target_grad_x = target[..., :, 1:] - target[..., :, :-1]
        target_grad_y = target[..., 1:, :] - target[..., :-1, :]
        roi_x = roi[..., :, 1:] * roi[..., :, :-1]
        roi_y = roi[..., 1:, :] * roi[..., :-1, :]
        delta_x = (grad_x - target_grad_x) * roi_x
        delta_y = (grad_y - target_grad_y) * roi_y
        # delta_x shape: (B, 1, H, W-1)
        # delta_y shape: (B, 1, H-1, W)
        fft_x = torch.fft.rfft2(delta_x, dim=(-2, -1), norm="ortho")
        fft_y = torch.fft.rfft2(delta_y, dim=(-2, -1), norm="ortho")
        freq_x = _freq_grid(
            delta_x.shape[-2],
            delta_x.shape[-1],
            delta_x.device,
            delta_x.dtype,
        ).pow(self.eil_gamma)
        freq_y = _freq_grid(
            delta_y.shape[-2],
            delta_y.shape[-1],
            delta_y.device,
            delta_y.dtype,
        ).pow(self.eil_gamma)
        energy_x = (freq_x * fft_x.abs().square()).mean(
            dim=(-2, -1),
            keepdim=True,
        )
        energy_y = (freq_y * fft_y.abs().square()).mean(
            dim=(-2, -1),
            keepdim=True,
        )
        roi_ratio = roi.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        return ((energy_x + energy_y) / roi_ratio).mean()

    def _background_tv(
        self,
        pred_prob: torch.Tensor,
        background: torch.Tensor,
    ) -> torch.Tensor:
        # pred_prob/background shape: (B, 1, H, W)
        delta_x = (pred_prob[..., :, 1:] - pred_prob[..., :, :-1]).abs()
        delta_y = (pred_prob[..., 1:, :] - pred_prob[..., :-1, :]).abs()
        background_x = background[..., :, 1:] * background[..., :, :-1]
        background_y = background[..., 1:, :] * background[..., :-1, :]
        total_variation = (
            (delta_x * background_x).sum()
            + (delta_y * background_y).sum()
        )
        return total_variation / (
            background_x.sum() + background_y.sum() + self.eps
        )

    def _hard_negative_focal(
        self,
        pred_prob: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # pred_prob/target shape: (B, 1, H, W)
        negative = (target < 0.5).float()
        probability = pred_prob.clamp(self.eps, 1.0 - self.eps)
        focal_negative = -(
            probability.pow(self.focal_gamma) * torch.log(1.0 - probability)
        )
        hard_gate = (
            (probability - self.fp_threshold)
            / (1.0 - self.fp_threshold + self.eps)
        ).clamp(0.0, 1.0)
        hard_boost = 1.0 + self.fp_hardness * hard_gate.square()
        weighted = (
            self.focal_alpha
            * focal_negative
            * hard_boost
            * negative
        )
        return weighted.sum() / (negative.sum() + self.eps)

    def _component_regularizer(
        self,
        pred_prob: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # pred_prob/target shape: (B, 1, H, W)
        negative = (target < 0.5).float()
        kernel_size = max(3, self.component_kernel_size | 1)
        local_mass = F.avg_pool2d(
            pred_prob * negative,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        isolation = (pred_prob * negative - local_mass).clamp(min=0.0)
        small_blob_weight = (1.0 - local_mass).clamp(0.0, 1.0).square()
        return (
            isolation * small_blob_weight
        ).sum() / (negative.sum() + self.eps)

    @staticmethod
    def _soft_erode(x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, H, W)
        return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, H, W)
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    def _soft_open(self, x: torch.Tensor) -> torch.Tensor:
        # x/output shape: (B, 1, H, W)
        return self._soft_dilate(self._soft_erode(x))

    def _soft_skeletonize(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, H, W)
        x = x.clamp(0.0, 1.0)
        skeleton = F.relu(x - self._soft_open(x))
        for _ in range(self.cldice_iters):
            x = self._soft_erode(x)
            delta = F.relu(x - self._soft_open(x))
            skeleton = skeleton + F.relu(delta - skeleton * delta)
        # skeleton shape: (B, 1, H, W)
        return skeleton

    def _cldice_loss(
        self,
        pred_prob: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # pred_prob/target shape: (B, 1, H, W)
        pred_skeleton = self._soft_skeletonize(pred_prob)
        target_skeleton = self._soft_skeletonize(target)
        topology_precision = (
            (pred_skeleton * target).sum(dim=(-2, -1))
            / (pred_skeleton.sum(dim=(-2, -1)) + self.eps)
        )
        topology_sensitivity = (
            (target_skeleton * pred_prob).sum(dim=(-2, -1))
            / (target_skeleton.sum(dim=(-2, -1)) + self.eps)
        )
        cldice = (
            2.0 * topology_precision * topology_sensitivity + self.eps
        ) / (topology_precision + topology_sensitivity + self.eps)
        return 1.0 - cldice.mean()

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = max(0.0, min(1.0, value))
        return value * value * (3.0 - 2.0 * value)

    def dynamic_weights(
        self,
        current_epoch: Optional[int],
        current_step: Optional[int],
        steps_per_epoch: Optional[int],
    ) -> Dict[str, float]:
        if current_epoch is None:
            return {
                "w_mse": self.mse_weight,
                "w_dice": self.dice_weight,
                "w_eil": self.lambda_eil,
                "w_tv": self.lambda_tv,
                "w_fp": self.lambda_fp,
                "w_cc": self.lambda_cc,
                "w_cldice": self.lambda_cldice,
            }
        epoch_value = float(current_epoch)
        if current_step is not None and steps_per_epoch and steps_per_epoch > 0:
            epoch_value += float(current_step) / float(steps_per_epoch)
        ramp = (
            epoch_value - float(self.schedule_warmup_epochs)
        ) / max(float(self.schedule_ramp_epochs), 1.0)
        schedule = self._smoothstep(ramp)
        schedule_squared = schedule * schedule
        return {
            "w_mse": self.mse_weight,
            "w_dice": self.dice_weight * (1.0 - 0.10 * schedule),
            "w_eil": self.lambda_eil * schedule,
            "w_cldice": self.lambda_cldice * schedule,
            "w_tv": self.lambda_tv * (0.35 + 1.15 * schedule_squared),
            "w_fp": self.lambda_fp * (0.35 + 1.45 * schedule_squared),
            "w_cc": self.lambda_cc * (0.25 + 1.35 * schedule_squared),
        }

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        current_epoch: Optional[int] = None,
        current_step: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
    ) -> torch.Tensor:
        # logits shape: (B, 1, H, W)
        # target shape: (B, H, W) or (B, 1, H, W)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        elif target.ndim == 4:
            target = target
        else:
            raise ValueError(
                f"target must have shape (B,H,W) or (B,1,H,W), got {tuple(target.shape)}"
            )
        target = target.float().clamp(0.0, 1.0)
        # target shape: (B, 1, H, W)
        pred_prob = torch.sigmoid(logits).float().clamp(
            self.eps,
            1.0 - self.eps,
        )
        # pred_prob shape: (B, 1, H, W)
        roi = self._dilated_roi(target)
        # roi shape: (B, 1, H, W)
        background = 1.0 - roi
        weights = self.dynamic_weights(
            current_epoch,
            current_step,
            steps_per_epoch,
        )
        loss_mse = self.mse(pred_prob, target)
        loss_dice = self._soft_dice_loss(pred_prob, target)
        loss_eil = self._masked_eil(pred_prob, target, roi)
        loss_tv = self._background_tv(pred_prob, background)
        loss_fp = self._hard_negative_focal(pred_prob, target)
        loss_cc = self._component_regularizer(pred_prob, target)
        loss_cldice = self._cldice_loss(pred_prob, target)
        total_loss = (
            weights["w_mse"] * loss_mse
            + weights["w_dice"] * loss_dice
            + weights["w_eil"] * loss_eil
            + weights["w_tv"] * loss_tv
            + weights["w_fp"] * loss_fp
            + weights["w_cc"] * loss_cc
            + weights["w_cldice"] * loss_cldice
        )
        self.last_loss_parts = {
            "LossMSE": loss_mse.detach().cpu().item(),
            "LossDice": loss_dice.detach().cpu().item(),
            "LossEIL": loss_eil.detach().cpu().item(),
            "LossTV": loss_tv.detach().cpu().item(),
            "LossFP": loss_fp.detach().cpu().item(),
            "LossCC": loss_cc.detach().cpu().item(),
            "LossCLDice": loss_cldice.detach().cpu().item(),
            "TotalLoss": total_loss.detach().cpu().item(),
        }
        return total_loss


def confusion_counts(pred: np.ndarray, target: np.ndarray) -> Tuple[int, int, int, int]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(pred, target).sum())
    tn = int(np.logical_and(~pred, ~target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    return tp, tn, fp, fn


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    eps = 1e-7
    return {
        "Dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "IoU": (tp + eps) / (tp + fp + fn + eps),
        "Sensitivity": (tp + eps) / (tp + fn + eps),
        "Specificity": (tn + eps) / (tn + fp + eps),
        "Precision": (tp + eps) / (tp + fp + eps),
        "Accuracy": (tp + tn + eps) / (tp + tn + fp + fn + eps),
    }


METRIC_NAMES = ["Dice", "IoU", "Sensitivity", "Specificity", "Precision", "Accuracy"]


def collect_gamma_values(model: nn.Module) -> Dict[str, float]:
    values: Dict[str, float] = {}
    swin = getattr(model, "swin", None)
    if swin is not None and hasattr(swin, "gamma"):
        values["swin_gamma"] = float(swin.gamma.detach().cpu().item())
    for stage_name in ("decode4", "decode3", "decode2", "decode1"):
        stage = getattr(model, stage_name, None)
        if stage is not None and hasattr(stage, "gamma"):
            values[f"{stage_name}_gamma"] = float(stage.gamma.detach().cpu().item())
    return values


def print_gamma_values(prefix: str, model: nn.Module) -> Dict[str, float]:
    values = collect_gamma_values(model)
    if values:
        text = " ".join(f"{name}={value:.6f}" for name, value in values.items())
        print(f"{prefix} {text}")
    return values


def scan_probability_thresholds(
    probabilities: list,
    targets: list,
    thresholds: list,
) -> list:
    if len(probabilities) != len(targets):
        raise ValueError(
            f"probabilities and targets must have equal length, got "
            f"{len(probabilities)} and {len(targets)}."
        )
    if not probabilities:
        raise ValueError("probabilities must not be empty.")
    rows = []
    for threshold in thresholds:
        counts = [0, 0, 0, 0]
        frame_metrics = []
        for probability, target in zip(probabilities, targets):
            if probability.shape != target.shape:
                raise ValueError(
                    f"Probability/target shape mismatch: "
                    f"{probability.shape} vs {target.shape}."
                )
            prediction = probability >= float(threshold)
            frame_counts = confusion_counts(prediction, target)
            counts = [left + right for left, right in zip(counts, frame_counts)]
            frame_metrics.append(metrics_from_counts(*frame_counts))
        row = {
            "Threshold": float(threshold),
            **metrics_from_counts(*counts),
        }
        for metric in METRIC_NAMES:
            values = np.asarray(
                [frame[metric] for frame in frame_metrics],
                dtype=np.float64,
            )
            row[f"{metric}_Mean"] = float(values.mean())
            row[f"{metric}_Std"] = float(values.std())
        rows.append(row)
    return rows


def run_threshold_scan(
    probabilities: list,
    targets: list,
    frame_metadata: list,
    output_dir: Path,
    data_module: Any,
    threshold_min: float,
    threshold_max: float,
    coarse_step: float,
    fine_step: float,
    fine_radius: float,
) -> Dict[str, Any]:
    if not 0.0 <= threshold_min < threshold_max <= 1.0:
        raise ValueError("Threshold scan range must satisfy 0 <= min < max <= 1.")
    if coarse_step <= 0.0 or fine_step <= 0.0 or fine_radius < 0.0:
        raise ValueError("Threshold scan steps must be positive and radius non-negative.")
    coarse_thresholds = np.round(
        np.arange(
            threshold_min,
            threshold_max + coarse_step * 0.5,
            coarse_step,
        ),
        6,
    ).tolist()
    coarse_rows = scan_probability_thresholds(
        probabilities,
        targets,
        coarse_thresholds,
    )
    coarse_best = max(coarse_rows, key=lambda row: row["Dice"])
    fine_min = max(threshold_min, coarse_best["Threshold"] - fine_radius)
    fine_max = min(threshold_max, coarse_best["Threshold"] + fine_radius)
    fine_thresholds = np.round(
        np.arange(fine_min, fine_max + fine_step * 0.5, fine_step),
        6,
    ).tolist()
    fine_rows = scan_probability_thresholds(
        probabilities,
        targets,
        fine_thresholds,
    )
    rows_by_threshold = {
        row["Threshold"]: row
        for row in [*coarse_rows, *fine_rows]
    }
    rows = [rows_by_threshold[key] for key in sorted(rows_by_threshold)]
    best_global = max(rows, key=lambda row: row["Dice"])
    best_mean = max(rows, key=lambda row: row["Dice_Mean"])

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(
        output_dir / "threshold_scan.csv",
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    prediction_root = output_dir / f"predictions_best_th{best_global['Threshold']:.3f}"
    for probability, metadata in zip(probabilities, frame_metadata):
        prediction = (probability >= best_global["Threshold"]).astype(np.uint8) * 255
        save_path = prediction_output_path(
            prediction_root,
            {key: [value] for key, value in metadata.items()},
            0,
        )
        data_module.cv2_imwrite_unicode(save_path, prediction)

    summary = {
        "frames": len(probabilities),
        "coarse_range": [threshold_min, threshold_max, coarse_step],
        "fine_range": [fine_min, fine_max, fine_step],
        "best_by_global_dice": best_global,
        "best_by_mean_dice": best_mean,
        "prediction_dir": str(prediction_root),
        "top_10_by_global_dice": sorted(
            rows,
            key=lambda row: row["Dice"],
            reverse=True,
        )[:10],
    }
    with open(
        output_dir / "threshold_scan_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def add_per_frame_summary(summary: Dict[str, float], rows: list) -> list:
    summary_rows = []
    for metric in METRIC_NAMES:
        values = np.array([row[metric] for row in rows], dtype=np.float32)
        mean = float(values.mean())
        std = float(values.std())
        summary[f"{metric}_Mean"] = mean
        summary[f"{metric}_Std"] = std
        summary_rows.append(
            {
                "Metric": metric,
                "Global": summary[metric],
                "Mean": mean,
                "Std": std,
                "Valid_Frames": len(values),
            }
        )
    return summary_rows


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float,
    use_amp: bool,
    scaler: torch.amp.GradScaler,
    current_epoch: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    component_totals: Dict[str, float] = {}
    counts = [0, 0, 0, 0]
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            if isinstance(criterion, OptimizedVesselLoss):
                loss = criterion(
                    logits,
                    masks,
                    current_epoch=current_epoch,
                    current_step=step,
                    steps_per_epoch=len(loader),
                )
            else:
                loss = criterion(logits, masks)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.detach().cpu().item() * images.shape[0]
        for name, value in getattr(criterion, "last_loss_parts", {}).items():
            if name == "TotalLoss":
                continue
            component_totals[name] = (
                component_totals.get(name, 0.0)
                + float(value) * images.shape[0]
            )
        pred = (torch.sigmoid(logits[:, 0]) >= threshold).detach().cpu().numpy()
        gt = masks.detach().cpu().numpy()
        batch_counts = confusion_counts(pred, gt)
        counts = [left + right for left, right in zip(counts, batch_counts)]
    metrics = metrics_from_counts(*counts)
    metrics["Loss"] = total_loss / len(loader.dataset)
    for name, value in component_totals.items():
        metrics[name] = value / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    data_module: Any,
    use_amp: bool,
    prediction_root: Optional[Path] = None,
    probability_png_root: Optional[Path] = None,
    probability_npy_root: Optional[Path] = None,
    metrics_dir: Optional[Path] = None,
    current_epoch: Optional[int] = None,
    threshold_scan_dir: Optional[Path] = None,
    threshold_scan_settings: Optional[Dict[str, float]] = None,
    tta_flips: bool = False,
    tta_horizontal: bool = False,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    component_totals: Dict[str, float] = {}
    counts = [0, 0, 0, 0]
    rows = []
    scan_probabilities = []
    scan_targets = []
    scan_metadata = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            if isinstance(criterion, OptimizedVesselLoss):
                loss = criterion(
                    logits,
                    masks,
                    current_epoch=current_epoch,
                )
            else:
                loss = criterion(logits, masks)
        total_loss += loss.detach().cpu().item() * images.shape[0]
        for name, value in getattr(criterion, "last_loss_parts", {}).items():
            if name == "TotalLoss":
                continue
            component_totals[name] = (
                component_totals.get(name, 0.0)
                + float(value) * images.shape[0]
            )
        if tta_flips or tta_horizontal:
            probability_views = [torch.sigmoid(logits[:, 0])]
            flip_dimensions = ((-1,),) if tta_horizontal else ((-1,), (-2,), (-2, -1))
            for dimensions in flip_dimensions:
                flipped_images = torch.flip(images, dims=dimensions)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    flipped_logits = model(flipped_images)
                restored = torch.flip(
                    torch.sigmoid(flipped_logits[:, 0]),
                    dims=dimensions,
                )
                probability_views.append(restored)
            probability_tensor = torch.stack(probability_views, dim=0).mean(dim=0)
        else:
            probability_tensor = torch.sigmoid(logits[:, 0])
        probabilities = probability_tensor.cpu().numpy().astype(np.float32)
        predictions = (probabilities >= threshold).astype(np.uint8)
        targets = masks.cpu().numpy().astype(np.uint8)
        if threshold_scan_dir is not None:
            scan_probabilities.extend(probabilities)
            scan_targets.extend(targets)
            scan_metadata.extend(
                {
                    "dataset_name": batch["dataset_name"][index],
                    "output_rel_path": batch["output_rel_path"][index],
                    "video_id": batch["video_id"][index],
                    "save_name": batch["save_name"][index],
                }
                for index in range(len(probabilities))
            )
        for index, pred in enumerate(predictions):
            frame_counts = confusion_counts(pred, targets[index])
            counts = [left + right for left, right in zip(counts, frame_counts)]
            row = {
                "dataset_name": batch["dataset_name"][index],
                "video_id": batch["video_id"][index],
                "frame_name": batch["frame_name"][index],
                "rel_path": batch["rel_path"][index],
                **metrics_from_counts(*frame_counts),
            }
            rows.append(row)
            if prediction_root is not None:
                save_path = prediction_output_path(prediction_root, batch, index)
                data_module.cv2_imwrite_unicode(save_path, pred * 255)
            if probability_png_root is not None:
                prob_png_path = prediction_output_path(probability_png_root, batch, index)
                prob_png = np.clip(probabilities[index] * 255.0, 0, 255).round().astype(np.uint8)
                data_module.cv2_imwrite_unicode(prob_png_path, prob_png)
            if probability_npy_root is not None:
                prob_npy_path = prediction_output_path(
                    probability_npy_root,
                    batch,
                    index,
                ).with_suffix(".npy")
                prob_npy_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(prob_npy_path, probabilities[index])
    summary = metrics_from_counts(*counts)
    summary["Loss"] = total_loss / len(loader.dataset)
    for name, value in component_totals.items():
        summary[name] = value / len(loader.dataset)
    summary["Frames"] = len(rows)
    summary_rows = add_per_frame_summary(summary, rows)
    if metrics_dir is not None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / "per_frame_metrics.csv", "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with open(metrics_dir / "summary_metrics.csv", "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["Metric", "Global", "Mean", "Std", "Valid_Frames"]
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        with open(metrics_dir / "summary_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
    if threshold_scan_dir is not None:
        settings = threshold_scan_settings or {}
        scan_summary = run_threshold_scan(
            scan_probabilities,
            scan_targets,
            scan_metadata,
            threshold_scan_dir,
            data_module,
            threshold_min=float(settings.get("min", 0.05)),
            threshold_max=float(settings.get("max", 0.95)),
            coarse_step=float(settings.get("coarse_step", 0.01)),
            fine_step=float(settings.get("fine_step", 0.001)),
            fine_radius=float(settings.get("fine_radius", 0.015)),
        )
        summary["ThresholdScanBest"] = scan_summary["best_by_global_dice"]
    return summary


def print_metrics(prefix: str, metrics: Dict[str, float]) -> None:
    names = ["Loss", *METRIC_NAMES]
    values = " ".join(f"{name}={metrics[name]:.4f}" for name in names if name in metrics)
    print(f"{prefix} {values}")
    if "Dice_Mean" in metrics:
        print(
            f"{prefix} per-frame Mean+-Std: "
            f"Dice={metrics['Dice_Mean']:.4f}+-{metrics['Dice_Std']:.4f} "
            f"IoU={metrics['IoU_Mean']:.4f}+-{metrics['IoU_Std']:.4f} "
            f"Sensitivity={metrics['Sensitivity_Mean']:.4f}+-{metrics['Sensitivity_Std']:.4f} "
            f"Precision={metrics['Precision_Mean']:.4f}+-{metrics['Precision_Std']:.4f}"
        )
    component_names = (
        "LossMSE",
        "LossDice",
        "LossEIL",
        "LossTV",
        "LossFP",
        "LossCC",
        "LossCLDice",
    )
    component_values = " ".join(
        f"{name}={metrics[name]:.4f}"
        for name in component_names
        if name in metrics
    )
    if component_values:
        print(f"{prefix} components: {component_values}")


def print_threshold_scan_best(best: Dict[str, float]) -> None:
    print(
        "Threshold scan best: "
        f"Threshold={best['Threshold']:.3f} "
        f"Dice={best['Dice']:.4f} "
        f"IoU={best['IoU']:.4f} "
        f"Sensitivity={best['Sensitivity']:.4f} "
        f"Precision={best['Precision']:.4f}"
    )
    print(
        "Threshold scan best per-frame Mean+-Std: "
        f"Dice={best['Dice_Mean']:.4f}+-{best['Dice_Std']:.4f} "
        f"IoU={best['IoU_Mean']:.4f}+-{best['IoU_Std']:.4f} "
        f"Sensitivity={best['Sensitivity_Mean']:.4f}+-"
        f"{best['Sensitivity_Std']:.4f} "
        f"Precision={best['Precision_Mean']:.4f}+-"
        f"{best['Precision_Std']:.4f}"
    )


def load_model_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)
    return checkpoint if isinstance(checkpoint, dict) else {}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    args: argparse.Namespace,
    metrics: Dict[str, float],
    model_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "model_name": model_name,
            "encoder": args.encoder,
            "metrics": metrics,
        },
        path,
    )


def run_experiment(
    args: argparse.Namespace,
    model_builder: Callable[[argparse.Namespace], nn.Module],
    model_name: str,
) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir or (PROJECT_DIR / "outputs" / model_name)).expanduser().resolve()
    resolve_data_paths(args)
    data_module = load_data_module(args.loader_file)
    if args.encoder is None and args.mode == "eval" and args.checkpoint:
        checkpoint_metadata = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(checkpoint_metadata, dict):
            args.encoder = checkpoint_metadata.get("encoder")
    if args.encoder is None:
        args.encoder = "resnet34"
    model = model_builder(args).to(device)
    if args.mode == "train" and args.init_checkpoint:
        init_checkpoint = Path(args.init_checkpoint).expanduser().resolve()
        if not init_checkpoint.is_file():
            raise FileNotFoundError(f"Training initialization checkpoint not found: {init_checkpoint}")
        init_metadata = load_model_checkpoint(model, str(init_checkpoint), device)
        print(
            "Initialized full training model from: "
            f"{init_checkpoint} (source_epoch={init_metadata.get('epoch', 'unknown')}); "
            "optimizer and scheduler start fresh."
        )
    if args.loss == "optimized":
        criterion = OptimizedVesselLoss(
            mse_weight=args.mse_weight,
            dice_weight=args.dice_weight,
            lambda_eil=args.lambda_eil,
            lambda_tv=args.lambda_tv,
            lambda_fp=args.lambda_fp,
            lambda_cc=args.lambda_cc,
            lambda_cldice=args.lambda_cldice,
            dilate_radius=args.dilate_radius,
            eil_gamma=args.eil_gamma,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            fp_threshold=args.fp_threshold,
            fp_hardness=args.fp_hardness,
            component_kernel_size=args.component_kernel_size,
            cldice_iters=args.cldice_iters,
            schedule_warmup_epochs=args.schedule_warmup_epochs,
            schedule_ramp_epochs=args.schedule_ramp_epochs,
        ).to(device)
    elif args.loss == "eil":
        criterion = ConnectivityMSELoss(
            mse_weight=args.mse_weight,
            lambda_eil=args.lambda_eil,
            eil_alpha=args.eil_alpha,
            eil_width=args.eil_width,
        ).to(device)
    else:
        criterion = DiceBCELoss(pos_weight=args.pos_weight).to(device)
    use_amp = args.amp and device.type == "cuda"
    if args.amp and not use_amp:
        print("AMP was requested, but it is only enabled for a CUDA device in this runner.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(
        f"Model={model_name} Encoder={args.encoder} Device={device} "
        f"AMP={use_amp} OutputDir={output_dir}"
    )
    print_gamma_values("Initial gamma:", model)

    if args.mode == "eval":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --mode eval.")
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        print(f"Evaluation loads checkpoint: {checkpoint_path}")
        print("Evaluation saves predictions and metrics only; it does not save new .pt weights.")
        load_model_checkpoint(model, str(checkpoint_path), device)
        print_gamma_values("Loaded gamma:", model)
        dataset = make_dataset(
            data_module,
            args.eval_image_root,
            args.eval_mask_root,
            args.eval_split_file,
            args,
            "val",
            dataset_names=args.eval_datasets,
            config_split="test",
        )
        loader = make_loader(dataset, data_module, args, shuffle=False, device=device)
        metrics = evaluate(
            model,
            loader,
            criterion,
            device,
            args.threshold,
            data_module,
            use_amp,
            prediction_root=output_dir / "predictions",
            probability_png_root=output_dir / "probabilities_png",
            probability_npy_root=output_dir / "probabilities_npy",
            metrics_dir=output_dir / "metrics",
            threshold_scan_dir=(
                output_dir / "threshold_scan"
                if args.threshold_scan
                else None
            ),
            threshold_scan_settings={
                "min": args.threshold_scan_min,
                "max": args.threshold_scan_max,
                "coarse_step": args.threshold_scan_coarse_step,
                "fine_step": args.threshold_scan_fine_step,
                "fine_radius": args.threshold_scan_fine_radius,
            },
            tta_flips=args.tta_flips,
            tta_horizontal=args.tta_horizontal,
        )
        print_metrics("Evaluation:", metrics)
        if "ThresholdScanBest" in metrics:
            best = metrics["ThresholdScanBest"]
            print_threshold_scan_best(best)
            print(f"Threshold scan saved to: {output_dir / 'threshold_scan'}")
        print(f"Predictions saved to: {output_dir / 'predictions'}")
        print(f"Probability PNGs saved to: {output_dir / 'probabilities_png'}")
        print(f"Probability NumPy files saved to: {output_dir / 'probabilities_npy'}")
        print(f"Metrics saved to: {output_dir / 'metrics'}")
        return

    train_dataset = make_dataset(
        data_module,
        args.train_image_root,
        args.train_mask_root,
        args.train_split_file,
        args,
        "train",
        dataset_names=args.train_datasets,
        config_split="train",
    )
    train_loader = make_loader(train_dataset, data_module, args, shuffle=True, device=device)
    val_loader = None
    if not args.no_val_during_train:
        val_dataset_names = args.val_datasets or args.eval_datasets
        if val_dataset_names is not None:
            val_image_root = None
            val_mask_root = None
            val_split_file = None
        elif args.val_image_root or args.val_mask_root:
            if not args.val_image_root or not args.val_mask_root:
                raise ValueError("--val-image-root and --val-mask-root must be provided together.")
            val_image_root = args.val_image_root
            val_mask_root = args.val_mask_root
            val_split_file = args.val_split_file
        else:
            val_image_root = args.eval_image_root
            val_mask_root = args.eval_mask_root
            val_split_file = args.eval_split_file
        val_dataset = make_dataset(
            data_module,
            val_image_root,
            val_mask_root,
            val_split_file,
            args,
            "val",
            dataset_names=val_dataset_names,
            config_split="test",
        )
        val_loader = make_loader(val_dataset, data_module, args, shuffle=False, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = create_lr_scheduler(
        optimizer,
        enabled=args.cosine_lr,
        epochs=args.epochs,
        eta_min=args.eta_min,
    )
    history = []
    best_dice = -1.0
    print(f"Training checkpoints will be saved to: {output_dir / 'checkpoints'}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.threshold,
            use_amp,
            scaler,
            current_epoch=epoch,
        )
        print_metrics(f"Epoch {epoch:03d} train:", train_metrics)
        epoch_record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
        }
        monitor_metrics = train_metrics
        if val_loader is not None:
            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                args.threshold,
                data_module,
                use_amp,
                current_epoch=epoch,
            )
            print_metrics(f"Epoch {epoch:03d} val:  ", val_metrics)
            epoch_record["val"] = val_metrics
            monitor_metrics = val_metrics
        gamma_values = print_gamma_values(f"Epoch {epoch:03d} gamma:", model)
        if gamma_values:
            epoch_record["gamma"] = gamma_values
        history.append(epoch_record)
        if scheduler is not None:
            scheduler.step()
            print(
                f"Epoch {epoch:03d} next learning rate: "
                f"{optimizer.param_groups[0]['lr']:.8f}"
            )
        save_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            args,
            monitor_metrics,
            model_name,
        )
        if monitor_metrics["Dice"] > best_dice:
            best_dice = monitor_metrics["Dice"]
            save_checkpoint(
                output_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                args,
                monitor_metrics,
                model_name,
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "training_history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
    print(f"Training complete. Best monitored Dice={best_dice:.4f}")
