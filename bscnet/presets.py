"""Fixed settings used for the final BSC-Net experiments."""

from argparse import Namespace
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
# The environment override is useful for an externally mounted data package;
# the checked-in config remains the portable default for a standard clone.
DATASET_CONFIG = Path(
    os.environ.get("BSCNET_DATASET_CONFIG", ROOT / "configs" / "datasets.json")
).expanduser()
DATA_LOADER = ROOT / "bscnet" / "data.py"

DATASETS = {
    "mosxav": {
        "seed": 2026,
        "threshold": 0.699,
        "checkpoint": ROOT / "checkpoints" / "bscnet_mosxav_best.pt",
    },
    "ica_nj": {
        "seed": 5,
        "threshold": 0.296,
        "checkpoint": ROOT / "checkpoints" / "bscnet_ica_nj_best.pt",
    },
}


def apply_paper_defaults(args: Namespace) -> Namespace:
    """Apply the final experimental setup without changing user CLI options."""
    args.encoder = "resnet34"
    args.data_dir = str(ROOT / "data")
    args.loader_file = str(DATA_LOADER)
    args.dataset_config = str(DATASET_CONFIG)

    args.branch_crop_prob = 0.6
    args.branch_crop_ratio = 0.3
    args.branch_crop_min_component_area = 1
    args.branch_crop_small_component_area = 50
    args.repeat_factor = 5

    args.loss = "optimized"
    args.mse_weight = 1.0
    args.dice_weight = 0.35
    args.lambda_eil = 0.05
    args.lambda_tv = 0.01
    args.lambda_fp = 0.10
    args.lambda_cc = 0.04
    args.lambda_cldice = 0.20
    args.dilate_radius = 3
    args.eil_gamma = 1.0
    args.cldice_iters = 8
    args.focal_alpha = 0.25
    args.focal_gamma = 2.0
    args.fp_threshold = 0.62
    args.fp_hardness = 1.5
    args.component_kernel_size = 9
    args.schedule_warmup_epochs = 15
    args.schedule_ramp_epochs = 12

    args.epochs = 50
    args.lr = 1e-4
    args.weight_decay = 1e-4
    args.cosine_lr = True
    args.eta_min = 1e-6
    args.width = 512
    args.height = 512
    return args
