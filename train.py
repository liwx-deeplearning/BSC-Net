"""Train the final BSC-Net configuration on MOSXAV or ICA_NJ."""

from bscnet.engine import build_parser, run_experiment
from bscnet.model import BSCNet
from bscnet.presets import DATASETS, ROOT, apply_paper_defaults


def build_model(args):
    return BSCNet(encoder_weights=args.encoder_weights)


def main() -> None:
    parser = build_parser("Train BSC-Net for X-ray angiography vessel segmentation.")
    parser.add_argument("--dataset", choices=DATASETS, default="mosxav")
    parser.set_defaults(
        mode="train",
        seed=None,
        batch_size=2,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        cudnn_benchmark=True,
    )
    args = apply_paper_defaults(parser.parse_args())
    args.mode = "train"
    if args.seed is None:
        args.seed = DATASETS[args.dataset]["seed"]
    args.train_datasets = [args.dataset]
    args.val_datasets = [args.dataset]
    args.eval_datasets = [args.dataset]
    if args.encoder_weights is None and args.init_checkpoint is None:
        local_weights = ROOT / "pretrained" / "resnet34-b627a593.pth"
        args.encoder_weights = str(local_weights) if local_weights.is_file() else "imagenet"
    if args.output_dir is None:
        args.output_dir = str(ROOT / "outputs" / args.dataset / "train")
    run_experiment(args, build_model, "BSC-Net")


if __name__ == "__main__":
    main()
