"""Evaluate a BSC-Net checkpoint on the configured MOSXAV or ICA_NJ test split."""

from bscnet.engine import build_parser, run_experiment
from bscnet.model import BSCNet
from bscnet.presets import DATASETS, ROOT, apply_paper_defaults


def build_model(_args):
    return BSCNet()


def main() -> None:
    parser = build_parser("Evaluate BSC-Net on a configured test split.")
    parser.add_argument("--dataset", choices=DATASETS, default="mosxav")
    parser.set_defaults(mode="eval", threshold=None, batch_size=2, num_workers=4)
    args = apply_paper_defaults(parser.parse_args())
    args.mode = "eval"
    args.eval_datasets = [args.dataset]
    args.checkpoint = args.checkpoint or str(DATASETS[args.dataset]["checkpoint"])
    if args.threshold is None:
        args.threshold = DATASETS[args.dataset]["threshold"]
    if args.output_dir is None:
        args.output_dir = str(ROOT / "outputs" / args.dataset / "evaluation")
    run_experiment(args, build_model, "BSC-Net")


if __name__ == "__main__":
    main()
