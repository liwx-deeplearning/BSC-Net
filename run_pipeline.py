"""Run the complete BSC-Net segmentation and quantitative-analysis workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train BSC-Net or load an existing checkpoint, evaluate segmentation, "
            "predict a dynamic sequence, repair masks, and calculate anatomical "
            "QCA and hemodynamic parameters."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Dynamic XCA frames to segment.")
    parser.add_argument(
        "--frame-root", type=Path, default=None,
        help="Root containing per-video frame folders; defaults to --input-dir.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--peak-frames", nargs="+", default=None, metavar="VIDEO=FRAME")
    parser.add_argument("--dataset", choices=("mosxav", "ica_nj"), default="mosxav")
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument(
        "--train", action="store_true",
        help="Train the final preset first and use the resulting best checkpoint.",
    )
    model_source.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Existing checkpoint; defaults to the packaged checkpoint for --dataset.",
    )
    parser.add_argument(
        "--train-init-checkpoint", type=Path, default=None,
        help="Optional model initialization checkpoint used only with --train.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands and output layout without running the workflow.",
    )
    return parser


def run(command: list[str], *, dry_run: bool) -> None:
    print(f"[pipeline] {subprocess.list2cmdline(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.train_init_checkpoint is not None and not args.train:
        parser.error("--train-init-checkpoint requires --train")
    output_root = args.output_dir.expanduser().resolve()
    frame_root = (args.frame_root or args.input_dir).expanduser().resolve()
    training_dir = output_root / "training"
    evaluation_dir = output_root / "segmentation_evaluation"
    inference_dir = output_root / "inference"
    postprocessed_dir = output_root / "postprocessed"
    analysis_root = output_root / "quantitative_analysis"
    morphology_dir = analysis_root / "morphology"
    hemodynamics_dir = analysis_root / "hemodynamics"
    figures_dir = analysis_root / "figures"

    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    if args.train:
        training_command = [
            sys.executable, str(PROJECT_ROOT / "train.py"),
            "--dataset", args.dataset,
            "--output-dir", str(training_dir),
            "--device", args.device,
        ]
        if args.train_init_checkpoint is not None:
            training_command.extend(
                ["--init-checkpoint", str(args.train_init_checkpoint.expanduser().resolve())]
            )
        run(training_command, dry_run=args.dry_run)
        checkpoint = training_dir / "checkpoints" / "best.pt"

    evaluation_command = [
        sys.executable, str(PROJECT_ROOT / "evaluate.py"),
        "--dataset", args.dataset,
        "--output-dir", str(evaluation_dir),
        "--device", args.device,
    ]
    if checkpoint is not None:
        evaluation_command.extend(["--checkpoint", str(checkpoint)])
    if args.threshold is not None:
        evaluation_command.extend(["--threshold", str(args.threshold)])
    run(evaluation_command, dry_run=args.dry_run)

    inference_command = [
        sys.executable, str(PROJECT_ROOT / "infer.py"),
        "--dataset", args.dataset,
        "--input-dir", str(args.input_dir.expanduser().resolve()),
        "--output-dir", str(inference_dir),
        "--device", args.device,
    ]
    if checkpoint is not None:
        inference_command.extend(["--checkpoint", str(checkpoint)])
    if args.threshold is not None:
        inference_command.extend(["--threshold", str(args.threshold)])
    run(inference_command, dry_run=args.dry_run)

    run(
        [
            sys.executable, str(PROJECT_ROOT / "postprocessing" / "vessel_repair.py"),
            "--base-mask-dir", str(inference_dir / "masks"),
            "--probability-dir", str(inference_dir / "probabilities_npy"),
            "--output-dir", str(postprocessed_dir),
        ], dry_run=args.dry_run,
    )

    morphology_command = [
        sys.executable,
        str(PROJECT_ROOT / "quantitative_analysis" / "morphology" / "structural_analysis.py"),
        "--videos", *args.videos,
        "--mask-root", str(postprocessed_dir),
        "--output-dir", str(morphology_dir),
    ]
    if args.peak_frames:
        morphology_command.extend(["--peak-frames", *args.peak_frames])
    run(morphology_command, dry_run=args.dry_run)

    hemodynamics_command = [
        sys.executable,
        str(PROJECT_ROOT / "quantitative_analysis" / "hemodynamics" / "hemodynamic_analysis.py"),
        "--videos", *args.videos,
        "--mask-root", str(postprocessed_dir),
        "--frame-root", str(frame_root),
        "--output-dir", str(hemodynamics_dir),
    ]
    if args.peak_frames:
        hemodynamics_command.extend(["--peak-frames", *args.peak_frames])
    run(hemodynamics_command, dry_run=args.dry_run)

    run(
        [
            sys.executable,
            str(PROJECT_ROOT / "quantitative_analysis" / "generate_figures.py"),
            "--morphology-dir", str(morphology_dir),
            "--hemodynamics-dir", str(hemodynamics_dir),
            "--output-dir", str(figures_dir),
        ],
        dry_run=args.dry_run,
    )

    print("\nComplete pipeline output layout")
    print(f"Segmentation predictions and metrics: {evaluation_dir}")
    print(f"Dynamic-sequence predictions:          {inference_dir}")
    print(f"Postprocessed masks:                   {postprocessed_dir}")
    print(f"Anatomical QCA results:                {morphology_dir}")
    print(f"Hemodynamic-surrogate results:         {hemodynamics_dir}")
    print(f"Quantitative-analysis figures:         {figures_dir}")


if __name__ == "__main__":
    main()
