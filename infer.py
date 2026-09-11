"""Run BSC-Net on a directory of unlabeled X-ray angiography images."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bscnet.data import cv2_imread_unicode, cv2_imwrite_unicode, read_image_rgb
from bscnet.engine import load_model_checkpoint, resolve_device
from bscnet.model import BSCNet
from bscnet.presets import DATASETS


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment coronary vessels in unlabeled X-ray angiography images."
    )
    parser.add_argument("--input-dir", required=True, help="Directory searched recursively for images.")
    parser.add_argument("--output-dir", required=True, help="Directory for binary masks and PNG/NumPy probabilities.")
    parser.add_argument("--dataset", choices=DATASETS, default="mosxav")
    parser.add_argument("--checkpoint", default=None, help="BSC-Net checkpoint; defaults to the selected dataset checkpoint.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    return parser


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint or DATASETS[args.dataset]["checkpoint"]).expanduser().resolve()
    threshold = DATASETS[args.dataset]["threshold"] if args.threshold is None else args.threshold

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1].")

    image_paths = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in: {input_dir}")

    device = resolve_device(args.device)
    model = BSCNet().to(device)
    load_model_checkpoint(model, str(checkpoint), device)
    model.eval()

    for image_path in image_paths:
        raw = cv2_imread_unicode(image_path)
        original_height, original_width = raw.shape[:2]
        image = read_image_rgb(image_path, image_size=(args.width, args.height))
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)
        probability = torch.sigmoid(model(tensor))
        probability = F.interpolate(
            probability,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy().astype(np.float32)
        prediction = (probability >= threshold).astype(np.uint8) * 255

        relative_path = image_path.relative_to(input_dir).with_suffix(".png")
        mask_path = output_dir / "masks" / relative_path
        probability_path = output_dir / "probabilities" / relative_path
        probability_npy_path = output_dir / "probabilities_npy" / relative_path.with_suffix(".npy")
        cv2_imwrite_unicode(mask_path, prediction)
        cv2_imwrite_unicode(
            probability_path,
            np.clip(probability * 255.0, 0, 255).round().astype(np.uint8),
        )
        probability_npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(probability_npy_path), probability)

    print(f"Segmented {len(image_paths)} image(s). Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
