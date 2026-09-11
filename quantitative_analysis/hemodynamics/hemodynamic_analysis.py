#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Calculate AT, TTP, maximum wash-in slope, CTFC-like delay, and relative propagation velocity."""


from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import math
import os
import heapq
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

# 中文显示与字号统一设置
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.titlesize"] = 10
matplotlib.rcParams["axes.labelsize"] = 9
matplotlib.rcParams["xtick.labelsize"] = 8
matplotlib.rcParams["ytick.labelsize"] = 8
matplotlib.rcParams["legend.fontsize"] = 7


# ===== Shared hemodynamic utilities =====
Array = np.ndarray


def smooth_and_normalize_curve(curve: Array, window: int = 3) -> Array:
    values = np.asarray(curve, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("curve must be 1D")
    if window < 1:
        raise ValueError("window must be >= 1")

    if window > 1 and values.size >= window:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        pad_left = window // 2
        pad_right = window - 1 - pad_left
        padded = np.pad(values, (pad_left, pad_right), mode="edge")
        values = np.convolve(padded, kernel, mode="valid").astype(np.float32)

    min_value = float(values.min())
    max_value = float(values.max())
    scale = max_value - min_value
    if scale <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / scale).astype(np.float32)


def detect_arrival_frame(
    curve: Array,
    threshold: float = 0.2,
    consecutive_frames: int = 2,
) -> int | None:
    values = np.asarray(curve, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("curve must be 1D")
    if consecutive_frames < 1:
        raise ValueError("consecutive_frames must be >= 1")

    for start in range(0, values.size - consecutive_frames + 1):
        window = values[start : start + consecutive_frames]
        if bool(np.all(window >= threshold)):
            return int(start)
    return None


def compute_pseudo_timi(
    t_start: int | None,
    t_arrival: int | None,
    frame_interval: float | None = None,
) -> dict[str, float | int | None]:
    if t_start is None or t_arrival is None:
        return {"frames": None, "seconds": None}
    frames = int(t_arrival) - int(t_start)
    if frames < 0:
        return {"frames": None, "seconds": None}
    seconds = None if frame_interval is None else float(frames) * float(frame_interval)
    return {"frames": frames, "seconds": seconds}


def estimate_delay_by_cross_correlation(
    curve_proximal: Array,
    curve_distal: Array,
    max_lag: int | None = None,
) -> tuple[int | None, float | None]:
    proximal = np.asarray(curve_proximal, dtype=np.float32)
    distal = np.asarray(curve_distal, dtype=np.float32)
    if proximal.ndim != 1 or distal.ndim != 1:
        raise ValueError("curves must be 1D")
    if proximal.size != distal.size:
        raise ValueError("curves must have the same length")
    if proximal.size < 2:
        return None, None

    n_frames = proximal.size
    if max_lag is None:
        max_lag = max(1, n_frames // 2)
    max_lag = min(int(max_lag), n_frames - 1)

    best_lag: int | None = None
    best_corr = -np.inf
    for lag in range(0, max_lag + 1):
        p = proximal[: n_frames - lag]
        d = distal[lag:]
        if p.size < 2:
            continue
        p_std = float(p.std())
        d_std = float(d.std())
        if p_std <= 1e-8 or d_std <= 1e-8:
            continue
        corr = float(np.mean(((p - p.mean()) / p_std) * ((d - d.mean()) / d_std)))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    if best_lag is None:
        return None, None
    return best_lag, best_corr


def compute_relative_velocity(
    delta_s: float | None,
    delta_t: float | None,
    pixel_spacing: float | None = None,
    frame_interval: float | None = None,
) -> dict[str, float | str | None]:
    if delta_s is None or delta_t is None or float(delta_t) <= 0:
        return {"value": None, "unit": None}

    distance = float(delta_s)
    delay = float(delta_t)

    if pixel_spacing is not None:
        distance *= float(pixel_spacing)
        distance_unit = "mm"
    else:
        distance_unit = "pixel"

    if frame_interval is not None:
        delay *= float(frame_interval)
        time_unit = "s"
    else:
        time_unit = "frame"

    return {"value": distance / delay, "unit": f"{distance_unit}/{time_unit}"}


def load_sequence_frames(sequence_dir: Path) -> tuple[Array, list[int]]:
    from PIL import Image

    frame_paths = sorted(sequence_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"no jpg frames found in {sequence_dir}")

    frames = []
    frame_ids = []
    for path in frame_paths:
        image = Image.open(path).convert("L")
        frames.append(np.asarray(image, dtype=np.float32) / 255.0)
        frame_ids.append(int(path.stem))
    return np.stack(frames, axis=0), frame_ids


def load_mask(path: Path) -> Array:
    from PIL import Image

    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.uint8)


def select_peak_mask_by_area(mask_dir: Path, threshold: int = 127) -> tuple[Path, list[dict[str, int | str]]]:
    mask_paths = sorted(mask_dir.glob("*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"no png masks found in {mask_dir}")

    candidates = []
    for path in mask_paths:
        mask = load_mask(path)
        area = int((mask > threshold).sum())
        candidates.append({"frame": int(path.stem), "file": path.name, "area": area})

    best = max(candidates, key=lambda item: int(item["area"]))
    return mask_dir / str(best["file"]), candidates


def postprocess_peak_mask(mask: Array, min_component_area: int = 50) -> tuple[Array, Array]:
    import cv2
    from skimage.morphology import skeletonize

    binary = (mask > 127).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    clean = np.zeros_like(closed, dtype=np.uint8)
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            clean[labels == label] = 1

    if int(clean.sum()) == 0:
        raise ValueError("mask has no component after postprocessing")

    skeleton = skeletonize(clean.astype(bool)).astype(np.uint8)
    if int(skeleton.sum()) == 0:
        raise ValueError("skeleton is empty")
    return clean.astype(bool), skeleton.astype(bool)


def extract_main_centerline(skeleton: Array) -> list[tuple[int, int]]:
    coords = [tuple(coord) for coord in np.argwhere(skeleton > 0)]
    coord_set = set(coords)
    if not coords:
        raise ValueError("skeleton is empty")

    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row, col in coords:
        current = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                item = (row + dr, col + dc)
                if item in coord_set:
                    current.append(item)
        neighbors[(row, col)] = current

    endpoints = [point for point, items in neighbors.items() if len(items) == 1]
    if len(endpoints) < 2:
        endpoints = [coords[0]]

    best_path: list[tuple[int, int]] = [coords[0]]
    best_distance = -1
    starts = endpoints if len(endpoints) >= 2 else [coords[0]]
    for start in starts:
        distances, parents = _bfs_with_parents(start, neighbors)
        targets = endpoints if len(endpoints) >= 2 else list(distances.keys())
        for target in targets:
            distance = distances.get(target, -1)
            if distance > best_distance:
                best_distance = distance
                best_path = _reconstruct_path(start, target, parents)

    return best_path


def define_centerline_rois(
    centerline: list[tuple[int, int]],
    image_shape: tuple[int, int],
    clean_mask: Array,
    roi_radius: int = 5,
    ring_inner_radius: int = 8,
    ring_outer_radius: int = 14,
) -> dict[str, Any]:
    if len(centerline) < 3:
        raise ValueError("centerline is too short")

    indices = {
        "a": max(0, min(len(centerline) - 1, int(round(0.10 * (len(centerline) - 1))))),
        "mid": max(0, min(len(centerline) - 1, int(round(0.50 * (len(centerline) - 1))))),
        "b": max(0, min(len(centerline) - 1, int(round(0.90 * (len(centerline) - 1))))),
    }

    rois = {}
    for name, index in indices.items():
        center = centerline[index]
        roi = _disk_mask(image_shape, center, roi_radius)
        if int((roi & clean_mask).sum()) >= 5:
            roi = roi & clean_mask
        ring = _ring_mask(image_shape, center, ring_inner_radius, ring_outer_radius)
        ring = ring & (~clean_mask)
        if int(ring.sum()) < 5:
            ring = _ring_mask(image_shape, center, ring_inner_radius, ring_outer_radius)
        rois[name] = {
            "center": {"row": int(center[0]), "col": int(center[1])},
            "index": int(index),
            "roi": roi,
            "ring": ring,
        }
    return rois


def compute_arc_length_for_indices(
    centerline: list[tuple[int, int]],
    index_a: int,
    index_b: int,
) -> float:
    lo = min(index_a, index_b)
    hi = max(index_a, index_b)
    if hi <= lo:
        return 0.0
    distance = 0.0
    for i in range(lo + 1, hi + 1):
        r0, c0 = centerline[i - 1]
        r1, c1 = centerline[i]
        distance += math.hypot(float(r1 - r0), float(c1 - c0))
    return float(distance)


def extract_roi_curve(frames: Array, roi: Array, background_ring: Array) -> Array:
    if frames.ndim != 3:
        raise ValueError("frames must have shape (T, H, W)")
    if roi.shape != frames.shape[1:] or background_ring.shape != frames.shape[1:]:
        raise ValueError("roi and background_ring must match frame shape")
    if int(roi.sum()) == 0:
        raise ValueError("roi is empty")
    if int(background_ring.sum()) == 0:
        raise ValueError("background ring is empty")

    vessel = frames[:, roi].mean(axis=1)
    background = frames[:, background_ring].mean(axis=1)
    curve = vessel - background
    peak_index = int(np.argmax(np.abs(curve)))
    if float(curve[peak_index]) < 0.0:
        curve = -curve
    return curve.astype(np.float32)


def run_video(
    video_id: str,
    mask_root: Path,
    frame_root: Path,
    output_dir: Path,
    frame_interval: float | None = None,
    pixel_spacing: float | None = None,
) -> dict[str, Any]:
    mask_dir = mask_root / video_id
    sequence_dir = frame_root / video_id

    frames, frame_ids = load_sequence_frames(sequence_dir)
    peak_mask_path, peak_candidates = select_peak_mask_by_area(mask_dir)
    peak_mask = load_mask(peak_mask_path)
    clean_mask, skeleton = postprocess_peak_mask(peak_mask)
    centerline = extract_main_centerline(skeleton)
    rois = define_centerline_rois(centerline, frames.shape[1:], clean_mask)

    curve_a = extract_roi_curve(frames, rois["a"]["roi"], rois["a"]["ring"])
    curve_b = extract_roi_curve(frames, rois["b"]["roi"], rois["b"]["ring"])
    curve_mid = extract_roi_curve(frames, rois["mid"]["roi"], rois["mid"]["ring"])

    curve_a_norm = smooth_and_normalize_curve(curve_a)
    curve_b_norm = smooth_and_normalize_curve(curve_b)
    curve_mid_norm = smooth_and_normalize_curve(curve_mid)

    t_a = detect_arrival_frame(curve_a_norm)
    t_b = detect_arrival_frame(curve_b_norm)
    if t_a is not None and t_b is not None and t_b < t_a:
        proximal_key, distal_key = "b", "a"
        proximal_curve = curve_b_norm
        distal_curve = curve_a_norm
        t_start = t_b
        t_arrival = t_a
    else:
        proximal_key, distal_key = "a", "b"
        proximal_curve = curve_a_norm
        distal_curve = curve_b_norm
        t_start = t_a
        t_arrival = t_b

    delta_t, corr_peak = estimate_delay_by_cross_correlation(proximal_curve, distal_curve)
    delta_s = compute_arc_length_for_indices(
        centerline,
        int(rois[proximal_key]["index"]),
        int(rois[distal_key]["index"]),
    )
    pseudo_timi = compute_pseudo_timi(t_start, t_arrival, frame_interval)
    velocity = compute_relative_velocity(delta_s, delta_t, pixel_spacing, frame_interval)

    qc = _quality_control(t_start, t_arrival, delta_t, corr_peak, delta_s)
    result = {
        "video_id": video_id,
        "selected_peak_mask": peak_mask_path.name,
        "selected_peak_frame": int(peak_mask_path.stem),
        "peak_candidates": peak_candidates,
        "frame_count": int(frames.shape[0]),
        "frame_ids": frame_ids,
        "centerline_points": int(len(centerline)),
        "clean_mask_area": int(clean_mask.sum()),
        "skeleton_length_pixels": int(skeleton.sum()),
        "proximal_key": proximal_key,
        "distal_key": distal_key,
        "roi_centers": {
            "proximal": rois[proximal_key]["center"],
            "mid": rois["mid"]["center"],
            "distal": rois[distal_key]["center"],
        },
        "delta_s_pixel": delta_s,
        "delta_t_frame": delta_t,
        "correlation_peak": corr_peak,
        "t_start_frame": t_start,
        "t_arrival_frame": t_arrival,
        "pseudo_timi": pseudo_timi,
        "relative_velocity": velocity,
        "qc": qc,
        "curves": {
            "proximal_norm": proximal_curve.tolist(),
            "mid_norm": curve_mid_norm.tolist(),
            "distal_norm": distal_curve.tolist(),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{video_id}_hemodynamics.json"
    csv_path = output_dir / f"{video_id}_hemodynamics.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_csv(csv_path, result)
    result["json_path"] = str(json_path)
    result["csv_path"] = str(csv_path)
    return result


def _bfs_with_parents(
    start: tuple[int, int],
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int] | None]]:
    distances = {start: 0}
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        point = queue.popleft()
        for item in neighbors[point]:
            if item in distances:
                continue
            distances[item] = distances[point] + 1
            parents[item] = point
            queue.append(item)
    return distances, parents


def _reconstruct_path(
    start: tuple[int, int],
    target: tuple[int, int],
    parents: dict[tuple[int, int], tuple[int, int] | None],
) -> list[tuple[int, int]]:
    path = [target]
    current = target
    while current != start:
        parent = parents.get(current)
        if parent is None:
            break
        path.append(parent)
        current = parent
    path.reverse()
    return path


def _disk_mask(shape: tuple[int, int], center: tuple[int, int], radius: int) -> Array:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    row, col = center
    return ((rows - row) ** 2 + (cols - col) ** 2) <= radius**2


def _ring_mask(
    shape: tuple[int, int],
    center: tuple[int, int],
    inner_radius: int,
    outer_radius: int,
) -> Array:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    row, col = center
    distance_sq = (rows - row) ** 2 + (cols - col) ** 2
    return (distance_sq >= inner_radius**2) & (distance_sq <= outer_radius**2)


def _quality_control(
    t_start: int | None,
    t_arrival: int | None,
    delta_t: int | None,
    corr_peak: float | None,
    delta_s: float,
) -> dict[str, Any]:
    issues = []
    if t_start is None:
        issues.append("missing_t_start")
    if t_arrival is None:
        issues.append("missing_t_arrival")
    if t_start is not None and t_arrival is not None and t_arrival < t_start:
        issues.append("arrival_before_start")
    if delta_t is None or delta_t <= 0:
        issues.append("invalid_delta_t")
    if corr_peak is None or corr_peak < 0.5:
        issues.append("low_correlation_peak")
    if delta_s <= 0:
        issues.append("invalid_delta_s")
    return {"status": "Pass" if not issues else "Fail", "issues": issues}


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    row = {
        "video_id": result["video_id"],
        "selected_peak_mask": result["selected_peak_mask"],
        "frame_count": result["frame_count"],
        "delta_s_pixel": result["delta_s_pixel"],
        "delta_t_frame": result["delta_t_frame"],
        "correlation_peak": result["correlation_peak"],
        "v_rel": result["relative_velocity"]["value"],
        "v_rel_unit": result["relative_velocity"]["unit"],
        "t_start_frame": result["t_start_frame"],
        "t_arrival_frame": result["t_arrival_frame"],
        "pseudo_timi_frame": result["pseudo_timi"]["frames"],
        "pseudo_timi_second": result["pseudo_timi"]["seconds"],
        "qc_status": result["qc"]["status"],
        "qc_issues": ";".join(result["qc"]["issues"]),
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


# ===== Shared quantitative-analysis utilities =====
Array = np.ndarray
Color = tuple[int, int, int]

NPG_COLORS = {
    "red": "#E64B35",
    "blue": "#4DBBD5",
    "green": "#00A087",
    "purple": "#3C5488",
    "orange": "#F39B7F",
    "slate": "#8491B4",
}


def _hex_rgb(value: str) -> Color:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


RED: Color = _hex_rgb(NPG_COLORS["red"])
BLUE: Color = _hex_rgb(NPG_COLORS["blue"])
GREEN: Color = _hex_rgb(NPG_COLORS["green"])
PURPLE: Color = _hex_rgb(NPG_COLORS["purple"])
ORANGE: Color = _hex_rgb(NPG_COLORS["orange"])
SLATE: Color = _hex_rgb(NPG_COLORS["slate"])
YELLOW: Color = ORANGE
DARK: Color = (35, 42, 52)
GRAY: Color = (105, 115, 128)


def choose_key_frame_indices(
    frame_count: int,
    t_proximal: int | None,
    t_distal: int | None,
) -> list[int]:
    if frame_count < 1:
        return []
    start = 0 if t_proximal is None else max(0, min(frame_count - 1, t_proximal))
    end = frame_count - 1 if t_distal is None else max(0, min(frame_count - 1, t_distal))
    if end < start:
        start, end = end, start
    middle = int(round((start + end) / 2.0))
    return sorted(set([start, middle, end]))


def summarize_metric_box(
    pseudo_timi_frames: int | None,
    relative_velocity: float | None,
    velocity_unit: str | None,
    qc_status: str,
) -> str:
    ctfc = "N/A" if pseudo_timi_frames is None else f"{pseudo_timi_frames} frames"
    velocity = (
        "N/A"
        if relative_velocity is None or velocity_unit is None
        else f"{relative_velocity:.2f} {velocity_unit}"
    )
    return f"CTFC-like = {ctfc}\nv_rel = {velocity}\nQC = {qc_status}"


def validate_mask_frame_subset(
    mask_frame_ids: list[int],
    dynamic_frame_ids: list[int],
) -> None:
    missing = sorted(set(mask_frame_ids) - set(dynamic_frame_ids))
    if missing:
        raise ValueError(f"Mask frame IDs are absent from dynamic sequence: {missing}")


def is_cross_correlation_overlap_sufficient(
    frame_count: int,
    lag: int | None,
    min_overlap_frames: int = 4,
    min_overlap_ratio: float = 0.4,
) -> bool:
    if lag is None or lag < 0 or frame_count < 1:
        return False
    overlap = frame_count - lag
    required = max(min_overlap_frames, int(math.ceil(frame_count * min_overlap_ratio)))
    return overlap >= required


def is_temporal_delay_reliable(
    delta_t: int | None,
    correlation_peak: float | None,
    correlation_threshold: float = 0.5,
) -> bool:
    return (
        delta_t is not None
        and delta_t > 0
        and correlation_peak is not None
        and correlation_peak >= correlation_threshold
    )


def svg_document(width: int, height: int, title: str, body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f"<title>{html.escape(title)}</title>\n"
        "<style>"
        "text{font-family:Arial,'Microsoft YaHei',sans-serif;}"
        ".title{font-size:46px;font-weight:700;fill:#232A34;}"
        ".subtitle{font-size:25px;fill:#8491B4;}"
        ".label{font-size:25px;font-weight:700;}"
        ".small{font-size:20px;fill:#8491B4;}"
        ".metric{font-size:27px;font-weight:700;fill:#232A34;}"
        "</style>\n"
        f"{body}\n</svg>\n"
    )


def load_gray_sequence(sequence_dir: Path) -> tuple[Array, list[int], list[Path]]:
    paths = sorted(sequence_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No JPG frames found: {sequence_dir}")
    frames: list[Array] = []
    frame_ids: list[int] = []
    for path in paths:
        frame = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        frames.append(frame)
        frame_ids.append(int(path.stem))
    return np.stack(frames, axis=0), frame_ids, paths


def load_mask_sequence(mask_dir: Path) -> tuple[list[Array], list[int], list[Path]]:
    paths = sorted(mask_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG masks found: {mask_dir}")
    masks: list[Array] = []
    frame_ids: list[int] = []
    for path in paths:
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        masks.append(mask)
        frame_ids.append(int(path.stem))
    return masks, frame_ids, paths


def extract_roi_curves_with_shared_polarity(
    frames: Array,
    rois: dict[str, dict[str, Any]],
    peak_frame_index: int,
    proximal_key: str = "a",
    baseline_frames: int = 5,
) -> tuple[dict[str, Array], float]:
    raw_curves: dict[str, Array] = {}
    for key in ("a", "mid", "b"):
        roi = np.asarray(rois[key]["roi"], dtype=bool)
        ring = np.asarray(rois[key]["ring"], dtype=bool)
        vessel = frames[:, roi].mean(axis=1)
        background = frames[:, ring].mean(axis=1)
        raw_curves[key] = (vessel - background).astype(np.float32)

    proximal = raw_curves[proximal_key]
    baseline_count = min(max(1, baseline_frames), proximal.size)
    baseline = float(np.median(proximal[:baseline_count]))
    peak_change = float(proximal[peak_frame_index] - baseline)
    if abs(peak_change) <= 1e-8:
        strongest = int(np.argmax(np.abs(proximal - baseline)))
        peak_change = float(proximal[strongest] - baseline)
    polarity = -1.0 if peak_change < 0.0 else 1.0
    return {
        key: (curve * polarity).astype(np.float32)
        for key, curve in raw_curves.items()
    }, polarity


def prepare_dynamic_curve(
    curve: Array,
    smoothing_window: int = 3,
    baseline_frames: int = 10,
    peak_fraction_threshold: float = 0.2,
    noise_sigma_multiplier: float = 3.0,
) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("curve must be 1D")
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be >= 1")
    if baseline_frames < 2:
        raise ValueError("baseline_frames must be >= 2")
    if not 0.0 <= peak_fraction_threshold <= 1.0:
        raise ValueError("peak_fraction_threshold must be within [0, 1]")

    smoothed = values.copy()
    if smoothing_window > 1 and values.size >= smoothing_window:
        kernel = np.ones(smoothing_window, dtype=np.float32) / float(smoothing_window)
        left = smoothing_window // 2
        right = smoothing_window - 1 - left
        padded = np.pad(values, (left, right), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid").astype(np.float32)

    baseline_count = min(baseline_frames, smoothed.size)
    baseline_values = smoothed[:baseline_count]
    baseline = float(np.median(baseline_values))
    noise_sigma = float(np.std(baseline_values))
    enhancement = np.maximum(smoothed - baseline, 0.0).astype(np.float32)
    peak = float(enhancement.max())
    if peak <= 1e-8:
        normalized = np.zeros_like(enhancement)
        threshold = 1.0
    else:
        normalized = (enhancement / peak).astype(np.float32)
        noise_threshold = noise_sigma_multiplier * noise_sigma / peak
        threshold = float(
            min(1.0, max(peak_fraction_threshold, noise_threshold))
        )
    return {
        "smoothed": smoothed,
        "baseline": baseline,
        "noise_sigma": noise_sigma,
        "enhancement": enhancement,
        "peak_enhancement": peak,
        "normalized": normalized,
        "threshold": threshold,
    }


def track_roi_centers(
    frames: Array,
    reference_center: tuple[int, int],
    reference_frame_index: int,
    template_radius: int = 8,
    search_radius: int = 12,
) -> list[tuple[int, int]]:
    if frames.ndim != 3:
        raise ValueError("frames must have shape (T, H, W)")
    if not 0 <= reference_frame_index < frames.shape[0]:
        raise ValueError("reference_frame_index is out of range")
    if template_radius < 1 or search_radius < 0:
        raise ValueError("tracking radii must be non-negative")

    height, width = frames.shape[1:]
    reference = (
        int(np.clip(reference_center[0], 0, height - 1)),
        int(np.clip(reference_center[1], 0, width - 1)),
    )
    centers: list[tuple[int, int] | None] = [None] * frames.shape[0]
    centers[reference_frame_index] = reference

    def track_direction(indices: range) -> None:
        previous_index = reference_frame_index
        for frame_index in indices:
            previous_center = centers[previous_index]
            if previous_center is None:
                raise RuntimeError("tracking lost its previous center")
            centers[frame_index] = _match_local_template(
                source=frames[previous_index],
                target=frames[frame_index],
                center=previous_center,
                template_radius=template_radius,
                search_radius=search_radius,
            )
            previous_index = frame_index

    track_direction(range(reference_frame_index - 1, -1, -1))
    track_direction(range(reference_frame_index + 1, frames.shape[0]))
    return [center for center in centers if center is not None]


def _match_local_template(
    source: Array,
    target: Array,
    center: tuple[int, int],
    template_radius: int,
    search_radius: int,
) -> tuple[int, int]:
    height, width = source.shape
    row, col = center
    top = max(0, row - template_radius)
    bottom = min(height, row + template_radius + 1)
    left = max(0, col - template_radius)
    right = min(width, col + template_radius + 1)
    template = np.asarray(source[top:bottom, left:right], dtype=np.float32)
    if template.size == 0:
        return center

    search_top = max(0, top - search_radius)
    search_bottom = min(height, bottom + search_radius)
    search_left = max(0, left - search_radius)
    search_right = min(width, right + search_radius)
    search = np.asarray(
        target[search_top:search_bottom, search_left:search_right],
        dtype=np.float32,
    )
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return center

    template_features = _tracking_features(template)
    search_features = _tracking_features(search)
    response = cv2.matchTemplate(
        search_features,
        template_features,
        cv2.TM_CCOEFF_NORMED,
    )
    if not np.isfinite(response).any():
        return center
    _, _, _, max_location = cv2.minMaxLoc(np.nan_to_num(response, nan=-1.0))
    matched_left = search_left + max_location[0]
    matched_top = search_top + max_location[1]
    matched_row = matched_top + (bottom - top) // 2
    matched_col = matched_left + (right - left) // 2
    return (
        int(np.clip(matched_row, 0, height - 1)),
        int(np.clip(matched_col, 0, width - 1)),
    )


def _tracking_features(image: Array) -> Array:
    values = np.asarray(image, dtype=np.float32)
    blurred = cv2.GaussianBlur(values, (0, 0), sigmaX=1.2)
    high_pass = values - blurred
    gradient_x = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    return (high_pass + gradient).astype(np.float32)


def extract_tracked_roi_curve_details(
    frames: Array,
    centers: list[tuple[int, int]],
    roi_radius: int,
    ring_inner_radius: int,
    ring_outer_radius: int,
) -> dict[str, Array]:
    if len(centers) != frames.shape[0]:
        raise ValueError("one tracked center is required per frame")
    vessel_values: list[float] = []
    background_values: list[float] = []
    for frame, center in zip(frames, centers):
        roi = _disk_mask_local(frame.shape, center, roi_radius)
        ring = _ring_mask_local(
            frame.shape,
            center,
            ring_inner_radius,
            ring_outer_radius,
        )
        vessel_values.append(float(frame[roi].mean()))
        background_values.append(float(frame[ring].mean()))
    vessel = np.asarray(vessel_values, dtype=np.float32)
    background = np.asarray(background_values, dtype=np.float32)
    return {
        "vessel": vessel,
        "background": background,
        "corrected": (vessel - background).astype(np.float32),
    }


def select_stable_distal_candidate(
    candidates: list[dict[str, Any]],
    proximal_arrival: int | None,
    correlation_threshold: float = 0.5,
    minimum_snr: float = 3.0,
) -> dict[str, Any]:
    if proximal_arrival is None:
        raise ValueError("proximal arrival is required")
    valid = [
        candidate
        for candidate in candidates
        if candidate.get("arrival") is not None
        and int(candidate["arrival"]) > proximal_arrival
        and candidate.get("correlation_peak") is not None
        and float(candidate["correlation_peak"]) >= correlation_threshold
        and float(candidate.get("snr", 0.0)) >= minimum_snr
    ]
    if not valid:
        raise ValueError("no reliable distal candidate")
    return max(
        valid,
        key=lambda candidate: (
            float(candidate["correlation_peak"]),
            min(float(candidate.get("snr", 0.0)), 20.0),
            float(candidate["fraction"]),
        ),
    )


def determine_curve_polarity(
    proximal_curve: Array,
    peak_frame_index: int,
    baseline_frames: int = 5,
) -> float:
    values = np.asarray(proximal_curve, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("proximal_curve must be a non-empty 1D array")
    if not 0 <= peak_frame_index < values.size:
        raise ValueError("peak_frame_index is out of range")
    baseline_count = min(max(1, baseline_frames), values.size)
    baseline = float(np.median(values[:baseline_count]))
    peak_change = float(values[peak_frame_index] - baseline)
    if abs(peak_change) <= 1e-8:
        strongest = int(np.argmax(np.abs(values - baseline)))
        peak_change = float(values[strongest] - baseline)
    return -1.0 if peak_change < 0.0 else 1.0


def _disk_mask_local(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
) -> Array:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    return (rows - center[0]) ** 2 + (cols - center[1]) ** 2 <= radius**2


def _ring_mask_local(
    shape: tuple[int, int],
    center: tuple[int, int],
    inner_radius: int,
    outer_radius: int,
) -> Array:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    distance_sq = (rows - center[0]) ** 2 + (cols - center[1]) ** 2
    return (distance_sq >= inner_radius**2) & (distance_sq <= outer_radius**2)


def postprocess_connected_mask(
    mask: Array,
    threshold: int = 127,
    close_kernel: int = 5,
) -> tuple[Array, Array]:
    # mask shape: (H, W)
    binary = (np.asarray(mask) > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_kernel, close_kernel),
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    connected, _ = connect_vessel_components(
        closed.astype(bool),
        max_gap=25.0,
        min_component_area=100,
        bridge_radius=2,
    )
    skeleton = skeletonize(connected).astype(bool)
    if not bool(skeleton.any()):
        raise ValueError("Connected mask produced an empty skeleton")
    return connected, skeleton


def connect_vessel_components(
    mask: Array,
    max_gap: float = 25.0,
    min_component_area: int = 100,
    bridge_radius: int = 2,
) -> tuple[Array, list[dict[str, Any]]]:
    if max_gap <= 0:
        raise ValueError("max_gap must be > 0")
    if min_component_area < 1:
        raise ValueError("min_component_area must be >= 1")
    if bridge_radius < 1:
        raise ValueError("bridge_radius must be >= 1")

    connected = np.asarray(mask, dtype=bool).copy()
    bridges: list[dict[str, Any]] = []
    while True:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            connected.astype(np.uint8),
            connectivity=8,
        )
        eligible = [
            label
            for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_component_area
        ]
        if len(eligible) < 2:
            break

        best: tuple[float, int, int, tuple[int, int], tuple[int, int]] | None = None
        boundaries: dict[int, Array] = {}
        for label in eligible:
            component = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(
                component,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            points = np.concatenate([contour.reshape(-1, 2) for contour in contours], axis=0)
            # OpenCV contour coordinates: (col, row)
            boundaries[label] = points

        for first_index, first_label in enumerate(eligible):
            first = boundaries[first_label]
            for second_label in eligible[first_index + 1 :]:
                second = boundaries[second_label]
                distance_sq = (
                    (first[:, None, 0] - second[None, :, 0]) ** 2
                    + (first[:, None, 1] - second[None, :, 1]) ** 2
                )
                flat_index = int(np.argmin(distance_sq))
                first_point_index, second_point_index = np.unravel_index(
                    flat_index,
                    distance_sq.shape,
                )
                gap = float(math.sqrt(float(distance_sq[first_point_index, second_point_index])))
                first_xy = tuple(int(value) for value in first[first_point_index])
                second_xy = tuple(int(value) for value in second[second_point_index])
                candidate = (gap, first_label, second_label, first_xy, second_xy)
                if best is None or candidate[0] < best[0]:
                    best = candidate

        if best is None or best[0] > max_gap:
            break
        gap, first_label, second_label, first_xy, second_xy = best
        bridge_layer = np.zeros_like(connected, dtype=np.uint8)
        cv2.line(
            bridge_layer,
            first_xy,
            second_xy,
            color=1,
            thickness=2 * bridge_radius + 1,
            lineType=cv2.LINE_AA,
        )
        connected |= bridge_layer.astype(bool)
        bridges.append(
            {
                "component_a": int(first_label),
                "component_b": int(second_label),
                "start": {"row": int(first_xy[1]), "col": int(first_xy[0])},
                "end": {"row": int(second_xy[1]), "col": int(second_xy[0])},
                "gap": gap,
                "bridge_radius": int(bridge_radius),
            }
        )
    return connected, bridges


def extract_diameter_aware_main_trunk(
    vessel_mask: Array,
    skeleton: Array,
    diameter_weight: float = 0.8,
    continuity_weight: float = 0.2,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    coords = [tuple(int(value) for value in coord) for coord in np.argwhere(skeleton)]
    coord_set = set(coords)
    if not coords:
        raise ValueError("skeleton is empty")
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row, col in coords:
        neighbors[(row, col)] = [
            (row + dr, col + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr != 0 or dc != 0) and (row + dr, col + dc) in coord_set
        ]
    endpoints = [point for point, items in neighbors.items() if len(items) == 1]
    if len(endpoints) < 2:
        raise ValueError("diameter-aware trunk requires at least two endpoints")

    distance_map = cv2.distanceTransform(
        np.asarray(vessel_mask, dtype=np.uint8),
        cv2.DIST_L2,
        5,
    )
    endpoint_radii = {
        endpoint: _endpoint_radius(endpoint, neighbors, distance_map)
        for endpoint in endpoints
    }
    proximal = max(endpoints, key=lambda point: endpoint_radii[point])
    distances, parents = _dijkstra_skeleton(proximal, neighbors)

    candidates: list[dict[str, Any]] = []
    for endpoint in endpoints:
        if endpoint == proximal or endpoint not in distances:
            continue
        path = _reconstruct_graph_path(proximal, endpoint, parents)
        radii = np.asarray([distance_map[row, col] for row, col in path], dtype=np.float32)
        arc_length = float(distances[endpoint])
        mean_radius = float(radii.mean())
        continuity = _path_direction_continuity(path)
        candidates.append(
            {
                "endpoint": endpoint,
                "path": path,
                "arc_length": arc_length,
                "mean_radius": mean_radius,
                "continuity": continuity,
                "distal_radius": float(endpoint_radii[endpoint]),
            }
        )
    if not candidates:
        raise ValueError("no distal endpoint candidate is reachable")

    max_length = max(candidate["arc_length"] for candidate in candidates)
    max_radius = max(candidate["mean_radius"] for candidate in candidates)
    for candidate in candidates:
        candidate["score"] = (
            candidate["arc_length"] / max(max_length, 1e-8)
            + diameter_weight
            * candidate["mean_radius"]
            / max(max_radius, 1e-8)
            + continuity_weight * candidate["continuity"]
        )
    selected = max(candidates, key=lambda candidate: candidate["score"])
    diagnostics = {
        "proximal_endpoint": {"row": proximal[0], "col": proximal[1]},
        "distal_endpoint": {
            "row": selected["endpoint"][0],
            "col": selected["endpoint"][1],
        },
        "proximal_radius": float(endpoint_radii[proximal]),
        "distal_radius": float(selected["distal_radius"]),
        "mean_path_radius": float(selected["mean_radius"]),
        "arc_length": float(selected["arc_length"]),
        "direction_continuity": float(selected["continuity"]),
        "score": float(selected["score"]),
        "candidate_count": len(candidates),
    }
    return selected["path"], diagnostics


def _endpoint_radius(
    endpoint: tuple[int, int],
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    distance_map: Array,
    sample_points: int = 20,
) -> float:
    values: list[float] = []
    previous: tuple[int, int] | None = None
    current = endpoint
    for _ in range(sample_points):
        values.append(float(distance_map[current]))
        next_points = [point for point in neighbors[current] if point != previous]
        if not next_points:
            break
        if len(next_points) > 1:
            next_point = max(next_points, key=lambda point: float(distance_map[point]))
        else:
            next_point = next_points[0]
        previous, current = current, next_point
    return float(np.mean(values))


def _dijkstra_skeleton(
    start: tuple[int, int],
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
) -> tuple[
    dict[tuple[int, int], float],
    dict[tuple[int, int], tuple[int, int] | None],
]:
    distances = {start: 0.0}
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    while queue:
        distance, point = heapq.heappop(queue)
        if distance > distances[point]:
            continue
        for neighbor in neighbors[point]:
            edge = math.hypot(neighbor[0] - point[0], neighbor[1] - point[1])
            candidate = distance + edge
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                parents[neighbor] = point
                heapq.heappush(queue, (candidate, neighbor))
    return distances, parents


def _reconstruct_graph_path(
    start: tuple[int, int],
    target: tuple[int, int],
    parents: dict[tuple[int, int], tuple[int, int] | None],
) -> list[tuple[int, int]]:
    path = [target]
    current = target
    while current != start:
        parent = parents.get(current)
        if parent is None:
            raise ValueError("target is not connected to start")
        path.append(parent)
        current = parent
    path.reverse()
    return path


def _path_direction_continuity(path: list[tuple[int, int]], step: int = 8) -> float:
    if len(path) < 2 * step + 1:
        return 1.0
    similarities: list[float] = []
    for index in range(step, len(path) - step, step):
        before = np.asarray(path[index], dtype=np.float32) - np.asarray(
            path[index - step],
            dtype=np.float32,
        )
        after = np.asarray(path[index + step], dtype=np.float32) - np.asarray(
            path[index],
            dtype=np.float32,
        )
        denominator = float(np.linalg.norm(before) * np.linalg.norm(after))
        if denominator <= 1e-8:
            continue
        similarities.append(max(-1.0, min(1.0, float(np.dot(before, after) / denominator))))
    if not similarities:
        return 1.0
    return float((np.mean(similarities) + 1.0) / 2.0)


def run_sequence(
    video_id: str,
    mask_root: Path,
    frame_root: Path,
    result_root: Path,
    bridge_max_gap: float = 25.0,
    bridge_min_component_area: int = 100,
    bridge_radius: int = 2,
    trunk_diameter_weight: float = 0.8,
    trunk_continuity_weight: float = 0.2,
    baseline_frames: int = 10,
    peak_fraction_threshold: float = 0.2,
    noise_sigma_multiplier: float = 3.0,
    correlation_qc_threshold: float = 0.5,
    tracking_template_radius: int = 8,
    tracking_search_radius: int = 12,
    distal_candidate_fractions: tuple[float, ...] = (
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ),
    distal_minimum_snr: float = 3.0,
) -> dict[str, Any]:
    masks, mask_ids, mask_paths = load_mask_sequence(mask_root / video_id)
    frames, frame_ids, frame_paths = load_gray_sequence(frame_root / video_id)
    validate_mask_frame_subset(mask_ids, frame_ids)

    areas = [int((mask > 127).sum()) for mask in masks]
    peak_local_index = int(np.argmax(areas))
    peak_frame_id = mask_ids[peak_local_index]
    peak_dynamic_index = frame_ids.index(peak_frame_id)
    binary = (masks[peak_local_index] > 127).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    connected_mask, bridge_connections = connect_vessel_components(
        closed.astype(bool),
        max_gap=bridge_max_gap,
        min_component_area=bridge_min_component_area,
        bridge_radius=bridge_radius,
    )
    skeleton = skeletonize(connected_mask).astype(bool)
    centerline, trunk_diagnostics = extract_diameter_aware_main_trunk(
        connected_mask,
        skeleton,
        diameter_weight=trunk_diameter_weight,
        continuity_weight=trunk_continuity_weight,
    )
    rois = define_centerline_rois(
        centerline,
        image_shape=frames.shape[1:],
        clean_mask=connected_mask,
        roi_radius=6,
        ring_inner_radius=9,
        ring_outer_radius=16,
    )

    proximal_key, distal_key = "a", "b"
    tracked_centers = {
        key: track_roi_centers(
            frames=frames,
            reference_center=(
                int(rois[key]["center"]["row"]),
                int(rois[key]["center"]["col"]),
            ),
            reference_frame_index=peak_dynamic_index,
            template_radius=tracking_template_radius,
            search_radius=tracking_search_radius,
        )
        for key in ("a", "mid")
    }
    curve_details = {
        key: extract_tracked_roi_curve_details(
            frames,
            tracked_centers[key],
            roi_radius=6,
            ring_inner_radius=9,
            ring_outer_radius=16,
        )
        for key in ("a", "mid")
    }
    curve_polarity = determine_curve_polarity(
        curve_details[proximal_key]["corrected"],
        peak_frame_index=peak_dynamic_index,
    )
    for detail in curve_details.values():
        detail["corrected"] = (
            np.asarray(detail["corrected"], dtype=np.float32) * curve_polarity
        )
    prepared_curves = {
        key: prepare_dynamic_curve(
            curve_details[key]["corrected"],
            smoothing_window=3,
            baseline_frames=baseline_frames,
            peak_fraction_threshold=peak_fraction_threshold,
            noise_sigma_multiplier=noise_sigma_multiplier,
        )
        for key in ("a", "mid")
    }
    curves_norm = {
        key: np.asarray(prepared_curves[key]["normalized"], dtype=np.float32)
        for key in ("a", "mid")
    }
    arrivals = {
        key: detect_arrival_frame(
            curves_norm[key],
            threshold=float(prepared_curves[key]["threshold"]),
            consecutive_frames=2,
        )
        for key in ("a", "mid")
    }
    t_proximal = arrivals[proximal_key]
    max_lag = max(1, frames.shape[0] // 2)
    distal_candidates: list[dict[str, Any]] = []
    for fraction in distal_candidate_fractions:
        index = int(round(fraction * (len(centerline) - 1)))
        center = centerline[index]
        centers = track_roi_centers(
            frames=frames,
            reference_center=center,
            reference_frame_index=peak_dynamic_index,
            template_radius=tracking_template_radius,
            search_radius=tracking_search_radius,
        )
        detail = extract_tracked_roi_curve_details(
            frames,
            centers,
            roi_radius=6,
            ring_inner_radius=9,
            ring_outer_radius=16,
        )
        detail["corrected"] = (
            np.asarray(detail["corrected"], dtype=np.float32) * curve_polarity
        )
        prepared = prepare_dynamic_curve(
            detail["corrected"],
            smoothing_window=3,
            baseline_frames=baseline_frames,
            peak_fraction_threshold=peak_fraction_threshold,
            noise_sigma_multiplier=noise_sigma_multiplier,
        )
        normalized = np.asarray(prepared["normalized"], dtype=np.float32)
        arrival = detect_arrival_frame(
            normalized,
            threshold=float(prepared["threshold"]),
            consecutive_frames=2,
        )
        lag, corr = estimate_delay_by_cross_correlation(
            curves_norm[proximal_key],
            normalized,
            max_lag=max_lag,
        )
        noise_sigma = float(prepared["noise_sigma"])
        snr = float(prepared["peak_enhancement"]) / max(noise_sigma, 1e-8)
        distal_candidates.append(
            {
                "fraction": float(fraction),
                "index": index,
                "center": center,
                "arrival": arrival,
                "delta_t": lag,
                "correlation_peak": corr,
                "snr": snr,
                "centers": centers,
                "detail": detail,
                "prepared": prepared,
                "normalized": normalized,
            }
        )
    distal_selection_reliable = True
    try:
        selected_distal = select_stable_distal_candidate(
            distal_candidates,
            proximal_arrival=t_proximal,
            correlation_threshold=correlation_qc_threshold,
            minimum_snr=distal_minimum_snr,
        )
    except ValueError:
        distal_selection_reliable = False
        selected_distal = max(
            distal_candidates,
            key=lambda candidate: (
                float(candidate["correlation_peak"] or -1.0),
                float(candidate["snr"]),
            ),
        )
    selected_index = int(selected_distal["index"])
    selected_center = centerline[selected_index]
    selected_roi = _disk_mask_local(frames.shape[1:], selected_center, 6)
    selected_roi &= connected_mask
    selected_ring = _ring_mask_local(frames.shape[1:], selected_center, 9, 16)
    selected_ring &= ~connected_mask
    rois[distal_key] = {
        "center": {"row": int(selected_center[0]), "col": int(selected_center[1])},
        "index": selected_index,
        "roi": selected_roi,
        "ring": selected_ring,
    }
    tracked_centers[distal_key] = selected_distal["centers"]
    curve_details[distal_key] = selected_distal["detail"]
    prepared_curves[distal_key] = selected_distal["prepared"]
    curves_norm[distal_key] = selected_distal["normalized"]
    arrivals[distal_key] = selected_distal["arrival"]

    proximal_curve = curves_norm[proximal_key]
    distal_curve = curves_norm[distal_key]
    t_distal = arrivals[distal_key]
    delta_t = selected_distal["delta_t"]
    correlation_peak = selected_distal["correlation_peak"]
    overlap_sufficient = is_cross_correlation_overlap_sufficient(
        frame_count=frames.shape[0],
        lag=delta_t,
    )
    if not overlap_sufficient:
        delta_t = None
    delta_s = compute_arc_length_for_indices(
        centerline,
        int(rois[proximal_key]["index"]),
        int(rois[distal_key]["index"]),
    )
    pseudo_timi = compute_pseudo_timi(t_proximal, t_distal)
    velocity_candidate = compute_relative_velocity(delta_s, delta_t)
    velocity = (
        velocity_candidate
        if is_temporal_delay_reliable(
            delta_t,
            correlation_peak,
            correlation_threshold=correlation_qc_threshold,
        )
        else {"value": None, "unit": None}
    )
    qc_issues = _quality_control(
        t_proximal=t_proximal,
        t_distal=t_distal,
        delta_t=delta_t,
        correlation_peak=correlation_peak,
        frame_count=frames.shape[0],
    )
    if not distal_selection_reliable:
        qc_issues.append("no_reliable_distal_candidate")
    qc_status = "Pass" if not qc_issues else "Low confidence"

    result: dict[str, Any] = {
        "video_id": video_id,
        "analysis_scope": "full dynamic sequence with tracked ROIs and stable distal selection",
        "frame_ids": frame_ids,
        "mask_candidate_frame_ids": mask_ids,
        "selected_peak_frame": peak_frame_id,
        "selected_peak_file": mask_paths[peak_local_index].name,
        "candidate_areas": [
            {"frame": frame_id, "area": area}
            for frame_id, area in zip(mask_ids, areas)
        ],
        "curve_polarity": curve_polarity,
        "roi_tracking": {
            "enabled": True,
            "reference_frame": peak_frame_id,
            "template_radius": tracking_template_radius,
            "search_radius": tracking_search_radius,
            "selected_distal_fraction": selected_distal["fraction"],
            "distal_selection_reliable": distal_selection_reliable,
            "distal_candidates": [
                {
                    key: candidate[key]
                    for key in (
                        "fraction",
                        "index",
                        "arrival",
                        "delta_t",
                        "correlation_peak",
                        "snr",
                    )
                }
                for candidate in distal_candidates
            ],
        },
        "curve_preparation": {
            key: {
                "baseline": prepared_curves[key]["baseline"],
                "noise_sigma": prepared_curves[key]["noise_sigma"],
                "peak_enhancement": prepared_curves[key]["peak_enhancement"],
                "arrival_threshold": prepared_curves[key]["threshold"],
            }
            for key in ("a", "mid", "b")
        },
        "cross_correlation_overlap_sufficient": overlap_sufficient,
        "connected_mask_area": int(connected_mask.sum()),
        "discarded_foreground_pixels": int(
            np.logical_and(binary.astype(bool), ~connected_mask).sum()
        ),
        "bridge_added_pixels": int(
            np.logical_and(connected_mask, ~closed.astype(bool)).sum()
        ),
        "bridge_connections": bridge_connections,
        "bridge_count": len(bridge_connections),
        "centerline_points": len(centerline),
        "trunk_selection": trunk_diagnostics,
        "proximal_key": proximal_key,
        "distal_key": distal_key,
        "roi_centers": {
            "proximal": rois[proximal_key]["center"],
            "mid": rois["mid"]["center"],
            "distal": rois[distal_key]["center"],
        },
        "tracked_roi_centers": {
            key: [
                {"row": int(center[0]), "col": int(center[1])}
                for center in tracked_centers[key]
            ]
            for key in ("a", "mid", "b")
        },
        "delta_s_pixel": delta_s,
        "t_proximal_local": t_proximal,
        "t_distal_local": t_distal,
        "t_proximal_frame_id": None if t_proximal is None else frame_ids[t_proximal],
        "t_distal_frame_id": None if t_distal is None else frame_ids[t_distal],
        "pseudo_timi": pseudo_timi,
        "delta_t_frame": delta_t,
        "correlation_peak": correlation_peak,
        "relative_velocity_candidate": velocity_candidate,
        "relative_velocity": velocity,
        "qc": {"status": qc_status, "issues": qc_issues},
        "curves": {
            "proximal_norm": proximal_curve.tolist(),
            "mid_norm": curves_norm["mid"].tolist(),
            "distal_norm": distal_curve.tolist(),
        },
    }

    sequence_result_dir = result_root / video_id
    sequence_result_dir.mkdir(parents=True, exist_ok=True)
    _write_json(sequence_result_dir / "metrics.json", result)
    _write_spatial_figure(
        path=sequence_result_dir / "spatial_propagation.png",
        result=result,
        peak_frame=frames[peak_dynamic_index],
        frames=frames,
        frame_ids=frame_ids,
        connected_mask=connected_mask,
        centerline=centerline,
    )
    _write_spatial_svg(
        path=sequence_result_dir / "spatial_propagation.svg",
        result=result,
        peak_frame=frames[peak_dynamic_index],
        connected_mask=connected_mask,
        centerline=centerline,
    )
    _write_temporal_figure(sequence_result_dir / "temporal_curves.png", result)
    _write_temporal_svg(sequence_result_dir / "temporal_curves.svg", result)
    _write_combined_figure(
        path=sequence_result_dir / "combined_spatial_temporal.png",
        result=result,
        peak_frame=frames[peak_dynamic_index],
        connected_mask=connected_mask,
        centerline=centerline,
    )
    _write_combined_svg(
        path=sequence_result_dir / "combined_spatial_temporal.svg",
        result=result,
        peak_frame=frames[peak_dynamic_index],
        connected_mask=connected_mask,
        centerline=centerline,
    )
    result["source_frame_paths"] = [str(path) for path in frame_paths]
    return result


def _quality_control(
    t_proximal: int | None,
    t_distal: int | None,
    delta_t: int | None,
    correlation_peak: float | None,
    frame_count: int,
) -> list[str]:
    issues: list[str] = []
    if t_proximal is None:
        issues.append("missing_proximal_arrival")
    if t_distal is None:
        issues.append("missing_distal_arrival")
    if t_proximal is not None and t_distal is not None and t_distal < t_proximal:
        issues.append("distal_before_proximal")
    if delta_t is None or delta_t <= 0:
        issues.append("invalid_cross_correlation_delay")
    if correlation_peak is None or correlation_peak < 0.5:
        issues.append("low_correlation_peak")
    if frame_count < 8:
        issues.append("short_peak_neighborhood_sequence")
    return issues


def _write_json(path: Path, result: dict[str, Any]) -> None:
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _gray_to_rgb(frame: Array) -> Image.Image:
    values = np.clip(np.asarray(frame) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(values, mode="L").convert("RGB")


def _image_data_uri(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/png" if image_format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def _mask_svg_paths(
    mask: Array,
    x: float,
    y: float,
    width: float,
    height: float,
) -> str:
    mask_u8 = np.asarray(mask, dtype=np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scale_x = width / mask.shape[1]
    scale_y = height / mask.shape[0]
    paths: list[str] = []
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        commands = [
            f"M {x + float(points[0, 0]) * scale_x:.2f} "
            f"{y + float(points[0, 1]) * scale_y:.2f}"
        ]
        commands.extend(
            f"L {x + float(col) * scale_x:.2f} {y + float(row) * scale_y:.2f}"
            for col, row in points[1:]
        )
        commands.append("Z")
        paths.append(" ".join(commands))
    return " ".join(paths)


def _centerline_svg_path(
    centerline: list[tuple[int, int]],
    image_shape: tuple[int, int],
    x: float,
    y: float,
    width: float,
    height: float,
) -> str:
    scale_x = width / image_shape[1]
    scale_y = height / image_shape[0]
    commands: list[str] = []
    for index, (row, col) in enumerate(centerline):
        command = "M" if index == 0 else "L"
        commands.append(
            f"{command} {x + col * scale_x:.2f} {y + row * scale_y:.2f}"
        )
    return " ".join(commands)


def _spatial_svg_group(
    result: dict[str, Any],
    peak_frame: Array,
    connected_mask: Array,
    centerline: list[tuple[int, int]],
    x: float,
    y: float,
    width: float,
    height: float,
) -> str:
    image_uri = _image_data_uri(_gray_to_rgb(peak_frame))
    mask_path = _mask_svg_paths(connected_mask, x, y, width, height)
    centerline_path = _centerline_svg_path(
        centerline,
        peak_frame.shape,
        x,
        y,
        width,
        height,
    )
    scale_x = width / peak_frame.shape[1]
    scale_y = height / peak_frame.shape[0]
    parts = [
        f'<image x="{x}" y="{y}" width="{width}" height="{height}" '
        f'xlink:href="{image_uri}"/>',
        f'<path d="{mask_path}" fill="{NPG_COLORS["blue"]}" fill-opacity="0.32" '
        f'stroke="{NPG_COLORS["blue"]}" stroke-width="2"/>',
        f'<path d="{centerline_path}" fill="none" stroke="{NPG_COLORS["orange"]}" '
        f'stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>',
    ]
    colors = {
        "proximal": NPG_COLORS["red"],
        "mid": NPG_COLORS["purple"],
        "distal": NPG_COLORS["green"],
    }
    for name in ("proximal", "mid", "distal"):
        center = result["roi_centers"][name]
        cx = x + float(center["col"]) * scale_x
        cy = y + float(center["row"]) * scale_y
        color = colors[name]
        parts.extend(
            [
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="17" fill="white" '
                f'fill-opacity="0.25" stroke="{color}" stroke-width="7"/>',
                f'<text x="{cx + 24:.2f}" y="{cy + 8:.2f}" class="label" fill="{color}">'
                f"{name}</text>",
            ]
        )
    mid = result["roi_centers"]["mid"]
    text_x = x + float(mid["col"]) * scale_x + 25
    text_y = y + float(mid["row"]) * scale_y + 58
    parts.append(
        f'<text x="{text_x:.2f}" y="{text_y:.2f}" class="label" '
        f'fill="{NPG_COLORS["orange"]}">Delta s = {result["delta_s_pixel"]:.1f} px</text>'
    )
    return "\n".join(parts)


def _temporal_svg_group(
    result: dict[str, Any],
    x: float,
    y: float,
    width: float,
    height: float,
    include_metrics: bool,
) -> str:
    left = x + 90
    top = y + 110
    metric_width = 440 if include_metrics else 0
    plot_width = width - 150 - metric_width
    plot_height = height - 250
    frame_count = len(result["frame_ids"])
    parts = [
        f'<text x="{left}" y="{y + 50}" class="title" font-size="34">'
        "Proximal / distal time-intensity curves</text>",
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        'fill="white" stroke="#232A34" stroke-width="3"/>',
        f'<text x="{left}" y="{top + plot_height + 75}" class="subtitle">local frame index</text>',
        f'<text x="{left - 55}" y="{top + 10}" class="small">1.0</text>',
        f'<text x="{left - 55}" y="{top + plot_height}" class="small">0.0</text>',
    ]
    for key, color, label in (
        ("proximal_norm", NPG_COLORS["red"], "proximal"),
        ("distal_norm", NPG_COLORS["green"], "distal"),
    ):
        commands: list[str] = []
        for index, value in enumerate(result["curves"][key]):
            px = left + plot_width * index / max(frame_count - 1, 1)
            py = top + plot_height * (1.0 - float(value))
            commands.append(f"{'M' if index == 0 else 'L'} {px:.2f} {py:.2f}")
        parts.append(
            f'<path d="{" ".join(commands)}" fill="none" stroke="{color}" '
            'stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        label_y = top + (45 if label == "proximal" else 85)
        parts.append(
            f'<text x="{left + plot_width - 160}" y="{label_y}" '
            f'class="label" fill="{color}">{label}</text>'
        )
    arrival_positions: dict[str, float] = {}
    for key, color, label in (
        ("t_proximal_local", NPG_COLORS["red"], "t_proximal"),
        ("t_distal_local", NPG_COLORS["green"], "t_distal"),
    ):
        value = result[key]
        if value is None:
            continue
        px = left + plot_width * int(value) / max(frame_count - 1, 1)
        arrival_positions[label] = px
        parts.extend(
            [
                f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" '
                f'y2="{top + plot_height}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{px + 8:.2f}" y="{top + plot_height + (35 if label == "t_proximal" else 68)}" '
                f'font-size="21" fill="{color}">{label}={value}</text>',
            ]
        )
    if "t_proximal" in arrival_positions and "t_distal" in arrival_positions:
        x1 = arrival_positions["t_proximal"]
        x2 = arrival_positions["t_distal"]
        arrow_y = top + 68
        parts.extend(
            [
                f'<line x1="{x1}" y1="{arrow_y}" x2="{x2}" y2="{arrow_y}" '
                f'stroke="{NPG_COLORS["purple"]}" stroke-width="5" '
                'marker-start="url(#arrowStart)" marker-end="url(#arrowEnd)"/>',
                f'<text x="{(x1 + x2) / 2 - 110:.2f}" y="{arrow_y - 22}" '
                f'class="label" fill="{NPG_COLORS["purple"]}">'
                f'arrival Delta t = {result["pseudo_timi"]["frames"]} frames</text>',
            ]
        )
    if include_metrics:
        metric_x = left + plot_width + 45
        metric_y = top + 125
        velocity = result["relative_velocity"]
        velocity_text = (
            "N/A"
            if velocity["value"] is None
            else f'{velocity["value"]:.2f} {velocity["unit"]}'
        )
        parts.extend(
            [
                f'<rect x="{metric_x}" y="{metric_y}" width="390" height="390" rx="24" '
                'fill="#F5F7FA" stroke="#8491B4" stroke-width="3"/>',
                f'<text x="{metric_x + 28}" y="{metric_y + 65}" class="metric">'
                f'CTFC-like = {result["pseudo_timi"]["frames"]} frames</text>',
                f'<text x="{metric_x + 28}" y="{metric_y + 120}" class="metric">'
                f"v_rel = {velocity_text}</text>",
                f'<text x="{metric_x + 28}" y="{metric_y + 175}" class="metric">'
                f'QC = {result["qc"]["status"]}</text>',
                f'<text x="{metric_x + 28}" y="{metric_y + 250}" class="small">'
                f'xcorr lag = {result["delta_t_frame"]} frames</text>',
                f'<text x="{metric_x + 28}" y="{metric_y + 292}" class="small">'
                f'corr peak = {result["correlation_peak"]:.3f}</text>',
                f'<text x="{metric_x + 28}" y="{metric_y + 334}" class="small">'
                "scope: full dynamic sequence</text>",
            ]
        )
    return "\n".join(parts)


def _overlay_spatial(
    frame: Array,
    connected_mask: Array,
    centerline: list[tuple[int, int]],
    result: dict[str, Any],
    size: tuple[int, int] = (640, 640),
) -> Image.Image:
    base = _gray_to_rgb(frame)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_array = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    overlay_array[connected_mask] = (30, 170, 235, 80)
    overlay = Image.fromarray(overlay_array, mode="RGBA")
    image = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    path_points = [(col, row) for row, col in centerline]
    if len(path_points) >= 2:
        draw.line(path_points, fill=YELLOW, width=3)
    centers = result["roi_centers"]
    for name, color in (("proximal", RED), ("mid", BLUE), ("distal", GREEN)):
        center = centers[name]
        x, y = int(center["col"]), int(center["row"])
        radius = 10
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=4)
        draw.text((x + 13, y - 12), name, fill=color, font=_font(18, bold=True))
    midpoint = path_points[len(path_points) // 2]
    draw.text(
        (midpoint[0] + 10, midpoint[1] + 10),
        f"Delta s = {result['delta_s_pixel']:.1f} px",
        fill=YELLOW,
        font=_font(20, bold=True),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


def _write_spatial_figure(
    path: Path,
    result: dict[str, Any],
    peak_frame: Array,
    frames: Array,
    frame_ids: list[int],
    connected_mask: Array,
    centerline: list[tuple[int, int]],
) -> None:
    canvas = Image.new("RGB", (1500, 950), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 28), f"{result['video_id']} | Spatial propagation", fill=DARK, font=_font(34, True))
    draw.text(
        (50, 75),
        "Peak frame + connected vessel mask + main centerline + proximal/distal ROIs",
        fill=GRAY,
        font=_font(20),
    )
    spatial = _overlay_spatial(peak_frame, connected_mask, centerline, result, (760, 760))
    canvas.paste(spatial, (50, 135))

    key_indices = choose_key_frame_indices(
        frame_count=len(frames),
        t_proximal=result["t_proximal_local"],
        t_distal=result["t_distal_local"],
    )
    labels = ["proximal onset", "middle propagation", "distal arrival"]
    for position, index in enumerate(key_indices):
        thumb = _gray_to_rgb(frames[index]).resize((300, 300), Image.Resampling.LANCZOS)
        x = 900
        y = 135 + position * 250
        thumb.thumbnail((300, 210), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x, y))
        label = labels[min(position, len(labels) - 1)]
        draw.text(
            (1220, y + 20),
            f"{label}\nframe {frame_ids[index]:05d}",
            fill=DARK,
            font=_font(20, True),
        )
    canvas.resize((2400, 1520), Image.Resampling.LANCZOS).save(path)


def _svg_defs() -> str:
    return (
        "<defs>"
        '<marker id="arrowStart" markerWidth="12" markerHeight="12" refX="2" refY="6" orient="auto">'
        f'<path d="M 12 0 L 0 6 L 12 12 Z" fill="{NPG_COLORS["purple"]}"/>'
        "</marker>"
        '<marker id="arrowEnd" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">'
        f'<path d="M 0 0 L 12 6 L 0 12 Z" fill="{NPG_COLORS["purple"]}"/>'
        "</marker>"
        "</defs>"
    )


def _write_spatial_svg(
    path: Path,
    result: dict[str, Any],
    peak_frame: Array,
    connected_mask: Array,
    centerline: list[tuple[int, int]],
) -> None:
    width, height = 2400, 1500
    body = [
        _svg_defs(),
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="70" y="75" class="title">{result["video_id"]} | Spatial propagation</text>',
        '<text x="70" y="120" class="subtitle">'
        "Peak frame + connected vessel mask + main centerline + ROIs</text>",
        _spatial_svg_group(
            result,
            peak_frame,
            connected_mask,
            centerline,
            x=150,
            y=180,
            width=1120,
            height=1120,
        ),
    ]
    path.write_text(
        svg_document(width, height, f"{result['video_id']} spatial propagation", "\n".join(body)),
        encoding="utf-8",
    )


def _plot_curves(
    result: dict[str, Any],
    size: tuple[int, int],
    include_metrics: bool,
) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    left, top = 85, 90
    right_margin = 340 if include_metrics else 120
    plot_width = width - left - right_margin
    plot_height = height - top - 130
    draw.rectangle((left, top, left + plot_width, top + plot_height), outline=DARK, width=2)
    draw.text((left, 28), "Proximal / distal time-intensity curves", fill=DARK, font=_font(28, True))
    draw.text((left, top + plot_height + 45), "local frame index", fill=GRAY, font=_font(18))
    draw.text((15, top - 5), "1.0", fill=GRAY, font=_font(16))
    draw.text((15, top + plot_height - 12), "0.0", fill=GRAY, font=_font(16))

    curves = [
        ("proximal_norm", RED, "proximal"),
        ("distal_norm", GREEN, "distal"),
    ]
    frame_count = len(result["frame_ids"])
    for key, color, label in curves:
        values = result["curves"][key]
        points = []
        for index, value in enumerate(values):
            x = left + int(round(plot_width * index / max(frame_count - 1, 1)))
            y = top + int(round(plot_height * (1.0 - float(value))))
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=5)
        draw.text(
            (left + plot_width - 110, top + (15 if label == "proximal" else 48)),
            label,
            fill=color,
            font=_font(18, True),
        )

    for key, color, label in (
        ("t_proximal_local", RED, "t_proximal"),
        ("t_distal_local", GREEN, "t_distal"),
    ):
        value = result[key]
        if value is None:
            continue
        x = left + int(round(plot_width * int(value) / max(frame_count - 1, 1)))
        draw.line((x, top, x, top + plot_height), fill=color, width=2)
        label_y = top + plot_height + (10 if label == "t_proximal" else 35)
        draw.text((x + 5, label_y), f"{label}={value}", fill=color, font=_font(16))

    t_proximal = result["t_proximal_local"]
    t_distal = result["t_distal_local"]
    if t_proximal is not None and t_distal is not None:
        x1 = left + int(round(plot_width * t_proximal / max(frame_count - 1, 1)))
        x2 = left + int(round(plot_width * t_distal / max(frame_count - 1, 1)))
        y = top + 45
        if x1 != x2:
            draw.line((x1, y, x2, y), fill=PURPLE, width=3)
            draw.polygon([(x1, y), (x1 + 10, y - 6), (x1 + 10, y + 6)], fill=PURPLE)
            draw.polygon([(x2, y), (x2 - 10, y - 6), (x2 - 10, y + 6)], fill=PURPLE)
        draw.text(
            (max(left + 10, (x1 + x2) // 2 - 75), y - 35),
            f"arrival Delta t = {t_distal - t_proximal} frames",
            fill=PURPLE,
            font=_font(18, True),
        )

    if include_metrics:
        metric_x = left + plot_width + 45
        metric_y = top + 80
        draw.rounded_rectangle(
            (metric_x, metric_y, width - 35, metric_y + 260),
            radius=18,
            fill=(243, 247, 252),
            outline=(190, 200, 215),
            width=2,
        )
        metric_text = summarize_metric_box(
            pseudo_timi_frames=result["pseudo_timi"]["frames"],
            relative_velocity=result["relative_velocity"]["value"],
            velocity_unit=result["relative_velocity"]["unit"],
            qc_status=result["qc"]["status"],
        )
        draw.multiline_text(
            (metric_x + 25, metric_y + 28),
            metric_text,
            fill=DARK,
            font=_font(22, True),
            spacing=18,
        )
        draw.multiline_text(
            (metric_x + 25, metric_y + 170),
            f"xcorr lag = {result['delta_t_frame']} frames\n"
            f"corr peak = {result['correlation_peak']:.3f}\n"
            f"scope: full dynamic sequence",
            fill=GRAY,
            font=_font(17),
            spacing=10,
        )
    return image


def _write_temporal_figure(path: Path, result: dict[str, Any]) -> None:
    image = _plot_curves(result, (1400, 760), include_metrics=True)
    image.resize((2400, 1300), Image.Resampling.LANCZOS).save(path)


def _write_temporal_svg(path: Path, result: dict[str, Any]) -> None:
    width, height = 2400, 1300
    body = [
        _svg_defs(),
        f'<rect width="{width}" height="{height}" fill="white"/>',
        _temporal_svg_group(result, x=40, y=35, width=2320, height=1190, include_metrics=True),
    ]
    path.write_text(
        svg_document(width, height, f"{result['video_id']} temporal curves", "\n".join(body)),
        encoding="utf-8",
    )


def _write_combined_figure(
    path: Path,
    result: dict[str, Any],
    peak_frame: Array,
    connected_mask: Array,
    centerline: list[tuple[int, int]],
) -> None:
    canvas = Image.new("RGB", (1800, 920), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (45, 25),
        f"{result['video_id']} | Spatial-temporal hemodynamic analysis",
        fill=DARK,
        font=_font(34, True),
    )
    spatial = _overlay_spatial(
        peak_frame,
        connected_mask,
        centerline,
        result,
        size=(780, 780),
    )
    temporal = _plot_curves(result, (930, 780), include_metrics=True)
    canvas.paste(spatial, (40, 105))
    canvas.paste(temporal, (830, 105))
    canvas.resize((2880, 1472), Image.Resampling.LANCZOS).save(path)


def _write_combined_svg(
    path: Path,
    result: dict[str, Any],
    peak_frame: Array,
    connected_mask: Array,
    centerline: list[tuple[int, int]],
) -> None:
    width, height = 3000, 1600
    body = [
        _svg_defs(),
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="70" y="75" class="title">{result["video_id"]} | Spatial-temporal hemodynamic analysis</text>',
        _spatial_svg_group(
            result,
            peak_frame,
            connected_mask,
            centerline,
            x=80,
            y=170,
            width=1280,
            height=1280,
        ),
        _temporal_svg_group(
            result,
            x=1410,
            y=145,
            width=1530,
            height=1300,
            include_metrics=True,
        ),
    ]
    path.write_text(
        svg_document(width, height, f"{result['video_id']} combined analysis", "\n".join(body)),
        encoding="utf-8",
    )


def _write_workflow_figure(
    path: Path,
    example_result: dict[str, Any],
    frame_root: Path,
    mask_root: Path,
    result_root: Path,
) -> None:
    video_id = str(example_result["video_id"])
    frame_paths = sorted((frame_root / video_id).glob("*.jpg"))
    mask_path = mask_root / video_id / str(example_result["selected_peak_file"])
    spatial_path = result_root / video_id / "combined_spatial_temporal.png"

    canvas = Image.new("RGB", (2000, 1120), (247, 249, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text((60, 35), "XCA Hemodynamic Analysis Workflow", fill=DARK, font=_font(42, True))
    draw.text(
        (60, 95),
        "From full dynamic sequence and peak-mask vessel segmentation to CTFC-like and relative velocity",
        fill=GRAY,
        font=_font(22),
    )

    box_y, box_h, box_w, gap = 190, 760, 340, 45
    titles = [
        "Step 1\nXCA sequence",
        "Step 2\nSegmentation +\nconnectivity cleanup",
        "Step 3\nMain centerline",
        "Step 4\nProximal / distal\nROIs and Delta s",
        "Step 5\nDelay Delta t and\nrelative velocity",
    ]
    for index, title in enumerate(titles):
        x = 45 + index * (box_w + gap)
        draw.rounded_rectangle(
            (x, box_y, x + box_w, box_y + box_h),
            radius=22,
            fill="white",
            outline=(205, 213, 224),
            width=3,
        )
        draw.multiline_text(
            (x + 24, box_y + 24),
            title,
            fill=DARK,
            font=_font(23, True),
            spacing=8,
        )
        if index < 4:
            arrow_x = x + box_w + 8
            arrow_y = box_y + box_h // 2
            draw.line((arrow_x, arrow_y, arrow_x + 28, arrow_y), fill=PURPLE, width=6)
            draw.polygon(
                [(arrow_x + 28, arrow_y), (arrow_x + 16, arrow_y - 10), (arrow_x + 16, arrow_y + 10)],
                fill=PURPLE,
            )

    first_x = 45
    thumb_y = box_y + 205
    selected = [0, len(frame_paths) // 2, len(frame_paths) - 1]
    for i, frame_index in enumerate(selected):
        thumb = Image.open(frame_paths[frame_index]).convert("RGB")
        thumb.thumbnail((245, 155), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (first_x + 48, thumb_y + i * 165))

    second_x = first_x + box_w + gap
    mask = Image.open(mask_path).convert("L").convert("RGB")
    mask.thumbnail((290, 290), Image.Resampling.LANCZOS)
    canvas.paste(mask, (second_x + 25, box_y + 250))
    draw.text(
        (second_x + 35, box_y + 565),
        "Disconnected vessel components\nbridged before trunk selection",
        fill=GRAY,
        font=_font(19),
        spacing=8,
    )

    combined = Image.open(spatial_path).convert("RGB")
    scale_x = combined.width / 1800.0
    scale_y = combined.height / 920.0
    spatial_crop = combined.crop(
        (
            int(40 * scale_x),
            int(105 * scale_y),
            int(820 * scale_x),
            int(885 * scale_y),
        )
    ).resize((300, 300), Image.Resampling.LANCZOS)
    third_x = second_x + box_w + gap
    fourth_x = third_x + box_w + gap
    canvas.paste(spatial_crop, (third_x + 20, box_y + 250))
    canvas.paste(spatial_crop, (fourth_x + 20, box_y + 250))
    draw.text(
        (third_x + 32, box_y + 570),
        "Orange: main centerline",
        fill=GRAY,
        font=_font(19),
    )
    draw.multiline_text(
        (fourth_x + 32, box_y + 570),
        f"Red: proximal ROI\nGreen: distal ROI\nDelta s = {example_result['delta_s_pixel']:.1f} px",
        fill=GRAY,
        font=_font(19),
        spacing=8,
    )

    fifth_x = fourth_x + box_w + gap
    curve_crop = combined.crop(
        (
            int(830 * scale_x),
            int(105 * scale_y),
            int(1760 * scale_x),
            int(885 * scale_y),
        )
    ).resize((300, 250), Image.Resampling.LANCZOS)
    canvas.paste(curve_crop, (fifth_x + 20, box_y + 245))
    metric_text = summarize_metric_box(
        example_result["pseudo_timi"]["frames"],
        example_result["relative_velocity"]["value"],
        example_result["relative_velocity"]["unit"],
        example_result["qc"]["status"],
    )
    draw.multiline_text(
        (fifth_x + 35, box_y + 535),
        metric_text,
        fill=DARK,
        font=_font(20, True),
        spacing=12,
    )
    draw.text(
        (60, 1015),
        "Note: peak-neighborhood masks define vessel geometry; temporal metrics use the full dynamic sequence.",
        fill=(155, 75, 45),
        font=_font(22, True),
    )
    canvas.resize((3000, 1680), Image.Resampling.LANCZOS).save(path)


def _write_workflow_svg(
    path: Path,
    example_result: dict[str, Any],
    frame_root: Path,
    mask_root: Path,
) -> None:
    width, height = 3200, 1800
    video_id = str(example_result["video_id"])
    frame_paths = sorted((frame_root / video_id).glob("*.jpg"))
    mask_path = mask_root / video_id / str(example_result["selected_peak_file"])
    panels = [
        ("Step 1", "XCA dynamic sequence"),
        ("Step 2", "Segmentation + connectivity cleanup"),
        ("Step 3", "Main centerline extraction"),
        ("Step 4", "Proximal / distal ROIs and Delta s"),
        ("Step 5", "Delay Delta t and relative velocity"),
    ]
    panel_width, panel_height, gap = 570, 1270, 55
    start_x, panel_y = 55, 250
    parts = [
        _svg_defs(),
        f'<rect width="{width}" height="{height}" fill="#F7F9FC"/>',
        '<text x="80" y="90" class="title">XCA Hemodynamic Analysis Workflow</text>',
        '<text x="80" y="140" class="subtitle">'
        "Nature/NPG palette | editable vector overlays and curves</text>",
    ]
    for index, (step, title) in enumerate(panels):
        x = start_x + index * (panel_width + gap)
        parts.extend(
            [
                f'<rect x="{x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" '
                'rx="28" fill="white" stroke="#8491B4" stroke-width="3"/>',
                f'<text x="{x + 35}" y="{panel_y + 65}" class="label" '
                f'fill="{NPG_COLORS["purple"]}">{step}</text>',
                f'<text x="{x + 35}" y="{panel_y + 115}" font-size="28" font-weight="700" '
                f'fill="#232A34">{html.escape(title)}</text>',
            ]
        )
        if index < len(panels) - 1:
            arrow_y = panel_y + panel_height / 2
            parts.append(
                f'<line x1="{x + panel_width + 10}" y1="{arrow_y}" '
                f'x2="{x + panel_width + gap - 12}" y2="{arrow_y}" '
                f'stroke="{NPG_COLORS["purple"]}" stroke-width="7" marker-end="url(#arrowEnd)"/>'
            )

    selected = [0, len(frame_paths) // 2, len(frame_paths) - 1]
    for position, frame_index in enumerate(selected):
        uri = _image_data_uri(Image.open(frame_paths[frame_index]).convert("RGB"))
        parts.append(
            f'<image x="{start_x + 95}" y="{panel_y + 220 + position * 300}" '
            f'width="380" height="260" preserveAspectRatio="xMidYMid meet" xlink:href="{uri}"/>'
        )

    second_x = start_x + panel_width + gap
    mask_uri = _image_data_uri(Image.open(mask_path).convert("L").convert("RGB"))
    parts.extend(
        [
            f'<image x="{second_x + 55}" y="{panel_y + 310}" width="460" height="460" '
            f'xlink:href="{mask_uri}"/>',
            f'<text x="{second_x + 55}" y="{panel_y + 830}" class="small">'
            "Disconnected vessel components bridged</text>",
        ]
    )

    # Reconstruct the peak-frame geometry for editable centerline and ROI panels.
    masks, mask_ids, _ = load_mask_sequence(mask_root / video_id)
    frames, frame_ids, _ = load_gray_sequence(frame_root / video_id)
    peak_index = frame_ids.index(int(example_result["selected_peak_frame"]))
    connected_mask, skeleton = postprocess_connected_mask(masks[mask_ids.index(frame_ids[peak_index])])
    centerline, _ = extract_diameter_aware_main_trunk(connected_mask, skeleton)
    for panel_index in (2, 3):
        panel_x = start_x + panel_index * (panel_width + gap)
        parts.append(
            _spatial_svg_group(
                example_result,
                frames[peak_index],
                connected_mask,
                centerline,
                x=panel_x + 45,
                y=panel_y + 260,
                width=480,
                height=480,
            )
        )
    fourth_x = start_x + 3 * (panel_width + gap)
    parts.append(
        f'<text x="{fourth_x + 55}" y="{panel_y + 825}" class="small">'
        f'Delta s = {example_result["delta_s_pixel"]:.1f} pixel</text>'
    )

    fifth_x = start_x + 4 * (panel_width + gap)
    parts.append(
        _temporal_svg_group(
            example_result,
            x=fifth_x + 5,
            y=panel_y + 215,
            width=550,
            height=720,
            include_metrics=False,
        )
    )
    velocity = example_result["relative_velocity"]
    velocity_text = (
        "N/A"
        if velocity["value"] is None
        else f'{velocity["value"]:.2f} {velocity["unit"]}'
    )
    parts.extend(
        [
            f'<text x="{fifth_x + 50}" y="{panel_y + 1020}" class="metric">'
            f'CTFC-like = {example_result["pseudo_timi"]["frames"]} frames</text>',
            f'<text x="{fifth_x + 50}" y="{panel_y + 1080}" class="metric">'
            f"v_rel = {velocity_text}</text>",
            f'<text x="{fifth_x + 50}" y="{panel_y + 1140}" class="metric">'
            f'QC = {example_result["qc"]["status"]}</text>',
            f'<text x="80" y="1680" font-size="25" font-weight="700" fill="{NPG_COLORS["red"]}">'
            "Note: peak-neighborhood masks define geometry; temporal metrics use the full dynamic sequence.</text>",
        ]
    )
    path.write_text(
        svg_document(width, height, "XCA hemodynamic analysis workflow", "\n".join(parts)),
        encoding="utf-8",
    )


def _write_summary_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "video_id",
        "selected_peak_frame",
        "connected_mask_area",
        "delta_s_pixel",
        "t_proximal_frame_id",
        "t_distal_frame_id",
        "ctfc_like_frame",
        "cross_correlation_delta_t_frame",
        "correlation_peak",
        "v_rel",
        "v_rel_unit",
        "qc_status",
        "qc_issues",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "video_id": result["video_id"],
                    "selected_peak_frame": result["selected_peak_frame"],
                    "connected_mask_area": result["connected_mask_area"],
                    "delta_s_pixel": result["delta_s_pixel"],
                    "t_proximal_frame_id": result["t_proximal_frame_id"],
                    "t_distal_frame_id": result["t_distal_frame_id"],
                    "ctfc_like_frame": result["pseudo_timi"]["frames"],
                    "cross_correlation_delta_t_frame": result["delta_t_frame"],
                    "correlation_peak": result["correlation_peak"],
                    "v_rel": result["relative_velocity"]["value"],
                    "v_rel_unit": result["relative_velocity"]["unit"],
                    "qc_status": result["qc"]["status"],
                    "qc_issues": ";".join(result["qc"]["issues"]),
                }
            )


# ===== Main AT/TTP/slope pipeline =====
Array = np.ndarray
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

ROI_KEYS = ("proximal", "mid", "distal")
ROI_COLORS = {
    "proximal": "#D9A7B8",
    "mid": "#7FA7C9",
    "distal": "#79B8B1",
}

ROI_LABELS = {
    "proximal": "Proximal ROI",
    "mid": "Mid ROI",
    "distal": "Distal ROI",
}

ROI_MARKERS = {"proximal": "o", "mid": "s", "distal": "^"}
ROI_LINESTYLES = {"proximal": "-", "mid": "--", "distal": "-."}
WASH_IN_COLOR = "#CFE8BF"

ROI_LABELS_CN = {
    "proximal": "近端",
    "mid": "中段",
    "distal": "远端",
}


@dataclass(frozen=True)
class AnalysisConfig:
    mask_threshold: int = 127
    close_kernel: int = 5
    bridge_max_gap: float = 25.0
    bridge_min_component_area: int = 100
    bridge_radius: int = 2
    trunk_diameter_weight: float = 0.8
    trunk_continuity_weight: float = 0.2
    roi_fracs: tuple[float, float, float] = (0.10, 0.50, 0.70)
    roi_radius: int = 6
    ring_inner_radius: int = 9
    ring_outer_radius: int = 16
    tracking_template_radius: int = 8
    tracking_search_radius: int = 12
    smoothing_window: int = 3
    baseline_frames: int = 10
    arrival_threshold: float = 0.20
    noise_sigma_multiplier: float = 3.0
    consecutive_frames: int = 2
    min_snr: float = 3.0
    min_peak_norm_at_ttp: float = 0.50
    min_smoothness: float = 0.25
    dpi: int = 180

    def validate(self) -> None:
        if not (0 <= self.mask_threshold <= 255):
            raise ValueError("mask_threshold must be within [0, 255]")
        if self.close_kernel < 1 or self.close_kernel % 2 == 0:
            raise ValueError("close_kernel must be a positive odd integer")
        if self.bridge_max_gap <= 0:
            raise ValueError("bridge_max_gap must be positive")
        if self.bridge_min_component_area < 1:
            raise ValueError("bridge_min_component_area must be >= 1")
        if self.bridge_radius < 1:
            raise ValueError("bridge_radius must be >= 1")
        if len(self.roi_fracs) != 3 or not (
            0.0 <= self.roi_fracs[0] < self.roi_fracs[1] < self.roi_fracs[2] <= 1.0
        ):
            raise ValueError("roi_fracs must be three increasing values in [0,1]")
        if self.roi_radius < 1:
            raise ValueError("roi_radius must be >= 1")
        if not (0 < self.ring_inner_radius < self.ring_outer_radius):
            raise ValueError("ring radii must satisfy 0 < inner < outer")
        if self.tracking_template_radius < 1 or self.tracking_search_radius < 0:
            raise ValueError("tracking radii are invalid")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")
        if not (0.0 <= self.arrival_threshold <= 1.0):
            raise ValueError("arrival_threshold must be within [0,1]")
        if self.baseline_frames < 2:
            raise ValueError("baseline_frames must be >= 2")
        if self.consecutive_frames < 1:
            raise ValueError("consecutive_frames must be >= 1")
        if self.min_snr <= 0:
            raise ValueError("min_snr must be positive")


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def numeric_key(path: Path | str) -> tuple[int, str]:
    name = Path(path).stem
    digits = "".join(ch for ch in name if ch.isdigit())
    return (int(digits) if digits else 0, str(path))


def list_image_paths(folder: Path, exts: tuple[str, ...]) -> list[Path]:
    if not folder.is_dir():
        return []
    paths = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    return sorted(paths, key=numeric_key)


def load_gray_sequence_any(sequence_dir: Path) -> tuple[Array, list[int], list[Path]]:
    paths = list_image_paths(sequence_dir, IMG_EXTS)
    if not paths:
        raise FileNotFoundError(f"No image frames found in {sequence_dir}")
    frames: list[Array] = []
    frame_ids: list[int] = []
    for path in paths:
        img = Image.open(path).convert("L")
        frames.append(np.asarray(img, dtype=np.float32) / 255.0)
        frame_ids.append(numeric_key(path)[0])
    return np.stack(frames, axis=0), frame_ids, paths


def load_mask_sequence_any(mask_dir: Path) -> tuple[list[Array], list[int], list[Path]]:
    paths = list_image_paths(mask_dir, MASK_EXTS)
    if not paths:
        raise FileNotFoundError(f"No mask images found in {mask_dir}")
    masks: list[Array] = []
    mask_ids: list[int] = []
    for path in paths:
        img = Image.open(path).convert("L")
        masks.append(np.asarray(img, dtype=np.uint8))
        mask_ids.append(numeric_key(path)[0])
    return masks, mask_ids, paths


# -----------------------------------------------------------------------------
# Geometry and ROI definitions
# -----------------------------------------------------------------------------
def disk_mask(shape: tuple[int, int], center: tuple[int, int], radius: int) -> Array:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    return ((rows - center[0]) ** 2 + (cols - center[1]) ** 2) <= radius**2


def ring_mask(
    shape: tuple[int, int],
    center: tuple[int, int],
    inner_radius: int,
    outer_radius: int,
) -> Array:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    d2 = (rows - center[0]) ** 2 + (cols - center[1]) ** 2
    return (d2 >= inner_radius**2) & (d2 <= outer_radius**2)


def define_rois_by_fractions(
    centerline: list[tuple[int, int]],
    image_shape: tuple[int, int],
    vessel_mask: Array,
    config: AnalysisConfig,
) -> dict[str, dict[str, Any]]:
    if len(centerline) < 3:
        raise ValueError("centerline is too short")
    rois: dict[str, dict[str, Any]] = {}
    for key, frac in zip(ROI_KEYS, config.roi_fracs):
        index = int(round(frac * (len(centerline) - 1)))
        index = max(0, min(len(centerline) - 1, index))
        center = centerline[index]

        roi = disk_mask(image_shape, center, config.roi_radius)
        vessel_roi = roi & vessel_mask
        # For the spatial overlay and reference ROI, restrict to vessel if enough pixels exist.
        if int(vessel_roi.sum()) >= 5:
            roi_for_overlay = vessel_roi
        else:
            roi_for_overlay = roi

        ring = ring_mask(
            image_shape,
            center,
            config.ring_inner_radius,
            config.ring_outer_radius,
        )
        ring_without_vessel = ring & (~vessel_mask)
        if int(ring_without_vessel.sum()) >= 5:
            ring_for_overlay = ring_without_vessel
        else:
            ring_for_overlay = ring

        rois[key] = {
            "center": center,
            "index": index,
            "fraction": float(frac),
            "roi": roi_for_overlay,
            "ring": ring_for_overlay,
        }
    return rois


def select_peak_mask(
    masks: list[Array],
    mask_ids: list[int],
    mask_paths: list[Path],
    threshold: int,
    manual_peak_frame: int | None = None,
) -> tuple[int, int, Path, list[int]]:
    areas = [int((mask > threshold).sum()) for mask in masks]
    if manual_peak_frame is not None and manual_peak_frame in mask_ids:
        idx = mask_ids.index(manual_peak_frame)
    else:
        idx = int(np.argmax(areas))
    return idx, mask_ids[idx], mask_paths[idx], areas


def build_connected_oriented_centerline(
    peak_mask: Array,
    config: AnalysisConfig,
) -> tuple[Array, Array, list[tuple[int, int]], dict[str, Any], list[dict[str, Any]], Array]:
    binary = (np.asarray(peak_mask) > config.mask_threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.close_kernel, config.close_kernel),
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    connected_mask, bridge_connections = connect_vessel_components(
        closed.astype(bool),
        max_gap=config.bridge_max_gap,
        min_component_area=config.bridge_min_component_area,
        bridge_radius=config.bridge_radius,
    )
    skeleton = skeletonize(connected_mask).astype(bool)
    if not bool(skeleton.any()):
        raise ValueError("empty skeleton after mask postprocessing")
    centerline, trunk_diag = extract_diameter_aware_main_trunk(
        connected_mask,
        skeleton,
        diameter_weight=config.trunk_diameter_weight,
        continuity_weight=config.trunk_continuity_weight,
    )
    return connected_mask, skeleton, centerline, trunk_diag, bridge_connections, closed.astype(bool)


# -----------------------------------------------------------------------------
# Curve metrics
# -----------------------------------------------------------------------------
def first_consecutive_above(values: Array, threshold: float, n: int) -> int | None:
    arr = np.asarray(values, dtype=np.float32)
    for start in range(0, arr.size - n + 1):
        if bool(np.all(arr[start : start + n] >= threshold)):
            return int(start)
    return None


def compute_curve_quality(values: Array, peak: float) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size < 2:
        return 0.0
    tv = float(np.sum(np.abs(np.diff(arr))))
    return float((2.0 * max(float(peak), 1e-8)) / max(tv, 1e-8))


def compute_att_ttp_slope_for_curve(
    corrected_curve: Array,
    config: AnalysisConfig,
) -> dict[str, Any]:
    prepared = prepare_dynamic_curve(
        corrected_curve,
        smoothing_window=config.smoothing_window,
        baseline_frames=config.baseline_frames,
        peak_fraction_threshold=config.arrival_threshold,
        noise_sigma_multiplier=config.noise_sigma_multiplier,
    )
    smoothed = np.asarray(prepared["smoothed"], dtype=np.float32)
    enhancement = np.asarray(prepared["enhancement"], dtype=np.float32)
    normalized = np.asarray(prepared["normalized"], dtype=np.float32)
    threshold = float(prepared["threshold"])
    peak_gray = float(prepared["peak_enhancement"])
    noise_sigma = float(prepared["noise_sigma"])
    baseline = float(prepared["baseline"])

    at = first_consecutive_above(normalized, threshold, config.consecutive_frames)
    if at is None:
        ttp = int(np.argmax(enhancement)) if enhancement.size else None
    else:
        ttp = at + int(np.argmax(enhancement[at:]))
    if ttp is None:
        peak_norm = 0.0
    else:
        peak_norm = float(normalized[ttp])

    if at is not None and ttp is not None and ttp > at:
        diff_segment = np.diff(enhancement[at : ttp + 1])
        slope_max = float(np.max(diff_segment)) if diff_segment.size else 0.0
        slope_mean = float((enhancement[ttp] - enhancement[at]) / max(ttp - at, 1))
    else:
        slope_max = 0.0
        slope_mean = 0.0

    auc = float(np.sum(enhancement))
    snr = float(peak_gray / max(noise_sigma, 1e-8))
    smoothness = compute_curve_quality(enhancement, peak_gray)

    qc: list[str] = []
    if at is None:
        qc.append("missing_arrival")
    if peak_gray <= 1e-8:
        qc.append("no_positive_enhancement")
    if snr < config.min_snr:
        qc.append("low_snr")
    if ttp is None or at is None or ttp < at:
        qc.append("invalid_ttp")
    if peak_norm < max(config.min_peak_norm_at_ttp, threshold):
        qc.append("weak_peak_after_arrival")
    if smoothness < config.min_smoothness:
        qc.append("oscillatory_curve")

    reliable = not qc
    return {
        "raw_corrected": np.asarray(corrected_curve, dtype=np.float32),
        "smoothed": smoothed,
        "enhancement": enhancement,
        "normalized": normalized,
        "baseline": baseline,
        "threshold": threshold,
        "AT": at,
        "TTP": ttp,
        "Peak": peak_gray,
        "Peak_norm_at_TTP": peak_norm,
        "slope": slope_max,
        "slope_mean": slope_mean,
        "AUC": auc,
        "noise_sigma": noise_sigma,
        "snr": snr,
        "smoothness": smoothness,
        "reliable": reliable,
        "qc": qc,
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def _img01(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.size == 0:
        return arr
    if float(np.nanmax(arr)) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _draw_gray(ax: plt.Axes, img: np.ndarray, title: str = "", title_size: int = 7) -> None:
    ax.imshow(_img01(img), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_size, pad=2)


def _draw_centerline(ax: plt.Axes, centerline: list[tuple[int, int]], color: str = "#E3B36F", lw: float = 1.5) -> None:
    pa = np.asarray(centerline)
    if pa.size == 0:
        return
    ax.plot(pa[:, 1], pa[:, 0], color=color, lw=lw, solid_capstyle="round")


def _draw_roi_point(ax: plt.Axes, center: tuple[int, int], label: str, color: str, fs: int = 6) -> None:
    row, col = center
    ax.add_patch(Circle((col, row), 4.0, facecolor=color, edgecolor="white", lw=0.8, zorder=5))
    ax.text(col + 6, row, label, color=color, fontsize=fs, fontweight="bold", va="center")


def _panel_background(ax: plt.Axes, facecolor: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    patch = FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        transform=ax.transAxes,
        fc=facecolor, ec="none", zorder=0,
    )
    ax.add_patch(patch)


def save_method_figure(path: Path, config: AnalysisConfig, example_result: dict[str, Any] | None = None) -> None:
    """Save a Chinese workflow figure.

    If example_result is provided, the workflow contains real example images inside
    the four panels; otherwise it falls back to a simple text-only workflow.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Fallback for self-test initialization or calls before videos are processed.
    if example_result is None:
        fig, ax = plt.subplots(figsize=(14, 4.8))
        ax.axis("off")
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 4.8)
        fig.patch.set_facecolor("white")
        boxes = [
            ("1. 完整XCA序列", "原始帧用于提取曲线"),
            ("2. 峰值mask修复", "闭运算 + 连通桥接"),
            ("3. 定向中心线", "粗近端 → 细远端"),
            ("4. 动态ROI", "10% / 50% / 70% + 跟踪"),
            ("5. AT / TTP / slope", "基线、阈值、洗入过程"),
        ]
        xs = np.linspace(0.6, 11.8, len(boxes))
        y = 2.6
        w = 2.25
        h = 1.55
        colors = ["#DCEEF8", "#E8DFF0", "#E4F1D9", "#FFF0BE", "#F8DDDD"]
        for i, ((title, subtitle), x, color) in enumerate(zip(boxes, xs, colors)):
            box = FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.18,rounding_size=0.25", fc=color, ec="none")
            ax.add_patch(box)
            ax.text(x + w / 2, y + 0.28, title, ha="center", va="center", fontsize=10, fontweight="bold")
            ax.text(x + w / 2, y - 0.28, subtitle, ha="center", va="center", fontsize=8)
            if i < len(boxes) - 1:
                ax.add_patch(FancyArrowPatch((x + w + 0.08, y), (xs[i + 1] - 0.1, y), arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#4B5563"))
        formula = (
            r"$C_{raw}(t)=I_{vessel}(t)-I_{background}(t)$" + "\n" +
            "极性统一：造影增强向上增长" + "\n" +
            r"AT：$C_{norm}\geq0.20$ 且连续2帧；TTP：AT后峰值；slope：最大洗入导数"
        )
        ax.text(7.0, 0.8, formula, ha="center", va="center", fontsize=10)
        ax.set_title("AT / TTP / slope 生理参数分析流程", fontsize=14, fontweight="bold", pad=12)
        fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        return

    fig = plt.figure(figsize=(15.5, 8.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.15, 1.0, 1.45], wspace=0.08)
    fig.patch.set_facecolor("white")

    frames = np.asarray(example_result["frames_gray"], dtype=np.float32)
    peak_frame = np.asarray(example_result["peak_frame"], dtype=np.float32)
    connected_mask = np.asarray(example_result["connected_mask"], dtype=np.float32)
    centerline = example_result["centerline"]
    rois = example_result["rois"]
    metrics = example_result["metrics"]
    video_id = example_result["video_id"]

    t_prox = metrics["proximal"]["AT"] if metrics["proximal"]["AT"] is not None else 0
    t_mid = metrics["mid"]["AT"] if metrics["mid"]["AT"] is not None else min(len(frames) - 1, len(frames) // 2)
    t_dist = metrics["distal"]["AT"] if metrics["distal"]["AT"] is not None else len(frames) - 1
    t_prox = int(np.clip(t_prox, 0, len(frames) - 1))
    t_mid = int(np.clip(t_mid, 0, len(frames) - 1))
    t_dist = int(np.clip(t_dist, 0, len(frames) - 1))

    outer1 = fig.add_subplot(gs[0, 0])
    _panel_background(outer1, "#DCEEF8")
    outer1.text(0.5, 0.97, "Step1：\nXCA序列提取", ha="center", va="top", fontsize=15, fontweight="bold", transform=outer1.transAxes)
    ax11 = outer1.inset_axes([0.08, 0.63, 0.84, 0.25]); _draw_gray(ax11, frames[t_prox], f"近端到达帧 | frame {t_prox}")
    ax12 = outer1.inset_axes([0.08, 0.35, 0.84, 0.25]); _draw_gray(ax12, frames[t_mid], f"中段到达帧 | frame {t_mid}")
    ax13 = outer1.inset_axes([0.08, 0.07, 0.84, 0.25]); _draw_gray(ax13, frames[t_dist], f"远端到达帧 | frame {t_dist}")

    outer2 = fig.add_subplot(gs[0, 1])
    _panel_background(outer2, "#E8DFF0")
    outer2.text(0.5, 0.97, "Step2：\n掩码修复与骨架识别", ha="center", va="top", fontsize=15, fontweight="bold", transform=outer2.transAxes)
    ax21 = outer2.inset_axes([0.10, 0.53, 0.80, 0.36]); _draw_gray(ax21, connected_mask, "修补后的mask", title_size=8)
    ax22 = outer2.inset_axes([0.10, 0.08, 0.80, 0.36]); _draw_gray(ax22, peak_frame, "主干中心线", title_size=8); _draw_centerline(ax22, centerline, lw=1.4)

    outer3 = fig.add_subplot(gs[0, 2])
    _panel_background(outer3, "#E4F1D9")
    outer3.text(0.5, 0.97, "Step3：\n灌注点寻找", ha="center", va="top", fontsize=15, fontweight="bold", transform=outer3.transAxes)
    ax31 = outer3.inset_axes([0.10, 0.63, 0.80, 0.25]); _draw_gray(ax31, frames[t_prox], f"Proximal arrival | frame {t_prox}"); _draw_roi_point(ax31, rois["proximal"]["center"], "proximal", ROI_COLORS["proximal"])
    ax32 = outer3.inset_axes([0.10, 0.35, 0.80, 0.25]); _draw_gray(ax32, frames[t_mid], f"Mid arrival | frame {t_mid}"); _draw_roi_point(ax32, rois["mid"]["center"], "mid", ROI_COLORS["mid"])
    ax33 = outer3.inset_axes([0.10, 0.07, 0.80, 0.25]); _draw_gray(ax33, frames[t_dist], f"Distal arrival | frame {t_dist}"); _draw_roi_point(ax33, rois["distal"]["center"], "distal", ROI_COLORS["distal"])

    outer4 = fig.add_subplot(gs[0, 3])
    _panel_background(outer4, "#FFF0BE")
    outer4.text(0.5, 0.97, "Step4：\nAT / TTP / slope计算", ha="center", va="top", fontsize=15, fontweight="bold", transform=outer4.transAxes)
    ax41 = outer4.inset_axes([0.08, 0.47, 0.84, 0.42])
    _draw_gray(ax41, peak_frame, "", title_size=8)
    _draw_centerline(ax41, centerline, lw=1.8)
    _draw_roi_point(ax41, rois["proximal"]["center"], "proximal", ROI_COLORS["proximal"], fs=7)
    _draw_roi_point(ax41, rois["mid"]["center"], "mid", ROI_COLORS["mid"], fs=7)
    _draw_roi_point(ax41, rois["distal"]["center"], "distal", ROI_COLORS["distal"], fs=7)

    formula_lines = [
        r"$C_{\mathrm{raw}}(t)=I_{\mathrm{vessel}}(t)-I_{\mathrm{background}}(t)$",
        r"$C_{\mathrm{enh}}(t)=\max\!\left(C_{\mathrm{smooth}}(t)-\mathrm{baseline},\,0\right)$",
        r"$AT=\min\{t\mid C_{\mathrm{norm}}(t)\geq0.20\}$",
        r"$\mathrm{for\ two\ consecutive\ frames}$",
        r"$TTP=\arg\max_{t\geq AT} C_{\mathrm{enh}}(t)$",
        r"$\mathrm{slope}=\max_{AT\leq t<TTP}\left[C_{\mathrm{enh}}(t+1)-C_{\mathrm{enh}}(t)\right]$",
    ]
    y0 = 0.34
    dy = 0.055
    for i, line in enumerate(formula_lines):
        outer4.text(0.50, y0 - i * dy, line, ha="center", va="center", fontsize=11.8, transform=outer4.transAxes)
    fig.suptitle("AT / TTP / slope 生理参数分析流程", fontsize=19, fontweight="bold", y=0.98)
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_tracked_centers(ax: plt.Axes, centers: list[tuple[int, int]], color: str) -> None:
    if not centers:
        return
    arr = np.asarray(centers)
    ax.plot(arr[:, 1], arr[:, 0], color=color, lw=0.8, alpha=0.45)



def save_video_result_figure(
    path: Path,
    video_result: dict[str, Any],
    config: AnalysisConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.asarray(video_result["peak_frame"], dtype=np.float32)
    centerline = video_result["centerline"]
    rois = video_result["rois"]
    metrics = video_result["metrics"]
    tracked = video_result["tracked_centers"]
    frame_ids = video_result["frame_ids"]
    video_id = video_result["video_id"]

    fig = plt.figure(figsize=(17.0, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 0.9])
    fig.patch.set_facecolor("white")

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(frame, cmap="gray", vmin=0, vmax=1)
    pa = np.asarray(centerline)
    ax0.plot(pa[:, 1], pa[:, 0], color="#E3B36F", lw=2.0, alpha=0.9)
    for key in ROI_KEYS:
        row, col = rois[key]["center"]
        color = ROI_COLORS[key]
        ax0.add_patch(Circle((col, row), config.roi_radius, fill=False, ec=color, lw=2.3))
        ax0.add_patch(Circle((col, row), config.ring_inner_radius, fill=False, ec=color, lw=0.7, ls="--", alpha=0.55))
        ax0.add_patch(Circle((col, row), config.ring_outer_radius, fill=False, ec=color, lw=0.7, ls="--", alpha=0.55))
        ax0.text(col + config.ring_outer_radius + 2, row, key, color=color, fontsize=10, fontweight="bold", va="center")
        plot_tracked_centers(ax0, tracked[key], color)
    ax0.set_title(f"{video_id}: oriented trunk + tracked ROIs", fontsize=11, fontweight="bold")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1])
    t = np.arange(len(frame_ids))
    for key in ROI_KEYS:
        color = ROI_COLORS[key]
        m = metrics[key]
        enh = np.asarray(m["enhancement"], dtype=np.float32)
        ax1.plot(t, enh, color=color, marker="o", ms=2.5, lw=1.5, label=f"{key} ({'ok' if m['reliable'] else 'low conf.'})")
        if m["AT"] is not None:
            ax1.axvline(m["AT"], color=color, ls="--", lw=1.0, alpha=0.65)
        if m["TTP"] is not None:
            ax1.axvline(m["TTP"], color=color, ls="-", lw=1.2, alpha=0.85)
    ax1.axhline(0, color="#6B7280", lw=0.8, alpha=0.5)
    ax1.set_xlabel("local frame index", fontsize=10)
    ax1.set_ylabel("contrast enhancement (gray, baseline-subtracted)", fontsize=10)
    ax1.set_title("Time-intensity curves: AT dashed, TTP solid", fontsize=11, fontweight="bold")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[2])
    ax2.axis("off")
    rows = [["ROI", "AT", "TTP", "Peak", "slope", "AUC", "QC"]]
    for key in ROI_KEYS:
        m = metrics[key]
        rows.append([
            key,
            "n/a" if m["AT"] is None else str(m["AT"]),
            "n/a" if m["TTP"] is None else str(m["TTP"]),
            f"{m['Peak']:.3f}",
            f"{m['slope']:.3f}",
            f"{m['AUC']:.3f}",
            "Y" if m["reliable"] else "-",
        ])
    rows.append(["", "", "", "", "", "", ""])
    rows.append(["dAT(D-P)", _fmt_none(video_result["summary"]["dAT"]), "", "", "", "", ""])
    rows.append(["dTTP(D-P)", _fmt_none(video_result["summary"]["dTTP"]), "", "", "", "", ""])
    rows.append(["QC", video_result["summary"]["qc_status"], "", "", "", "", ""])
    table = ax2.table(cellText=rows, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.30)
    ax2.set_title("Metrics (unit: frame / gray)", fontsize=11, fontweight="bold")

    fig.suptitle(f"{video_id} AT / TTP / wash-in slope", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)



def save_combined_result_figure(
    path: Path,
    results: list[dict[str, Any]],
    config: AnalysisConfig,
) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(results)
    fig = plt.figure(figsize=(16.0, max(5.0, 4.2 * n)))
    gs = fig.add_gridspec(n, 3, width_ratios=[1.0, 1.35, 0.95], hspace=0.35, wspace=0.28)
    fig.patch.set_facecolor("white")

    for row, result in enumerate(results):
        frame = np.asarray(result["peak_frame"], dtype=np.float32)
        centerline = result["centerline"]
        rois = result["rois"]
        metrics = result["metrics"]
        video_id = result["video_id"]

        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(frame, cmap="gray", vmin=0, vmax=1)
        pa = np.asarray(centerline)
        ax0.plot(pa[:, 1], pa[:, 0], color="#E3B36F", lw=2.0)
        for key in ROI_KEYS:
            r, c = rois[key]["center"]
            color = ROI_COLORS[key]
            ax0.add_patch(Circle((c, r), config.roi_radius, fill=False, ec=color, lw=2.0))
            ax0.text(c + 10, r, key, color=color, fontsize=9, fontweight="bold", va="center")
        ax0.set_title(f"{video_id}: ROI on oriented trunk", fontweight="bold", fontsize=10)
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[row, 1])
        t = np.arange(len(result["frame_ids"]))
        for key in ROI_KEYS:
            m = metrics[key]
            color = ROI_COLORS[key]
            ax1.plot(t, np.asarray(m["enhancement"]), color=color, lw=1.5, marker="o", ms=2.2, label=key)
            if m["AT"] is not None:
                ax1.axvline(m["AT"], color=color, ls="--", lw=0.9, alpha=0.7)
            if m["TTP"] is not None:
                ax1.axvline(m["TTP"], color=color, ls="-", lw=1.0, alpha=0.7)
        ax1.set_title("Enhancement curves", fontweight="bold", fontsize=10)
        ax1.set_xlabel("local frame index", fontsize=9)
        ax1.set_ylabel("enhancement", fontsize=9)
        ax1.grid(alpha=0.25)
        ax1.legend(fontsize=8)

        ax2 = fig.add_subplot(gs[row, 2])
        ax2.axis("off")
        rows = [["ROI", "AT", "TTP", "Peak", "slope", "QC"]]
        for key in ROI_KEYS:
            m = metrics[key]
            rows.append([
                key,
                _fmt_none(m["AT"]),
                _fmt_none(m["TTP"]),
                f"{m['Peak']:.3f}",
                f"{m['slope']:.3f}",
                "Y" if m["reliable"] else "-",
            ])
        rows.append(["dAT", _fmt_none(result["summary"]["dAT"]), "", "", "", ""])
        rows.append(["dTTP", _fmt_none(result["summary"]["dTTP"]), "", "", "", ""])
        table = ax2.table(cellText=rows, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.20)
        ax2.set_title("Metrics", fontweight="bold", fontsize=10)

    fig.suptitle("AT / TTP / wash-in slope results", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def _style_tic_axis(ax: plt.Axes) -> None:
    """Apply the restrained paper style used by the retained Fig. 5(c)."""
    ax.spines["top"].set_linewidth(0.9)
    ax.spines["right"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["left"].set_linewidth(0.9)
    ax.tick_params(direction="out", width=0.8, length=3.5)
    ax.grid(False)


def _plot_paper_tic_panel(
    ax: plt.Axes,
    result: dict[str, Any],
    *,
    show_legend: bool,
) -> None:
    """Draw one normalized TIC panel using the established paper design."""
    frames = np.asarray(result["frame_ids"], dtype=np.int32)
    metrics = result["metrics"]
    distal = metrics["distal"]
    distal_at = distal["AT"]
    distal_ttp = distal["TTP"]
    if distal_at is not None and distal_ttp is not None and distal_ttp >= distal_at:
        ax.axvspan(
            int(frames[int(distal_at)]),
            int(frames[int(distal_ttp)]),
            color=WASH_IN_COLOR,
            alpha=0.58,
            linewidth=0,
            zorder=0,
        )

    for key in ROI_KEYS:
        metric = metrics[key]
        curve = np.asarray(metric["normalized"], dtype=np.float64)
        ax.plot(
            frames,
            curve,
            color=ROI_COLORS[key],
            linestyle=ROI_LINESTYLES[key],
            marker=ROI_MARKERS[key],
            markevery=5,
            linewidth=1.5,
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=0.7,
            label=ROI_LABELS[key],
        )
        if metric["AT"] is not None:
            index = int(metric["AT"])
            ax.scatter(
                [frames[index]], [curve[index]], s=28, marker="D",
                color="black", zorder=6,
            )
        if metric["TTP"] is not None:
            index = int(metric["TTP"])
            ax.scatter(
                [frames[index]], [curve[index]], s=60, marker="*",
                color="black", zorder=7,
            )

    ax.set_title(f"{result['video_id'].upper()} TICs")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized enhancement")
    ax.set_ylim(-0.04, 1.08)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        handles.extend([
            Patch(facecolor=WASH_IN_COLOR, edgecolor="none", alpha=0.65),
            Line2D([0], [0], marker="D", color="black", linewidth=0, markersize=5),
            Line2D([0], [0], marker="*", color="black", linewidth=0, markersize=8),
        ])
        labels.extend(["Distal wash-in interval", "AT", "TTP"])
        ax.legend(handles, labels, loc="lower right", frameon=False, ncol=2)
    _style_tic_axis(ax)


def save_video_result_figure(
    path: Path,
    video_result: dict[str, Any],
    config: AnalysisConfig,
) -> None:
    """Save the per-case paper-style normalized TIC, without diagnostic panels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
    }):
        fig, ax = plt.subplots(figsize=(5.0, 3.8), dpi=config.dpi)
        _plot_paper_tic_panel(ax, video_result, show_legend=True)
        fig.tight_layout()
        fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)


def save_combined_result_figure(
    path: Path,
    results: list[dict[str, Any]],
    config: AnalysisConfig,
) -> None:
    """Save a side-by-side Fig. 5(c)-style TIC summary for all requested cases."""
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
    }):
        fig, axes = plt.subplots(
            1, len(results), figsize=(5.0 * len(results), 3.8),
            dpi=config.dpi, squeeze=False,
        )
        for index, result in enumerate(results):
            _plot_paper_tic_panel(
                axes[0, index], result, show_legend=index == 0,
            )
        fig.text(
            0.055, 0.985, "Fig. 5(c) Time-Intensity Curve Analysis",
            ha="left", va="top", fontsize=11, fontstyle="italic",
        )
        fig.subplots_adjust(
            left=0.09, right=0.98, top=0.85, bottom=0.15, wspace=0.24,
        )
        fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)


def _fmt_none(v: Any) -> str:
    return "n/a" if v is None else str(v)


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def write_metrics_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video", "roi", "AT", "TTP", "Peak_gray", "slope_max_gray_per_frame",
            "slope_mean_gray_per_frame", "AUC_gray_frame", "SNR", "smoothness", "reliable", "qc_issues",
            "ctfc_like_frame", "relative_propagation_velocity_pixel_per_frame",
        ])
        for result in results:
            for key in ROI_KEYS:
                m = result["metrics"][key]
                writer.writerow([
                    result["video_id"], key, m["AT"], m["TTP"], f"{m['Peak']:.6g}",
                    f"{m['slope']:.6g}", f"{m['slope_mean']:.6g}", f"{m['AUC']:.6g}",
                    f"{m['snr']:.6g}", f"{m['smoothness']:.6g}", int(bool(m["reliable"])),
                    ";".join(m["qc"]), "", "",
                ])
            s = result["summary"]
            writer.writerow([
                result["video_id"], "D-P", s["dAT"], s["dTTP"], "", "", "", "", "", "", "",
                s["qc_status"], s["ctfc_like_frame"], s["relative_propagation_velocity_pixel_per_frame"],
            ])



# -----------------------------------------------------------------------------
# Main per-video processing
# -----------------------------------------------------------------------------
def process_video(
    video_id: str,
    mask_root: Path,
    frame_root: Path,
    out_root: Path,
    config: AnalysisConfig,
    manual_peak_frame: int | None = None,
) -> dict[str, Any]:
    config.validate()
    mask_dir = mask_root / video_id
    frame_dir = frame_root / video_id
    masks, mask_ids, mask_paths = load_mask_sequence_any(mask_dir)
    frames, frame_ids, frame_paths = load_gray_sequence_any(frame_dir)

    missing = sorted(set(mask_ids) - set(frame_ids))
    if missing:
        raise ValueError(f"Mask frame IDs are absent from dynamic sequence for {video_id}: {missing}")

    peak_mask_index, peak_frame_id, peak_mask_path, areas = select_peak_mask(
        masks, mask_ids, mask_paths, config.mask_threshold, manual_peak_frame,
    )
    peak_dynamic_index = frame_ids.index(peak_frame_id)
    peak_mask = masks[peak_mask_index]
    if peak_mask.shape != frames.shape[1:]:
        peak_mask = cv2.resize(
            peak_mask,
            (frames.shape[2], frames.shape[1]),
            interpolation=cv2.INTER_NEAREST,
        )

    connected_mask, skeleton, centerline, trunk_diag, bridge_connections, closed = build_connected_oriented_centerline(
        peak_mask,
        config,
    )
    rois = define_rois_by_fractions(
        centerline=centerline,
        image_shape=frames.shape[1:],
        vessel_mask=connected_mask,
        config=config,
    )

    tracked_centers: dict[str, list[tuple[int, int]]] = {}
    curve_details: dict[str, dict[str, Array]] = {}
    for key in ROI_KEYS:
        tracked_centers[key] = track_roi_centers(
            frames=frames,
            reference_center=tuple(int(x) for x in rois[key]["center"]),
            reference_frame_index=peak_dynamic_index,
            template_radius=config.tracking_template_radius,
            search_radius=config.tracking_search_radius,
        )
        curve_details[key] = extract_tracked_roi_curve_details(
            frames=frames,
            centers=tracked_centers[key],
            roi_radius=config.roi_radius,
            ring_inner_radius=config.ring_inner_radius,
            ring_outer_radius=config.ring_outer_radius,
        )

    polarity = determine_curve_polarity(
        curve_details["proximal"]["corrected"],
        peak_frame_index=peak_dynamic_index,
        baseline_frames=min(5, config.baseline_frames),
    )

    metrics: dict[str, dict[str, Any]] = {}
    for key in ROI_KEYS:
        corrected = np.asarray(curve_details[key]["corrected"], dtype=np.float32) * float(polarity)
        curve_details[key]["corrected_polarity_aligned"] = corrected
        metrics[key] = compute_att_ttp_slope_for_curve(corrected, config)

    # Add physiological-order QC. This is not used to swap labels; it only flags bad curves.
    p_at = metrics["proximal"]["AT"]
    d_at = metrics["distal"]["AT"]
    p_ttp = metrics["proximal"]["TTP"]
    d_ttp = metrics["distal"]["TTP"]
    if p_at is not None and d_at is not None and d_at <= p_at:
        metrics["distal"]["qc"].append("distal_arrival_not_later_than_proximal")
        metrics["distal"]["reliable"] = False
    if p_ttp is not None and d_ttp is not None and d_ttp <= p_ttp:
        metrics["distal"]["qc"].append("distal_ttp_not_later_than_proximal")
        metrics["distal"]["reliable"] = False

    dAT = None
    dTTP = None
    if metrics["proximal"]["reliable"] and metrics["distal"]["reliable"]:
        if p_at is not None and d_at is not None:
            dAT = int(d_at) - int(p_at)
        if p_ttp is not None and d_ttp is not None:
            dTTP = int(d_ttp) - int(p_ttp)

    delta_s_pm = compute_arc_length_for_indices(centerline, int(rois["proximal"]["index"]), int(rois["mid"]["index"]))
    delta_s_pd = compute_arc_length_for_indices(centerline, int(rois["proximal"]["index"]), int(rois["distal"]["index"]))
    ctfc_like_frame = dAT
    relative_velocity = (
        None
        if ctfc_like_frame is None or ctfc_like_frame <= 0
        else float(delta_s_pd) / float(ctfc_like_frame)
    )

    qc_issues: list[str] = []
    for key in ROI_KEYS:
        if not metrics[key]["reliable"]:
            qc_issues.append(f"{key}:" + "/".join(metrics[key]["qc"]))
    qc_status = "Pass" if not qc_issues else "Low confidence"

    result: dict[str, Any] = {
        "video_id": video_id,
        "frame_ids": frame_ids,
        "frame_paths": [str(p) for p in frame_paths],
        "mask_ids": mask_ids,
        "mask_paths": [str(p) for p in mask_paths],
        "selected_peak_frame": int(peak_frame_id),
        "selected_peak_mask": str(peak_mask_path),
        "peak_frame": frames[peak_dynamic_index],
        "mask_areas": areas,
        "connected_mask_area": int(connected_mask.sum()),
        "connected_mask": connected_mask,
        "frames_gray": frames,
        "bridge_connections": bridge_connections,
        "bridge_count": len(bridge_connections),
        "trunk_diagnostics": trunk_diag,
        "centerline": centerline,
        "skeleton_points": int(skeleton.sum()),
        "rois": rois,
        "tracked_centers": tracked_centers,
        "curve_polarity": float(polarity),
        "curve_details": curve_details,
        "metrics": metrics,
        "summary": {
            "dAT": dAT,
            "dTTP": dTTP,
            "ctfc_like_frame": ctfc_like_frame,
            "relative_propagation_velocity_pixel_per_frame": relative_velocity,
            "delta_s_proximal_mid_pixel": float(delta_s_pm),
            "delta_s_proximal_distal_pixel": float(delta_s_pd),
            "qc_status": qc_status,
            "qc_issues": qc_issues,
        },
        "config": asdict(config),
    }

    video_out = out_root / video_id
    video_out.mkdir(parents=True, exist_ok=True)
    # Keep JSON compact: figures need the image array in memory, but the report file
    # only needs geometry, curves, and metrics.
    def _metric_for_json(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in item.items()
            if key not in {"raw_corrected", "smoothed", "enhancement", "normalized"}
        }

    json_result = {
        "video_id": result["video_id"],
        "frame_ids": result["frame_ids"],
        "selected_peak_frame": result["selected_peak_frame"],
        "selected_peak_mask": result["selected_peak_mask"],
        "mask_areas": result["mask_areas"],
        "connected_mask_area": result["connected_mask_area"],
        "bridge_count": result["bridge_count"],
        "bridge_connections": result["bridge_connections"],
        "trunk_diagnostics": result["trunk_diagnostics"],
        "centerline_points": len(result["centerline"]),
        "rois": {
            key: {
                "center": result["rois"][key]["center"],
                "index": int(result["rois"][key]["index"]),
                "fraction": float(result["rois"][key]["fraction"]),
            }
            for key in ROI_KEYS
        },
        "curve_polarity": result["curve_polarity"],
        "metrics": {key: _metric_for_json(result["metrics"][key]) for key in ROI_KEYS},
        "curves": {
            f"{key}_norm": np.asarray(result["metrics"][key]["normalized"], dtype=np.float32).tolist()
            for key in ROI_KEYS
        },
        "summary": result["summary"],
        "config": result["config"],
    }
    (video_out / "metrics.json").write_text(
        json.dumps(to_jsonable(json_result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_video_result_figure(video_out / f"{video_id}_hemodynamic_analysis.png", result, config)
    return result


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_roi_fracs(text: str) -> tuple[float, float, float]:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--roi-fracs must contain three comma-separated floats")
    return (parts[0], parts[1], parts[2])


def parse_peak_frames(values: list[str] | None) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values or []:
        video, separator, frame = value.partition("=")
        if not separator or not video or not frame.isdigit():
            raise argparse.ArgumentTypeError(
                "--peak-frames values must use VIDEO=FRAME, e.g. v09=46"
            )
        result[video.lower()] = int(frame)
    return result


def print_results(results: list[dict[str, Any]]) -> None:
    print("\nHemodynamic analysis results")
    for result in results:
        print(f"\n{result['video_id'].upper()}")
        print("ROI        AT    TTP    Max wash-in slope")
        for key in ROI_KEYS:
            metric = result["metrics"][key]
            print(
                f"{key:<10} {str(metric['AT']):>4}  {str(metric['TTP']):>5}  "
                f"{float(metric['slope']):>17.6f}"
            )
        summary = result["summary"]
        print(f"CTFC-like delay: {summary['ctfc_like_frame']} frames")
        velocity = summary["relative_propagation_velocity_pixel_per_frame"]
        print(
            "Relative contrast-propagation velocity: "
            + ("N/A" if velocity is None else f"{velocity:.6f} pixel/frame")
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Calculate AT, TTP, maximum wash-in slope, CTFC-like delay, and relative propagation velocity."
    )
    parser.add_argument("--videos", nargs="+", default=["v09", "v11"], help="video IDs, e.g. v09 v11")
    parser.add_argument("--mask-root", type=Path, default=project_root / "outputs" / "mosxav_raw" / "postprocessed", help="root containing video mask folders")
    parser.add_argument("--frame-root", type=Path, default=project_root / "data" / "MOSXAV_raw" / "test" / "JPEGImages", help="root containing video frame folders")
    parser.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        type=Path,
        default=project_root / "outputs" / "quantitative_analysis" / "hemodynamics",
        help="output root",
    )
    parser.add_argument("--peak-frame", type=int, default=None, help="manual peak frame ID if needed")
    parser.add_argument(
        "--peak-frames",
        nargs="+",
        default=None,
        metavar="VIDEO=FRAME",
        help="per-video peak frames, e.g. v09=46 v11=52",
    )
    parser.add_argument("--roi-fracs", type=parse_roi_fracs, default=(0.10, 0.50, 0.70), help="proximal,mid,distal fractions, default 0.10,0.50,0.70")
    parser.add_argument("--roi-radius", type=int, default=6)
    parser.add_argument("--ring-inner", type=int, default=9)
    parser.add_argument("--ring-outer", type=int, default=16)
    parser.add_argument("--baseline-frames", type=int, default=10)
    parser.add_argument("--arrival-threshold", type=float, default=0.20)
    parser.add_argument("--min-snr", type=float, default=3.0)
    args = parser.parse_args()
    peak_frames = parse_peak_frames(args.peak_frames)

    config = AnalysisConfig(
        roi_fracs=args.roi_fracs,
        roi_radius=args.roi_radius,
        ring_inner_radius=args.ring_inner,
        ring_outer_radius=args.ring_outer,
        baseline_frames=args.baseline_frames,
        arrival_threshold=args.arrival_threshold,
        min_snr=args.min_snr,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for video in args.videos:
        print(f"[process] {video}")
        result = process_video(
            video_id=video,
            mask_root=args.mask_root,
            frame_root=args.frame_root,
            out_root=args.output_dir,
            config=config,
            manual_peak_frame=peak_frames.get(video.lower(), args.peak_frame),
        )
        results.append(result)
        print(f"  peak frame={result['selected_peak_frame']} qc={result['summary']['qc_status']} dAT={result['summary']['dAT']} dTTP={result['summary']['dTTP']}")
    write_metrics_csv(args.output_dir / "summary_metrics.csv", results)
    save_combined_result_figure(
        args.output_dir / "Fig_hemodynamic_TICs.png",
        results,
        config,
    )
    print_results(results)
    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
