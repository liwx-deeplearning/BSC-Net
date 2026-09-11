"""Calculate centerline diameter and local stenosis profiles from vessel masks."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize

@dataclass
class LesionResult:
    """Summary of one accepted continuous stenotic interval."""

    start_index: int
    end_index: int
    peak_index: int
    start_s_px: float
    end_s_px: float
    peak_s_px: float
    peak_s_norm: float
    peak_point_rc: Tuple[int, int]
    mld_px: float
    rvd_px: float
    ds_max_fraction: float
    ds_max_percent: float
    lesion_length_px: float

    @property
    def MLD_px(self) -> float:
        """Compatibility alias matching the reporting terminology."""
        return self.mld_px

    @property
    def RVD_px(self) -> float:
        """Compatibility alias matching the reporting terminology."""
        return self.rvd_px

    @property
    def DS_max_fraction(self) -> float:
        """Compatibility alias matching the reporting terminology."""
        return self.ds_max_fraction

    @property
    def DS_max_percent(self) -> float:
        """Compatibility alias matching the reporting terminology."""
        return self.ds_max_percent


@dataclass
class StenosisProfileResult:
    """Local-reference stenosis analysis for one ordered centerline profile."""

    arc_length_px: np.ndarray
    normalized_position: np.ndarray
    d_ref_profile_px: np.ndarray
    stenosis_profile_fraction: np.ndarray
    stenosis_profile_percent: np.ndarray
    lesions: List[LesionResult]
    primary_lesion: Optional[LesionResult]
    has_lesion: bool
    reference_intercept: float
    reference_slope: float


@dataclass
class VesselCaseResult:
    case_name: str
    image_path: Path
    gray_image: np.ndarray
    mask: np.ndarray
    skeleton: np.ndarray
    distance_map: np.ndarray
    path_coords_rc: np.ndarray
    raw_diameter_profile_px: np.ndarray
    diameter_profile_px: np.ndarray
    d_min_px: float
    d_ref_px: float
    stenosis_ratio: float
    d_min_point_rc: Tuple[int, int]
    arc_length_px: np.ndarray
    normalized_position: np.ndarray
    d_ref_profile_px: np.ndarray
    stenosis_profile_fraction: np.ndarray
    stenosis_profile_percent: np.ndarray
    lesions: List[LesionResult]
    primary_lesion: Optional[LesionResult]
    has_lesion: bool
    reference_intercept: float
    reference_slope: float


def read_image_unicode(path: Path, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    """Read image with unicode path support on Windows."""
    if not path.is_file():
        raise FileNotFoundError(f"Image path does not exist: {path}")

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Failed to read image bytes: {path}")

    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image


def to_binary_mask(image: np.ndarray) -> np.ndarray:
    """Convert image to uint8 binary mask {0, 1}."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape: {image.shape}")

    if np.unique(image).size <= 3:
        return (image > 0).astype(np.uint8)

    _, binary = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(np.uint8)


def keep_largest_component(mask01: np.ndarray) -> np.ndarray:
    """Keep only the largest connected vessel component to reduce noise."""
    n_labels, labels = cv2.connectedComponents(mask01, connectivity=8)
    if n_labels <= 1:
        return mask01

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest_id = int(np.argmax(counts))
    return (labels == largest_id).astype(np.uint8)


def build_graph_from_skeleton(skeleton: np.ndarray) -> Tuple[np.ndarray, List[List[Tuple[int, float]]]]:
    """Build 8-neighborhood weighted graph for skeleton pixels."""
    coords = np.argwhere(skeleton)
    if len(coords) < 2:
        raise ValueError("Skeleton has too few points for centerline analysis.")

    mapping: Dict[Tuple[int, int], int] = {
        (int(rc[0]), int(rc[1])): idx for idx, rc in enumerate(coords.tolist())
    }
    graph: List[List[Tuple[int, float]]] = [[] for _ in range(len(coords))]

    for idx, (r, c) in enumerate(coords.tolist()):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                j = mapping.get((nr, nc))
                if j is None:
                    continue
                weight = 1.41421356 if (dr != 0 and dc != 0) else 1.0
                graph[idx].append((j, weight))

    return coords, graph


def dijkstra(graph: Sequence[Sequence[Tuple[int, float]]], src: int) -> Tuple[np.ndarray, np.ndarray]:
    """Dijkstra shortest-path on skeleton graph."""
    n = len(graph)
    dist = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int32)
    dist[src] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + float(w)
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))

    return dist, parent


def reconstruct_path(parent: np.ndarray, src: int, dst: int) -> List[int]:
    """Reconstruct path indices from src to dst using parent links."""
    path: List[int] = []
    cur = dst
    while cur != -1:
        path.append(cur)
        if cur == src:
            break
        cur = int(parent[cur])
    path.reverse()
    if not path or path[0] != src:
        raise ValueError("Unable to reconstruct centerline path.")
    return path


def select_thickest_endpoint(
    endpoints: np.ndarray, coords: np.ndarray, distance_map: np.ndarray
) -> int:
    """Pick endpoint index with maximum EDT radius as the centerline root."""
    if endpoints.size == 0:
        raise ValueError("No endpoints available for thickest-endpoint selection.")

    endpoint_coords = coords[endpoints]
    endpoint_radii = distance_map[endpoint_coords[:, 0], endpoint_coords[:, 1]]
    max_pos = int(np.argmax(endpoint_radii))
    return int(endpoints[max_pos])


def extract_main_centerline_path(skeleton: np.ndarray, distance_map: np.ndarray) -> np.ndarray:
    """Extract centerline path with index-0 anchored at the thickest endpoint."""
    coords, graph = build_graph_from_skeleton(skeleton)
    degrees = np.asarray([len(nbrs) for nbrs in graph], dtype=np.int32)
    endpoints = np.where(degrees == 1)[0]

    if len(endpoints) >= 2:
        # Force root at thickest terminal branch end so index-0 is anatomically meaningful.
        root = select_thickest_endpoint(endpoints=endpoints, coords=coords, distance_map=distance_map)
        dist, parent = dijkstra(graph, root)

        candidates = [
            int(ep)
            for ep in endpoints.tolist()
            if int(ep) != root and np.isfinite(dist[int(ep)]) and float(dist[int(ep)]) > 0.0
        ]
        if candidates:
            # From the thickest root, choose the farthest reachable endpoint as main path tip.
            dst = max(candidates, key=lambda idx: float(dist[idx]))
            path_idx = reconstruct_path(parent, root, int(dst))
            return coords[np.asarray(path_idx, dtype=np.int32)]

    # Loop-like vessel (no endpoints): use thickest skeleton point as pseudo-root,
    # then connect it to the farthest point to preserve root->distal ordering.
    radii_all = distance_map[coords[:, 0], coords[:, 1]]
    root = int(np.argmax(radii_all))
    dist, parent = dijkstra(graph, root)
    dst = int(np.argmax(dist))
    path_idx = reconstruct_path(parent, root, dst)
    return coords[np.asarray(path_idx, dtype=np.int32)]


def smooth_diameter_profile(diameter_profile: np.ndarray, max_window: int = 21, polyorder: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay smoothing with robust window fallback for short profiles."""
    profile = np.asarray(diameter_profile, dtype=np.float32)
    n = int(profile.size)
    if n < 5:
        return profile

    window = min(max_window, n)
    if window % 2 == 0:
        window -= 1
    if window < 5:
        return profile

    order = min(polyorder, window - 1)
    if order < 1:
        return profile

    smoothed = savgol_filter(profile, window_length=window, polyorder=order, mode="interp")
    return np.maximum(smoothed.astype(np.float32), 1e-6)


def compute_arc_length(path_coords_rc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return cumulative Euclidean arc length and its [0, 1] normalization."""
    coords = np.asarray(path_coords_rc, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 2:
        raise ValueError("path_coords_rc must have shape (N, 2) with N >= 2")
    if not np.all(np.isfinite(coords)):
        raise ValueError("path_coords_rc must contain only finite coordinates")

    step_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    arc_length = np.concatenate(([0.0], np.cumsum(step_lengths, dtype=np.float64)))
    total_length = float(arc_length[-1])
    if total_length <= 0.0:
        raise ValueError("Centerline path must have positive arc length")
    normalized = arc_length / total_length
    return arc_length, normalized


def fit_reference_taper(
    normalized_position: np.ndarray,
    diameter: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """Fit D_ref(s) = intercept + slope * s_norm using valid centerline points."""
    x = np.asarray(normalized_position, dtype=np.float64)
    y = np.asarray(diameter, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError("normalized_position and diameter must be matching 1D arrays with >= 2 points")

    finite = np.isfinite(x) & np.isfinite(y)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != x.shape:
            raise ValueError("valid_mask must match normalized_position shape")
        finite &= mask
    valid_indices = np.flatnonzero(finite)
    if valid_indices.size < 2 or np.ptp(x[valid_indices]) <= 1e-12:
        fallback = y[np.isfinite(y)]
        if fallback.size == 0:
            raise ValueError("diameter must contain finite values")
        intercept = float(np.median(fallback))
        slope = 0.0
    else:
        slope, intercept = np.polyfit(x[valid_indices], y[valid_indices], deg=1)
        slope = float(slope)
        intercept = float(intercept)

    reference = np.maximum(intercept + slope * x, 1e-6)
    return reference.astype(np.float64), intercept, slope


def _merge_candidate_intervals(candidate_mask: np.ndarray, merge_gap_points: int) -> List[Tuple[int, int]]:
    indices = np.flatnonzero(candidate_mask)
    if indices.size == 0:
        return []

    intervals: List[Tuple[int, int]] = []
    start = int(indices[0])
    end = start
    for index in indices[1:]:
        current = int(index)
        gap_points = current - end - 1
        if gap_points <= merge_gap_points:
            end = current
        else:
            intervals.append((start, end))
            start = current
            end = current
    intervals.append((start, end))
    return intervals


def analyze_stenosis_profile(
    path_coords_rc: np.ndarray,
    diameter_profile_px: np.ndarray,
    reference_exclusion_threshold: float = 0.20,
    reference_max_iterations: int = 5,
    stenosis_threshold: float = 0.20,
    min_lesion_points: int = 3,
    merge_gap_points: int = 2,
    min_lesion_length_px: Optional[float] = None,
) -> StenosisProfileResult:
    """Fit a local taper reference and detect persistent stenotic intervals."""
    diameter = np.asarray(diameter_profile_px, dtype=np.float64)
    coords = np.asarray(path_coords_rc)
    if diameter.ndim != 1 or diameter.size != len(coords) or diameter.size < 2:
        raise ValueError("diameter_profile_px must be 1D and match path_coords_rc length")
    if not np.all(np.isfinite(diameter)) or np.any(diameter <= 0.0):
        raise ValueError("diameter_profile_px must contain finite positive values")
    if not 0.0 < reference_exclusion_threshold < 1.0:
        raise ValueError("reference_exclusion_threshold must be between 0 and 1")
    if reference_max_iterations < 1:
        raise ValueError("reference_max_iterations must be >= 1")
    if not 0.0 <= stenosis_threshold < 1.0:
        raise ValueError("stenosis_threshold must be in [0, 1)")
    if min_lesion_points < 1 or merge_gap_points < 0:
        raise ValueError("min_lesion_points must be >= 1 and merge_gap_points must be >= 0")
    if min_lesion_length_px is not None and min_lesion_length_px < 0.0:
        raise ValueError("min_lesion_length_px must be >= 0 when provided")

    arc_length, normalized = compute_arc_length(coords)
    normal_mask = np.ones(diameter.shape, dtype=bool)
    reference, intercept, slope = fit_reference_taper(normalized, diameter, normal_mask)

    for _ in range(reference_max_iterations):
        deficit = np.maximum(0.0, (reference - diameter) / reference)
        updated_mask = normal_mask & (deficit < reference_exclusion_threshold)
        if np.array_equal(updated_mask, normal_mask):
            break
        if int(updated_mask.sum()) < 2 or np.ptp(normalized[updated_mask]) <= 1e-12:
            break
        normal_mask = updated_mask
        reference, intercept, slope = fit_reference_taper(normalized, diameter, normal_mask)

    reference, intercept, slope = fit_reference_taper(normalized, diameter, normal_mask)
    stenosis_fraction = np.maximum(0.0, (reference - diameter) / reference)
    stenosis_fraction = np.nan_to_num(stenosis_fraction, nan=0.0, posinf=0.0, neginf=0.0)
    stenosis_percent = 100.0 * stenosis_fraction

    candidate_mask = stenosis_fraction >= (stenosis_threshold - 1e-12)
    lesions: List[LesionResult] = []
    for start, end in _merge_candidate_intervals(candidate_mask, merge_gap_points):
        point_count = end - start + 1
        lesion_length = float(arc_length[end] - arc_length[start])
        if point_count < min_lesion_points:
            continue
        if min_lesion_length_px is not None and lesion_length < min_lesion_length_px:
            continue

        peak_index = start + int(np.argmax(stenosis_fraction[start : end + 1]))
        peak_point = (int(coords[peak_index, 0]), int(coords[peak_index, 1]))
        ds_max = float(stenosis_fraction[peak_index])
        lesions.append(
            LesionResult(
                start_index=start,
                end_index=end,
                peak_index=peak_index,
                start_s_px=float(arc_length[start]),
                end_s_px=float(arc_length[end]),
                peak_s_px=float(arc_length[peak_index]),
                peak_s_norm=float(normalized[peak_index]),
                peak_point_rc=peak_point,
                mld_px=float(diameter[peak_index]),
                rvd_px=float(reference[peak_index]),
                ds_max_fraction=ds_max,
                ds_max_percent=100.0 * ds_max,
                lesion_length_px=lesion_length,
            )
        )

    primary = max(lesions, key=lambda lesion: lesion.ds_max_fraction) if lesions else None
    return StenosisProfileResult(
        arc_length_px=arc_length,
        normalized_position=normalized,
        d_ref_profile_px=reference,
        stenosis_profile_fraction=stenosis_fraction,
        stenosis_profile_percent=stenosis_percent,
        lesions=lesions,
        primary_lesion=primary,
        has_lesion=primary is not None,
        reference_intercept=intercept,
        reference_slope=slope,
    )


def enforce_black_axes_frame(ax: plt.Axes, linewidth: float = 0.9) -> None:
    """Apply a solid black frame, ticks, labels, and legend border to a data axis."""
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(linewidth)
    ax.tick_params(axis="both", colors="black", width=linewidth)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    legend = ax.get_legend()
    if legend is not None and legend.get_frame().get_visible():
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(linewidth)


def analyze_case(
    case_name: str,
    image_path: Path,
    ref_percentile: Optional[float] = None,
    *,
    reference_exclusion_threshold: float = 0.20,
    reference_max_iterations: int = 5,
    stenosis_threshold: float = 0.20,
    min_lesion_points: int = 3,
    merge_gap_points: int = 2,
    min_lesion_length_px: Optional[float] = None,
) -> VesselCaseResult:
    """Analyze one vessel mask using a local taper reference and interval lesions."""
    del ref_percentile  # Retained only for call-site compatibility with the legacy API.
    gray = read_image_unicode(image_path, flags=cv2.IMREAD_GRAYSCALE)
    mask = keep_largest_component(to_binary_mask(gray))

    if int(mask.sum()) == 0:
        raise ValueError(f"Mask foreground is empty: {image_path}")

    skeleton = skeletonize(mask.astype(bool))
    if int(skeleton.sum()) < 2:
        raise ValueError(f"Skeleton points are too few for analysis: {image_path}")

    distance_map = distance_transform_edt(mask.astype(bool)).astype(np.float32)
    path_coords = extract_main_centerline_path(skeleton, distance_map)

    radii = distance_map[path_coords[:, 0], path_coords[:, 1]]
    diam_raw = 2.0 * radii
    diam = smooth_diameter_profile(diam_raw)
    stenosis = analyze_stenosis_profile(
        path_coords,
        diam,
        reference_exclusion_threshold=reference_exclusion_threshold,
        reference_max_iterations=reference_max_iterations,
        stenosis_threshold=stenosis_threshold,
        min_lesion_points=min_lesion_points,
        merge_gap_points=merge_gap_points,
        min_lesion_length_px=min_lesion_length_px,
    )

    if stenosis.primary_lesion is not None:
        primary = stenosis.primary_lesion
        d_min = primary.mld_px
        d_ref = primary.rvd_px
        sr = primary.ds_max_fraction
        d_min_point = primary.peak_point_rc
    else:
        fallback_index = int(np.argmin(diam))
        d_min = float(diam[fallback_index])
        d_ref = float(stenosis.d_ref_profile_px[fallback_index])
        sr = 0.0
        d_min_point = (
            int(path_coords[fallback_index, 0]),
            int(path_coords[fallback_index, 1]),
        )

    return VesselCaseResult(
        case_name=case_name,
        image_path=image_path,
        gray_image=gray,
        mask=mask,
        skeleton=skeleton,
        distance_map=distance_map,
        path_coords_rc=path_coords,
        raw_diameter_profile_px=diam_raw,
        diameter_profile_px=diam,
        d_min_px=d_min,
        d_ref_px=d_ref,
        stenosis_ratio=sr,
        d_min_point_rc=d_min_point,
        arc_length_px=stenosis.arc_length_px,
        normalized_position=stenosis.normalized_position,
        d_ref_profile_px=stenosis.d_ref_profile_px,
        stenosis_profile_fraction=stenosis.stenosis_profile_fraction,
        stenosis_profile_percent=stenosis.stenosis_profile_percent,
        lesions=stenosis.lesions,
        primary_lesion=stenosis.primary_lesion,
        has_lesion=stenosis.has_lesion,
        reference_intercept=stenosis.reference_intercept,
        reference_slope=stenosis.reference_slope,
    )


def _select_mask(mask_dir: Path, peak_frame: int | None) -> Path:
    """Select a requested mask, or the frame with the largest foreground area."""
    if peak_frame is not None:
        requested = mask_dir / f"{peak_frame:05d}.png"
        if not requested.is_file():
            raise FileNotFoundError(f"Requested mask not found: {requested}")
        return requested

    candidates = sorted(mask_dir.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"No PNG masks found in: {mask_dir}")
    return max(
        candidates,
        key=lambda path: int(to_binary_mask(read_image_unicode(path)).sum()),
    )


def _case_summary(result: VesselCaseResult) -> dict[str, object]:
    diameter = np.asarray(result.diameter_profile_px, dtype=np.float64)
    return {
        "case": result.case_name,
        "source_mask": str(result.image_path),
        "centerline_length_pixel": float(result.arc_length_px[-1]),
        "mean_apparent_diameter_pixel": float(diameter.mean()),
        "minimum_apparent_diameter_pixel": float(diameter.min()),
        "maximum_apparent_diameter_pixel": float(diameter.max()),
        "minimum_lumen_diameter_mld_pixel": float(result.d_min_px),
        "reference_vessel_diameter_rvd_pixel": float(result.d_ref_px),
        "maximum_diameter_stenosis_percent": float(100.0 * result.stenosis_ratio),
        "lesion_count": len(result.lesions),
    }


def _write_case_outputs(result: VesselCaseResult, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _case_summary(result)
    profile_path = output_dir / "diameter_profile.csv"
    with profile_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "centerline_arc_length_pixel",
                "normalized_centerline_position",
                "apparent_diameter_pixel",
                "reference_diameter_pixel",
                "diameter_stenosis_percent",
            ]
        )
        writer.writerows(
            zip(
                result.arc_length_px,
                result.normalized_position,
                result.diameter_profile_px,
                result.d_ref_profile_px,
                result.stenosis_profile_percent,
            )
        )
    detailed_metrics = dict(summary)
    detailed_metrics["lesions"] = [
        {
            "start_index": int(lesion.start_index),
            "end_index": int(lesion.end_index),
            "peak_index": int(lesion.peak_index),
            "minimum_lumen_diameter_mld_pixel": float(lesion.mld_px),
            "reference_vessel_diameter_rvd_pixel": float(lesion.rvd_px),
            "maximum_diameter_stenosis_percent": float(100.0 * lesion.ds_max_fraction),
        }
        for lesion in result.lesions
    ]
    (output_dir / "metrics.json").write_text(
        json.dumps(detailed_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=180)
    axes[0].imshow(result.gray_image, cmap="gray")
    path = np.asarray(result.path_coords_rc)
    axes[0].plot(path[:, 1], path[:, 0], color="#21B6A8", linewidth=1.2)
    axes[0].scatter(
        [result.d_min_point_rc[1]],
        [result.d_min_point_rc[0]],
        color="#D81B60",
        s=24,
        label="MLD",
    )
    axes[0].set_title(f"{result.case_name}: centerline and MLD")
    axes[0].axis("off")
    axes[0].legend(frameon=False, loc="lower right")

    x = result.normalized_position
    axes[1].plot(x, result.diameter_profile_px, label="Apparent lumen diameter", color="#3B82F6")
    axes[1].plot(x, result.d_ref_profile_px, label="Reference diameter", color="#10B981", linestyle="--")
    for index, lesion in enumerate(result.lesions):
        axes[1].axvspan(
            x[lesion.start_index],
            x[lesion.end_index],
            color="#D81B60",
            alpha=0.16,
            label="Stenotic interval" if index == 0 else None,
        )
    axes[1].set_xlabel("Normalized centerline position")
    axes[1].set_ylabel("Diameter (pixel)")
    axes[1].set_title(
        f"MLD={result.d_min_px:.2f} px, DS={100.0 * result.stenosis_ratio:.1f}%"
    )
    axes[1].legend(frameon=False)
    enforce_black_axes_frame(axes[1])
    fig.tight_layout()
    figure_path = output_dir / "diameter_stenosis_analysis.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return summary


def _parse_peak_frames(values: list[str] | None) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values or []:
        case, separator, frame = value.partition("=")
        if not separator or not case or not frame.isdigit():
            raise argparse.ArgumentTypeError(
                "--peak-frames values must use CASE=FRAME, e.g. v09=46"
            )
        parsed[case.lower()] = int(frame)
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate quantitative coronary angiography morphology: apparent "
            "diameter, MLD, RVD, and diameter stenosis (DS)."
        )
    )
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--peak-frames",
        nargs="+",
        default=None,
        metavar="CASE=FRAME",
        help="optional per-case frame IDs; otherwise the largest-area mask is used",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    peak_frames = _parse_peak_frames(args.peak_frames)
    summaries: list[dict[str, object]] = []
    print("\nQuantitative coronary angiography: anatomical morphology")
    for video in args.videos:
        key = video.lower()
        mask_path = _select_mask(args.mask_root / video, peak_frames.get(key))
        result = analyze_case(video, mask_path)
        summary = _write_case_outputs(result, args.output_dir / video)
        summaries.append(summary)
        print(f"\n{video.upper()} (mask: {mask_path.name})")
        print(
            "Apparent lumen diameter: "
            f"mean={summary['mean_apparent_diameter_pixel']:.3f} pixel, "
            f"range={summary['minimum_apparent_diameter_pixel']:.3f}-"
            f"{summary['maximum_apparent_diameter_pixel']:.3f} pixel"
        )
        print(
            "QCA morphology: "
            f"MLD={summary['minimum_lumen_diameter_mld_pixel']:.3f} pixel, "
            f"RVD={summary['reference_vessel_diameter_rvd_pixel']:.3f} pixel, "
            f"diameter stenosis (DS)="
            f"{summary['maximum_diameter_stenosis_percent']:.2f}%"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]) if summaries else ["case"])
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nMorphology outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
