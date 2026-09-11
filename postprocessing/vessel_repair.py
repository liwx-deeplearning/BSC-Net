"""Graph-guided vessel repair for binary segmentation masks."""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology
from skimage.graph import route_through_array


@dataclass(frozen=True)
class PostprocessV3Config:
    support_threshold: float = 0.18
    short_gap_px: int = 12
    medium_gap_px: int = 40
    max_growth_px: int = 50
    max_angle_deg: float = 50.0
    max_area_growth_ratio: float = 0.10
    min_path_mean_probability: float = 0.48
    min_path_supported_fraction: float = 0.78
    tangent_length_px: int = 7
    path_margin_px: int = 8
    maximum_repair_width: int = 9
    adaptive_repair_width: bool = True
    repair_width_scale: float = 1.0
    lateral_support_ratio: float = 0.65
    minimum_isolated_component_area: int = 32
    enable_endpoint_to_segment: bool = True
    enable_omnidirectional_orphan_link: bool = False
    # One-ended growth cannot prove that it closes a discontinuity.  Keep it
    # opt-in so the default operation only accepts topology-improving bridges.
    enable_one_sided_growth: bool = False

    def __post_init__(self) -> None:
        probability_values = (
            self.support_threshold,
            self.min_path_mean_probability,
            self.min_path_supported_fraction,
        )
        if any(value < 0.0 or value > 1.0 for value in probability_values):
            raise ValueError("probability parameters must be in range [0, 1]")
        if self.short_gap_px < 1:
            raise ValueError("short_gap_px must be positive")
        if self.medium_gap_px < self.short_gap_px:
            raise ValueError("medium_gap_px must be >= short_gap_px")
        if self.max_growth_px < 1:
            raise ValueError("max_growth_px must be positive")
        if not 0.0 < self.max_angle_deg < 90.0:
            raise ValueError("max_angle_deg must be in range (0, 90)")
        if self.max_area_growth_ratio < 0.0:
            raise ValueError("max_area_growth_ratio must be non-negative")
        if self.maximum_repair_width < 1:
            raise ValueError("maximum_repair_width must be positive")
        if self.repair_width_scale <= 0.0:
            raise ValueError("repair_width_scale must be positive")
        if not 0.0 < self.lateral_support_ratio <= 1.0:
            raise ValueError("lateral_support_ratio must be in range (0, 1]")
        if self.minimum_isolated_component_area < 0:
            raise ValueError("minimum_isolated_component_area must be non-negative")


@dataclass(frozen=True)
class Endpoint:
    row: int
    col: int
    component_id: int
    outward_row: float
    outward_col: float
    radius: float


@dataclass(frozen=True)
class PathCandidate:
    source_component: int
    target_component: int
    path: np.ndarray
    score: float
    width: int
    start_width: int
    end_width: int
    candidate_type: str


def _validate_inputs(
    base_mask: np.ndarray,
    probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if base_mask.ndim != 2 or probability.ndim != 2:
        raise ValueError("base_mask and probability must be 2D arrays")
    if base_mask.shape != probability.shape:
        raise ValueError("base_mask and probability must have the same shape")
    if not np.isfinite(probability).all():
        raise ValueError("probability must contain only finite values")
    if float(probability.min()) < 0.0 or float(probability.max()) > 1.0:
        raise ValueError("probability values must be in range [0, 1]")
    # mask/probability shape: (H, W)
    return (
        (base_mask > 0).astype(np.uint8),
        probability.astype(np.float32, copy=False),
    )


def _neighbors(
    skeleton: np.ndarray,
    point: tuple[int, int],
) -> list[tuple[int, int]]:
    row, col = point
    result: list[tuple[int, int]] = []
    for delta_row in (-1, 0, 1):
        for delta_col in (-1, 0, 1):
            if delta_row == 0 and delta_col == 0:
                continue
            next_row = row + delta_row
            next_col = col + delta_col
            if (
                0 <= next_row < skeleton.shape[0]
                and 0 <= next_col < skeleton.shape[1]
                and skeleton[next_row, next_col]
            ):
                result.append((next_row, next_col))
    return result


def _trace_inward(
    skeleton: np.ndarray,
    start: tuple[int, int],
    length: int,
) -> tuple[int, int] | None:
    path = [start]
    previous: tuple[int, int] | None = None
    current = start
    for _ in range(length):
        candidates = [
            point
            for point in _neighbors(skeleton, current)
            if point != previous and point not in path
        ]
        if not candidates:
            break
        next_point = max(
            candidates,
            key=lambda point: (
                (point[0] - start[0]) ** 2
                + (point[1] - start[1]) ** 2
            ),
        )
        previous, current = current, next_point
        path.append(current)
        if len(_neighbors(skeleton, current)) >= 3:
            break
    return path[-1] if len(path) >= 2 else None


def _extract_graph(
    mask: np.ndarray,
    tangent_length_px: int,
) -> tuple[np.ndarray, np.ndarray, list[Endpoint]]:
    # mask/skeleton/labels shape: (H, W)
    skeleton = morphology.skeletonize(mask.astype(bool))
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    kernel = np.array(
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
        dtype=np.uint8,
    )
    neighbor_count = cv2.filter2D(
        skeleton.astype(np.uint8),
        -1,
        kernel,
    )

    endpoints: list[Endpoint] = []
    for row, col in np.argwhere(skeleton & (neighbor_count == 1)):
        inward = _trace_inward(
            skeleton,
            (int(row), int(col)),
            tangent_length_px,
        )
        if inward is None:
            continue
        outward = np.array(
            [row - inward[0], col - inward[1]],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(outward))
        if norm <= 1e-7:
            continue
        outward /= norm
        window = distance[
            max(0, row - tangent_length_px) : row + tangent_length_px + 1,
            max(0, col - tangent_length_px) : col + tangent_length_px + 1,
        ]
        endpoints.append(
            Endpoint(
                row=int(row),
                col=int(col),
                component_id=int(labels[row, col]),
                outward_row=float(outward[0]),
                outward_col=float(outward[1]),
                radius=max(1.0, float(window.max())),
            )
        )
    return skeleton, labels, endpoints


def _path_cost(probability: np.ndarray, support_threshold: float) -> np.ndarray:
    clipped = np.clip(probability, 1e-5, 1.0)
    cost = -np.log(clipped)
    cost[probability < support_threshold] += 25.0
    return cost.astype(np.float64)


def _route(
    cost: np.ndarray,
    start: tuple[int, int],
    target: tuple[int, int],
    margin: int,
) -> np.ndarray | None:
    row_min = max(0, min(start[0], target[0]) - margin)
    row_max = min(cost.shape[0], max(start[0], target[0]) + margin + 1)
    col_min = max(0, min(start[1], target[1]) - margin)
    col_max = min(cost.shape[1], max(start[1], target[1]) + margin + 1)
    local_start = (start[0] - row_min, start[1] - col_min)
    local_target = (target[0] - row_min, target[1] - col_min)
    local_cost = cost[row_min:row_max, col_min:col_max]
    try:
        path, _ = route_through_array(
            local_cost,
            local_start,
            local_target,
            fully_connected=True,
            geometric=True,
        )
    except ValueError:
        return None
    path_array = np.asarray(path, dtype=np.int32)
    path_array[:, 0] += row_min
    path_array[:, 1] += col_min
    return path_array


def _alignment(endpoint: Endpoint, target: tuple[int, int]) -> float:
    direction = np.array(
        [target[0] - endpoint.row, target[1] - endpoint.col],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-7:
        return -1.0
    direction /= norm
    return float(
        direction[0] * endpoint.outward_row
        + direction[1] * endpoint.outward_col
    )


def _nearest_target(
    endpoint: Endpoint,
    target_pixels: np.ndarray,
    max_distance: int,
    cosine_limit: float,
) -> tuple[int, int] | None:
    if target_pixels.size == 0:
        return None
    deltas = target_pixels - np.array([endpoint.row, endpoint.col])
    distances = np.linalg.norm(deltas, axis=1)
    order = np.argsort(distances)
    for index in order:
        if distances[index] > max_distance:
            break
        target = tuple(int(value) for value in target_pixels[index])
        if _alignment(endpoint, target) >= cosine_limit:
            return target
    return None


def _evaluate_path(
    path: np.ndarray,
    mask: np.ndarray,
    probability: np.ndarray,
    labels: np.ndarray,
    source_component: int,
    target_component: int,
    config: PostprocessV3Config,
) -> tuple[float, int] | None:
    rows = path[:, 0]
    cols = path[:, 1]
    encountered = set(int(value) for value in labels[rows, cols])
    encountered.discard(0)
    if encountered - {source_component, target_component}:
        return None

    missing = mask[rows, cols] == 0
    if not np.any(missing):
        return None
    path_probability = probability[rows[missing], cols[missing]]
    mean_probability = float(path_probability.mean())
    supported_fraction = float(
        np.mean(path_probability >= config.support_threshold)
    )
    if (
        mean_probability < config.min_path_mean_probability
        or supported_fraction < config.min_path_supported_fraction
    ):
        return None
    score = (
        mean_probability
        + 0.25 * supported_fraction
        - 0.004 * float(len(path))
    )
    return score, int(np.count_nonzero(missing))


def _connection_candidates(
    mask: np.ndarray,
    probability: np.ndarray,
    config: PostprocessV3Config,
) -> list[PathCandidate]:
    skeleton, labels, endpoints = _extract_graph(
        mask,
        config.tangent_length_px,
    )
    component_ids = [
        int(value)
        for value in np.unique(labels)
        if value != 0
    ]
    component_sizes = {
        component_id: int(np.count_nonzero(labels == component_id))
        for component_id in component_ids
    }
    cost = _path_cost(probability, config.support_threshold)
    cosine_limit = float(np.cos(np.deg2rad(config.max_angle_deg)))
    endpoints_by_component = {
        component_id: [
            endpoint
            for endpoint in endpoints
            if endpoint.component_id == component_id
        ]
        for component_id in component_ids
    }
    skeleton_pixels = {
        component_id: np.argwhere(skeleton & (labels == component_id))
        for component_id in component_ids
    }

    candidates: list[PathCandidate] = []
    for endpoint in endpoints:
        for target_component in component_ids:
            if target_component == endpoint.component_id:
                continue

            target: tuple[int, int] | None = None
            candidate_type = "endpoint_to_endpoint"
            target_endpoints = endpoints_by_component[target_component]
            if (
                config.enable_omnidirectional_orphan_link
                and config.enable_endpoint_to_segment
            ):
                target = _nearest_target(
                    endpoint,
                    skeleton_pixels[target_component],
                    config.medium_gap_px,
                    -1.0,
                )
                candidate_type = "endpoint_to_segment"
            valid_endpoints = [
                target_endpoint
                for target_endpoint in target_endpoints
                if config.enable_omnidirectional_orphan_link
                or (
                    _alignment(
                        endpoint,
                        (target_endpoint.row, target_endpoint.col),
                    )
                    >= cosine_limit
                    and _alignment(
                        target_endpoint,
                        (endpoint.row, endpoint.col),
                    )
                    >= cosine_limit
                )
            ]
            if target is None and valid_endpoints:
                nearest_endpoint = min(
                    valid_endpoints,
                    key=lambda item: np.hypot(
                        item.row - endpoint.row,
                        item.col - endpoint.col,
                    ),
                )
                distance = float(
                    np.hypot(
                        nearest_endpoint.row - endpoint.row,
                        nearest_endpoint.col - endpoint.col,
                    )
                )
                if distance <= config.medium_gap_px:
                    target = (
                        nearest_endpoint.row,
                        nearest_endpoint.col,
                    )

            if target is None and config.enable_endpoint_to_segment:
                target = _nearest_target(
                    endpoint,
                    skeleton_pixels[target_component],
                    config.short_gap_px,
                    (
                        -1.0
                        if config.enable_omnidirectional_orphan_link
                        else cosine_limit
                    ),
                )
                candidate_type = "endpoint_to_segment"
            if target is None:
                continue

            path = _route(
                cost,
                (endpoint.row, endpoint.col),
                target,
                config.path_margin_px,
            )
            if path is None:
                continue
            evaluated = _evaluate_path(
                path,
                mask,
                probability,
                labels,
                endpoint.component_id,
                target_component,
                config,
            )
            if evaluated is None:
                continue
            score, _ = evaluated
            target_radius = float(
                cv2.distanceTransform(mask, cv2.DIST_L2, 5)[target]
            )
            width = max(
                1,
                int(round(endpoint.radius + max(1.0, target_radius) - 1.0)),
            )
            width = min(width, config.maximum_repair_width)
            if width % 2 == 0:
                width = max(1, width - 1)
            start_width = min(
                config.maximum_repair_width,
                max(
                    1,
                    int(
                        round(
                            (2.0 * endpoint.radius - 1.0)
                            * config.repair_width_scale
                        )
                    ),
                ),
            )
            end_width = min(
                config.maximum_repair_width,
                max(
                    1,
                    int(
                        round(
                            (2.0 * max(1.0, target_radius) - 1.0)
                            * config.repair_width_scale
                        )
                    ),
                ),
            )
            # Prefer attaching smaller orphan fragments to larger trees.
            if component_sizes[endpoint.component_id] < component_sizes[target_component]:
                score += 0.05
            candidates.append(
                PathCandidate(
                    source_component=endpoint.component_id,
                    target_component=target_component,
                    path=path,
                    score=score,
                    width=width,
                    start_width=start_width,
                    end_width=end_width,
                    candidate_type=candidate_type,
                )
            )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _path_mask(
    shape: tuple[int, int],
    path: np.ndarray,
    width: int,
    probability: np.ndarray,
    support_threshold: float,
    start_width: int | None = None,
    end_width: int | None = None,
    adaptive_width: bool = False,
) -> np.ndarray:
    centerline = np.zeros(shape, dtype=np.uint8)
    if (
        adaptive_width
        and start_width is not None
        and end_width is not None
        and len(path) > 1
    ):
        widths = np.linspace(start_width, end_width, len(path))
        for (row, col), point_width in zip(path, widths):
            radius = max(0, int(round(float(point_width))) // 2)
            cv2.circle(
                centerline,
                (int(col), int(row)),
                radius,
                1,
                thickness=-1,
            )
    else:
        centerline[path[:, 0], path[:, 1]] = 1
    if width > 1 and not adaptive_width:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (width, width),
        )
        centerline = cv2.dilate(centerline, kernel, iterations=1)
    # The centerline has already passed the stricter path-probability test in
    # _evaluate_path.  A vessel boundary is normally less confident than its
    # centre, so the locally interpolated tube may retain only probability-
    # supported lateral pixels at a lower, explicitly bounded threshold.
    centerline[probability < support_threshold] = 0
    return centerline


def _accept_connections(
    mask: np.ndarray,
    probability: np.ndarray,
    config: PostprocessV3Config,
    max_area: int,
) -> np.ndarray:
    result = mask.copy()
    for _ in range(8):
        candidates = _connection_candidates(result, probability, config)
        accepted = False
        for candidate in candidates:
            addition = _path_mask(
                result.shape,
                candidate.path,
                candidate.width,
                probability,
                (
                    config.support_threshold * config.lateral_support_ratio
                    if config.adaptive_repair_width
                    else config.support_threshold
                ),
                start_width=candidate.start_width,
                end_width=candidate.end_width,
                adaptive_width=config.adaptive_repair_width,
            )
            proposed = np.maximum(result, addition)
            if int(proposed.sum()) > max_area:
                continue
            before_components = cv2.connectedComponents(
                result,
                connectivity=8,
            )[0]
            after_components = cv2.connectedComponents(
                proposed,
                connectivity=8,
            )[0]
            if after_components >= before_components:
                continue
            result = proposed
            accepted = True
            break
        if not accepted:
            break
    return result


def _one_sided_candidates(
    mask: np.ndarray,
    probability: np.ndarray,
    config: PostprocessV3Config,
) -> Iterable[tuple[Endpoint, np.ndarray]]:
    skeleton, labels, endpoints = _extract_graph(
        mask,
        config.tangent_length_px,
    )
    del skeleton
    cost = _path_cost(probability, config.support_threshold)
    cosine_limit = float(np.cos(np.deg2rad(config.max_angle_deg)))
    supported_pixels = np.argwhere(
        (probability >= config.support_threshold) & (mask == 0)
    )

    for endpoint in endpoints:
        if supported_pixels.size == 0:
            continue
        deltas = supported_pixels - np.array([endpoint.row, endpoint.col])
        distances = np.linalg.norm(deltas, axis=1)
        outward = np.array(
            [endpoint.outward_row, endpoint.outward_col],
            dtype=np.float64,
        )
        alignments = np.divide(
            deltas @ outward,
            distances,
            out=np.full_like(distances, -1.0),
            where=distances > 1e-7,
        )
        valid = (
            (distances >= max(4.0, endpoint.radius * 2.0))
            & (distances <= config.max_growth_px)
            & (alignments >= cosine_limit)
        )
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            continue
        terminal_probability = probability[
            supported_pixels[indices, 0],
            supported_pixels[indices, 1],
        ]
        candidate_score = terminal_probability + 0.005 * distances[indices]
        order = indices[np.argsort(candidate_score)[::-1]]
        for index in order[:20]:
            target = tuple(
                int(value) for value in supported_pixels[index]
            )
            path = _route(
                cost,
                (endpoint.row, endpoint.col),
                target,
                config.path_margin_px,
            )
            if path is None:
                continue
            rows = path[:, 0]
            cols = path[:, 1]
            # A one-sided path must grow into new territory. Touching any
            # existing component after leaving the source would create a
            # same-component shortcut or an unscored cross-connection.
            touched = labels[rows[2:], cols[2:]]
            if np.any(touched != 0):
                continue
            path_probability = probability[rows[1:], cols[1:]]
            if (
                float(path_probability.mean())
                < config.min_path_mean_probability
                or float(
                    np.mean(
                        path_probability
                        >= config.support_threshold
                    )
                )
                < config.min_path_supported_fraction
            ):
                continue
            yield endpoint, path
            break


def _grow_one_sided(
    mask: np.ndarray,
    probability: np.ndarray,
    config: PostprocessV3Config,
    max_area: int,
) -> np.ndarray:
    result = mask.copy()
    candidates = list(_one_sided_candidates(result, probability, config))
    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    for endpoint, path in candidates:
        width = max(1, int(round(2.0 * endpoint.radius - 1.0)))
        width = min(width, config.maximum_repair_width)
        if width % 2 == 0:
            width = max(1, width - 1)
        addition = _path_mask(
            result.shape,
            path,
            width,
            probability,
            config.support_threshold * config.lateral_support_ratio,
        )
        proposed = np.maximum(result, addition)
        if int(proposed.sum()) <= max_area:
            result = proposed
    return result


def _remove_isolated_islands(
    mask: np.ndarray,
    minimum_area: int,
) -> np.ndarray:
    """Remove tiny, disconnected islands that cannot represent a vessel tree.

    Coronary branches may be thin, but a branch should remain connected to the
    visible coronary tree.  Therefore this deliberately conservative rule only
    removes *separate* components at or below ``minimum_area`` pixels; it never
    erodes a branch that is already connected to the main tree.
    """
    if minimum_area == 0:
        return mask.copy()
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    result = mask.copy()
    if component_count <= 1:
        return result
    largest_component = 1 + int(
        np.argmax(statistics[1:, cv2.CC_STAT_AREA])
    )
    for component_id in range(1, component_count):
        if (
            component_id != largest_component
            and int(statistics[component_id, cv2.CC_STAT_AREA])
            <= minimum_area
        ):
            result[labels == component_id] = 0
    return result


def repair_overlay(
    base_mask: np.ndarray,
    refined_mask: np.ndarray,
) -> np.ndarray:
    """Render a lossless audit overlay for a postprocessing result.

    Persistent prediction pixels are white, accepted bridge pixels are
    magenta, and removed isolated false-positive pixels are cyan.  The colors
    make both kinds of intervention inspectable without changing the binary
    mask used by downstream quantitative analysis.
    """
    base = base_mask > 0
    refined = refined_mask > 0
    overlay = np.zeros((*base.shape, 3), dtype=np.uint8)
    overlay[base & refined] = (255, 255, 255)
    overlay[~base & refined] = (255, 0, 255)  # magenta, RGB
    overlay[base & ~refined] = (0, 255, 255)  # cyan, RGB
    return overlay


def refine_mask_v3(
    base_mask: np.ndarray,
    probability: np.ndarray,
    config: PostprocessV3Config | None = None,
) -> np.ndarray:
    """Repair vessel topology using graph candidates and probability paths.

    Args:
        base_mask: Shape ``(H, W)``, values encoded as ``{0, 1}`` or
            ``{0, 255}``.
        probability: Shape ``(H, W)``, vessel probability in ``[0, 1]``.
        config: Version-3 repair configuration.

    Returns:
        Binary mask with shape ``(H, W)``, dtype ``uint8`` and values
        ``{0, 1}``.
    """
    if config is None:
        config = PostprocessV3Config()
    mask, probability_f32 = _validate_inputs(base_mask, probability)
    # mask/probability_f32 shape: (H, W)
    base_area = int(mask.sum())
    if base_area == 0:
        return mask
    max_area = int(
        np.ceil(base_area * (1.0 + config.max_area_growth_ratio))
    )
    # Remove only demonstrably isolated micro-islands before searching for
    # endpoint pairs.  This prevents a false-positive speck from becoming a
    # tempting bridge target.
    cleaned = _remove_isolated_islands(
        mask,
        config.minimum_isolated_component_area,
    )
    result = _accept_connections(
        cleaned,
        probability_f32,
        config,
        max_area,
    )
    if config.enable_one_sided_growth:
        result = _grow_one_sided(
            result,
            probability_f32,
            config,
            max_area,
        )
    return result.astype(np.uint8, copy=False)


def _read_mask(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read mask: {path}")
    return image


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise ValueError(f"Failed to encode mask: {path}")
    encoded.tofile(str(path))


def _write_overlay(path: Path, overlay_rgb: np.ndarray) -> None:
    """Write an RGB audit overlay through OpenCV's BGR encoder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", overlay_rgb[:, :, ::-1])
    if not ok:
        raise ValueError(f"Failed to encode overlay: {path}")
    encoded.tofile(str(path))


def _write_repair_audit(
    path: Path,
    base_mask: np.ndarray,
    refined_mask: np.ndarray,
) -> None:
    """Write a readable before/audit/zoom panel for changed predictions."""
    base = base_mask > 0
    refined = refined_mask > 0
    added = ~base & refined
    removed = base & ~refined
    changed = added | removed
    if not np.any(changed):
        return

    overlay = repair_overlay(base, refined)
    base_rgb = np.repeat((base.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
    height, width = base.shape

    def change_zoom(change_mask: np.ndarray) -> np.ndarray:
        rows, cols = np.where(change_mask)
        padding = 24
        top = max(0, int(rows.min()) - padding)
        bottom = min(height, int(rows.max()) + padding + 1)
        left = max(0, int(cols.min()) - padding)
        right = min(width, int(cols.max()) + padding + 1)
        box_color = (255, 215, 0)
        for panel in (base_rgb, overlay):
            cv2.rectangle(panel, (left, top), (right - 1, bottom - 1), box_color, 2)
        zoom = overlay[top:bottom, left:right]
        zoom_scale = min(width / max(1, zoom.shape[1]), height / max(1, zoom.shape[0]))
        zoom_size = (
            max(1, int(round(zoom.shape[1] * zoom_scale))),
            max(1, int(round(zoom.shape[0] * zoom_scale))),
        )
        zoom = cv2.resize(zoom, zoom_size, interpolation=cv2.INTER_NEAREST)
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        y0 = (height - zoom.shape[0]) // 2
        x0 = (width - zoom.shape[1]) // 2
        panel[y0:y0 + zoom.shape[0], x0:x0 + zoom.shape[1]] = zoom
        return panel

    panels = [base_rgb, overlay]
    labels = ["Prediction", "Repair audit"]
    if np.any(added):
        panels.append(change_zoom(added))
        labels.append("Added pixels (zoom)")
    if np.any(removed):
        panels.append(change_zoom(removed))
        labels.append("Removed pixels (zoom)")

    header = 42
    canvas = np.zeros((height + header, width * len(panels), 3), dtype=np.uint8)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        canvas[header:, index * width:(index + 1) * width] = panel
        cv2.putText(
            canvas,
            label,
            (index * width + 14, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    _write_overlay(path, canvas)


def process_dataset(
    base_mask_dir: Path,
    probability_dir: Path,
    output_dir: Path,
    config: PostprocessV3Config,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for mask_path in sorted(base_mask_dir.rglob("*.png")):
        relative_path = mask_path.relative_to(base_mask_dir)
        probability_path = probability_dir / relative_path.with_suffix(".npy")
        if not probability_path.exists():
            raise FileNotFoundError(
                f"Missing probability map for {relative_path}"
            )
        base_mask = _read_mask(mask_path)
        probability = np.load(probability_path)
        # base_mask/probability shape: (H, W)
        refined = refine_mask_v3(base_mask, probability, config)
        _write_mask(output_dir / relative_path, refined)
        _write_overlay(
            output_dir / "repair_overlays" / relative_path,
            repair_overlay(base_mask, refined),
        )
        _write_repair_audit(
            output_dir / "repair_audits" / relative_path,
            base_mask,
            refined,
        )

        base_binary = (base_mask > 0).astype(np.uint8)
        base_components = cv2.connectedComponents(
            base_binary,
            connectivity=8,
        )[0] - 1
        refined_components = cv2.connectedComponents(
            refined,
            connectivity=8,
        )[0] - 1
        base_area = int(base_binary.sum())
        refined_area = int(refined.sum())
        added_area = int(np.count_nonzero((refined > 0) & (base_binary == 0)))
        removed_area = int(np.count_nonzero((refined == 0) & (base_binary > 0)))
        records.append(
            {
                "frame": relative_path.as_posix(),
                "base_area": base_area,
                "refined_area": refined_area,
                "added_area": added_area,
                "removed_area": removed_area,
                "net_area_change": refined_area - base_area,
                "area_growth_ratio": (
                    (refined_area - base_area) / base_area
                    if base_area
                    else 0.0
                ),
                "base_components": int(base_components),
                "refined_components": int(refined_components),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parameters.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "processing_metrics.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = list(records[0]) if records else ["frame"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graph-guided vessel repair for binary segmentation masks."
    )
    parser.add_argument("--base-mask-dir", type=Path, required=True)
    parser.add_argument("--probability-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-threshold", type=float, default=0.18)
    parser.add_argument("--short-gap-px", type=int, default=12)
    parser.add_argument("--medium-gap-px", type=int, default=40)
    parser.add_argument("--max-growth-px", type=int, default=50)
    parser.add_argument("--max-angle-deg", type=float, default=50.0)
    parser.add_argument("--max-area-growth-ratio", type=float, default=0.10)
    parser.add_argument(
        "--min-path-mean-probability",
        type=float,
        default=0.48,
    )
    parser.add_argument(
        "--min-path-supported-fraction",
        type=float,
        default=0.78,
    )
    parser.add_argument("--path-margin-px", type=int, default=8)
    parser.add_argument("--maximum-repair-width", type=int, default=9)
    parser.add_argument(
        "--adaptive-repair-width",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--repair-width-scale", type=float, default=1.0)
    parser.add_argument(
        "--lateral-support-ratio",
        type=float,
        default=0.65,
        help="Minimum lateral support relative to the centreline threshold.",
    )
    parser.add_argument(
        "--minimum-isolated-component-area",
        type=int,
        default=32,
        help="Remove disconnected components no larger than this many pixels.",
    )
    parser.add_argument(
        "--enable-omnidirectional-orphan-link",
        action="store_true",
    )
    parser.add_argument(
        "--enable-one-sided-growth",
        action="store_true",
        help="Allow unpaired endpoint extension (disabled by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = PostprocessV3Config(
        support_threshold=args.support_threshold,
        short_gap_px=args.short_gap_px,
        medium_gap_px=args.medium_gap_px,
        max_growth_px=args.max_growth_px,
        max_angle_deg=args.max_angle_deg,
        max_area_growth_ratio=args.max_area_growth_ratio,
        min_path_mean_probability=args.min_path_mean_probability,
        min_path_supported_fraction=args.min_path_supported_fraction,
        path_margin_px=args.path_margin_px,
        maximum_repair_width=args.maximum_repair_width,
        adaptive_repair_width=args.adaptive_repair_width,
        repair_width_scale=args.repair_width_scale,
        lateral_support_ratio=args.lateral_support_ratio,
        minimum_isolated_component_area=args.minimum_isolated_component_area,
        enable_omnidirectional_orphan_link=(
            args.enable_omnidirectional_orphan_link
        ),
        enable_one_sided_growth=args.enable_one_sided_growth,
    )
    records = process_dataset(
        args.base_mask_dir,
        args.probability_dir,
        args.output_dir,
        config,
    )
    changed = sum(int(record["added_area"]) > 0 for record in records)
    added = sum(int(record["added_area"]) for record in records)
    print(
        f"Processed {len(records)} frames; changed {changed}; "
        f"added {added} pixels."
    )


if __name__ == "__main__":
    main()
