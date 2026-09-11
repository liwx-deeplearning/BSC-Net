import json
import random
import re
from bisect import bisect_right
from pathlib import Path
from typing import Iterable, Optional, Tuple, Set, Dict, Any, List, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def cv2_imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    """
    Windows 中文路径安全版 imread。

    cv2.imread 在中文路径下可能返回 None，
    因此使用 np.fromfile + cv2.imdecode。
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    data = np.fromfile(str(path), dtype=np.uint8)

    if data.size == 0:
        raise FileNotFoundError(f"File is empty or cannot be read: {path}")

    img = cv2.imdecode(data, flags)

    if img is None:
        raise RuntimeError(f"cv2.imdecode failed: {path}")

    return img


def cv2_imwrite_unicode(path: Path, img: np.ndarray):
    """
    Windows 中文路径安全版 imwrite。
    训练/预测保存时也建议使用这个函数。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ok, encoded = cv2.imencode(path.suffix, img)

    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")

    encoded.tofile(str(path))


def numeric_sort_key(path: Path):
    """
    按文件名中的数字排序。
    例如 00025.jpg -> 25
    """
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def read_static_split_file(split_file: Optional[str]):
    """
    读取静态 train.txt / test.txt。

    同时支持两种格式：

    1. 视频级 split:
        v00
        v01

    2. 帧级 split:
        v00/00025.png
        v00/00030.jpg
        JPEGImages_Static/v00/00025.jpg

    返回:
        allowed_video_ids:
            set[str] or None
            例如 {"v00", "v01"}

        allowed_frame_keys:
            set[str] or None
            格式为 "v00/00025"，不带扩展名。
    """
    if split_file is None:
        return None, None

    split_path = Path(split_file)

    if not split_path.exists():
        raise FileNotFoundError(f"split_file not found: {split_path}")

    allowed_video_ids: Set[str] = set()
    allowed_frame_keys: Set[str] = set()

    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            token = line.split()[0]
            parts = Path(token).parts

            video_id = None
            frame_name = None

            for i, part in enumerate(parts):
                if re.match(r"^v\d+$", part, flags=re.IGNORECASE):
                    video_id = part

                    if i + 1 < len(parts):
                        frame_name = parts[i + 1]

                    break

            # 情况 1：整行就是 v00
            if video_id is not None and frame_name is None:
                allowed_video_ids.add(video_id)
                continue

            # 情况 2：v00/00025.png 或 JPEGImages_Static/v00/00025.jpg
            if video_id is not None and frame_name is not None:
                frame_stem = Path(frame_name).stem
                allowed_frame_keys.add(f"{video_id}/{frame_stem}")
                continue

            # 兜底：如果没有识别到 vxx，则当作视频级 ID
            allowed_video_ids.add(token)

    if len(allowed_video_ids) == 0:
        allowed_video_ids = None

    if len(allowed_frame_keys) == 0:
        allowed_frame_keys = None

    return allowed_video_ids, allowed_frame_keys


def read_image_rgb(path: Path, image_size: Optional[Tuple[int, int]] = None):
    """
    读取静态 XCA 原图。

    Parameters
    ----------
    path:
        图像路径。

    image_size:
        OpenCV resize 使用 (width, height)。

    Returns
    -------
    img:
        [H, W, 3], float32, range 0~1, RGB。
    """
    img = cv2_imread_unicode(path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if image_size is not None:
        img = cv2.resize(img, image_size, interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32) / 255.0

    return img


def read_binary_vessel_mask(path: Path, image_size: Optional[Tuple[int, int]] = None):
    """
    读取静态血管 mask。

    适用于：
        黑色背景 = 0
        血管区域 = 非 0 像素

    支持灰度 mask 或 RGB/RGBA 彩色 mask。

    Returns
    -------
    mask:
        [H, W], uint8, 0/1。
    """
    raw = cv2_imread_unicode(path, flags=cv2.IMREAD_UNCHANGED)

    # RGBA -> RGB/BGR
    if raw.ndim == 3 and raw.shape[2] == 4:
        raw = raw[:, :, :3]

    if image_size is not None:
        raw = cv2.resize(raw, image_size, interpolation=cv2.INTER_NEAREST)

    # 灰度 mask: 0/255 或 0/1
    if raw.ndim == 2:
        return (raw > 0).astype(np.uint8)

    # 彩色 mask: 黑背景 + 非黑血管
    if raw.ndim == 3:
        return np.any(raw > 0, axis=-1).astype(np.uint8)

    raise ValueError(f"Unsupported mask shape {raw.shape}: {path}")


def make_mask_skeleton(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    # binary shape: (H, W)
    skeleton = np.zeros_like(binary, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    current = binary.copy()

    while cv2.countNonZero(current) > 0:
        opened = cv2.morphologyEx(current, cv2.MORPH_OPEN, element)
        detail = cv2.subtract(current, opened)
        skeleton = cv2.bitwise_or(skeleton, detail)
        current = cv2.erode(current, element)

    # skeleton shape: (H, W)
    return (skeleton > 0).astype(np.uint8)


def count_skeleton_neighbors(skeleton: np.ndarray) -> np.ndarray:
    binary = (skeleton > 0).astype(np.uint8)
    # binary shape: (H, W)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(binary, ddepth=cv2.CV_16S, kernel=kernel) - binary
    # neighbor_count shape: (H, W)
    return neighbor_count


def _unique_yx_points(points: Iterable[Tuple[int, int]]) -> np.ndarray:
    point_list = list(points)
    if len(point_list) == 0:
        return np.empty((0, 2), dtype=np.int32)
    unique_points = sorted(set((int(y), int(x)) for y, x in point_list))
    return np.asarray(unique_points, dtype=np.int32)


def _component_center_point(
    labels: np.ndarray,
    component_id: int,
    centroid_xy: Tuple[float, float],
) -> Tuple[int, int]:
    # labels shape: (H, W)
    yx = np.argwhere(labels == component_id)
    if yx.size == 0:
        return 0, 0

    center_x, center_y = centroid_xy
    rounded_y = int(round(center_y))
    rounded_x = int(round(center_x))
    h, w = labels.shape[:2]
    rounded_y = int(np.clip(rounded_y, 0, h - 1))
    rounded_x = int(np.clip(rounded_x, 0, w - 1))

    if labels[rounded_y, rounded_x] == component_id:
        return rounded_y, rounded_x

    distances = (yx[:, 0] - center_y) ** 2 + (yx[:, 1] - center_x) ** 2
    nearest_index = int(np.argmin(distances))
    return int(yx[nearest_index, 0]), int(yx[nearest_index, 1])


def make_mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    # binary shape: (H, W)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    boundary = ((dilated - eroded) > 0) & (binary > 0)
    # boundary shape: (H, W)
    return boundary.astype(np.uint8)


def find_branch_crop_candidate_groups(
    mask: np.ndarray,
    min_component_area: int = 1,
    small_component_area: int = 50,
) -> Dict[str, np.ndarray]:
    binary = (mask > 0).astype(np.uint8)
    # binary shape: (H, W)
    empty = np.empty((0, 2), dtype=np.int32)
    groups: Dict[str, np.ndarray] = {
        "endpoint": empty,
        "thin": empty,
        "small_component": empty,
        "boundary": empty,
        "branch": empty,
    }
    if binary.sum() == 0:
        return groups

    skeleton = make_mask_skeleton(binary)
    # skeleton shape: (H, W)
    if skeleton.sum() > 0:
        neighbor_count = count_skeleton_neighbors(skeleton)
        endpoint_yx = np.argwhere((skeleton > 0) & (neighbor_count == 1))
        branch_yx = np.argwhere((skeleton > 0) & (neighbor_count >= 3))
        groups["endpoint"] = _unique_yx_points((int(y), int(x)) for y, x in endpoint_yx)
        groups["branch"] = _unique_yx_points((int(y), int(x)) for y, x in branch_yx)

        distance = cv2.distanceTransform(binary, distanceType=cv2.DIST_L2, maskSize=3)
        skeleton_radius = distance[skeleton > 0]
        if skeleton_radius.size > 0:
            radius_threshold = float(np.percentile(skeleton_radius, 25))
            thin_yx = np.argwhere((skeleton > 0) & (distance <= radius_threshold))
            groups["thin"] = _unique_yx_points((int(y), int(x)) for y, x in thin_yx)

    boundary_yx = np.argwhere(make_mask_boundary(binary) > 0)
    groups["boundary"] = _unique_yx_points((int(y), int(x)) for y, x in boundary_yx)

    small_points: List[Tuple[int, int]] = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    for component_id in range(1, num_labels):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if min_component_area <= area <= small_component_area:
            small_points.append(
                _component_center_point(
                    labels,
                    component_id,
                    (float(centroids[component_id][0]), float(centroids[component_id][1])),
                )
            )
    groups["small_component"] = _unique_yx_points(small_points)

    # group arrays shape: (N, 2), each row is (y, x)
    return groups


def find_branch_crop_candidates(
    mask: np.ndarray,
    min_component_area: int = 1,
    small_component_area: int = 50,
) -> np.ndarray:
    points: List[Tuple[int, int]] = []
    groups = find_branch_crop_candidate_groups(
        mask,
        min_component_area=min_component_area,
        small_component_area=small_component_area,
    )
    for name in ("endpoint", "branch", "thin", "small_component"):
        points.extend((int(y), int(x)) for y, x in groups[name])

    # candidates shape: (N, 2), each row is (y, x)
    return _unique_yx_points(points)


def _crop_bounds(
    shape: Tuple[int, int],
    center_yx: Tuple[int, int],
    crop_ratio: float,
) -> Tuple[int, int, int, int]:
    h, w = shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image/mask spatial shape: {(h, w)}")

    crop_ratio = float(np.clip(crop_ratio, 0.05, 1.0))
    crop_h = max(16, int(round(h * crop_ratio)))
    crop_w = max(16, int(round(w * crop_ratio)))
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)

    center_y, center_x = int(center_yx[0]), int(center_yx[1])
    top = int(np.clip(center_y - crop_h // 2, 0, h - crop_h))
    left = int(np.clip(center_x - crop_w // 2, 0, w - crop_w))
    bottom = top + crop_h
    right = left + crop_w
    return top, bottom, left, right


def crop_resize_around_center(
    img: np.ndarray,
    mask: np.ndarray,
    center_yx: Tuple[int, int],
    crop_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape[:2]
    # img shape: (H, W, C), mask shape: (H, W)
    top, bottom, left, right = _crop_bounds((h, w), center_yx, crop_ratio)

    img_crop = img[top:bottom, left:right]
    mask_crop = mask[top:bottom, left:right]
    # crop shapes: image (crop_h, crop_w, C), mask (crop_h, crop_w)

    resized_img = cv2.resize(img_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)
    # output shapes: image (H, W, C), mask (H, W)
    return resized_img.astype(img.dtype, copy=False), (resized_mask > 0).astype(np.uint8)


def make_tubed_skeleton(mask: np.ndarray, tube_radius: int = 1) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    # binary shape: (H, W)
    if binary.sum() == 0:
        return np.zeros_like(binary, dtype=np.uint8)

    skeleton = np.zeros_like(binary, dtype=np.uint8)
    current = (binary * 255).copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(current) > 0:
        eroded = cv2.erode(current, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(current, opened))
        current = eroded

    skeleton = (skeleton > 0).astype(np.uint8)
    if tube_radius > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        skeleton = cv2.dilate(skeleton, kernel, iterations=int(tube_radius))
        skeleton = ((skeleton > 0) & (binary > 0)).astype(np.uint8)
    # skeleton shape: (H, W), values: {0, 1}
    return skeleton


def make_distance_weighted_skeleton(
    mask: np.ndarray,
    tube_radius: int = 3,
    distance_alpha: float = 2.0,
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    # binary shape: (H, W)
    if binary.sum() == 0:
        return np.zeros_like(binary, dtype=np.float32)

    center_skeleton = make_tubed_skeleton(binary, tube_radius=0)
    # center_skeleton shape: (H, W), values: {0, 1}
    if center_skeleton.sum() == 0:
        return np.zeros_like(binary, dtype=np.float32)

    radius = max(int(tube_radius), 1)
    kernel = np.ones((3, 3), dtype=np.uint8)
    tube = cv2.dilate(center_skeleton, kernel, iterations=radius)
    tube = ((tube > 0) & (binary > 0)).astype(np.uint8)
    # tube shape: (H, W), values: {0, 1}

    distance_input = (center_skeleton == 0).astype(np.uint8)
    distance_to_skeleton = cv2.distanceTransform(distance_input, cv2.DIST_L2, 3)
    normalized_distance = np.clip(distance_to_skeleton / float(radius), 0.0, 1.0)
    # normalized_distance shape: (H, W), range: [0, 1]

    weights = tube.astype(np.float32) * (
        1.0 + float(distance_alpha) * normalized_distance.astype(np.float32)
    )
    # weights shape: (H, W), center skeleton weight 1, tube edge weight up to 1 + alpha
    return weights.astype(np.float32, copy=False)


def apply_branch_aware_crop(
    img: np.ndarray,
    mask: np.ndarray,
    crop_prob: float = 0.3,
    crop_ratio: float = 0.5,
    min_component_area: int = 1,
    small_component_area: int = 50,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    # img shape: (H, W, C), mask shape: (H, W)
    if crop_prob <= 0.0 or random.random() >= crop_prob:
        return img, mask, False

    candidates = find_branch_crop_candidates(
        mask,
        min_component_area=min_component_area,
        small_component_area=small_component_area,
    )
    if candidates.shape[0] == 0:
        return img, mask, False

    selected_index = random.randrange(candidates.shape[0])
    center_yx = (int(candidates[selected_index, 0]), int(candidates[selected_index, 1]))
    cropped_img, cropped_mask = crop_resize_around_center(
        img,
        mask,
        center_yx=center_yx,
        crop_ratio=crop_ratio,
    )
    # cropped_img shape: (H, W, C), cropped_mask shape: (H, W)
    return cropped_img, cropped_mask, True


class XCAStaticAugmenter:
    """
    静态 XCA 在线增强。

    与动态增强保持相同思想：
    - 图像和 mask 使用同一组空间变换；
    - mask 使用最近邻插值；
    - 强度增强只作用于图像。
    """

    def __init__(
        self,
        p_hflip: float = 0.5,
        p_vflip: float = 0.0,
        max_rotate_degree: float = 8.0,
        max_translate_ratio: float = 0.04,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        p_intensity: float = 0.6,
        noise_std: float = 0.01,
        brightness_limit: float = 0.06,
        contrast_limit: float = 0.06,
    ):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scale_range = scale_range
        self.p_intensity = p_intensity
        self.noise_std = noise_std
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def _sample_spatial_params(self, h: int, w: int):
        do_hflip = random.random() < self.p_hflip
        do_vflip = random.random() < self.p_vflip

        angle = random.uniform(-self.max_rotate_degree, self.max_rotate_degree)
        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        tx = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * w
        ty = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * h

        center = (w / 2.0, h / 2.0)
        affine_mat = cv2.getRotationMatrix2D(center, angle, scale)
        affine_mat[0, 2] += tx
        affine_mat[1, 2] += ty

        return {
            "do_hflip": do_hflip,
            "do_vflip": do_vflip,
            "affine_mat": affine_mat,
        }

    def _sample_intensity_params(self):
        if random.random() >= self.p_intensity:
            return {
                "contrast": 1.0,
                "brightness": 0.0,
                "noise_std": 0.0,
            }

        contrast = 1.0 + random.uniform(-self.contrast_limit, self.contrast_limit)
        brightness = random.uniform(-self.brightness_limit, self.brightness_limit)

        return {
            "contrast": contrast,
            "brightness": brightness,
            "noise_std": self.noise_std,
        }

    def _apply_spatial_image(self, img: np.ndarray, params: Dict[str, Any]):
        h, w = img.shape[:2]
        out = img

        if params["do_hflip"]:
            out = cv2.flip(out, 1)

        if params["do_vflip"]:
            out = cv2.flip(out, 0)

        out = cv2.warpAffine(
            out,
            params["affine_mat"],
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        return out

    def _apply_spatial_mask(self, mask: np.ndarray, params: Dict[str, Any]):
        h, w = mask.shape[:2]
        out = mask

        if params["do_hflip"]:
            out = cv2.flip(out, 1)

        if params["do_vflip"]:
            out = cv2.flip(out, 0)

        out = cv2.warpAffine(
            out,
            params["affine_mat"],
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return (out > 0).astype(np.uint8)

    def _apply_intensity(self, img: np.ndarray, params: Dict[str, Any]):
        out = img.copy()
        out = out * params["contrast"] + params["brightness"]

        if params["noise_std"] > 0:
            noise = np.random.normal(
                0.0,
                params["noise_std"],
                size=out.shape,
            ).astype(np.float32)
            out = out + noise

        out = np.clip(out, 0.0, 1.0)

        return out

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        h, w = img.shape[:2]

        spatial_params = self._sample_spatial_params(h, w)
        intensity_params = self._sample_intensity_params()

        img = self._apply_spatial_image(img, spatial_params)
        mask = self._apply_spatial_mask(mask, spatial_params)
        img = self._apply_intensity(img, intensity_params)

        return img, mask


class XCAStaticDataset(Dataset):
    """
    静态单帧 XCA dataloader。

    输入结构:
        image_root/
        ├── v00/
        │   ├── 00025.jpg
        │   └── ...
        mask_root/
        ├── v00/
        │   ├── 00025.png
        │   └── ...

    单样本输出:
        image:      [3, H, W]
        mask:       [H, W]
        video_id:   str
        frame_name: str，例如 00025.jpg
        frame_stem: str，例如 00025
        rel_path:   str，例如 v00/00025.jpg
        save_name:  str，例如 00025.png
    """

    def __init__(
        self,
        image_root: str,
        mask_root: str,
        split_file: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = (512, 512),
        mode: str = "train",
        augment: bool = True,
        augment_params: Optional[Dict[str, Any]] = None,
        branch_crop_params: Optional[Dict[str, Any]] = None,
        skeleton_params: Optional[Dict[str, Any]] = None,
        require_mask: bool = True,
        repeat_factor: int = 1,
        dataset_name: str = "dataset",
    ):
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.split_file = split_file
        self.image_size = image_size
        self.mode = mode
        self.augment = augment and (mode == "train")
        self.require_mask = require_mask
        self.repeat_factor = max(1, int(repeat_factor))
        self.dataset_name = str(dataset_name)

        if not self.image_root.exists():
            raise FileNotFoundError(f"image_root not found: {self.image_root}")

        if not self.mask_root.exists():
            raise FileNotFoundError(f"mask_root not found: {self.mask_root}")

        if augment_params is None:
            augment_params = {}

        self.augmenter = XCAStaticAugmenter(**augment_params)
        if branch_crop_params is None:
            branch_crop_params = {}
        self.branch_crop_enabled = bool(branch_crop_params.get("enabled", False)) and (
            mode == "train"
        )
        self.branch_crop_prob = float(branch_crop_params.get("prob", 0.3))
        self.branch_crop_ratio = float(branch_crop_params.get("ratio", 0.5))
        self.branch_crop_min_component_area = int(
            branch_crop_params.get("min_component_area", 1)
        )
        self.branch_crop_small_component_area = int(
            branch_crop_params.get("small_component_area", 50)
        )
        if skeleton_params is None:
            skeleton_params = {}
        self.skeleton_enabled = bool(skeleton_params.get("enabled", False))
        self.skeleton_tube_radius = int(skeleton_params.get("tube_radius", 1))
        self.skeleton_loss_mode = str(skeleton_params.get("mode", "center"))
        if self.skeleton_loss_mode not in {"center", "distance"}:
            raise ValueError(
                "skeleton_params['mode'] must be 'center' or 'distance', "
                f"got {self.skeleton_loss_mode!r}"
            )
        self.skeleton_distance_alpha = float(skeleton_params.get("distance_alpha", 2.0))

        self.allowed_video_ids, self.allowed_frame_keys = read_static_split_file(split_file)
        self.samples = self._scan_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                "No valid static samples found. Please check image_root, mask_root, split_file."
            )

        print(f"[XCAStaticDataset] mode={mode}")
        print(f"  dataset_name={self.dataset_name}")
        print(f"  image_root={self.image_root}")
        print(f"  mask_root={self.mask_root}")
        print(f"  split_file={self.split_file}")
        print(f"  samples={len(self.samples)}")
        print(f"  repeat_factor={self.repeat_factor}")
        print(f"  effective_length={len(self)}")
        print(f"  image_size={self.image_size}")
        print(f"  augment={self.augment}")
        print(f"  branch_crop_enabled={self.branch_crop_enabled}")
        if self.branch_crop_enabled:
            print(f"  branch_crop_prob={self.branch_crop_prob}")
            print(f"  branch_crop_ratio={self.branch_crop_ratio}")
            print(f"  branch_crop_small_component_area={self.branch_crop_small_component_area}")
        print(f"  skeleton_enabled={self.skeleton_enabled}")
        if self.skeleton_enabled:
            print(f"  skeleton_tube_radius={self.skeleton_tube_radius}")
            print(f"  skeleton_loss_mode={self.skeleton_loss_mode}")
            if self.skeleton_loss_mode == "distance":
                print(f"  skeleton_distance_alpha={self.skeleton_distance_alpha}")
        print(f"  require_mask={self.require_mask}")

    def _find_mask_path(self, relative_image_path: Path):
        mask_dir = self.mask_root / relative_image_path.parent
        stem = relative_image_path.stem

        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            candidate = mask_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate

        return None

    def _is_allowed_by_split(self, video_id: str, img_path: Path):
        if self.allowed_video_ids is None and self.allowed_frame_keys is None:
            return True

        frame_key = f"{video_id}/{img_path.stem}"

        allowed_by_video = (
            self.allowed_video_ids is not None
            and video_id in self.allowed_video_ids
        )

        allowed_by_frame = (
            self.allowed_frame_keys is not None
            and frame_key in self.allowed_frame_keys
        )

        return allowed_by_video or allowed_by_frame

    def _scan_samples(self):
        samples: List[Dict[str, Any]] = []

        img_paths = [
            path
            for path in self.image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]
        img_paths = sorted(
            img_paths,
            key=lambda path: path.relative_to(self.image_root).as_posix().lower(),
        )

        for img_path in img_paths:
            relative_image_path = img_path.relative_to(self.image_root)
            video_id = (
                relative_image_path.parts[0]
                if len(relative_image_path.parts) > 1
                else self.dataset_name
            )
            if not self._is_allowed_by_split(video_id, relative_image_path):
                continue

            mask_path = self._find_mask_path(relative_image_path)
            if self.require_mask and mask_path is None:
                continue

            mask_rel_path = (
                mask_path.relative_to(self.mask_root).as_posix()
                if mask_path is not None
                else None
            )
            output_rel_path = (
                mask_rel_path
                if mask_rel_path is not None
                else relative_image_path.with_suffix(".png").as_posix()
            )
            samples.append({
                "dataset_name": self.dataset_name,
                "video_id": video_id,
                "img_path": img_path,
                "mask_path": mask_path,
                "frame_name": img_path.name,
                "frame_stem": img_path.stem,
                "rel_path": relative_image_path.as_posix(),
                "mask_rel_path": mask_rel_path,
                "output_rel_path": output_rel_path,
                "save_name": Path(output_rel_path).name,
            })

        return samples

    def __len__(self):
        return len(self.samples) * self.repeat_factor

    def __getitem__(self, index):
        index = index % len(self.samples)
        sample = self.samples[index]

        img = read_image_rgb(
            sample["img_path"],
            image_size=self.image_size,
        )

        if sample["mask_path"] is not None:
            mask = read_binary_vessel_mask(
                sample["mask_path"],
                image_size=self.image_size,
            )
        else:
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

        if self.branch_crop_enabled:
            img, mask, _ = apply_branch_aware_crop(
                img,
                mask,
                crop_prob=self.branch_crop_prob,
                crop_ratio=self.branch_crop_ratio,
                min_component_area=self.branch_crop_min_component_area,
                small_component_area=self.branch_crop_small_component_area,
            )

        if self.augment:
            img, mask = self.augmenter(img, mask)

        skeleton = None
        if self.skeleton_enabled:
            if self.skeleton_loss_mode == "distance":
                skeleton = make_distance_weighted_skeleton(
                    mask,
                    tube_radius=self.skeleton_tube_radius,
                    distance_alpha=self.skeleton_distance_alpha,
                )
                # skeleton shape: (H, W), dtype: float32, continuous target weights
            else:
                skeleton = make_tubed_skeleton(mask, tube_radius=self.skeleton_tube_radius)
                # skeleton shape: (H, W), dtype: uint8, values: {0, 1}

        img = np.transpose(img, (2, 0, 1))  # [3, H, W]

        image_tensor = torch.from_numpy(img).float()
        mask_tensor = torch.from_numpy(mask).long()

        sample_dict = {
            "image": image_tensor,
            "mask": mask_tensor,
            "dataset_name": sample["dataset_name"],
            "video_id": sample["video_id"],
            "frame_name": sample["frame_name"],
            "frame_stem": sample["frame_stem"],
            "rel_path": sample["rel_path"],
            "mask_rel_path": sample["mask_rel_path"],
            "output_rel_path": sample["output_rel_path"],
            "save_name": sample["save_name"],
            "img_path": str(sample["img_path"]),
            "mask_path": str(sample["mask_path"]) if sample["mask_path"] is not None else None,
        }
        if skeleton is not None:
            sample_dict["skeleton"] = torch.from_numpy(skeleton).float()
        return sample_dict


class MultiXCAStaticDataset(Dataset):
    """Concatenate named XCA datasets without changing native sample paths."""

    def __init__(
        self,
        datasets: Sequence[XCAStaticDataset],
        repeat_factor: int = 1,
    ) -> None:
        if len(datasets) == 0:
            raise ValueError("datasets must not be empty")
        self.datasets = list(datasets)
        self.repeat_factor = max(1, int(repeat_factor))
        self.cumulative_sizes: List[int] = []
        running_size = 0
        for dataset in self.datasets:
            running_size += len(dataset)
            self.cumulative_sizes.append(running_size)
        self.base_length = running_size

    def __len__(self) -> int:
        return self.base_length * self.repeat_factor

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        base_index = index % self.base_length
        dataset_index = bisect_right(self.cumulative_sizes, base_index)
        previous_size = 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
        return self.datasets[dataset_index][base_index - previous_size]


def _resolve_config_path(config_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_dir / path


def build_multi_dataset_from_config(
    config_path: Path,
    dataset_names: Sequence[str],
    split: str,
    image_size: Optional[Tuple[int, int]] = (512, 512),
    mode: str = "train",
    augment: bool = True,
    augment_params: Optional[Dict[str, Any]] = None,
    branch_crop_params: Optional[Dict[str, Any]] = None,
    skeleton_params: Optional[Dict[str, Any]] = None,
    require_mask: bool = True,
    repeat_factor: int = 1,
) -> MultiXCAStaticDataset:
    """Build any selected train/test combination from a shared JSON config."""
    if isinstance(dataset_names, str) or len(dataset_names) == 0:
        raise ValueError("dataset_names must be a non-empty sequence of names")
    if len(set(dataset_names)) != len(dataset_names):
        raise ValueError(f"dataset_names contains duplicates: {dataset_names}")

    config_path = Path(config_path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    config_dir = config_path.parent
    configured_datasets = config.get("datasets", {})

    datasets: List[XCAStaticDataset] = []
    for dataset_name in dataset_names:
        if dataset_name not in configured_datasets:
            raise KeyError(f"Unknown dataset {dataset_name!r} in {config_path}")
        dataset_config = configured_datasets[dataset_name]
        if split not in dataset_config:
            raise KeyError(
                f"Dataset {dataset_name!r} has no split {split!r} in {config_path}"
            )
        split_config = dataset_config[split]
        split_file_value = split_config.get("split_file")
        split_file = (
            str(_resolve_config_path(config_dir, split_file_value))
            if split_file_value is not None
            else None
        )
        datasets.append(
            XCAStaticDataset(
                image_root=str(
                    _resolve_config_path(config_dir, split_config["image_root"])
                ),
                mask_root=str(
                    _resolve_config_path(config_dir, split_config["mask_root"])
                ),
                split_file=split_file,
                image_size=image_size,
                mode=mode,
                augment=augment,
                augment_params=augment_params,
                branch_crop_params=branch_crop_params,
                skeleton_params=skeleton_params,
                require_mask=require_mask,
                repeat_factor=1,
                dataset_name=dataset_name,
            )
        )
    return MultiXCAStaticDataset(datasets, repeat_factor=repeat_factor)


def xca_static_collate_fn(batch):
    """
    静态 batch collate。

    输出:
        image: [B, 3, H, W]
        mask:  [B, H, W]
        其余 ID 字段为 list[str]。
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.stack([item["mask"] for item in batch], dim=0)

    batch_dict = {
        "image": images,
        "mask": masks,
        "dataset_name": [item["dataset_name"] for item in batch],
        "video_id": [item["video_id"] for item in batch],
        "frame_name": [item["frame_name"] for item in batch],
        "frame_stem": [item["frame_stem"] for item in batch],
        "rel_path": [item["rel_path"] for item in batch],
        "mask_rel_path": [item["mask_rel_path"] for item in batch],
        "output_rel_path": [item["output_rel_path"] for item in batch],
        "save_name": [item["save_name"] for item in batch],
        "img_path": [item["img_path"] for item in batch],
        "mask_path": [item["mask_path"] for item in batch],
    }
    if "skeleton" in batch[0]:
        batch_dict["skeleton"] = torch.stack([item["skeleton"] for item in batch], dim=0)
    return batch_dict
