"""RealSense point-cloud picking page for MotorStudio."""

from __future__ import annotations

import logging
import math
import json
import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from PyQt6.QtCore import QEvent, QSignalBlocker, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from MotorStudio.utils.i18n import tr
from MotorStudio.utils.lift_platform_defaults import load_lift_platform_defaults
from MotorStudio.utils.style import SCENE_COLORS
from MotorStudio.utils.theme_manager import ThemeManager
from MotorStudio.utils.tcp_offset_store import get_tcp_offset_path
from el_a3_sdk import (
    DEFAULT_LIFT_ACCELERATION,
    DEFAULT_LIFT_PULSES_PER_CM,
    DEFAULT_LIFT_SPEED_RPM,
)
from el_a3_sdk.protocol import DEFAULT_JOINT_LIMITS

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    from vtkmodules.vtkRenderingCore import vtkPointPicker

    HAS_PYVISTA = True
except Exception as _exc:
    pv = None
    QtInteractor = None
    vtkPointPicker = None
    HAS_PYVISTA = False
    logging.getLogger("MotorStudio.realsense_panel").warning(
        "pyvista / pyvistaqt 不可用，RealSense 点云页禁用: %s", _exc
    )


logger = logging.getLogger("MotorStudio.realsense_panel")


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()
        parent = self.parentWidget()
        while parent is not None and not isinstance(parent, QScrollArea):
            parent = parent.parentWidget()
        if parent is not None:
            QApplication.sendEvent(parent.viewport(), event)


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()
        parent = self.parentWidget()
        while parent is not None and not isinstance(parent, QScrollArea):
            parent = parent.parentWidget()
        if parent is not None:
            QApplication.sendEvent(parent.viewport(), event)


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()
        parent = self.parentWidget()
        while parent is not None and not isinstance(parent, QScrollArea):
            parent = parent.parentWidget()
        if parent is not None:
            QApplication.sendEvent(parent.viewport(), event)


M_TO_CM = 100.0
BOOK_PHOTOS_DIR = Path(__file__).resolve().parents[2] / "assets" / "book_photos"
DEFAULT_BOOK_TEMPLATE_PATH = BOOK_PHOTOS_DIR / "net2.png"
BOOK_TEMPLATE_OPTIONS = [
    ("net2.png", BOOK_PHOTOS_DIR / "net2.png"),
    ("test2.jpeg", BOOK_PHOTOS_DIR / "test2.jpeg"),
    ("test3.jpeg", BOOK_PHOTOS_DIR / "test3.jpeg"),
    ("test4.jpeg", BOOK_PHOTOS_DIR / "test4.jpeg"),
]
WORKFLOW_DEFAULTS_CONFIG_VERSION = 1
WORKFLOW_DEFAULTS_PATH = get_tcp_offset_path().with_name(
    "motorstudio_book_workflow_defaults.json"
)
BOOK_SPINE_INFERENCE_DIR = (
    Path(__file__).resolve().parents[2] / "book_spine_inference_light"
)
BOOK_SPINE_INFER_PATH = BOOK_SPINE_INFERENCE_DIR / "infer.py"
BOOK_SPINE_WEIGHTS_PATH = BOOK_SPINE_INFERENCE_DIR / "weights" / "best.pt"
BOOK_SPINE_SEGMENT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "recordings"
    / "realsense"
    / "book_spine_segmentation"
)
BOOK_SPINE_SEGMENT_IMGSZ = 640
BOOK_SPINE_SEGMENT_CONF = 0.60
BOOK_SPINE_SEGMENT_IOU = 0.6
BOOK_SPINE_SEGMENT_MAX_DET = 30
BOOK_SPINE_SEGMENT_RETINA_MASKS = False
BOOK_SPINE_SEGMENT_SAVE_CROPS = False
BOOK_SPINE_SEGMENT_DEVICE = "cpu"


def _book_spine_thread_count() -> int:
    try:
        value = int(os.environ.get("MOTORSTUDIO_BOOK_SPINE_THREADS", "2") or 2)
    except (TypeError, ValueError):
        value = 2
    return max(1, min(2, value))


BOOK_SPINE_SEGMENT_THREADS = _book_spine_thread_count()
BOOK_SPINE_SEGMENT_CAPTURE_WIDTH = 640
BOOK_SPINE_SEGMENT_CAPTURE_HEIGHT = 480
BOOK_SPINE_SEGMENT_CAPTURE_DEPTH_WIDTH = 640
BOOK_SPINE_SEGMENT_CAPTURE_DEPTH_HEIGHT = 480
BOOK_SPINE_SEGMENT_WARMUP = 8
for _thread_env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_env_name, str(BOOK_SPINE_SEGMENT_THREADS))
_BOOK_SPINE_SEGMENT_MODEL = None
_BOOK_SPINE_SEGMENT_MODEL_PATH: Optional[Path] = None


def _load_book_workflow_defaults(mode: str) -> dict:
    try:
        with WORKFLOW_DEFAULTS_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    values = payload.get(mode, {})
    return values if isinstance(values, dict) else {}


def load_book_workflow_defaults(mode: str) -> dict:
    return _load_book_workflow_defaults(mode)


def _save_book_workflow_defaults(mode: str, values: dict):
    payload = {}
    try:
        with WORKFLOW_DEFAULTS_PATH.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        if isinstance(loaded, dict):
            payload = loaded
    except Exception:
        payload = {}
    payload["version"] = WORKFLOW_DEFAULTS_CONFIG_VERSION
    payload[mode] = values
    WORKFLOW_DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with WORKFLOW_DEFAULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_book_workflow_defaults(mode: str, values: dict):
    _save_book_workflow_defaults(mode, values)


def load_book_debug_pose_deg(
    mode: str,
    fallback: Sequence[float] = (0.0, 35.0, -45.0, 0.0, 0.0, 0.0),
) -> list[float]:
    defaults = _load_book_workflow_defaults(mode)
    pose = list(float(v) for v in fallback[:6])
    for idx in range(6):
        key = f"debug_joint_{idx + 1}"
        if key not in defaults:
            continue
        try:
            pose[idx] = float(defaults[key])
        except (TypeError, ValueError):
            pass
    return pose


def save_book_debug_pose_deg(mode: str, joints_deg: Sequence[float]) -> None:
    defaults = _load_book_workflow_defaults(mode)
    for idx, value in enumerate(list(joints_deg)[:6]):
        defaults[f"debug_joint_{idx + 1}"] = float(value)
    _save_book_workflow_defaults(mode, defaults)


BOOK_DEBUG_POSE_WORKFLOW_MODES = ("takeout", "putback", "putback2", "tail_putback")
BOOK_GRIPPER_WORKFLOW_MODE = "book_gripper"
BOOK_GRIPPER_DEFAULT_KEYS = (
    "gripper_open",
    "gripper_close",
    "gripper_effort",
    "gripper_start_effort",
    "gripper_start_boost",
    "gripper_close_speed",
    "gripper_open_speed",
    "gripper_kp",
    "gripper_kd",
    "gripper_timeout",
    "gripper_target_tolerance",
    "gripper_stall_tolerance",
    "gripper_stall_time",
    "gripper_min_monitor",
    "gripper_hold_margin",
    "gripper_command_lead",
    "gripper_stall_lead_threshold",
    "gripper_step_interval",
)


def _load_book_spine_infer_module():
    if not BOOK_SPINE_INFER_PATH.exists():
        raise FileNotFoundError(f"未找到书脊分割脚本: {BOOK_SPINE_INFER_PATH}")
    module_name = "_motorstudio_book_spine_inference_light"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(
        module_name,
        BOOK_SPINE_INFER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载书脊分割脚本: {BOOK_SPINE_INFER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_book_spine_segment_model(weights_path: Path = BOOK_SPINE_WEIGHTS_PATH):
    global _BOOK_SPINE_SEGMENT_MODEL, _BOOK_SPINE_SEGMENT_MODEL_PATH
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"未找到书脊分割模型权重: {weights_path}")
    if _BOOK_SPINE_SEGMENT_MODEL is not None and _BOOK_SPINE_SEGMENT_MODEL_PATH == weights_path:
        return _BOOK_SPINE_SEGMENT_MODEL
    from ultralytics import YOLO
    try:
        import torch

        torch.set_num_threads(BOOK_SPINE_SEGMENT_THREADS)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    if cv2 is not None:
        try:
            cv2.setNumThreads(BOOK_SPINE_SEGMENT_THREADS)
        except Exception:
            pass

    _BOOK_SPINE_SEGMENT_MODEL = YOLO(str(weights_path))
    try:
        _BOOK_SPINE_SEGMENT_MODEL.to(BOOK_SPINE_SEGMENT_DEVICE)
    except Exception:
        pass
    _BOOK_SPINE_SEGMENT_MODEL_PATH = weights_path
    return _BOOK_SPINE_SEGMENT_MODEL


# Camera -> robot base extrinsic:
# camera +X -> robot +Y
# camera +Y -> robot -Z
# camera +Z -> robot -X
# camera origin in robot base: x=+5 cm, y=-28.5 cm, z=+15 cm.
CAMERA_TO_ROBOT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "config"
    / "camera_to_robot_transform.json"
)
DEFAULT_CAMERA_TO_ROBOT_MATRIX = np.array(
    [
        [0.0, 0.0, -1.0, 0.05],
        [1.0, 0.0, 0.0, -0.285],
        [0.0, -1.0, 0.0, 0.15],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def _validated_array(values, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numbers")
    return array


def _load_camera_to_robot_transform(
    path: Path = CAMERA_TO_ROBOT_CONFIG_PATH,
) -> np.ndarray:
    """Load camera->robot extrinsic from JSON, falling back to built-in values."""

    matrix = DEFAULT_CAMERA_TO_ROBOT_MATRIX.copy()
    if not path.exists():
        return matrix

    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError("JSON root must be an object")

        if "matrix" in payload:
            matrix = _validated_array(payload["matrix"], (4, 4), "matrix")
        else:
            rotation_values = payload.get(
                "rotation",
                payload.get("rotation_matrix", matrix[:3, :3]),
            )
            rotation = _validated_array(rotation_values, (3, 3), "rotation")

            translation = matrix[:3, 3]
            if "translation_m" in payload:
                translation = _validated_array(
                    payload["translation_m"],
                    (3,),
                    "translation_m",
                )
            elif "translation_cm" in payload:
                translation = (
                    _validated_array(payload["translation_cm"], (3,), "translation_cm")
                    / M_TO_CM
                )
            elif "xyz_m" in payload:
                translation = _validated_array(payload["xyz_m"], (3,), "xyz_m")

            matrix = np.eye(4, dtype=float)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = translation

        return matrix.astype(float, copy=True)
    except Exception as exc:
        logger.warning(
            "加载相机到机械臂外参失败，使用内置默认值: %s (%s)",
            path,
            exc,
        )
        return matrix


CAMERA_TO_ROBOT_MATRIX = _load_camera_to_robot_transform()
ROBOT_TO_CAMERA_MATRIX = np.linalg.inv(CAMERA_TO_ROBOT_MATRIX)
CAMERA_TO_ROBOT_ROTATION = CAMERA_TO_ROBOT_MATRIX[:3, :3]
CAMERA_TO_ROBOT_TRANSLATION_M = CAMERA_TO_ROBOT_MATRIX[:3, 3]


def _rpy_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cr, sr = math.cos(rx), math.sin(rx)
    cp, sp = math.cos(ry), math.sin(ry)
    cy, sy = math.cos(rz), math.sin(rz)
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=float,
    )
    rotation_y = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=float,
    )
    rotation_z = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rotation_z @ rotation_y @ rotation_x


def camera_point_to_robot_target(
    camera_point_m: Sequence[float],
) -> np.ndarray:
    """Map a RealSense camera-space point into robot base coordinates."""

    camera_point = np.asarray(camera_point_m, dtype=float).reshape(3)
    camera_point_h = np.ones(4, dtype=float)
    camera_point_h[:3] = camera_point
    return (CAMERA_TO_ROBOT_MATRIX @ camera_point_h)[:3]


def robot_point_to_camera_target(
    robot_point_m: Sequence[float],
) -> np.ndarray:
    """Map a robot base point into RealSense camera-space coordinates."""

    robot_point = np.asarray(robot_point_m, dtype=float).reshape(3)
    robot_point_h = np.ones(4, dtype=float)
    robot_point_h[:3] = robot_point
    return (ROBOT_TO_CAMERA_MATRIX @ robot_point_h)[:3]


@dataclass(frozen=True)
class BookGapResult:
    left_index: int
    right_index: int
    gap_width_px: float
    line_top_uv: tuple[int, int]
    line_bottom_uv: tuple[int, int]
    midpoint_uv: tuple[int, int]
    camera_point_m: np.ndarray
    robot_point_m: np.ndarray


def _format_vec_m(values: Sequence[float]) -> str:
    vals = np.asarray(values, dtype=float).reshape(3)
    return f"X={vals[0]:.4f} m, Y={vals[1]:.4f} m, Z={vals[2]:.4f} m"


def _format_vec_cm(values: Sequence[float]) -> str:
    vals = np.asarray(values, dtype=float).reshape(3) * M_TO_CM
    return f"X={vals[0]:.2f} cm, Y={vals[1]:.2f} cm, Z={vals[2]:.2f} cm"


def _crop_polygon(crop) -> Optional[np.ndarray]:
    polygon = getattr(crop, "polygon_xy", None)
    if polygon is not None:
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
        if len(points) >= 3:
            return points
    bbox = getattr(crop, "bbox_xyxy", None)
    if bbox is None:
        return None
    x1, y1, x2, y2 = np.asarray(bbox, dtype=float).reshape(4)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=float,
    )


def _polygon_x_range_at_y(polygon: np.ndarray, y: float) -> Optional[tuple[float, float]]:
    points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(points) < 3:
        return None
    intersections: list[float] = []
    for start, end in zip(points, np.roll(points, -1, axis=0), strict=False):
        x1, y1 = float(start[0]), float(start[1])
        x2, y2 = float(end[0]), float(end[1])
        if math.isclose(y1, y2):
            continue
        if (y1 <= y < y2) or (y2 <= y < y1):
            t = (float(y) - y1) / (y2 - y1)
            intersections.append(x1 + t * (x2 - x1))
    if len(intersections) < 2:
        return None
    intersections.sort()
    return float(intersections[0]), float(intersections[-1])


def _find_largest_book_gap(
    crops: Sequence[object],
    frame,
    *,
    max_depth_m: float,
    min_overlap_px: float = 20.0,
    samples_per_gap: int = 15,
) -> BookGapResult:
    spines: list[dict[str, object]] = []
    for crop in crops:
        polygon = _crop_polygon(crop)
        if polygon is None:
            continue
        min_xy = polygon.min(axis=0)
        max_xy = polygon.max(axis=0)
        width = float(max_xy[0] - min_xy[0])
        height = float(max_xy[1] - min_xy[1])
        if width <= 1.0 or height <= 1.0:
            continue
        spines.append(
            {
                "index": int(getattr(crop, "index", len(spines) + 1)),
                "polygon": polygon,
                "center_x": float((min_xy[0] + max_xy[0]) * 0.5),
                "y_min": float(min_xy[1]),
                "y_max": float(max_xy[1]),
            }
        )
    spines.sort(key=lambda item: float(item["center_x"]))
    if len(spines) < 2:
        raise ValueError("书脊数量少于 2，无法计算书缝。")

    best: Optional[dict[str, object]] = None
    for left, right in zip(spines, spines[1:], strict=False):
        overlap_top = max(float(left["y_min"]), float(right["y_min"]))
        overlap_bottom = min(float(left["y_max"]), float(right["y_max"]))
        if overlap_bottom - overlap_top < min_overlap_px:
            continue
        y_values = np.linspace(overlap_top, overlap_bottom, max(3, int(samples_per_gap)))
        line_points: list[tuple[float, float]] = []
        widths: list[float] = []
        for y in y_values:
            left_range = _polygon_x_range_at_y(np.asarray(left["polygon"]), float(y))
            right_range = _polygon_x_range_at_y(np.asarray(right["polygon"]), float(y))
            if left_range is None or right_range is None:
                continue
            left_edge = float(left_range[1])
            right_edge = float(right_range[0])
            gap_width = right_edge - left_edge
            if gap_width <= 0.0:
                continue
            widths.append(gap_width)
            line_points.append(((left_edge + right_edge) * 0.5, float(y)))
        if len(line_points) < 2:
            continue
        avg_width = float(np.mean(widths))
        if best is None or avg_width > float(best["gap_width_px"]):
            best = {
                "left_index": int(left["index"]),
                "right_index": int(right["index"]),
                "gap_width_px": avg_width,
                "line_points": line_points,
            }
    if best is None:
        raise ValueError("未找到有效的相邻书缝。")

    line_points = list(best["line_points"])
    top_x, top_y = line_points[0]
    bottom_x, bottom_y = line_points[-1]
    mid_x = float(np.mean([point[0] for point in line_points]))
    mid_y = float(np.mean([point[1] for point in line_points]))
    height, width = frame.depth_m.shape

    def clamp_uv(x: float, y: float) -> tuple[int, int]:
        u = max(0, min(width - 1, int(round(x))))
        v = max(0, min(height - 1, int(round(y))))
        return u, v

    midpoint_uv = clamp_uv(mid_x, mid_y)
    camera_point = np.asarray(
        frame.point_at_pixel(
            midpoint_uv[0],
            midpoint_uv[1],
            search_radius=12,
            max_depth_m=max_depth_m,
        ),
        dtype=float,
    )
    robot_point = camera_point_to_robot_target(camera_point)
    return BookGapResult(
        left_index=int(best["left_index"]),
        right_index=int(best["right_index"]),
        gap_width_px=float(best["gap_width_px"]),
        line_top_uv=clamp_uv(top_x, top_y),
        line_bottom_uv=clamp_uv(bottom_x, bottom_y),
        midpoint_uv=midpoint_uv,
        camera_point_m=camera_point,
        robot_point_m=robot_point,
    )


def _normalize_book_workflow_mode(workflow_mode: str) -> str:
    mode = str(workflow_mode).lower()
    if mode in {"image_recognition", "image", "vision", "spine_segmentation", "book_spine_segmentation"}:
        return "image_recognition"
    if mode in {"putback2", "putback_2", "return2", "book_putback2"}:
        return "putback2"
    if mode in {"putback", "return", "put_back"}:
        return "putback"
    if mode in {"tail_putback", "tail_return", "end_putback", "book_tail_putback"}:
        return "tail_putback"
    return "takeout"


def _is_putback_workflow_mode(workflow_mode: str) -> bool:
    return _normalize_book_workflow_mode(workflow_mode) in {"putback", "tail_putback"}


def _is_takeout_based_workflow_mode(workflow_mode: str) -> bool:
    return _normalize_book_workflow_mode(workflow_mode) in {"takeout", "putback2"}


def _is_image_recognition_workflow_mode(workflow_mode: str) -> bool:
    return _normalize_book_workflow_mode(workflow_mode) == "image_recognition"


def _book_pick_point_from_polygon(
    polygon: np.ndarray,
    workflow_mode: str = "takeout",
    putback_target_line: str = "right",
    putback_right_edge_ratio: float = 0.75,
) -> tuple[int, int]:
    points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(points) < 4:
        raise ValueError("书脊识别结果缺少四边形角点。")
    tl, tr, br, bl = points[:4]
    if _normalize_book_workflow_mode(workflow_mode) == "tail_putback":
        return int(round(float(br[0]))), int(round(float(br[1])))
    if _is_putback_workflow_mode(workflow_mode):
        line = str(putback_target_line).lower()
        if line == "left":
            top, bottom = tl, bl
        elif line == "middle":
            top = (tl + tr) * 0.5
            bottom = (bl + br) * 0.5
        else:
            top, bottom = tr, br
        ratio = max(0.0, min(1.0, float(putback_right_edge_ratio)))
        point = top * (1.0 - ratio) + bottom * ratio
        return int(round(float(point[0]))), int(round(float(point[1])))

    left = tl * (5.0 / 6.0) + bl * (1.0 / 6.0)
    right = tr * (5.0 / 6.0) + br * (1.0 / 6.0)
    point = (left + right) * 0.5
    return int(round(float(point[0]))), int(round(float(point[1])))


def _filter_depth_min(point_cloud, depth_min_m: float):
    if depth_min_m <= 0.0:
        return point_cloud

    points = np.asarray(point_cloud.points_xyz_m)
    keep = points[:, 2] >= depth_min_m
    return type(point_cloud)(
        points_xyz_m=points[keep],
        colors_rgb=(
            None
            if point_cloud.colors_rgb is None
            else np.asarray(point_cloud.colors_rgb)[keep]
        ),
        pixels_uv=np.asarray(point_cloud.pixels_uv)[keep],
        intrinsics=point_cloud.intrinsics,
        timestamp_ms=point_cloud.timestamp_ms,
        frame_number=point_cloud.frame_number,
    )


class RealSenseCaptureWorker(QThread):
    """Capture one RealSense RGB-D frame and convert it to a point cloud."""

    capture_finished = pyqtSignal(object, object)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        *,
        serial: Optional[str],
        width: int,
        height: int,
        fps: int,
        warmup: int,
        timeout_ms: int,
        align_depth_to_color: bool,
        depth_min_m: float,
        depth_max_m: float,
        stride: int,
        include_color: bool,
        depth_width: int,
        depth_height: int,
        parent=None,
    ):
        super().__init__(parent)
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup = warmup
        self.timeout_ms = timeout_ms
        self.align_depth_to_color = align_depth_to_color
        self.depth_min_m = depth_min_m
        self.depth_max_m = depth_max_m
        self.stride = stride
        self.include_color = include_color
        self.depth_width = depth_width
        self.depth_height = depth_height

    def run(self):
        try:
            from el_a3_sdk.realsense import RealSenseD435

            with RealSenseD435(
                width=self.width,
                height=self.height,
                fps=self.fps,
                serial=self.serial or None,
                align_depth_to_color=self.align_depth_to_color,
                depth_width=self.depth_width,
                depth_height=self.depth_height,
            ) as camera:
                if self.isInterruptionRequested():
                    return
                camera.warmup(frame_count=self.warmup, timeout_ms=self.timeout_ms)
                if self.isInterruptionRequested():
                    return
                frame = camera.get_frame(timeout_ms=self.timeout_ms)
                if self.isInterruptionRequested():
                    return

            point_cloud = frame.to_point_cloud(
                max_depth_m=self.depth_max_m,
                stride=self.stride,
                include_color=self.include_color,
            )
            point_cloud = _filter_depth_min(point_cloud, self.depth_min_m)
            if point_cloud.size == 0:
                raise RuntimeError("点云为空，请调整深度范围或相机视角。")
            self.capture_finished.emit(frame, point_cloud)
        except Exception as exc:
            self.error_occurred.emit(str(exc))


def _make_book_match_config():
    from el_a3_sdk.realsense import BookSpineMatchConfig

    return BookSpineMatchConfig(
        match_confidence=0.65,
        center_tolerance_ratio=0.03,
        min_center_tolerance_px=45,
        frame_max_side=0,
        template_max_side=1200,
        min_good_matches=12,
        min_inliers=8,
        sift_ratio_test=0.72,
        sift_features=4000,
        acquire_match_confidence=0.50,
        acquire_min_good_matches=8,
        acquire_min_inliers=5,
        acquire_tile_columns=3,
        acquire_tile_overlap_ratio=0.20,
        search_scales=(1.0, 1.5, 2.0),
        max_scaled_frame_side=1800,
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=8,
        roi_expand_ratio=0.45,
        roi_min_pad=60,
        roi_reacquire_after_misses=3,
        polygon_hold_frames=4,
        min_polygon_area_ratio=0.0002,
        max_polygon_area_ratio=0.75,
        min_polygon_fill_ratio=0.35,
        max_polygon_skew_ratio=3.0,
        max_polygon_jump_ratio=0.25,
        max_polygon_area_change_ratio=0.75,
        polygon_smoothing_alpha=0.25,
        keep_last_good_on_reject=True,
        center_smoothing_alpha=0.25,
        green_confirm_frames=2,
        red_confirm_frames=5,
    )


class BookSpineDetectWorker(QThread):
    """Keep capturing until the book spine is confidently detected."""

    detection_finished = pyqtSignal(object, object, object)
    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        *,
        serial: Optional[str],
        width: int,
        height: int,
        fps: int,
        warmup: int,
        timeout_ms: int,
        align_depth_to_color: bool,
        depth_min_m: float,
        depth_max_m: float,
        stride: int,
        include_color: bool,
        depth_width: int,
        depth_height: int,
        template_path: Path,
        config: object,
        parent=None,
    ):
        super().__init__(parent)
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup = warmup
        self.timeout_ms = timeout_ms
        self.align_depth_to_color = align_depth_to_color
        self.depth_min_m = depth_min_m
        self.depth_max_m = depth_max_m
        self.stride = stride
        self.include_color = include_color
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.template_path = template_path
        self.config = config

    def run(self):
        try:
            from el_a3_sdk.realsense import BookSpineMatcher, RealSenseD435, match_with_fallback

            with RealSenseD435(
                width=self.width,
                height=self.height,
                fps=self.fps,
                serial=self.serial or None,
                align_depth_to_color=self.align_depth_to_color,
                depth_width=self.depth_width,
                depth_height=self.depth_height,
            ) as camera:
                if self.isInterruptionRequested():
                    return
                camera.warmup(frame_count=self.warmup, timeout_ms=self.timeout_ms)
                matcher = BookSpineMatcher.from_template_path(self.template_path, self.config)

                while not self.isInterruptionRequested():
                    frame = camera.get_frame(timeout_ms=self.timeout_ms)
                    if self.isInterruptionRequested():
                        return

                    point_cloud = frame.to_point_cloud(
                        max_depth_m=self.depth_max_m,
                        stride=self.stride,
                        include_color=self.include_color,
                    )
                    point_cloud = _filter_depth_min(point_cloud, self.depth_min_m)
                    if point_cloud.size == 0:
                        self.status_message.emit("点云为空，继续重采集...")
                        continue

                    result = match_with_fallback(
                        frame.color_bgr,
                        matcher.backends,
                        frame_max_side=self.config.frame_max_side,
                        search_rect=None,
                        match_confidence=self.config.match_confidence,
                        search_scales=self.config.search_scales,
                        max_scaled_frame_side=self.config.max_scaled_frame_side,
                        use_clahe=self.config.use_clahe,
                        clahe_clip_limit=self.config.clahe_clip_limit,
                        clahe_tile_grid_size=self.config.clahe_tile_grid_size,
                        min_good_matches=self.config.min_good_matches,
                        min_inliers=self.config.min_inliers,
                        polygon_validator=lambda _polygon, _shape: True,
                    )
                    if result.polygon is None:
                        self.status_message.emit(
                            f"识别未达标，重采集中: score={result.match_confidence:.2f} "
                            f"good={result.good_count} inliers={result.inlier_count}"
                        )
                        continue

                    self.detection_finished.emit(frame, point_cloud, result)
                    return
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class BookSpineSegmentWorker(QThread):
    """Capture one color frame and run YOLO book-spine segmentation."""

    segmentation_finished = pyqtSignal(object)
    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        *,
        serial: Optional[str],
        width: int,
        height: int,
        fps: int,
        warmup: int,
        timeout_ms: int,
        align_depth_to_color: bool,
        depth_min_m: float,
        depth_max_m: float,
        stride: int,
        include_color: bool,
        depth_width: int,
        depth_height: int,
        parent=None,
    ):
        super().__init__(parent)
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup = warmup
        self.timeout_ms = timeout_ms
        self.align_depth_to_color = align_depth_to_color
        self.depth_min_m = depth_min_m
        self.depth_max_m = depth_max_m
        self.stride = stride
        self.include_color = include_color
        self.depth_width = depth_width
        self.depth_height = depth_height

    def run(self):
        try:
            from el_a3_sdk.realsense import RealSenseD435

            with RealSenseD435(
                width=self.width,
                height=self.height,
                fps=self.fps,
                serial=self.serial or None,
                align_depth_to_color=self.align_depth_to_color,
                depth_width=self.depth_width,
                depth_height=self.depth_height,
            ) as camera:
                if self.isInterruptionRequested():
                    return
                self.status_message.emit("正在预热并拍摄图像...")
                camera.warmup(frame_count=self.warmup, timeout_ms=self.timeout_ms)
                if self.isInterruptionRequested():
                    return
                frame = camera.get_frame(timeout_ms=self.timeout_ms)
                if self.isInterruptionRequested():
                    return

            self.status_message.emit("正在加载模型并分割书脊...")
            infer_module = _load_book_spine_infer_module()
            model = _load_book_spine_segment_model()
            self.status_message.emit(
                f"正在分割书脊(imgsz={BOOK_SPINE_SEGMENT_IMGSZ}, "
                f"max_det={BOOK_SPINE_SEGMENT_MAX_DET}, threads={BOOK_SPINE_SEGMENT_THREADS})..."
            )
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            capture_dir = BOOK_SPINE_SEGMENT_OUTPUT_DIR / "captures"
            capture_dir.mkdir(parents=True, exist_ok=True)
            image_path = capture_dir / f"book_spine_{timestamp}_frame{frame.frame_number}.jpg"
            if cv2 is None:
                raise RuntimeError("书脊分割需要 opencv-python。")
            cv2.imwrite(str(image_path), frame.color_bgr)
            overlay_path, crops = infer_module.segment_image(
                model=model,
                image_path=image_path,
                output_dir=BOOK_SPINE_SEGMENT_OUTPUT_DIR,
                imgsz=BOOK_SPINE_SEGMENT_IMGSZ,
                conf=BOOK_SPINE_SEGMENT_CONF,
                iou=BOOK_SPINE_SEGMENT_IOU,
                pad=8,
                retina_masks=BOOK_SPINE_SEGMENT_RETINA_MASKS,
                max_det=BOOK_SPINE_SEGMENT_MAX_DET,
                device=BOOK_SPINE_SEGMENT_DEVICE,
                save_crops=BOOK_SPINE_SEGMENT_SAVE_CROPS,
            )
            gap_result = None
            gap_error = ""
            try:
                gap_result = _find_largest_book_gap(
                    crops,
                    frame,
                    max_depth_m=self.depth_max_m,
                )
            except Exception as exc:
                gap_error = str(exc)
            payload = {
                "ok": True,
                "frame": frame,
                "image_path": image_path,
                "overlay_path": overlay_path,
                "spine_count": len(crops),
                "crops": crops,
                "gap_result": gap_result,
                "gap_error": gap_error,
            }
            self.segmentation_finished.emit(payload)
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class RealSensePointPanel(QWidget):
    """Point cloud capture, target picking, and MoveL confirmation page."""

    DEFAULT_SERIAL = None
    # D435-class devices commonly support aligned RGB-D at 1280x720@30.
    DEFAULT_WIDTH = 1280
    DEFAULT_HEIGHT = 720
    DEFAULT_FPS = 30
    DEFAULT_WARMUP = 30
    DEFAULT_TIMEOUT_MS = 5000
    DEFAULT_DEPTH_MIN_M = 0.0
    DEFAULT_DEPTH_MAX_M = 2.0
    DEFAULT_STRIDE = 1
    DEFAULT_MAX_POINTS = 150000
    DEFAULT_POINT_SIZE = 2.0
    DEFAULT_ALIGN_DEPTH_TO_COLOR = True
    DEFAULT_INCLUDE_COLOR = True
    DEFAULT_FLIP_VIEW = True
    DEFAULT_DEPTH_WIDTH = 1280
    DEFAULT_DEPTH_HEIGHT = 720
    DEFAULT_ROD_PORT = "/dev/rodmotor"
    DEFAULT_ROD_BAUD = 921600
    DEFAULT_ROD_TIMEOUT_S = 0.3
    DEFAULT_ROD_SPEED = 1000
    DEFAULT_ROD_ACC = 50
    DEFAULT_ROD_TORQUE = 1.0
    DEFAULT_TARGET_COMPENSATION_CM = (0.0, -1.0, -5.0)
    DEFAULT_PUTBACK2_TARGET_COMPENSATION_CM = (-1.0, 1.0, 1.0)
    DEFAULT_ROD_GRASP_DEG = 115.0
    DEFAULT_PUTBACK2_ROD_GRASP_DEG = -90.0
    DEFAULT_PREGRASP_OFFSET_CM = (10.0, 0.0, 0.0)
    DEFAULT_GRIPPER_OPEN_DEG = 0.0
    DEFAULT_GRIPPER_CLOSE_DEG = 108.5
    DEFAULT_GRIPPER_HOLD_EFFORT_NM = 0.25
    DEFAULT_GRIPPER_START_EFFORT_NM = 0.30
    DEFAULT_GRIPPER_START_BOOST_S = 0.4
    DEFAULT_GRIPPER_KP = 18.0
    DEFAULT_GRIPPER_KD = 2.0
    DEFAULT_GRIPPER_CLOSE_SPEED_DEG_S = 16.7
    DEFAULT_GRIPPER_OPEN_SPEED_DEG_S = 16.7
    DEFAULT_PUTBACK2_GRIPPER_CLOSE_SPEED_DEG_S = 40.0
    DEFAULT_GRASP_RPY_DEG = (75.0, 0.0, 90.0)
    DEFAULT_PUTBACK2_GRASP_RPY_DEG = (65.0, 0.0, 90.0)
    DEFAULT_FLOW_MOVEL_DURATION_S = 2.0
    GRIPPER_CLOSE_TOLERANCE_DEG = 2.0
    GRIPPER_CLOSE_STALL_TOLERANCE_DEG = 1.5
    GRIPPER_CLOSE_STALL_S = 2.5
    GRIPPER_CLOSE_MIN_MONITOR_S = 2.0
    GRIPPER_CLOSE_TIMEOUT_S = 10.0
    GRIPPER_CLOSE_COMMAND_LEAD_S = 0.25
    GRIPPER_CLOSE_STALL_LEAD_THRESHOLD_DEG = 8.0
    GRIPPER_CLOSE_STEP_DEG = 3.0
    GRIPPER_CLOSE_STEP_INTERVAL_S = 0.18
    DEFAULT_GRIPPER_HOLD_MARGIN_DEG = 0.5
    DEFAULT_PUTBACK_TARGET_RPY_DEG = (90.0, 0.0, 60.0)
    DEFAULT_PUTBACK_INSERT_RPY_DEG = (90.0, 0.0, 90.0)
    DEFAULT_PUTBACK_PREPUSH_OFFSET_CM = (10.0, 0.0, 0.0)
    DEFAULT_PUTBACK_PREPUSH_RPY_OFFSET_DEG = (0.0, 0.0, 0.0)
    DEFAULT_PUTBACK_PUSH_OFFSET_CM = (0.0, 0.0, 0.0)
    DEFAULT_PUTBACK_PUSH_RPY_OFFSET_DEG = (0.0, 0.0, 0.0)
    DEFAULT_PUTBACK_PUSH_OUT_OFFSET_CM = (0.0, 5.0, 0.0)
    DEFAULT_PUTBACK_PUSH_OUT_RPY_OFFSET_DEG = (0.0, 0.0, 0.0)
    DEFAULT_PUTBACK_LEAVE_PUSH_Y_OFFSET_CM = 10.0
    DEFAULT_PUTBACK_INSERT_PREPOSE_X_OFFSET_CM = 10.0
    DEFAULT_PUTBACK_INSERT_PREPOSE_Y_OFFSET_CM = 1.0
    DEFAULT_PUTBACK_INSERT_X_OFFSET_CM = -3.0
    DEFAULT_PUTBACK_INSERT_Y_OFFSET_CM = 1.0
    DEFAULT_PUTBACK_LEAVE_INSERT_X_OFFSET_CM = 10.0
    DEFAULT_PUTBACK_GRIPPER_OPEN_DEG = 30.0
    DEFAULT_DEBUG_JOINTS_DEG = (0.0, 35.0, -45.0, 0.0, 0.0, 0.0)
    DEFAULT_TURN_STAGE1_JOINTS_DEG = (-90.0, 35.0, -40.0, 0.0, 0.0, 0.0)
    LIFT_POSITION_LOWEST = "lowest"
    LIFT_POSITION_RETURN = "return"
    LIFT_POSITION_TAKE = "take"
    DEFAULT_TAKEOUT_LIFT_POSITION = LIFT_POSITION_TAKE
    DEFAULT_TARGET_RELATIVE_OFFSET_CM = (0.0, 0.0, 0.0)
    DEFAULT_TARGET_RELATIVE_RPY_DEG = (90.0, 0.0, 90.0)
    DEFAULT_TARGET_RELATIVE_DURATION_S = 2.0
    DEFAULT_POST_GRIPPER_MOVEJ_OFFSET_CM = (0.0, 0.0, 0.0)
    DEFAULT_POST_GRIPPER_MOVEJ_RPY_OFFSET_DEG = (0.0, 0.0, 0.0)
    DEFAULT_POST_GRIPPER_MOVEJ_DURATION_S = 2.0
    DEFAULT_TAIL_PLACE_OFFSET_CM = (0.0, 0.0, 0.0)
    DEFAULT_TAIL_PLACE_RPY_OFFSET_DEG = (0.0, 0.0, 0.0)
    DEFAULT_TAIL_FINE_TUNE_OFFSET_CM = (0.0, 0.0, 0.0)
    DEFAULT_TAIL_FINE_TUNE_RPY_OFFSET_DEG = (0.0, 0.0, 0.0)
    DEFAULT_TAIL_FINE_TUNE_MOVEL_DURATION_S = 1.0
    DEFAULT_TAIL_PREPLACE_MOTION_MODE = "movej"
    DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_DEG = 12.0
    DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_SPEED_DEG_S = 6.0
    DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_TIMEOUT_S = 4.0
    DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_TARGET_TOLERANCE_DEG = 1.5
    DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_STALL_TOLERANCE_DEG = 0.8
    DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_STALL_TIME_S = 1.2
    DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_MIN_MONITOR_S = 0.4
    DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_HOLD_MARGIN_DEG = 0.2
    DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_COMMAND_LEAD_S = 0.12
    DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_STALL_LEAD_THRESHOLD_DEG = 2.0
    DEFAULT_TAIL_BACKOFF_OFFSET_CM = (5.0, 0.0, 0.0)
    DEFAULT_TAIL_BACKOFF_RPY_OFFSET_DEG = (0.0, 0.0, 0.0)
    DEFAULT_TAIL_BACKOFF_MOVEL_DURATION_S = 1.0
    DEFAULT_TAIL_PREPUSH_MOTION_MODE = "movel"
    DEFAULT_TAIL_PREPUSH_OFFSET_CM = (5.0, 0.0, 0.0)
    DEFAULT_TAIL_PREPUSH_RPY_OFFSET_DEG = (0.0, 0.0, 0.0)
    DEFAULT_TAIL_PREPUSH_MOVEL_DURATION_S = 2.0
    DEFAULT_TAIL_PUSH_OFFSET_CM = (5.0, 0.0, 0.0)
    DEFAULT_TAIL_PUSH_RPY_OFFSET_DEG = (0.0, 0.0, 0.0)
    DEFAULT_TAIL_PUSH_MOVEL_DURATION_S = 2.0
    DEFAULT_TAIL_FINAL_HOME_MODE = "zero"
    DEFAULT_TAIL_FINAL_HOME_DURATION_S = 3.0
    DEFAULT_PUTBACK2_RESET_RPY_DEG = (90.0, 0.0, 90.0)
    DEFAULT_PUTBACK2_RESET_RPY_DURATION_S = 2.0
    DEFAULT_PUTBACK2_X_OFFSET_CM = -5.0
    DEFAULT_PUTBACK2_X_MOVE_DURATION_S = 2.0
    DEFAULT_PUTBACK2_GRIPPER_PARTIAL_OPEN_DEG = 90.0
    DEFAULT_PUTBACK2_GRIPPER_PARTIAL_OPEN_SPEED_DEG_S = 8.0
    DEFAULT_PUTBACK2_ROD_RELEASE_DEG = 90.0
    DEFAULT_PUTBACK2_Z_OFFSET_CM = -10.0
    DEFAULT_PUTBACK2_Z_MOVE_DURATION_S = 2.0
    DEFAULT_PUTBACK2_GRIPPER_FULL_OPEN_DEG = 0.0
    DEFAULT_PUTBACK2_GRIPPER_FULL_OPEN_SPEED_DEG_S = 8.0
    DEFAULT_PUTBACK2_AFTER_OPEN_X1_OFFSET_CM = 10.0
    DEFAULT_PUTBACK2_AFTER_OPEN_X1_DURATION_S = 2.0
    DEFAULT_PUTBACK2_COMBO_CLOSE_DEG = 108.5
    DEFAULT_PUTBACK2_COMBO_CLOSE_SPEED_DEG_S = 40.0
    DEFAULT_PUTBACK2_COMBO_Y_OFFSET_CM = 2.0
    DEFAULT_PUTBACK2_COMBO_Y_DURATION_S = 2.0
    DEFAULT_PUTBACK2_COMBO_WAIT_S = 2.0
    DEFAULT_PUTBACK2_COMBO_X_OFFSET_CM = -10.0
    DEFAULT_PUTBACK2_COMBO_X_DURATION_S = 2.0
    DEFAULT_PUTBACK2_COMBO_FULL_OPEN_DEG = 0.0
    DEFAULT_PUTBACK2_COMBO_FULL_OPEN_SPEED_DEG_S = 8.0
    DEFAULT_PUTBACK2_AFTER_COMBO_X_OFFSET_CM = 10.0
    DEFAULT_PUTBACK2_AFTER_COMBO_X_DURATION_S = 2.0
    DEFAULT_TAKEOUT_SPECIFIED_POSE_CM = (0.0, 0.0, 0.0)
    DEFAULT_TAKEOUT_SPECIFIED_POSE_RPY_DEG = (0.0, 0.0, 0.0)
    DEFAULT_TAKEOUT_SPECIFIED_POSE_DURATION_S = 3.0
    DEFAULT_TAKEOUT_STEP13_FINE_TUNE_OFFSET_CM = (0.0, 0.0, 0.0)
    DEFAULT_TAKEOUT_STEP13_FINE_TUNE_RPY_OFFSET_DEG = (0.0, 0.0, 0.0)
    DEFAULT_TAKEOUT_STEP13_FINE_TUNE_DURATION_S = 3.0
    DEFAULT_FINAL_JOINTS_DEG = (-119.64, 67.09, 67.22, 2.23, 51.75, -14.65)
    DEFAULT_TURN_DURATION_S = 6.0
    DEFAULT_DEBUG_MOVE_DURATION_S = 3.0
    HEADER_MOVE_DURATION_S = 2.5
    DEFAULT_FINAL_MOVE_DURATION_S = 8.0
    FLOW_BUTTON_HEIGHT = 26
    FLOW_BUTTON_WIDTH_EXTRA = 22

    move_l_requested = pyqtSignal(list, float)
    move_l_block_requested = pyqtSignal(list, float)
    move_j_block_requested = pyqtSignal(list, float)
    end_pose_block_requested = pyqtSignal(list, float)
    gripper_requested = pyqtSignal(float, float, float, float)
    gripper_close_monitor_requested = pyqtSignal(object)
    rod_connect_requested = pyqtSignal(str, int, float)
    rod_write_requested = pyqtSignal(float, int, int, float)
    lift_move_distance_requested = pyqtSignal(float, int, int, float, int)
    workflow_stop_requested = pyqtSignal()
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None, workflow_mode: str = "takeout"):
        super().__init__(parent)
        self._workflow_mode = _normalize_book_workflow_mode(workflow_mode)
        self._capture_worker: Optional[RealSenseCaptureWorker] = None
        self._detect_worker: Optional[BookSpineDetectWorker] = None
        self._segment_worker: Optional[BookSpineSegmentWorker] = None
        self._frame = None
        self._point_cloud = None
        self._display_indices: Optional[np.ndarray] = None
        self._display_points: Optional[np.ndarray] = None
        self._cloud_actor = None
        self._selected_actor = None
        self._picker = vtkPointPicker() if HAS_PYVISTA else None
        self._filter_installed = False
        self._selected_display_index: Optional[int] = None
        self._selected_display_point_m: Optional[np.ndarray] = None
        self._selected_raw_index: Optional[int] = None
        self._selected_camera_point_m: Optional[np.ndarray] = None
        self._selected_robot_target_raw_m: Optional[np.ndarray] = None
        self._target_robot_point_m: Optional[np.ndarray] = None
        self._tail_putback_bottom_point_m: Optional[np.ndarray] = None
        self._tail_putback_step4_target_point_m: Optional[np.ndarray] = None
        self._book_spine_pick: Optional[object] = None
        self._segmentation_result: Optional[dict] = None
        self._tcp_offset = np.zeros(6, dtype=float)
        self._header_zero_duration_s = float(self.HEADER_MOVE_DURATION_S)
        self._current_end_pose = None
        self._rpy_initialized = False
        self._arm_enabled = False
        self._rod_connected = False
        self._flow_active = False
        self._flow_auto_run = False
        self._flow_step_index = 0
        self._flow_waiting_motion = False
        self._flow_waiting_kind: Optional[str] = None
        self._flow_rollback_waiting = False
        self._flow_pose_history: list[tuple[int, str, list[float]]] = []
        self._flow_rollback_entry: Optional[tuple[int, str, list[float]]] = None
        self._flow_approach_pose: Optional[list[float]] = None
        self._flow_last_pose: Optional[list[float]] = None
        self._flow_pending_pose: Optional[list[float]] = None
        self._rod_current_angle_deg: Optional[float] = None
        self._rod_target_angle_deg: Optional[float] = None
        self._rod_wait_tolerance_deg = 1.5
        self._current_gripper_angle_deg: Optional[float] = None
        self._gripper_close_target_deg: Optional[float] = None
        self._gripper_close_last_cmd_deg: Optional[float] = None
        self._gripper_close_started_at_s = 0.0
        self._gripper_close_last_cmd_s = 0.0
        self._gripper_close_last_angle_deg: Optional[float] = None
        self._gripper_close_stable_since_s: Optional[float] = None
        self._gripper_close_effort_nm = 0.0
        self._manual_gripper_open_target_deg: Optional[float] = None
        self._workflow_default_edit_enabled = False
        self._workflow_default_controls: dict[str, QWidget] = {}
        self._workflow_default_edit_snapshot: dict[str, object] = {}
        self._workflow_defaults = _load_book_workflow_defaults(self._workflow_mode)
        self._loading_workflow_defaults = False
        self._takeout_step_buttons: list[QPushButton] = []
        self._viewer_visible = False
        self._viewer_widget = None
        self._serial = self.DEFAULT_SERIAL
        self._width = self.DEFAULT_WIDTH
        self._height = self.DEFAULT_HEIGHT
        self._depth_width = self.DEFAULT_DEPTH_WIDTH
        self._depth_height = self.DEFAULT_DEPTH_HEIGHT
        self._fps = self.DEFAULT_FPS
        self._warmup = self.DEFAULT_WARMUP
        self._timeout_ms = self.DEFAULT_TIMEOUT_MS
        self._depth_min_m = self.DEFAULT_DEPTH_MIN_M
        self._depth_max_m = self.DEFAULT_DEPTH_MAX_M
        self._stride = self.DEFAULT_STRIDE
        self._max_points = self.DEFAULT_MAX_POINTS
        self._point_size = self.DEFAULT_POINT_SIZE
        self._align_depth_to_color = self.DEFAULT_ALIGN_DEPTH_TO_COLOR
        self._include_color = self.DEFAULT_INCLUDE_COLOR
        self._flip_view = self.DEFAULT_FLIP_VIEW
        self._init_ui()

    def _default_debug_joints_deg(self) -> list[float]:
        return load_book_debug_pose_deg(self._workflow_mode, self.DEFAULT_DEBUG_JOINTS_DEG)

    def _init_ui(self):
        self._viewer_widget = self._create_viewer_widget()
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        if not self._is_image_recognition_workflow():
            root.addWidget(self._create_workflow_header())

        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        self._hidden_control_groups = [
            self._create_capture_group(),
            self._create_result_group(),
            self._create_move_group(),
            self._create_book_group(),
        ]
        for group in self._hidden_control_groups:
            group.hide()
        if self._is_image_recognition_workflow():
            controls_layout.addWidget(self._create_image_recognition_group())
        else:
            controls_layout.addWidget(self._create_book_grasp_group())
        controls_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(controls_container)
        self.workflow_scroll_area = scroll
        root.addWidget(scroll, 1)

        self.status_label = QLabel(tr("pc.ready"))
        self.status_label.hide()
        self._update_move_button_state()

    def _create_workflow_header(self):
        self.workflow_header_group = QGroupBox("流程控制")
        layout = QGridLayout(self.workflow_header_group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        management_row = QHBoxLayout()
        management_row.setSpacing(4)
        self.flow_steps_btn = QPushButton(tr("pc.workflow_steps"))
        self.flow_steps_btn.clicked.connect(self._show_flow_steps_dialog)
        self._compact_flow_button(self.flow_steps_btn)
        management_row.addWidget(self.flow_steps_btn)
        self.flow_auto_run_btn = QPushButton("完整执行")
        self.flow_auto_run_btn.setObjectName("enableBtn")
        self.flow_auto_run_btn.clicked.connect(self._start_auto_book_grasp_flow)
        self._compact_flow_button(
            self.flow_auto_run_btn,
            ["完整执行", "自动执行中"],
        )
        if self._workflow_mode != "takeout":
            self.flow_auto_run_btn.hide()
        management_row.addWidget(self.flow_auto_run_btn)
        self.flow_emergency_stop_btn = QPushButton("急停")
        self.flow_emergency_stop_btn.setObjectName("emergencyStop")
        self.flow_emergency_stop_btn.clicked.connect(self._request_workflow_stop)
        self._compact_flow_button(self.flow_emergency_stop_btn)
        if self._workflow_mode != "takeout":
            self.flow_emergency_stop_btn.hide()
        management_row.addWidget(self.flow_emergency_stop_btn)
        self.flow_debug_move_btn = QPushButton("运动到图书调试位")
        self.flow_debug_move_btn.clicked.connect(self._move_to_debug_pose_from_header)
        self._compact_flow_button(self.flow_debug_move_btn)
        management_row.addWidget(self.flow_debug_move_btn)
        self.flow_zero_move_btn = QPushButton("归零")
        self.flow_zero_move_btn.clicked.connect(self._move_to_zero_pose_from_header)
        self._compact_flow_button(self.flow_zero_move_btn)
        management_row.addWidget(self.flow_zero_move_btn)
        self.flow_gripper_close_btn = QPushButton("夹爪关闭")
        self.flow_gripper_close_btn.clicked.connect(self._close_gripper_from_header)
        self._compact_flow_button(self.flow_gripper_close_btn)
        management_row.addWidget(self.flow_gripper_close_btn)
        self.flow_gripper_open_btn = QPushButton("夹爪打开")
        self.flow_gripper_open_btn.clicked.connect(self._open_gripper_from_header)
        self._compact_flow_button(self.flow_gripper_open_btn)
        management_row.addWidget(self.flow_gripper_open_btn)
        self.flow_defaults_btn = QPushButton("修改默认值")
        self.flow_defaults_btn.clicked.connect(self._begin_workflow_default_edit)
        self._compact_flow_button(
            self.flow_defaults_btn,
            ["修改默认值", "修改默认值中"],
        )
        management_row.addWidget(self.flow_defaults_btn)
        self.flow_defaults_confirm_btn = QPushButton("确认")
        self.flow_defaults_confirm_btn.setObjectName("enableBtn")
        self.flow_defaults_confirm_btn.clicked.connect(self._confirm_workflow_default_edit)
        self._compact_flow_button(self.flow_defaults_confirm_btn)
        self.flow_defaults_confirm_btn.hide()
        management_row.addWidget(self.flow_defaults_confirm_btn)
        self.flow_defaults_cancel_btn = QPushButton("取消")
        self.flow_defaults_cancel_btn.clicked.connect(self._cancel_workflow_default_edit)
        self._compact_flow_button(self.flow_defaults_cancel_btn)
        self.flow_defaults_cancel_btn.hide()
        management_row.addWidget(self.flow_defaults_cancel_btn)
        management_row.addStretch()
        layout.addLayout(management_row, 0, 0, 1, 5)

        self.flow_status_label = QLabel(tr("pc.workflow_pending"))
        self.flow_status_label.setWordWrap(True)
        self._stabilize_flow_label(self.flow_status_label)
        layout.addWidget(self.flow_status_label, 1, 0, 1, 5)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.flow_next_btn = QPushButton(tr("pc.workflow_next"))
        self.flow_next_btn.setObjectName("enableBtn")
        self.flow_next_btn.clicked.connect(self._execute_next_book_grasp_step)
        self._compact_flow_button(
            self.flow_next_btn,
            [
                tr("pc.workflow_next"),
                "开始执行",
                "确认执行第 12 步",
                "确认执行第 16 步",
                "流程完成，点击重置",
            ],
        )
        btn_row.addWidget(self.flow_next_btn)
        self.flow_back_btn = QPushButton(tr("pc.workflow_back_step"))
        self.flow_back_btn.clicked.connect(self._return_to_previous_flow_step)
        self._compact_flow_button(self.flow_back_btn)
        btn_row.addWidget(self.flow_back_btn)
        self.flow_back_pose_btn = QPushButton(tr("pc.workflow_back_pose"))
        self.flow_back_pose_btn.clicked.connect(self._rollback_to_previous_flow_pose)
        self._compact_flow_button(self.flow_back_pose_btn)
        btn_row.addWidget(self.flow_back_pose_btn)
        self.flow_reset_btn = QPushButton(tr("pc.workflow_reset"))
        self.flow_reset_btn.clicked.connect(self._reset_book_grasp_flow)
        self._compact_flow_button(self.flow_reset_btn)
        btn_row.addWidget(self.flow_reset_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row, 2, 0, 1, 5)

        self._install_workflow_header_wheel_filter()
        self._update_flow_button_state()
        return self.workflow_header_group

    def _install_workflow_header_wheel_filter(self):
        self.workflow_header_group.installEventFilter(self)
        for child in self.workflow_header_group.findChildren(QWidget):
            child.installEventFilter(self)

    def _create_viewer_widget(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        self.rgb_preview_label = QLabel(container)
        self.rgb_preview_label.setFixedSize(300, 210)
        self.rgb_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rgb_preview_label.setScaledContents(False)
        self.rgb_preview_label.setStyleSheet(
            "QLabel { background: rgba(15, 18, 24, 210); border: 1px solid rgba(255, 255, 255, 120); }"
        )
        self.rgb_preview_label.hide()
        container.installEventFilter(self)
        if HAS_PYVISTA:
            self._plotter = None
            self._plotter_placeholder = QLabel(tr("pc.viewer_loading"))
            self._plotter_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._plotter_placeholder.setWordWrap(True)
            layout.addWidget(self._plotter_placeholder)
        else:
            self._plotter = None
            self.no_plot_label = QLabel(tr("pc.no_pyvista"))
            self.no_plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.no_plot_label.setWordWrap(True)
            layout.addWidget(self.no_plot_label)
        return container

    def _ensure_plotter(self):
        if not HAS_PYVISTA or self._viewer_widget is None or self._plotter is not None:
            return

        layout = self._viewer_widget.layout()
        if layout is None:
            layout = QVBoxLayout(self._viewer_widget)
            layout.setContentsMargins(0, 0, 0, 0)

        pv.global_theme.allow_empty_mesh = True
        self._plotter = QtInteractor(self._viewer_widget, multi_samples=4)
        self._plotter.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        sc = SCENE_COLORS[ThemeManager.instance().theme]
        self._plotter.set_background(sc["bg_bottom"], top=sc["bg_top"])

        if hasattr(self, "_plotter_placeholder") and self._plotter_placeholder is not None:
            layout.removeWidget(self._plotter_placeholder)
            self._plotter_placeholder.deleteLater()
            self._plotter_placeholder = None

        layout.addWidget(self._plotter.interactor)
        self._position_rgb_preview()
        self._install_event_filter()
        self._reset_scene()

    def viewer_widget(self):
        return self._viewer_widget

    def show_viewer(self):
        self._viewer_visible = True
        self._ensure_plotter()
        if self._point_cloud is not None and self._display_points is not None:
            self._display_cloud(self._point_cloud)
        else:
            self._reset_scene()
        self._update_pick_mode_state()

    def hide_viewer(self):
        self._viewer_visible = False
        self._update_pick_mode_state()

    def _create_capture_group(self):
        self.capture_group = QGroupBox(tr("pc.capture_group"))
        layout = QGridLayout(self.capture_group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        self.pick_mode_btn = QPushButton(tr("pc.pick_mode"))
        self.pick_mode_btn.setCheckable(True)
        self.pick_mode_btn.setChecked(True)
        self.pick_mode_btn.toggled.connect(
            lambda _checked: self._update_pick_mode_state()
        )
        layout.addWidget(self.pick_mode_btn, 0, 0)

        self.capture_btn = QPushButton(tr("pc.capture"))
        self.capture_btn.setObjectName("enableBtn")
        self.capture_btn.clicked.connect(self._start_capture)
        layout.addWidget(self.capture_btn, 0, 1)
        layout.setColumnStretch(2, 1)

        return self.capture_group

    def _create_book_group(self):
        self.book_group = QGroupBox(tr("pc.book_group"))
        layout = QGridLayout(self.book_group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        return self.book_group

    def _create_image_recognition_group(self):
        self.image_recognition_group = QGroupBox(tr("pc.image_recognition_group"))
        layout = QGridLayout(self.image_recognition_group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        self.image_detect_btn = QPushButton(tr("pc.image_detect"))
        self.image_detect_btn.setObjectName("enableBtn")
        self.image_detect_btn.clicked.connect(self._start_book_spine_segmentation)
        layout.addWidget(self.image_detect_btn, 0, 0)
        layout.setColumnStretch(1, 1)

        self.image_status_label = QLabel(tr("pc.image_ready"))
        self.image_status_label.setWordWrap(True)
        self._stabilize_flow_label(self.image_status_label)
        layout.addWidget(self.image_status_label, 1, 0, 1, 5)

        self.segment_preview_label = QLabel("分割结果预览")
        self.segment_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.segment_preview_label.setMinimumSize(420, 320)
        self.segment_preview_label.setStyleSheet(
            "QLabel { background: rgba(15, 18, 24, 210); border: 1px solid rgba(255, 255, 255, 120); }"
        )
        self.segment_preview_label.setScaledContents(False)
        layout.addWidget(self.segment_preview_label, 2, 0, 1, 5)

        self.segment_count_label = QLabel("识别到书脊数量: --")
        self.segment_count_label.setWordWrap(True)
        layout.addWidget(self.segment_count_label, 3, 0, 1, 5)
        self.segment_gap_label = QLabel("Step2 最大书缝: --")
        self.segment_gap_label.setWordWrap(True)
        layout.addWidget(self.segment_gap_label, 4, 0, 1, 5)
        self.segment_gap_line_label = QLabel("中线像素: --")
        self.segment_gap_line_label.setWordWrap(True)
        layout.addWidget(self.segment_gap_line_label, 5, 0, 1, 5)
        self.segment_gap_pixel_label = QLabel("中线中点像素: --")
        self.segment_gap_pixel_label.setWordWrap(True)
        layout.addWidget(self.segment_gap_pixel_label, 6, 0, 1, 5)
        self.segment_gap_camera_label = QLabel("中点点云坐标: --")
        self.segment_gap_camera_label.setWordWrap(True)
        layout.addWidget(self.segment_gap_camera_label, 7, 0, 1, 5)
        self.segment_gap_robot_label = QLabel("机械臂末端点坐标: --")
        self.segment_gap_robot_label.setWordWrap(True)
        layout.addWidget(self.segment_gap_robot_label, 8, 0, 1, 5)
        self.segment_output_label = QLabel("输出路径: --")
        self.segment_output_label.setWordWrap(True)
        layout.addWidget(self.segment_output_label, 9, 0, 1, 5)

        hint = QLabel("Step1 分割书脊；Step2 根据相邻书脊轮廓计算最大书缝中线和中点坐标。")
        hint.setWordWrap(True)
        layout.addWidget(hint, 10, 0, 1, 5)
        return self.image_recognition_group

    def _create_result_group(self):
        self.result_group = QGroupBox(tr("pc.result_group"))
        layout = QFormLayout(self.result_group)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.pixel_label = QLabel(tr("pc.pixel"))
        self.robot_point_cm_label = QLabel(tr("pc.move_target_point_cm"))
        self.target_point_cm_label = QLabel(tr("pc.target_point_cm"))

        self.pixel_value = QLabel("--")
        self.move_target_point_cm_value = QLabel("--")
        self.target_point_cm_value = QLabel("--")

        layout.addRow(self.pixel_label, self.pixel_value)
        layout.addRow(self.robot_point_cm_label, self.move_target_point_cm_value)
        layout.addRow(self.target_point_cm_label, self.target_point_cm_value)
        return self.result_group

    def _create_move_group(self):
        self.move_group = QGroupBox(tr("pc.move_group"))
        layout = QGridLayout(self.move_group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        self.rpy_labels = []
        self.rpy_spins = []
        for col, key in enumerate(("pc.rx", "pc.ry", "pc.rz")):
            label = QLabel(tr(key))
            spin = self._make_float_spin(-180.0, 180.0, 0.0, 1.0, "°")
            spin.valueChanged.connect(self._on_rpy_changed)
            self.rpy_labels.append(label)
            self.rpy_spins.append(spin)
            layout.addWidget(label, 0, col * 2)
            layout.addWidget(spin, 0, col * 2 + 1)

        self.duration_label = QLabel(tr("pc.duration"))
        self.duration_spin = self._make_float_spin(0.5, 30.0, 2.0, 0.5, " s")
        layout.addWidget(self.duration_label, 1, 0)
        layout.addWidget(self.duration_spin, 1, 1)

        row = QHBoxLayout()
        self.read_rpy_btn = QPushButton(tr("pc.read_rpy"))
        self.read_rpy_btn.clicked.connect(self._fill_current_rpy)
        row.addWidget(self.read_rpy_btn)

        self.move_btn = QPushButton(tr("pc.confirm_move"))
        self.move_btn.setObjectName("enableBtn")
        self.move_btn.clicked.connect(self._on_confirm_move)
        row.addWidget(self.move_btn)
        row.addStretch()
        layout.addLayout(row, 1, 2, 1, 4)
        return self.move_group

    def _create_book_grasp_group(self):
        self.grasp_group = QGroupBox(self._workflow_group_title())
        layout = QGridLayout(self.grasp_group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        row = 0
        if self._is_takeout_based_workflow():
            step1, step1_layout = self._create_step_group("步骤1：采集点云")
            layout.addWidget(step1, row, 0, 1, 5)
            step1_layout.addWidget(QLabel("点云采集由顶部流程按钮触发"), 0, 0, 1, 5)
            self._add_takeout_step_button(step1_layout, 0, 1)

            row += 1
            step2, step2_layout = self._create_step_group("步骤2：识别书籍")
            layout.addWidget(step2, row, 0, 1, 5)
            template_row = QHBoxLayout()
            template_row.setContentsMargins(0, 0, 0, 0)
            template_row.setSpacing(2)
            template_label = QLabel("识别模板:")
            template_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            template_row.addWidget(template_label)
            self.template_combo = NoWheelComboBox()
            for label, path in BOOK_TEMPLATE_OPTIONS:
                self.template_combo.addItem(label, str(path))
            self.template_combo.setEditable(False)
            self.template_combo.setFixedWidth(150)
            template_row.addWidget(self.template_combo)
            template_row.addStretch()
            step2_layout.addLayout(template_row, 0, 0, 1, 5)
            self._add_takeout_step_button(step2_layout, 1, 1)

            row += 1
            step3, step3_layout = self._create_step_group("步骤3：解算目标点")
            layout.addWidget(step3, row, 0, 1, 5)
            self.book_status_label = QLabel(tr("pc.book_ready"))
            self.book_status_label.setWordWrap(True)
            self._stabilize_flow_label(self.book_status_label)
            step3_layout.addWidget(self.book_status_label, 0, 0, 1, 5)
            self.flow_target_label = QLabel(tr("pc.workflow_target"))
            self.flow_target_label.setWordWrap(True)
            self._stabilize_flow_label(self.flow_target_label, lines=3)
            step3_layout.addWidget(self.flow_target_label, 1, 0, 1, 5)
            step3_layout.addWidget(QLabel("目标点补偿XYZ:"), 2, 0)
            comp_row = QHBoxLayout()
            comp_row.setContentsMargins(0, 0, 0, 0)
            comp_row.setSpacing(4)
            self.target_comp_xyz_spins = []
            target_comp_defaults = (
                self.DEFAULT_PUTBACK2_TARGET_COMPENSATION_CM
                if self._workflow_mode == "putback2"
                else self.DEFAULT_TARGET_COMPENSATION_CM
            )
            for value in target_comp_defaults:
                spin = self._make_float_spin(-50.0, 50.0, value, 0.5, " cm")
                self.target_comp_xyz_spins.append(spin)
                comp_row.addWidget(spin)
            comp_row.addStretch()
            step3_layout.addLayout(comp_row, 2, 1, 1, 4)
            self._bind_workflow_target_refresh(self.target_comp_xyz_spins)
            self._add_takeout_step_button(step3_layout, 2, 3)

            row += 1
            row = self._populate_takeout_controls(layout, row)
        else:
            if not hasattr(self, "template_combo"):
                self.template_combo = NoWheelComboBox()
                for label, path in BOOK_TEMPLATE_OPTIONS:
                    self.template_combo.addItem(label, str(path))
                self.template_combo.setEditable(False)
            if hasattr(self, "book_status_label"):
                self._stabilize_flow_label(self.book_status_label)
            if hasattr(self, "flow_target_label"):
                self._stabilize_flow_label(self.flow_target_label, lines=3)
            row = self._populate_putback_controls(layout, row)

        self._setup_workflow_default_controls()
        self._update_flow_button_state()
        return self.grasp_group

    def _populate_takeout_controls(self, layout: QGridLayout, row: int) -> int:
        base_row = row
        group, group_layout = self._create_step_group("步骤4：到达预备抓取位姿")
        layout.addWidget(group, base_row, 0, 1, 5)
        self.pregrasp_offset_xyz_spins = []
        for value in self.DEFAULT_PREGRASP_OFFSET_CM:
            spin = self._make_float_spin(-50.0, 50.0, value, 0.5, " cm")
            self.pregrasp_offset_xyz_spins.append(spin)
        self._add_row_spins(group_layout, 0, "预备偏移XYZ:", self.pregrasp_offset_xyz_spins)

        self.pregrasp_rpy_spins = []
        pregrasp_rpy_defaults = (
            self.DEFAULT_PUTBACK2_GRASP_RPY_DEG
            if self._workflow_mode == "putback2"
            else self.DEFAULT_GRASP_RPY_DEG
        )
        for value in pregrasp_rpy_defaults:
            spin = self._make_float_spin(-180.0, 180.0, value, 1.0, "°")
            self.pregrasp_rpy_spins.append(spin)
        self.grasp_rpy_spins = self.pregrasp_rpy_spins
        self._add_row_spins(group_layout, 1, "预备姿态Rx/Ry/Rz:", self.pregrasp_rpy_spins)
        self._bind_workflow_target_refresh(self.pregrasp_rpy_spins)

        self.flow_ik_duration_spin = self._make_float_spin(0.5, 30.0, 2.0, 0.5, " s")
        self._add_labelled_spin(group_layout, 2, "MoveJ时间:", self.flow_ik_duration_spin, 0, 1)
        self.pregrasp_prepare_tools_check = QCheckBox("打开夹爪，并将杆电机运动到 0°")
        self.pregrasp_prepare_tools_check.setChecked(True)
        group_layout.addWidget(self.pregrasp_prepare_tools_check, 3, 0, 1, 5)
        self._add_takeout_step_button(group_layout, 3, 4)

        step5, step5_layout = self._create_step_group("步骤5：到达抓取位姿")
        layout.addWidget(step5, base_row + 1, 0, 1, 5)
        step5_layout.addWidget(QLabel("执行后自动记录抓取位姿"), 0, 0, 1, 5)
        self._add_takeout_step_button(step5_layout, 4, 1)

        step6, step6_layout = self._create_step_group("步骤6：杆电机到夹取位")
        layout.addWidget(step6, base_row + 2, 0, 1, 5)
        step6_layout.addWidget(QLabel("串口:"), 0, 0)
        self.rod_port_edit = QLineEdit(self.DEFAULT_ROD_PORT)
        self.rod_port_edit.setFixedWidth(150)
        step6_layout.addWidget(self.rod_port_edit, 0, 1)
        step6_layout.addWidget(QLabel("杆夹取位:"), 0, 2)
        rod_grasp_default = (
            self.DEFAULT_PUTBACK2_ROD_GRASP_DEG
            if self._workflow_mode == "putback2"
            else self.DEFAULT_ROD_GRASP_DEG
        )
        self.rod_grasp_spin = self._make_float_spin(-180.0, 180.0, rod_grasp_default, 1.0, "°")
        step6_layout.addWidget(self.rod_grasp_spin, 0, 3)
        step6_layout.addWidget(QLabel("速度:"), 1, 0)
        self.rod_speed_spin = self._make_int_spin(1, 10000, self.DEFAULT_ROD_SPEED, 100)
        step6_layout.addWidget(self.rod_speed_spin, 1, 1)
        step6_layout.addWidget(QLabel("加速度:"), 1, 2)
        self.rod_acc_spin = self._make_int_spin(1, 10000, self.DEFAULT_ROD_ACC, 10)
        step6_layout.addWidget(self.rod_acc_spin, 1, 3)
        step6_layout.setColumnStretch(4, 1)
        self._add_takeout_step_button(step6_layout, 5, 2)

        step7, step7_layout = self._create_step_group("步骤7：微调抓取位姿")
        layout.addWidget(step7, base_row + 3, 0, 1, 5)
        step7_layout.addWidget(QLabel("微调XYZ:"), 0, 0)
        target_offset_row = QHBoxLayout()
        target_offset_row.setContentsMargins(0, 0, 0, 0)
        target_offset_row.setSpacing(4)
        self.relative_target_xyz_spins = []
        for value in self.DEFAULT_TARGET_RELATIVE_OFFSET_CM:
            spin = self._make_float_spin(-50.0, 50.0, value, 0.5, " cm")
            self.relative_target_xyz_spins.append(spin)
            target_offset_row.addWidget(spin)
        target_offset_row.addStretch()
        step7_layout.addLayout(target_offset_row, 0, 1, 1, 4)
        step7_layout.addWidget(QLabel("微调姿态Rx/Ry/Rz:"), 1, 0)
        target_rpy_row = QHBoxLayout()
        target_rpy_row.setContentsMargins(0, 0, 0, 0)
        target_rpy_row.setSpacing(4)
        self.relative_target_rpy_spins = []
        for value in self.DEFAULT_TARGET_RELATIVE_RPY_DEG:
            spin = self._make_float_spin(-180.0, 180.0, value, 1.0, "°")
            self.relative_target_rpy_spins.append(spin)
            target_rpy_row.addWidget(spin)
        target_rpy_row.addStretch()
        step7_layout.addLayout(target_rpy_row, 1, 1, 1, 4)
        self.relative_target_duration_spin = self._make_float_spin(
            0.5,
            30.0,
            self.DEFAULT_TARGET_RELATIVE_DURATION_S,
            0.5,
            " s",
        )
        self._add_compact_value_row(
            step7_layout,
            2,
            "MoveL时间:",
            self.relative_target_duration_spin,
        )
        self._add_takeout_step_button(step7_layout, 6, 3)

        step8, step8_layout = self._create_step_group("步骤8：夹爪带监测持续关闭")
        layout.addWidget(step8, base_row + 4, 0, 1, 5)
        step8_layout.addWidget(QLabel("夹爪角度、速度、力矩和监测参数在夹爪管理页签中设置"), 0, 0, 1, 5)
        self._add_takeout_step_button(step8_layout, 7, 1)

        if self._workflow_mode == "putback2":
            step9, step9_layout = self._create_step_group("步骤9：姿态恢复到默认RPY")
            layout.addWidget(step9, base_row + 5, 0, 1, 5)
            self.putback2_reset_rpy_spins = []
            for value in self.DEFAULT_PUTBACK2_RESET_RPY_DEG:
                spin = self._make_float_spin(-180.0, 180.0, value, 1.0, "°")
                self.putback2_reset_rpy_spins.append(spin)
            self._add_row_spins(step9_layout, 0, "目标Rx/Ry/Rz:", self.putback2_reset_rpy_spins)
            self.putback2_reset_rpy_duration_spin = self._make_float_spin(
                0.5,
                30.0,
                self.DEFAULT_PUTBACK2_RESET_RPY_DURATION_S,
                0.5,
                " s",
            )
            self._add_compact_value_row(
                step9_layout,
                1,
                "MoveJ时间:",
                self.putback2_reset_rpy_duration_spin,
            )

            step10, step10_layout = self._create_step_group("步骤10：沿基坐标X+移动")
            layout.addWidget(step10, base_row + 6, 0, 1, 5)
            self.putback2_x_offset_spin = self._make_float_spin(
                -50.0,
                50.0,
                self.DEFAULT_PUTBACK2_X_OFFSET_CM,
                0.5,
                " cm",
            )
            self._add_compact_value_row(step10_layout, 0, "X偏移:", self.putback2_x_offset_spin)
            self.putback2_x_move_duration_spin = self._make_float_spin(
                0.5,
                30.0,
                self.DEFAULT_PUTBACK2_X_MOVE_DURATION_S,
                0.5,
                " s",
            )
            self._add_compact_value_row(
                step10_layout,
                1,
                "MoveL时间:",
                self.putback2_x_move_duration_spin,
            )

            step11, step11_layout = self._create_step_group("步骤11：夹爪开一点")
            layout.addWidget(step11, base_row + 7, 0, 1, 5)
            self.putback2_gripper_partial_open_spin = self._make_float_spin(
                -30.0,
                140.0,
                self.DEFAULT_PUTBACK2_GRIPPER_PARTIAL_OPEN_DEG,
                1.0,
                "°",
            )
            self._add_compact_value_row(
                step11_layout,
                0,
                "开一点角度:",
                self.putback2_gripper_partial_open_spin,
            )
            self.putback2_gripper_partial_open_speed_spin = self._make_float_spin(
                1.0,
                60.0,
                self.DEFAULT_PUTBACK2_GRIPPER_PARTIAL_OPEN_SPEED_DEG_S,
                0.5,
                "°/s",
            )
            self._add_compact_value_row(
                step11_layout,
                1,
                "张开速度:",
                self.putback2_gripper_partial_open_speed_spin,
            )

            step12, step12_layout = self._create_step_group("步骤12：杆电机运动到90°")
            layout.addWidget(step12, base_row + 8, 0, 1, 5)
            self.putback2_rod_release_spin = self._make_float_spin(
                -180.0,
                180.0,
                self.DEFAULT_PUTBACK2_ROD_RELEASE_DEG,
                1.0,
                "°",
            )
            self._add_compact_value_row(
                step12_layout,
                0,
                "杆目标角度:",
                self.putback2_rod_release_spin,
            )
            step12_layout.addWidget(QLabel("速度/加速度: 同步骤6"), 1, 0, 1, 5)

            step13, step13_layout = self._create_step_group("步骤13：机械臂Z方向移动")
            layout.addWidget(step13, base_row + 9, 0, 1, 5)
            self.putback2_z_offset_spin = self._make_float_spin(
                -50.0,
                50.0,
                self.DEFAULT_PUTBACK2_Z_OFFSET_CM,
                0.5,
                " cm",
            )
            self._add_compact_value_row(step13_layout, 0, "Z偏移:", self.putback2_z_offset_spin)
            self.putback2_z_move_duration_spin = self._make_float_spin(
                0.5,
                30.0,
                self.DEFAULT_PUTBACK2_Z_MOVE_DURATION_S,
                0.5,
                " s",
            )
            self._add_compact_value_row(
                step13_layout,
                1,
                "MoveL时间:",
                self.putback2_z_move_duration_spin,
            )

            step14, step14_layout = self._create_step_group("步骤14：夹爪全部张开")
            layout.addWidget(step14, base_row + 10, 0, 1, 5)
            self.putback2_gripper_full_open_spin = self._make_float_spin(
                -30.0,
                140.0,
                self.DEFAULT_PUTBACK2_GRIPPER_FULL_OPEN_DEG,
                1.0,
                "°",
            )
            self._add_compact_value_row(
                step14_layout,
                0,
                "全开角度:",
                self.putback2_gripper_full_open_spin,
            )
            self.putback2_gripper_full_open_speed_spin = self._make_float_spin(
                1.0,
                60.0,
                self.DEFAULT_PUTBACK2_GRIPPER_FULL_OPEN_SPEED_DEG_S,
                0.5,
                "°/s",
            )
            self._add_compact_value_row(
                step14_layout,
                1,
                "张开速度:",
                self.putback2_gripper_full_open_speed_spin,
            )

            step15, step15_layout = self._create_step_group("步骤15：机械臂X+移动")
            layout.addWidget(step15, base_row + 11, 0, 1, 5)
            self.putback2_after_open_x1_offset_spin = self._make_float_spin(
                -50.0,
                50.0,
                self.DEFAULT_PUTBACK2_AFTER_OPEN_X1_OFFSET_CM,
                0.5,
                " cm",
            )
            self._add_compact_value_row(
                step15_layout,
                0,
                "X偏移:",
                self.putback2_after_open_x1_offset_spin,
            )
            self.putback2_after_open_x1_duration_spin = self._make_float_spin(
                0.5,
                30.0,
                self.DEFAULT_PUTBACK2_AFTER_OPEN_X1_DURATION_S,
                0.5,
                " s",
            )
            self._add_compact_value_row(
                step15_layout,
                1,
                "MoveL时间:",
                self.putback2_after_open_x1_duration_spin,
            )

            step16, step16_layout = self._create_step_group("步骤16：闭合-Y+等待-X-全开")
            layout.addWidget(step16, base_row + 12, 0, 1, 5)
            self.putback2_combo_close_spin = self._make_float_spin(
                -30.0,
                140.0,
                self.DEFAULT_PUTBACK2_COMBO_CLOSE_DEG,
                1.0,
                "°",
            )
            self._add_compact_value_row(step16_layout, 0, "闭合角度:", self.putback2_combo_close_spin)
            self.putback2_combo_close_speed_spin = self._make_float_spin(
                1.0,
                60.0,
                self.DEFAULT_PUTBACK2_COMBO_CLOSE_SPEED_DEG_S,
                0.5,
                "°/s",
            )
            self._add_compact_value_row(step16_layout, 1, "闭合速度:", self.putback2_combo_close_speed_spin)
            self.putback2_combo_y_offset_spin = self._make_float_spin(
                -50.0,
                50.0,
                self.DEFAULT_PUTBACK2_COMBO_Y_OFFSET_CM,
                0.5,
                " cm",
            )
            self._add_compact_value_row(step16_layout, 2, "Y偏移:", self.putback2_combo_y_offset_spin)
            self.putback2_combo_y_duration_spin = self._make_float_spin(
                0.5,
                30.0,
                self.DEFAULT_PUTBACK2_COMBO_Y_DURATION_S,
                0.5,
                " s",
            )
            self._add_compact_value_row(step16_layout, 3, "Y MoveL时间:", self.putback2_combo_y_duration_spin)
            self.putback2_combo_wait_spin = self._make_float_spin(
                0.0,
                30.0,
                self.DEFAULT_PUTBACK2_COMBO_WAIT_S,
                0.5,
                " s",
            )
            self._add_compact_value_row(step16_layout, 4, "等待时间:", self.putback2_combo_wait_spin)
            self.putback2_combo_x_offset_spin = self._make_float_spin(
                -50.0,
                50.0,
                self.DEFAULT_PUTBACK2_COMBO_X_OFFSET_CM,
                0.5,
                " cm",
            )
            self._add_compact_value_row(step16_layout, 5, "X偏移:", self.putback2_combo_x_offset_spin)
            self.putback2_combo_x_duration_spin = self._make_float_spin(
                0.5,
                30.0,
                self.DEFAULT_PUTBACK2_COMBO_X_DURATION_S,
                0.5,
                " s",
            )
            self._add_compact_value_row(step16_layout, 6, "X MoveL时间:", self.putback2_combo_x_duration_spin)
            self.putback2_combo_full_open_spin = self._make_float_spin(
                -30.0,
                140.0,
                self.DEFAULT_PUTBACK2_COMBO_FULL_OPEN_DEG,
                1.0,
                "°",
            )
            self._add_compact_value_row(step16_layout, 7, "全开角度:", self.putback2_combo_full_open_spin)
            self.putback2_combo_full_open_speed_spin = self._make_float_spin(
                1.0,
                60.0,
                self.DEFAULT_PUTBACK2_COMBO_FULL_OPEN_SPEED_DEG_S,
                0.5,
                "°/s",
            )
            self._add_compact_value_row(step16_layout, 8, "全开速度:", self.putback2_combo_full_open_speed_spin)

            step17, step17_layout = self._create_step_group("步骤17：机械臂X+移动")
            layout.addWidget(step17, base_row + 13, 0, 1, 5)
            self.putback2_after_combo_x_offset_spin = self._make_float_spin(
                -50.0,
                50.0,
                self.DEFAULT_PUTBACK2_AFTER_COMBO_X_OFFSET_CM,
                0.5,
                " cm",
            )
            self._add_compact_value_row(
                step17_layout,
                0,
                "X偏移:",
                self.putback2_after_combo_x_offset_spin,
            )
            self.putback2_after_combo_x_duration_spin = self._make_float_spin(
                0.5,
                30.0,
                self.DEFAULT_PUTBACK2_AFTER_COMBO_X_DURATION_S,
                0.5,
                " s",
            )
            self._add_compact_value_row(
                step17_layout,
                1,
                "MoveL时间:",
                self.putback2_after_combo_x_duration_spin,
            )
            return base_row + 14

        step9, step9_layout = self._create_step_group("步骤9：当前位姿偏移IK+MoveJ")
        layout.addWidget(step9, base_row + 5, 0, 1, 5)
        self.post_gripper_movej_xyz_spins = []
        for value in self.DEFAULT_POST_GRIPPER_MOVEJ_OFFSET_CM:
            spin = self._make_float_spin(-50.0, 50.0, value, 0.5, " cm")
            self.post_gripper_movej_xyz_spins.append(spin)
        self._add_row_spins(step9_layout, 0, "偏移XYZ:", self.post_gripper_movej_xyz_spins)
        self.post_gripper_movej_rpy_spins = []
        for value in self.DEFAULT_POST_GRIPPER_MOVEJ_RPY_OFFSET_DEG:
            spin = self._make_float_spin(-180.0, 180.0, value, 1.0, "°")
            self.post_gripper_movej_rpy_spins.append(spin)
        self._add_row_spins(step9_layout, 1, "偏移Rx/Ry/Rz:", self.post_gripper_movej_rpy_spins)
        self.post_gripper_movej_duration_spin = self._make_float_spin(
            0.5,
            30.0,
            self.DEFAULT_POST_GRIPPER_MOVEJ_DURATION_S,
            0.5,
            " s",
        )
        self._add_compact_value_row(
            step9_layout,
            2,
            "MoveJ时间:",
            self.post_gripper_movej_duration_spin,
        )
        self._add_takeout_step_button(step9_layout, 8, 3)

        step10, step10_layout = self._create_step_group("步骤10：回到图书调试位")
        layout.addWidget(step10, base_row + 6, 0, 1, 5)
        self.debug_joint_spins = []
        for value in self._default_debug_joints_deg():
            spin = self._make_float_spin(-360.0, 360.0, value, 1.0, "°")
            self.debug_joint_spins.append(spin)
        self._add_indexed_joint_spins(step10_layout, 0, "目标关节:", self.debug_joint_spins)
        self.flow_debug_duration_spin = self._make_float_spin(3.0, 20.0, self.DEFAULT_DEBUG_MOVE_DURATION_S, 0.5, " s")
        self._add_compact_value_row(step10_layout, 2, "MoveJ时间:", self.flow_debug_duration_spin)
        self._add_takeout_step_button(step10_layout, 9, 3)

        step_lift, step_lift_layout = self._create_step_group("步骤11：升降台从取书位移动到还书位")
        layout.addWidget(step_lift, base_row + 7, 0, 1, 5)
        step_lift_layout.addWidget(
            QLabel("当前位置固定: 取书位"),
            0,
            0,
            1,
            2,
        )
        step_lift_layout.addWidget(QLabel("目标位置: 还书位"), 0, 2, 1, 3)
        self._add_takeout_step_button(step_lift_layout, 10, 1)

        step11, step11_layout = self._create_step_group("步骤12：慢速转身到预备放书位置")
        layout.addWidget(step11, base_row + 8, 0, 1, 5)
        self.turn_stage1_joint_spins = []
        for value in self.DEFAULT_TURN_STAGE1_JOINTS_DEG:
            spin = self._make_float_spin(-360.0, 360.0, value, 1.0, "°")
            self.turn_stage1_joint_spins.append(spin)
        self._add_indexed_joint_spins(step11_layout, 0, "目标关节:", self.turn_stage1_joint_spins)
        self.flow_turn_duration_spin = self._make_float_spin(3.0, 20.0, self.DEFAULT_TURN_DURATION_S, 0.5, " s")
        self._add_compact_value_row(step11_layout, 2, "转身时间:", self.flow_turn_duration_spin)
        self._add_takeout_step_button(step11_layout, 11, 3)

        step12b, step12b_layout = self._create_step_group("步骤13：IK+MoveJ到指定位姿")
        layout.addWidget(step12b, base_row + 9, 0, 1, 5)
        self.takeout_specified_pose_xyz_spins = []
        for value in self.DEFAULT_TAKEOUT_SPECIFIED_POSE_CM:
            spin = self._make_float_spin(-200.0, 200.0, value, 0.5, " cm")
            self.takeout_specified_pose_xyz_spins.append(spin)
        self._add_row_spins(
            step12b_layout,
            0,
            "目标XYZ:",
            self.takeout_specified_pose_xyz_spins,
        )
        self.takeout_specified_pose_rpy_spins = []
        for value in self.DEFAULT_TAKEOUT_SPECIFIED_POSE_RPY_DEG:
            spin = self._make_float_spin(-360.0, 360.0, value, 1.0, "°")
            self.takeout_specified_pose_rpy_spins.append(spin)
        self._add_row_spins(
            step12b_layout,
            1,
            "目标Rx/Ry/Rz:",
            self.takeout_specified_pose_rpy_spins,
        )
        self.takeout_specified_pose_duration_spin = self._make_float_spin(
            0.5,
            30.0,
            self.DEFAULT_TAKEOUT_SPECIFIED_POSE_DURATION_S,
            0.5,
            " s",
        )
        self._add_compact_value_row(
            step12b_layout,
            2,
            "MoveJ时间:",
            self.takeout_specified_pose_duration_spin,
        )
        self._add_takeout_step_button(step12b_layout, 12, 3)

        step13b, step13b_layout = self._create_step_group("步骤14：基于步骤13位姿微调")
        layout.addWidget(step13b, base_row + 10, 0, 1, 5)
        step13b_layout.addWidget(
            QLabel("基于步骤13后的末端位姿，在机械臂基坐标系下做 MoveL 微调"),
            0,
            0,
            1,
            5,
        )
        self.takeout_step13_fine_tune_xyz_spins = []
        for value in self.DEFAULT_TAKEOUT_STEP13_FINE_TUNE_OFFSET_CM:
            spin = self._make_float_spin(-200.0, 200.0, value, 0.5, " cm")
            self.takeout_step13_fine_tune_xyz_spins.append(spin)
        self._add_row_spins(
            step13b_layout,
            1,
            "微调XYZ:",
            self.takeout_step13_fine_tune_xyz_spins,
        )
        self.takeout_step13_fine_tune_rpy_spins = []
        for value in self.DEFAULT_TAKEOUT_STEP13_FINE_TUNE_RPY_OFFSET_DEG:
            spin = self._make_float_spin(-180.0, 180.0, value, 1.0, "°")
            self.takeout_step13_fine_tune_rpy_spins.append(spin)
        self._add_row_spins(
            step13b_layout,
            2,
            "微调Rx/Ry/Rz:",
            self.takeout_step13_fine_tune_rpy_spins,
        )
        self.takeout_step13_fine_tune_duration_spin = self._make_float_spin(
            0.5,
            30.0,
            self.DEFAULT_TAKEOUT_STEP13_FINE_TUNE_DURATION_S,
            0.5,
            " s",
        )
        self._add_compact_value_row(
            step13b_layout,
            3,
            "MoveL时间:",
            self.takeout_step13_fine_tune_duration_spin,
        )
        self._add_takeout_step_button(step13b_layout, 13, 4)

        step14b, step14b_layout = self._create_step_group("步骤15：夹爪打开")
        layout.addWidget(step14b, base_row + 11, 0, 1, 5)
        step14b_layout.addWidget(
            QLabel("打开参数使用夹爪管理页签，机械臂保持当前位姿"),
            0,
            0,
            1,
            5,
        )
        self._add_takeout_step_button(step14b_layout, 14, 1)

        step12, step12_layout = self._create_step_group("步骤16：MoveJ到最终放置构型")
        layout.addWidget(step12, base_row + 12, 0, 1, 5)
        self.final_joint_spins = []
        for value in self.DEFAULT_FINAL_JOINTS_DEG:
            spin = self._make_float_spin(-360.0, 360.0, value, 1.0, "°")
            self.final_joint_spins.append(spin)
        self._add_indexed_joint_spins(step12_layout, 0, "目标关节:", self.final_joint_spins)
        self.flow_final_duration_spin = self._make_float_spin(
            3.0,
            30.0,
            self.DEFAULT_FINAL_MOVE_DURATION_S,
            0.5,
            " s",
        )
        self._add_compact_value_row(step12_layout, 2, "MoveJ时间:", self.flow_final_duration_spin)
        self._add_takeout_step_button(step12_layout, 15, 3)

        return base_row + 13

    def _populate_putback_controls(self, layout: QGridLayout, row: int) -> int:
        base_row = row
        step1, step1_layout = self._create_step_group("步骤1：采集点云")
        layout.addWidget(step1, base_row, 0, 1, 5)
        step1_layout.addWidget(QLabel("点云采集由顶部流程按钮触发"), 0, 0, 1, 5)

        step2, step2_layout = self._create_step_group("步骤2：识别书籍")
        layout.addWidget(step2, base_row + 1, 0, 1, 5)
        self.template_combo.setFixedWidth(150)
        self._add_left_value_row(step2_layout, 0, "识别模板:", self.template_combo)

        step3_title = (
            "步骤3：解算底边点"
            if self._workflow_mode == "tail_putback"
            else "步骤3：解算目标点"
        )
        step3, step3_layout = self._create_step_group(step3_title)
        layout.addWidget(step3, base_row + 2, 0, 1, 5)
        self.book_status_label = QLabel(tr("pc.book_ready"))
        self.book_status_label.setWordWrap(True)
        self._stabilize_flow_label(self.book_status_label)
        step3_layout.addWidget(self.book_status_label, 0, 0, 1, 5)

        self.flow_target_label = QLabel(tr("pc.workflow_target"))
        self.flow_target_label.setWordWrap(True)
        self._stabilize_flow_label(self.flow_target_label, lines=3)
        step3_layout.addWidget(self.flow_target_label, 1, 0, 1, 5)

        if self._workflow_mode == "tail_putback":
            step3_layout.addWidget(
                QLabel("底边点固定为识别框右边线最底下的点"),
                2,
                0,
                1,
                5,
            )
            tail_step4, tail_step4_layout = self._create_step_group("步骤4：解算目标点")
            layout.addWidget(tail_step4, base_row + 3, 0, 1, 5)
            self.tail_book_height_spin = self._make_float_spin(
                0.0,
                100.0,
                24.0,
                0.5,
                " cm",
            )
            self.tail_book_thickness_spin = self._make_float_spin(
                0.0,
                20.0,
                1.5,
                0.1,
                " cm",
            )
            self.tail_gripper_height_spin = self._make_float_spin(
                0.0,
                20.0,
                2.0,
                0.1,
                " cm",
            )
            self._add_left_value_row(
                tail_step4_layout,
                0,
                "书本高度:",
                self.tail_book_height_spin,
            )
            self._add_left_value_row(
                tail_step4_layout,
                1,
                "书本厚度:",
                self.tail_book_thickness_spin,
            )
            self._add_left_value_row(
                tail_step4_layout,
                2,
                "夹爪高度:",
                self.tail_gripper_height_spin,
            )
            target_layout = tail_step4_layout
            target_base_row = 3
            motion_row_offset = 1
        else:
            target_layout = step3_layout
            target_base_row = 2
            motion_row_offset = 0

        self.putback_target_rpy_spins = []
        for value in self.DEFAULT_PUTBACK_TARGET_RPY_DEG:
            spin = self._make_float_spin(-180.0, 180.0, value, 1.0, "°")
            self.putback_target_rpy_spins.append(spin)
        self._add_left_spins_row(
            target_layout,
            target_base_row,
            "目标姿态Rx/Ry/Rz:",
            self.putback_target_rpy_spins,
        )
        self._bind_workflow_target_refresh(self.putback_target_rpy_spins)

        self.putback_target_comp_xyz_spins = []
        for value in (0.0, 0.0, 0.0):
            spin = self._make_float_spin(-50.0, 50.0, value, 0.5, " cm")
            self.putback_target_comp_xyz_spins.append(spin)
        self._add_left_spins_row(
            target_layout,
            target_base_row + 1,
            "目标补偿XYZ:",
            self.putback_target_comp_xyz_spins,
        )
        self._bind_workflow_target_refresh(self.putback_target_comp_xyz_spins)

        if self._workflow_mode == "tail_putback":
            for spin in (
                self.tail_book_height_spin,
                self.tail_book_thickness_spin,
                self.tail_gripper_height_spin,
            ):
                spin.valueChanged.connect(lambda *_args: self._on_workflow_target_changed())
            tail_step4_layout.addWidget(
                QLabel(
                    "目标点 = 底边点X, 底边点Y + 书本厚度/2, 底边点Z + 书本高度 - 夹爪高度/2"
                ),
                target_base_row + 2,
                0,
                1,
                5,
            )
            tail_step5, tail_step5_layout = self._create_step_group("步骤5：目标点姿态")
            layout.addWidget(tail_step5, base_row + 4, 0, 1, 5)
            tail_step5_layout.addWidget(QLabel("目标姿态与目标补偿已在步骤4中设置"), 0, 0, 1, 5)

            tail_step6, tail_step6_layout = self._create_step_group("步骤6：到预备放书点")
            layout.addWidget(tail_step6, base_row + 5, 0, 1, 5)
            self.tail_preplace_motion_combo = NoWheelComboBox()
            self.tail_preplace_motion_combo.addItem("MoveJ", "movej")
            self.tail_preplace_motion_combo.addItem("MoveL", "movel")
            self.tail_preplace_motion_combo.setFixedWidth(100)
            self._add_left_value_row(
                tail_step6_layout,
                0,
                "运动方式:",
                self.tail_preplace_motion_combo,
            )
            self.tail_preplace_xyz_spins = []
            for value in (0.0, 0.0, 0.0):
                spin = self._make_float_spin(-50.0, 50.0, value, 0.5, " cm")
                self.tail_preplace_xyz_spins.append(spin)
            self._add_left_spins_row(
                tail_step6_layout,
                1,
                "预备偏移XYZ:",
                self.tail_preplace_xyz_spins,
            )
            self.tail_preplace_rpy_spins = []
            for value in (0.0, 0.0, 0.0):
                spin = self._make_float_spin(-180.0, 180.0, value, 1.0, "°")
                self.tail_preplace_rpy_spins.append(spin)
            self._add_left_spins_row(
                tail_step6_layout,
                2,
                "预备偏移Rx/Ry/Rz:",
                self.tail_preplace_rpy_spins,
            )
            self.tail_preplace_movej_duration_spin = self._make_float_spin(
                0.5,
                30.0,
                2.0,
                0.5,
                " s",
            )
            self._add_left_value_row(
                tail_step6_layout,
                3,
                "运动时间:",
                self.tail_preplace_movej_duration_spin,
            )
            tail_step7, tail_step7_layout = self._create_step_group("步骤7：运动到放置点")
            layout.addWidget(tail_step7, base_row + 6, 0, 1, 5)
            tail_step7_layout.addWidget(
                QLabel("放置点 = 步骤4补偿目标点 + 下方位姿偏移"),
                0,
                0,
                1,
                5,
            )
            self.tail_place_motion_combo = NoWheelComboBox()
            self.tail_place_motion_combo.addItem("MoveL", "movel")
            self.tail_place_motion_combo.addItem("MoveJ", "movej")
            self.tail_place_motion_combo.setFixedWidth(100)
            self._add_left_value_row(
                tail_step7_layout,
                1,
                "运动方式:",
                self.tail_place_motion_combo,
            )
            self.tail_place_offset_xyz_spins = []
            for value in self.DEFAULT_TAIL_PLACE_OFFSET_CM:
                spin = self._make_float_spin(-50.0, 50.0, value, 0.5, " cm")
                self.tail_place_offset_xyz_spins.append(spin)
            self._add_left_spins_row(
                tail_step7_layout,
                2,
                "放置偏移XYZ:",
                self.tail_place_offset_xyz_spins,
            )
            self.tail_place_rpy_offset_spins = []
            for value in self.DEFAULT_TAIL_PLACE_RPY_OFFSET_DEG:
                spin = self._make_float_spin(-180.0, 180.0, value, 1.0, "°")
                self.tail_place_rpy_offset_spins.append(spin)
            self._add_left_spins_row(
                tail_step7_layout,
                3,
                "放置姿态偏移Rx/Ry/Rz:",
                self.tail_place_rpy_offset_spins,
            )
            self.tail_target_movel_duration_spin = self._make_float_spin(
                0.5,
                30.0,
                2.0,
                0.5,
                " s",
            )
            self._add_left_value_row(
                tail_step7_layout,
                4,
                "运动时间:",
                self.tail_target_movel_duration_spin,
            )
            tail_step8, tail_step8_layout = self._create_step_group("步骤8：微调书本位置")
            layout.addWidget(tail_step8, base_row + 7, 0, 1, 5)
            tail_step8_layout.addWidget(
                QLabel("基于步骤7完成后的位姿微调，可选择 MoveL 或 MoveJ"),
                0,
                0,
                1,
                5,
            )
            self.tail_fine_tune_motion_combo = NoWheelComboBox()
            self.tail_fine_tune_motion_combo.addItem("MoveL", "movel")
            self.tail_fine_tune_motion_combo.addItem("MoveJ", "movej")
            self.tail_fine_tune_motion_combo.setFixedWidth(100)
            self._add_left_value_row(
                tail_step8_layout,
                1,
                "运动方式:",
                self.tail_fine_tune_motion_combo,
            )
            self.tail_fine_tune_xyz_spins = []
            for value in self.DEFAULT_TAIL_FINE_TUNE_OFFSET_CM:
                spin = self._make_float_spin(-50.0, 50.0, value, 0.5, " cm")
                self.tail_fine_tune_xyz_spins.append(spin)
            self._add_left_spins_row(
                tail_step8_layout,
                2,
                "微调XYZ:",
                self.tail_fine_tune_xyz_spins,
            )
            self.tail_fine_tune_rpy_spins = []
            for value in self.DEFAULT_TAIL_FINE_TUNE_RPY_OFFSET_DEG:
                spin = self._make_float_spin(-180.0, 180.0, value, 1.0, "°")
                self.tail_fine_tune_rpy_spins.append(spin)
            self._add_left_spins_row(
                tail_step8_layout,
                3,
                "微调姿态Rx/Ry/Rz:",
                self.tail_fine_tune_rpy_spins,
            )
            self.tail_fine_tune_movel_duration_spin = self._make_float_spin(
                0.2,
                30.0,
                self.DEFAULT_TAIL_FINE_TUNE_MOVEL_DURATION_S,
                0.2,
                " s",
            )
            self._add_left_value_row(
                tail_step8_layout,
                4,
                "运动时间:",
                self.tail_fine_tune_movel_duration_spin,
            )
            tail_step9, tail_step9_layout = self._create_step_group("步骤9：夹爪微微张开")
            layout.addWidget(tail_step9, base_row + 8, 0, 1, 5)
            self.tail_gripper_slight_open_spin = self._make_float_spin(
                -30.0,
                140.0,
                self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_DEG,
                1.0,
                "°",
            )
            self.tail_gripper_slight_open_speed_spin = self._make_float_spin(
                0.5,
                30.0,
                self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_SPEED_DEG_S,
                0.5,
                "°/s",
            )
            self.tail_gripper_slight_open_timeout_spin = self._make_float_spin(
                0.5,
                15.0,
                self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_TIMEOUT_S,
                0.5,
                " s",
            )
            self.tail_gripper_slight_open_target_tolerance_spin = self._make_float_spin(
                0.1,
                10.0,
                self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_TARGET_TOLERANCE_DEG,
                0.1,
                "°",
            )
            self.tail_gripper_slight_open_stall_tolerance_spin = self._make_float_spin(
                0.1,
                10.0,
                self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_STALL_TOLERANCE_DEG,
                0.1,
                "°",
            )
            self.tail_gripper_slight_open_stall_time_spin = self._make_float_spin(
                0.1,
                10.0,
                self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_STALL_TIME_S,
                0.1,
                " s",
            )
            self.tail_gripper_slight_open_min_monitor_spin = self._make_float_spin(
                0.0,
                10.0,
                self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_MIN_MONITOR_S,
                0.1,
                " s",
            )
            self.tail_gripper_slight_open_hold_margin_spin = self._make_float_spin(
                0.0,
                5.0,
                self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_HOLD_MARGIN_DEG,
                0.1,
                "°",
            )
            self.tail_gripper_slight_open_command_lead_spin = self._make_float_spin(
                0.0,
                2.0,
                self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_COMMAND_LEAD_S,
                0.05,
                " s",
            )
            self.tail_gripper_slight_open_stall_lead_threshold_spin = self._make_float_spin(
                0.0,
                15.0,
                self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_STALL_LEAD_THRESHOLD_DEG,
                0.5,
                "°",
            )
            self._add_left_value_pairs(
                tail_step9_layout,
                0,
                (
                    ("微张增量:", self.tail_gripper_slight_open_spin),
                    ("微张速度:", self.tail_gripper_slight_open_speed_spin),
                    ("超时:", self.tail_gripper_slight_open_timeout_spin),
                    ("到位容差:", self.tail_gripper_slight_open_target_tolerance_spin),
                    ("卡滞容差:", self.tail_gripper_slight_open_stall_tolerance_spin),
                    ("卡滞时间:", self.tail_gripper_slight_open_stall_time_spin),
                    ("最短监测:", self.tail_gripper_slight_open_min_monitor_spin),
                    ("锁定余量:", self.tail_gripper_slight_open_hold_margin_spin),
                    ("指令提前:", self.tail_gripper_slight_open_command_lead_spin),
                    ("卡滞阈值:", self.tail_gripper_slight_open_stall_lead_threshold_spin),
                ),
                columns=2,
            )
            tail_step10, tail_step10_layout = self._create_step_group("步骤10：基于当前位姿后移")
            layout.addWidget(tail_step10, base_row + 9, 0, 1, 5)
            tail_step10_layout.addWidget(
                QLabel("基于步骤9后的当前位姿做 MoveL 后移，偏移使用机械臂基坐标系"),
                0,
                0,
                1,
                5,
            )
            self.tail_backoff_xyz_spins = []
            for value in self.DEFAULT_TAIL_BACKOFF_OFFSET_CM:
                spin = self._make_float_spin(-50.0, 50.0, value, 0.5, " cm")
                self.tail_backoff_xyz_spins.append(spin)
            self._add_left_spins_row(
                tail_step10_layout,
                1,
                "基坐标后移XYZ:",
                self.tail_backoff_xyz_spins,
            )
            self.tail_backoff_rpy_spins = []
            for value in self.DEFAULT_TAIL_BACKOFF_RPY_OFFSET_DEG:
                spin = self._make_float_spin(-180.0, 180.0, value, 1.0, "°")
                self.tail_backoff_rpy_spins.append(spin)
            self._add_left_spins_row(
                tail_step10_layout,
                2,
                "后移姿态偏移Rx/Ry/Rz:",
                self.tail_backoff_rpy_spins,
            )
            self.tail_backoff_movel_duration_spin = self._make_float_spin(
                0.2,
                30.0,
                self.DEFAULT_TAIL_BACKOFF_MOVEL_DURATION_S,
                0.2,
                " s",
            )
            self._add_left_value_row(
                tail_step10_layout,
                3,
                "MoveL时间:",
                self.tail_backoff_movel_duration_spin,
            )

            tail_step11, tail_step11_layout = self._create_step_group(
                "步骤11：夹爪直接闭合"
            )
            layout.addWidget(tail_step11, base_row + 10, 0, 1, 5)
            tail_step11_layout.addWidget(
                QLabel("闭合参数使用夹爪管理页签，机械臂保持当前位姿"),
                0,
                0,
                1,
                5,
            )

            tail_step12, tail_step12_layout = self._create_step_group(
                "步骤12：运动到预备推书位"
            )
            layout.addWidget(tail_step12, base_row + 11, 0, 1, 5)
            tail_step12_layout.addWidget(
                QLabel("基于步骤11后的位姿，在机械臂坐标系下做位姿偏移"),
                0,
                0,
                1,
                5,
            )
            self.tail_prepush_motion_combo = NoWheelComboBox()
            self.tail_prepush_motion_combo.addItem("MoveL", "movel")
            self.tail_prepush_motion_combo.addItem("MoveJ", "movej")
            self.tail_prepush_motion_combo.setFixedWidth(100)
            self._add_left_value_row(
                tail_step12_layout,
                1,
                "运动方式:",
                self.tail_prepush_motion_combo,
            )
            self.tail_prepush_offset_xyz_spins = []
            for value in self.DEFAULT_TAIL_PREPUSH_OFFSET_CM:
                spin = self._make_float_spin(-50.0, 50.0, value, 0.5, " cm")
                self.tail_prepush_offset_xyz_spins.append(spin)
            self._add_left_spins_row(
                tail_step12_layout,
                2,
                "预备推书偏移XYZ:",
                self.tail_prepush_offset_xyz_spins,
            )
            self.tail_prepush_rpy_offset_spins = []
            for value in self.DEFAULT_TAIL_PREPUSH_RPY_OFFSET_DEG:
                spin = self._make_float_spin(-180.0, 180.0, value, 1.0, "°")
                self.tail_prepush_rpy_offset_spins.append(spin)
            self._add_left_spins_row(
                tail_step12_layout,
                3,
                "姿态偏移Rx/Ry/Rz:",
                self.tail_prepush_rpy_offset_spins,
            )
            self.tail_prepush_movel_duration_spin = self._make_float_spin(
                0.5,
                30.0,
                self.DEFAULT_TAIL_PREPUSH_MOVEL_DURATION_S,
                0.5,
                " s",
            )
            self._add_left_value_row(
                tail_step12_layout,
                4,
                "运动时间:",
                self.tail_prepush_movel_duration_spin,
            )

            tail_step13, tail_step13_layout = self._create_step_group(
                "步骤13：运动到推书位"
            )
            layout.addWidget(tail_step13, base_row + 12, 0, 1, 5)
            tail_step13_layout.addWidget(
                QLabel("基于步骤12后的位姿，在机械臂坐标系下做 MoveL 偏移"),
                0,
                0,
                1,
                5,
            )
            self.tail_push_offset_xyz_spins = []
            for value in self.DEFAULT_TAIL_PUSH_OFFSET_CM:
                spin = self._make_float_spin(-50.0, 50.0, value, 0.5, " cm")
                self.tail_push_offset_xyz_spins.append(spin)
            self._add_left_spins_row(
                tail_step13_layout,
                1,
                "推书偏移XYZ:",
                self.tail_push_offset_xyz_spins,
            )
            self.tail_push_rpy_offset_spins = []
            for value in self.DEFAULT_TAIL_PUSH_RPY_OFFSET_DEG:
                spin = self._make_float_spin(-180.0, 180.0, value, 1.0, "°")
                self.tail_push_rpy_offset_spins.append(spin)
            self._add_left_spins_row(
                tail_step13_layout,
                2,
                "姿态偏移Rx/Ry/Rz:",
                self.tail_push_rpy_offset_spins,
            )
            self.tail_push_movel_duration_spin = self._make_float_spin(
                0.5,
                30.0,
                self.DEFAULT_TAIL_PUSH_MOVEL_DURATION_S,
                0.5,
                " s",
            )
            self._add_left_value_row(
                tail_step13_layout,
                3,
                "MoveL时间:",
                self.tail_push_movel_duration_spin,
            )

            tail_step14, tail_step14_layout = self._create_step_group(
                "步骤14：回到归零位置或图书调试位"
            )
            layout.addWidget(tail_step14, base_row + 13, 0, 1, 5)
            self.tail_final_home_combo = NoWheelComboBox()
            self.tail_final_home_combo.addItem("归零位置", "zero")
            self.tail_final_home_combo.addItem("图书调试位置", "debug")
            self.tail_final_home_combo.setFixedWidth(120)
            self._set_workflow_control_value(
                self.tail_final_home_combo,
                self.DEFAULT_TAIL_FINAL_HOME_MODE,
            )
            self._add_left_value_row(
                tail_step14_layout,
                0,
                "回位目标:",
                self.tail_final_home_combo,
            )
            self.tail_final_home_duration_spin = self._make_float_spin(
                0.5,
                30.0,
                self.DEFAULT_TAIL_FINAL_HOME_DURATION_S,
                0.5,
                " s",
            )
            self._add_left_value_row(
                tail_step14_layout,
                1,
                "MoveJ时间:",
                self.tail_final_home_duration_spin,
            )
            self._ensure_debug_pose_controls()
            return base_row + 14
        else:
            self.putback_target_line_combo = NoWheelComboBox()
            self.putback_target_line_combo.addItem("中间线", "middle")
            self.putback_target_line_combo.addItem("右边线", "right")
            self.putback_target_line_combo.addItem("左边线", "left")
            self.putback_target_line_combo.setCurrentIndex(1)
            self.putback_target_line_combo.setFixedWidth(90)
            self.putback_target_line_combo.currentIndexChanged.connect(
                lambda *_args: self._refresh_book_pick_from_last_detection()
            )
            self._add_left_value_row(
                target_layout,
                target_base_row + 2,
                "目标线:",
                self.putback_target_line_combo,
            )

            self.putback_pick_ratio_combo = NoWheelComboBox()
            self.putback_pick_ratio_combo.addItem("下四分之一", "0.75")
            self.putback_pick_ratio_combo.addItem("下三分之一", "0.6666666666666666")
            self.putback_pick_ratio_combo.addItem("二分之一", "0.50")
            self.putback_pick_ratio_combo.addItem("上三分之一", "0.3333333333333333")
            self.putback_pick_ratio_combo.setFixedWidth(100)
            self.putback_pick_ratio_combo.currentIndexChanged.connect(
                lambda *_args: self._refresh_book_pick_from_last_detection()
            )
            self._add_left_value_row(
                target_layout,
                target_base_row + 3,
                "取点位置:",
                self.putback_pick_ratio_combo,
            )

        step4, step4_layout = self._create_step_group("步骤4：到达预备推书点")
        if self._workflow_mode == "tail_putback":
            step4.setTitle("步骤5：到达预备推书点")
        layout.addWidget(step4, base_row + 3 + motion_row_offset, 0, 1, 5)
        self.putback_prepush_xyz_spins = []
        for value in self.DEFAULT_PUTBACK_PREPUSH_OFFSET_CM:
            spin = self._make_float_spin(-50.0, 50.0, value, 0.5, " cm")
            self.putback_prepush_xyz_spins.append(spin)
        self._add_left_spins_row(
            step4_layout,
            0,
            "预备偏移XYZ:",
            self.putback_prepush_xyz_spins,
        )
        self.putback_prepush_rpy_offset_spins = []
        for value in self.DEFAULT_PUTBACK_PREPUSH_RPY_OFFSET_DEG:
            spin = self._make_float_spin(-180.0, 180.0, value, 1.0, "°")
            self.putback_prepush_rpy_offset_spins.append(spin)
        self._add_left_spins_row(
            step4_layout,
            1,
            "姿态偏移Rx/Ry/Rz:",
            self.putback_prepush_rpy_offset_spins,
        )
        self.flow_ik_duration_spin = self._make_float_spin(
            0.5,
            30.0,
            2.0,
            0.5,
            " s",
        )
        self._add_left_value_row(
            step4_layout,
            2,
            "IK+MoveJ时间:",
            self.flow_ik_duration_spin,
        )

        step5, step5_layout = self._create_step_group("步骤5：MoveL到推书点")
        if self._workflow_mode == "tail_putback":
            step5.setTitle("步骤6：MoveL到推书点")
        layout.addWidget(step5, base_row + 4 + motion_row_offset, 0, 1, 5)
        self.putback_push_xyz_spins = []
        for value in self.DEFAULT_PUTBACK_PUSH_OFFSET_CM:
            spin = self._make_float_spin(-50.0, 50.0, value, 0.5, " cm")
            self.putback_push_xyz_spins.append(spin)
        self._add_left_spins_row(
            step5_layout,
            0,
            "推书偏移XYZ:",
            self.putback_push_xyz_spins,
        )
        self.putback_push_rpy_offset_spins = []
        for value in self.DEFAULT_PUTBACK_PUSH_RPY_OFFSET_DEG:
            spin = self._make_float_spin(-180.0, 180.0, value, 1.0, "°")
            self.putback_push_rpy_offset_spins.append(spin)
        self._add_left_spins_row(
            step5_layout,
            1,
            "姿态偏移Rx/Ry/Rz:",
            self.putback_push_rpy_offset_spins,
        )
        self.flow_movel_duration_spin = self._make_float_spin(
            0.5,
            30.0,
            2.0,
            0.5,
            " s",
        )
        self._add_left_value_row(
            step5_layout,
            2,
            "MoveL时间:",
            self.flow_movel_duration_spin,
        )

        step6, step6_layout = self._create_step_group("步骤6：推开书本")
        if self._workflow_mode == "tail_putback":
            step6.setTitle("步骤7：推开书本")
        layout.addWidget(step6, base_row + 5 + motion_row_offset, 0, 1, 5)
        self.putback_push_out_xyz_spins = []
        for value in self.DEFAULT_PUTBACK_PUSH_OUT_OFFSET_CM:
            spin = self._make_float_spin(-50.0, 50.0, value, 0.5, " cm")
            self.putback_push_out_xyz_spins.append(spin)
        self._add_left_spins_row(
            step6_layout,
            0,
            "推开偏移XYZ:",
            self.putback_push_out_xyz_spins,
        )
        self.putback_push_out_rpy_offset_spins = []
        for value in self.DEFAULT_PUTBACK_PUSH_OUT_RPY_OFFSET_DEG:
            spin = self._make_float_spin(-180.0, 180.0, value, 1.0, "°")
            self.putback_push_out_rpy_offset_spins.append(spin)
        self._add_left_spins_row(
            step6_layout,
            1,
            "姿态偏移Rx/Ry/Rz:",
            self.putback_push_out_rpy_offset_spins,
        )
        self.putback_push_movel_duration_spin = self._make_float_spin(
            0.5,
            30.0,
            2.0,
            0.5,
            " s",
        )
        self._add_left_value_row(
            step6_layout,
            2,
            "MoveL时间:",
            self.putback_push_movel_duration_spin,
        )

        step7, step7_layout = self._create_step_group("步骤7：离开推书位置")
        if self._workflow_mode == "tail_putback":
            step7.setTitle("步骤8：离开推书位置")
        layout.addWidget(step7, base_row + 6 + motion_row_offset, 0, 1, 5)
        self.putback_leave_push_y_spin = self._make_float_spin(
            -50.0,
            50.0,
            self.DEFAULT_PUTBACK_LEAVE_PUSH_Y_OFFSET_CM,
            0.5,
            " cm",
        )
        self._add_left_value_row(
            step7_layout,
            0,
            "离开推书Y+:",
            self.putback_leave_push_y_spin,
        )
        self._add_left_value_row(step7_layout, 1, "MoveL时间:", "同步骤5")

        step8, step8_layout = self._create_step_group("步骤8：到达插入预备位")
        if self._workflow_mode == "tail_putback":
            step8.setTitle("步骤9：到达插入预备位")
        layout.addWidget(step8, base_row + 7 + motion_row_offset, 0, 1, 5)
        self.putback_insert_rpy_spins = []
        for value in self.DEFAULT_PUTBACK_INSERT_RPY_DEG:
            spin = self._make_float_spin(-180.0, 180.0, value, 1.0, "°")
            self.putback_insert_rpy_spins.append(spin)
        self._add_left_spins_row(
            step8_layout,
            0,
            "插入姿态Rx/Ry/Rz:",
            self.putback_insert_rpy_spins,
        )

        self.putback_insert_prepose_spins = []
        self.putback_insert_prepose_x_spin = self._make_float_spin(
            -50.0,
            50.0,
            self.DEFAULT_PUTBACK_INSERT_PREPOSE_X_OFFSET_CM,
            1.0,
            " cm",
        )
        self.putback_insert_prepose_y_spin = self._make_float_spin(
            -50.0,
            50.0,
            self.DEFAULT_PUTBACK_INSERT_PREPOSE_Y_OFFSET_CM,
            0.5,
            " cm",
        )
        self.putback_insert_prepose_spins.extend(
            [
                self.putback_insert_prepose_x_spin,
                self.putback_insert_prepose_y_spin,
            ]
        )
        self._add_left_spins_row(
            step8_layout,
            1,
            "插入预备XY:",
            self.putback_insert_prepose_spins,
        )
        self._add_left_value_row(step8_layout, 2, "IK+MoveJ时间:", "同步骤4")

        step9, step9_layout = self._create_step_group("步骤9：MoveL到插入位")
        if self._workflow_mode == "tail_putback":
            step9.setTitle("步骤10：MoveL到插入位")
        layout.addWidget(step9, base_row + 8 + motion_row_offset, 0, 1, 5)
        self.putback_insert_xy_spins = []
        self.putback_insert_x_spin = self._make_float_spin(
            -50.0,
            50.0,
            self.DEFAULT_PUTBACK_INSERT_X_OFFSET_CM,
            0.5,
            " cm",
        )
        self.putback_insert_y_spin = self._make_float_spin(
            -50.0,
            50.0,
            self.DEFAULT_PUTBACK_INSERT_Y_OFFSET_CM,
            0.5,
            " cm",
        )
        self.putback_insert_xy_spins.extend(
            [
                self.putback_insert_x_spin,
                self.putback_insert_y_spin,
            ]
        )
        self._add_left_spins_row(step9_layout, 0, "插入位XY:", self.putback_insert_xy_spins)
        self._add_left_value_row(step9_layout, 1, "MoveL时间:", "同步骤5")

        step10, step10_layout = self._create_step_group("步骤10：打开夹爪")
        if self._workflow_mode == "tail_putback":
            step10.setTitle("步骤11：打开夹爪")
        layout.addWidget(step10, base_row + 9 + motion_row_offset, 0, 1, 5)
        self.putback_gripper_open_spin = self._make_float_spin(
            -30.0,
            140.0,
            self.DEFAULT_PUTBACK_GRIPPER_OPEN_DEG,
            1.0,
            "°",
        )
        self._add_left_value_row(
            step10_layout,
            0,
            "打开夹爪:",
            self.putback_gripper_open_spin,
        )

        step11, step11_layout = self._create_step_group("步骤11：离开插入位")
        if self._workflow_mode == "tail_putback":
            step11.setTitle("步骤12：离开插入位")
        layout.addWidget(step11, base_row + 10 + motion_row_offset, 0, 1, 5)
        self.putback_leave_insert_x_spin = self._make_float_spin(
            -50.0,
            50.0,
            self.DEFAULT_PUTBACK_LEAVE_INSERT_X_OFFSET_CM,
            0.5,
            " cm",
        )
        self._add_left_value_row(
            step11_layout,
            0,
            "离开插入X+:",
            self.putback_leave_insert_x_spin,
        )
        self._add_left_value_row(step11_layout, 1, "MoveL时间:", "同步骤5")

        step12, step12_layout = self._create_step_group("步骤12：MoveJ回到图书调试位")
        if self._workflow_mode == "tail_putback":
            step12.setTitle("步骤13：MoveJ回到图书调试位")
        layout.addWidget(step12, base_row + 11 + motion_row_offset, 0, 1, 5)
        self.debug_joint_spins = []
        for value in self._default_debug_joints_deg():
            spin = self._make_float_spin(-360.0, 360.0, value, 1.0, "°")
            self.debug_joint_spins.append(spin)
        self._add_left_indexed_joint_spins(
            step12_layout,
            0,
            "目标关节:",
            self.debug_joint_spins,
        )
        self.flow_debug_duration_spin = self._make_float_spin(
            3.0,
            20.0,
            self.DEFAULT_DEBUG_MOVE_DURATION_S,
            0.5,
            " s",
        )
        self._add_left_value_row(
            step12_layout,
            3,
            "MoveJ时间:",
            self.flow_debug_duration_spin,
        )

        return base_row + 12 + motion_row_offset

    def _ensure_debug_pose_controls(self):
        if not hasattr(self, "debug_joint_spins"):
            self.debug_joint_spins = []
            for value in self._default_debug_joints_deg():
                spin = self._make_float_spin(-360.0, 360.0, value, 1.0, "°")
                self.debug_joint_spins.append(spin)
        if not hasattr(self, "flow_debug_duration_spin"):
            self.flow_debug_duration_spin = self._make_float_spin(
                3.0,
                20.0,
                self.DEFAULT_DEBUG_MOVE_DURATION_S,
                0.5,
                " s",
            )

    def set_book_debug_pose_deg(self, joints_deg: Sequence[float]):
        self._ensure_debug_pose_controls()
        values = list(joints_deg)[:6]
        self._loading_workflow_defaults = True
        try:
            for spin, value in zip(self.debug_joint_spins, values):
                spin.setValue(float(value))
        finally:
            self._loading_workflow_defaults = False

    def set_book_gripper_defaults(self, values: dict):
        self._ensure_library_gripper_controls()
        if not self._workflow_default_controls:
            self._workflow_default_controls = self._collect_workflow_default_controls()
        self._loading_workflow_defaults = True
        try:
            for key, value in values.items():
                widget = self._workflow_default_controls.get(key)
                if widget is not None:
                    self._set_workflow_control_value(widget, value)
        finally:
            self._loading_workflow_defaults = False
        self._workflow_defaults.update(values)

    def _load_shared_gripper_defaults(self) -> dict:
        stored = _load_book_workflow_defaults(BOOK_GRIPPER_WORKFLOW_MODE)
        return {key: stored[key] for key in BOOK_GRIPPER_DEFAULT_KEYS if key in stored}

    def _refresh_shared_gripper_defaults(self):
        values = self._load_shared_gripper_defaults()
        if values:
            self.set_book_gripper_defaults(values)
            return
        self._ensure_library_gripper_controls()

    def _ensure_library_gripper_controls(self):
        if not hasattr(self, "gripper_open_spin"):
            self.gripper_open_spin = self._make_float_spin(
                -30.0,
                140.0,
                self.DEFAULT_GRIPPER_OPEN_DEG,
                1.0,
                "°",
            )
        if not hasattr(self, "gripper_close_spin"):
            self.gripper_close_spin = self._make_float_spin(
                -30.0,
                140.0,
                self.DEFAULT_GRIPPER_CLOSE_DEG,
                1.0,
                "°",
            )
        if not hasattr(self, "gripper_effort_spin"):
            self.gripper_effort_spin = self._make_float_spin(
                0.0,
                5.0,
                self.DEFAULT_GRIPPER_HOLD_EFFORT_NM,
                0.01,
                " Nm",
            )
        if not hasattr(self, "gripper_start_effort_spin"):
            self.gripper_start_effort_spin = self._make_float_spin(
                0.0,
                5.0,
                self.DEFAULT_GRIPPER_START_EFFORT_NM,
                0.01,
                " Nm",
            )
        if not hasattr(self, "gripper_start_boost_spin"):
            self.gripper_start_boost_spin = self._make_float_spin(
                0.0,
                3.0,
                self.DEFAULT_GRIPPER_START_BOOST_S,
                0.05,
                " s",
            )
        if not hasattr(self, "gripper_close_speed_spin"):
            self.gripper_close_speed_spin = self._make_float_spin(
                1.0,
                60.0,
                (
                    self.DEFAULT_PUTBACK2_GRIPPER_CLOSE_SPEED_DEG_S
                    if self._workflow_mode == "putback2"
                    else self.DEFAULT_GRIPPER_CLOSE_SPEED_DEG_S
                ),
                0.5,
                "°/s",
            )
        if not hasattr(self, "gripper_open_speed_spin"):
            self.gripper_open_speed_spin = self._make_float_spin(
                1.0,
                60.0,
                self.DEFAULT_GRIPPER_OPEN_SPEED_DEG_S,
                0.5,
                "°/s",
            )
        if not hasattr(self, "gripper_kp_spin"):
            self.gripper_kp_spin = self._make_float_spin(
                0.0,
                200.0,
                self.DEFAULT_GRIPPER_KP,
                1.0,
                "",
            )
        if not hasattr(self, "gripper_kd_spin"):
            self.gripper_kd_spin = self._make_float_spin(
                0.0,
                50.0,
                self.DEFAULT_GRIPPER_KD,
                0.5,
                "",
            )
        if not hasattr(self, "gripper_timeout_spin"):
            self.gripper_timeout_spin = self._make_float_spin(
                1.0,
                30.0,
                self.GRIPPER_CLOSE_TIMEOUT_S,
                0.5,
                " s",
            )
        if not hasattr(self, "gripper_target_tolerance_spin"):
            self.gripper_target_tolerance_spin = self._make_float_spin(
                0.1,
                10.0,
                self.GRIPPER_CLOSE_TOLERANCE_DEG,
                0.1,
                "°",
            )
        if not hasattr(self, "gripper_stall_tolerance_spin"):
            self.gripper_stall_tolerance_spin = self._make_float_spin(
                0.1,
                10.0,
                self.GRIPPER_CLOSE_STALL_TOLERANCE_DEG,
                0.1,
                "°",
            )
        if not hasattr(self, "gripper_stall_time_spin"):
            self.gripper_stall_time_spin = self._make_float_spin(
                0.1,
                10.0,
                self.GRIPPER_CLOSE_STALL_S,
                0.1,
                " s",
            )
        if not hasattr(self, "gripper_min_monitor_spin"):
            self.gripper_min_monitor_spin = self._make_float_spin(
                0.0,
                10.0,
                self.GRIPPER_CLOSE_MIN_MONITOR_S,
                0.1,
                " s",
            )
        if not hasattr(self, "gripper_hold_margin_spin"):
            self.gripper_hold_margin_spin = self._make_float_spin(
                0.0,
                10.0,
                self.DEFAULT_GRIPPER_HOLD_MARGIN_DEG,
                0.1,
                "°",
            )
        if not hasattr(self, "gripper_command_lead_spin"):
            self.gripper_command_lead_spin = self._make_float_spin(
                0.0,
                3.0,
                self.GRIPPER_CLOSE_COMMAND_LEAD_S,
                0.05,
                " s",
            )
        if not hasattr(self, "gripper_stall_lead_threshold_spin"):
            self.gripper_stall_lead_threshold_spin = self._make_float_spin(
                0.0,
                30.0,
                self.GRIPPER_CLOSE_STALL_LEAD_THRESHOLD_DEG,
                0.5,
                "°",
            )
        if not hasattr(self, "gripper_step_interval_spin"):
            self.gripper_step_interval_spin = self._make_float_spin(
                0.02,
                1.0,
                self.GRIPPER_CLOSE_STEP_INTERVAL_S,
                0.01,
                " s",
            )

    @staticmethod
    def _make_int_spin(lo: int, hi: int, value: int, step: int) -> QSpinBox:
        spin = NoWheelSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        spin.setFixedWidth(RealSensePointPanel._spinbox_text_width(
            spin, [str(lo), str(hi), str(value)], extra=34
        ))
        return spin

    @staticmethod
    def _make_float_spin(
        lo: float,
        hi: float,
        value: float,
        step: float,
        suffix: str,
    ) -> QDoubleSpinBox:
        spin = NoWheelDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(3 if abs(step) < 0.1 else 2)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        decimals = 3 if abs(step) < 0.1 else 2
        samples = [
            f"{lo:.{decimals}f}{suffix}",
            f"{hi:.{decimals}f}{suffix}",
            f"{value:.{decimals}f}{suffix}",
        ]
        spin.setFixedWidth(RealSensePointPanel._spinbox_text_width(spin, samples, extra=34))
        return spin

    @staticmethod
    def _spinbox_text_width(widget: QWidget, samples: Sequence[str], extra: int = 34) -> int:
        fm = widget.fontMetrics()
        widest = max((fm.horizontalAdvance(text) for text in samples), default=0)
        return widest + extra

    @staticmethod
    def _add_section_title(layout: QGridLayout, row: int, title: str) -> int:
        title_label = QLabel(title)
        title_label.setWordWrap(True)
        title_label.setStyleSheet("font-weight: 600; padding-top: 4px;")
        layout.addWidget(title_label, row, 0, 1, 5)
        row += 1

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(divider, row, 0, 1, 5)
        return row + 1

    @staticmethod
    def _create_step_group(title: str) -> tuple[QGroupBox, QGridLayout]:
        group = QGroupBox(title)
        layout = QGridLayout(group)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(4)
        return group, layout

    def _add_takeout_step_button(self, layout: QGridLayout, step_index: int, row: int):
        if self._workflow_mode != "takeout":
            return
        button = QPushButton("执行本步")
        button.setObjectName("enableBtn")
        button.clicked.connect(
            lambda _checked=False, idx=step_index: self._execute_single_takeout_step(idx)
        )
        self._compact_flow_button(button, width_texts=("执行本步",))
        self._takeout_step_buttons.append(button)
        layout.addWidget(button, row, 4)

    @staticmethod
    def _button_text_width(widget: QWidget, texts: Sequence[str], extra: int = 28) -> int:
        fm = widget.fontMetrics()
        widest = max((fm.horizontalAdvance(text) for text in texts), default=0)
        return widest + extra

    def _compact_flow_button(
        self,
        button: QPushButton,
        width_texts: Optional[Sequence[str]] = None,
    ):
        button.setProperty("compactFlowButton", "true")
        button.setFixedHeight(self.FLOW_BUTTON_HEIGHT)
        button.style().unpolish(button)
        button.style().polish(button)
        if width_texts is not None:
            button.setMinimumWidth(
                self._button_text_width(
                    button,
                    width_texts,
                    extra=self.FLOW_BUTTON_WIDTH_EXTRA,
                )
            )

    @staticmethod
    def _wrapped_label_height(label: QLabel, lines: int) -> int:
        return label.fontMetrics().lineSpacing() * lines + 6

    def _stabilize_flow_label(self, label: QLabel, lines: int = 2):
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        label.setMaximumWidth(420)
        label.setFixedHeight(self._wrapped_label_height(label, lines))

    @staticmethod
    def _add_labelled_spin(
        layout: QGridLayout,
        row: int,
        label: str,
        spin: QDoubleSpinBox,
        label_col: int,
        spin_col: int,
        *,
        align_right: bool = True,
    ):
        label_widget = QLabel(label)
        if align_right:
            label_widget.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(label_widget, row, label_col)
        layout.addWidget(spin, row, spin_col)
        return label_widget

    @staticmethod
    def _add_row_spins(
        layout: QGridLayout,
        row: int,
        label: str,
        spins: Sequence[QDoubleSpinBox],
        *,
        label_col: int = 0,
        spin_col: int = 1,
    ):
        label_widget = QLabel(label)
        label_widget.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(label_widget, row, label_col)
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        for spin in spins:
            row_layout.addWidget(spin)
        row_layout.addStretch()
        layout.addLayout(row_layout, row, spin_col, 1, 4)

    @staticmethod
    def _add_compact_joint_spins(
        layout: QGridLayout,
        row: int,
        label: str,
        spins: Sequence[QDoubleSpinBox],
    ):
        label_widget = QLabel(label)
        label_widget.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(label_widget, row, 0, 2, 1)
        for idx, spin in enumerate(spins):
            spin_row = row + idx // 3
            spin_col = 1 + idx % 3
            layout.addWidget(spin, spin_row, spin_col)

    @staticmethod
    def _add_indexed_joint_spins(
        layout: QGridLayout,
        row: int,
        label: str,
        spins: Sequence[QDoubleSpinBox],
    ):
        label_widget = QLabel(label)
        label_widget.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(label_widget, row, 0, 2, 1)

        joint_grid = QGridLayout()
        joint_grid.setContentsMargins(0, 0, 0, 0)
        joint_grid.setHorizontalSpacing(8)
        joint_grid.setVerticalSpacing(4)
        for idx, spin in enumerate(spins):
            cell = QHBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(3)
            joint_label = QLabel(f"J{idx + 1}")
            joint_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            joint_label.setFixedWidth(18)
            cell.addWidget(joint_label)
            cell.addWidget(spin)
            cell.addStretch()
            joint_grid.addLayout(cell, idx // 3, idx % 3)
        layout.addLayout(joint_grid, row, 1, 2, 4)
        layout.setColumnStretch(4, 1)

    @staticmethod
    def _add_compact_value_row(layout: QGridLayout, row: int, label: str, value):
        label_widget = QLabel(label)
        label_widget.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(label_widget, row, 0)

        value_row = QHBoxLayout()
        value_row.setContentsMargins(0, 0, 0, 0)
        value_row.setSpacing(4)
        if isinstance(value, str):
            value_widget = QLabel(value)
            value_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            value_row.addWidget(value_widget)
        else:
            value_row.addWidget(value)
        value_row.addStretch()
        layout.addLayout(value_row, row, 1, 1, 4)

    @staticmethod
    def _add_left_value_row(layout: QGridLayout, row: int, label: str, value):
        value_row = QHBoxLayout()
        value_row.setContentsMargins(0, 0, 0, 0)
        value_row.setSpacing(4)
        label_widget = QLabel(label)
        label_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        value_row.addWidget(label_widget)
        if isinstance(value, str):
            value_widget = QLabel(value)
            value_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            value_row.addWidget(value_widget)
        else:
            value_row.addWidget(value)
        value_row.addStretch()
        layout.addLayout(value_row, row, 0, 1, 5)

    @staticmethod
    def _add_left_value_pairs(
        layout: QGridLayout,
        row: int,
        pairs: Sequence[tuple[str, object]],
        *,
        columns: int = 2,
    ):
        columns = max(1, int(columns))
        for idx, (label, value) in enumerate(pairs):
            pair_row = row + idx // columns
            pair_col = (idx % columns) * 2
            value_row = QHBoxLayout()
            value_row.setContentsMargins(0, 0, 0, 0)
            value_row.setSpacing(4)
            label_widget = QLabel(label)
            label_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            value_row.addWidget(label_widget)
            if isinstance(value, str):
                value_widget = QLabel(value)
                value_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                value_row.addWidget(value_widget)
            else:
                value_row.addWidget(value)
            value_row.addStretch()
            layout.addLayout(value_row, pair_row, pair_col, 1, 2)
        layout.setColumnStretch(columns * 2, 1)

    @staticmethod
    def _add_left_spins_row(
        layout: QGridLayout,
        row: int,
        label: str,
        spins: Sequence[QDoubleSpinBox],
    ):
        value_row = QHBoxLayout()
        value_row.setContentsMargins(0, 0, 0, 0)
        value_row.setSpacing(4)
        label_widget = QLabel(label)
        label_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        value_row.addWidget(label_widget)
        for spin in spins:
            value_row.addWidget(spin)
        value_row.addStretch()
        layout.addLayout(value_row, row, 0, 1, 5)

    @staticmethod
    def _add_left_indexed_joint_spins(
        layout: QGridLayout,
        row: int,
        label: str,
        spins: Sequence[QDoubleSpinBox],
    ):
        label_row = QHBoxLayout()
        label_row.setContentsMargins(0, 0, 0, 0)
        label_row.setSpacing(4)
        label_widget = QLabel(label)
        label_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label_row.addWidget(label_widget)
        label_row.addStretch()
        layout.addLayout(label_row, row, 0, 1, 5)

        joint_grid = QGridLayout()
        joint_grid.setContentsMargins(0, 0, 0, 0)
        joint_grid.setHorizontalSpacing(8)
        joint_grid.setVerticalSpacing(4)
        for idx, spin in enumerate(spins):
            cell = QHBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(3)
            joint_label = QLabel(f"J{idx + 1}")
            joint_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            joint_label.setFixedWidth(18)
            cell.addWidget(joint_label)
            cell.addWidget(spin)
            cell.addStretch()
            joint_grid.addLayout(cell, idx // 3, idx % 3)
        layout.addLayout(joint_grid, row + 1, 0, 2, 5)

    @staticmethod
    def _add_step_group(
        parent_layout: QGridLayout,
        row: int,
        title: str,
    ) -> tuple[QGridLayout, int]:
        group, group_layout = RealSensePointPanel._create_step_group(title)
        parent_layout.addWidget(group, row, 0, 1, 5)
        return group_layout, row + 1

    def _setup_workflow_default_controls(self):
        self._ensure_library_gripper_controls()
        self._workflow_default_controls = self._collect_workflow_default_controls()
        self._loading_workflow_defaults = True
        try:
            self._apply_workflow_defaults(
                self._workflow_defaults
            )
        finally:
            self._loading_workflow_defaults = False
        self._connect_workflow_default_controls()

    def _collect_workflow_default_controls(self) -> dict[str, QWidget]:
        controls: dict[str, QWidget] = {}

        def add(name: str, widget):
            if widget is not None:
                controls[name] = widget

        def add_many(prefix: str, widgets):
            for idx, widget in enumerate(widgets or []):
                add(f"{prefix}_{idx + 1}", widget)

        if hasattr(self, "template_combo"):
            add("template", self.template_combo)

        if self._is_takeout_based_workflow():
            add_many("target_comp_xyz", getattr(self, "target_comp_xyz_spins", []))
            add_many("pregrasp_offset_xyz", getattr(self, "pregrasp_offset_xyz_spins", []))
            add_many("pregrasp_rpy", getattr(self, "pregrasp_rpy_spins", []))
            add("flow_ik_duration", getattr(self, "flow_ik_duration_spin", None))
            add("pregrasp_prepare_tools", getattr(self, "pregrasp_prepare_tools_check", None))
            add("rod_port", getattr(self, "rod_port_edit", None))
            add("rod_grasp", getattr(self, "rod_grasp_spin", None))
            add("rod_speed", getattr(self, "rod_speed_spin", None))
            add("rod_acc", getattr(self, "rod_acc_spin", None))
            add_many("relative_target_xyz", getattr(self, "relative_target_xyz_spins", []))
            add_many("relative_target_rpy", getattr(self, "relative_target_rpy_spins", []))
            add("relative_target_duration", getattr(self, "relative_target_duration_spin", None))
            add("gripper_open", getattr(self, "gripper_open_spin", None))
            add("gripper_close", getattr(self, "gripper_close_spin", None))
            add("gripper_effort", getattr(self, "gripper_effort_spin", None))
            add("gripper_start_effort", getattr(self, "gripper_start_effort_spin", None))
            add("gripper_start_boost", getattr(self, "gripper_start_boost_spin", None))
            add("gripper_close_speed", getattr(self, "gripper_close_speed_spin", None))
            add("gripper_open_speed", getattr(self, "gripper_open_speed_spin", None))
            add("gripper_kp", getattr(self, "gripper_kp_spin", None))
            add("gripper_kd", getattr(self, "gripper_kd_spin", None))
            add("gripper_timeout", getattr(self, "gripper_timeout_spin", None))
            add("gripper_target_tolerance", getattr(self, "gripper_target_tolerance_spin", None))
            add("gripper_stall_tolerance", getattr(self, "gripper_stall_tolerance_spin", None))
            add("gripper_stall_time", getattr(self, "gripper_stall_time_spin", None))
            add("gripper_min_monitor", getattr(self, "gripper_min_monitor_spin", None))
            add("gripper_hold_margin", getattr(self, "gripper_hold_margin_spin", None))
            add("gripper_command_lead", getattr(self, "gripper_command_lead_spin", None))
            add(
                "gripper_stall_lead_threshold",
                getattr(self, "gripper_stall_lead_threshold_spin", None),
            )
            add("gripper_step_interval", getattr(self, "gripper_step_interval_spin", None))
            if self._workflow_mode == "putback2":
                add_many("putback2_reset_rpy", getattr(self, "putback2_reset_rpy_spins", []))
                add(
                    "putback2_reset_rpy_duration",
                    getattr(self, "putback2_reset_rpy_duration_spin", None),
                )
                add("putback2_x_offset", getattr(self, "putback2_x_offset_spin", None))
                add(
                    "putback2_x_move_duration",
                    getattr(self, "putback2_x_move_duration_spin", None),
                )
                add(
                    "putback2_gripper_partial_open",
                    getattr(self, "putback2_gripper_partial_open_spin", None),
                )
                add(
                    "putback2_gripper_partial_open_speed",
                    getattr(self, "putback2_gripper_partial_open_speed_spin", None),
                )
                add(
                    "putback2_rod_release",
                    getattr(self, "putback2_rod_release_spin", None),
                )
                add("putback2_z_offset", getattr(self, "putback2_z_offset_spin", None))
                add(
                    "putback2_z_move_duration",
                    getattr(self, "putback2_z_move_duration_spin", None),
                )
                add(
                    "putback2_gripper_full_open",
                    getattr(self, "putback2_gripper_full_open_spin", None),
                )
                add(
                    "putback2_gripper_full_open_speed",
                    getattr(self, "putback2_gripper_full_open_speed_spin", None),
                )
                add(
                    "putback2_after_open_x1_offset",
                    getattr(self, "putback2_after_open_x1_offset_spin", None),
                )
                add(
                    "putback2_after_open_x1_duration",
                    getattr(self, "putback2_after_open_x1_duration_spin", None),
                )
                add(
                    "putback2_combo_close",
                    getattr(self, "putback2_combo_close_spin", None),
                )
                add(
                    "putback2_combo_close_speed",
                    getattr(self, "putback2_combo_close_speed_spin", None),
                )
                add(
                    "putback2_combo_y_offset",
                    getattr(self, "putback2_combo_y_offset_spin", None),
                )
                add(
                    "putback2_combo_y_duration",
                    getattr(self, "putback2_combo_y_duration_spin", None),
                )
                add(
                    "putback2_combo_wait",
                    getattr(self, "putback2_combo_wait_spin", None),
                )
                add(
                    "putback2_combo_x_offset",
                    getattr(self, "putback2_combo_x_offset_spin", None),
                )
                add(
                    "putback2_combo_x_duration",
                    getattr(self, "putback2_combo_x_duration_spin", None),
                )
                add(
                    "putback2_combo_full_open",
                    getattr(self, "putback2_combo_full_open_spin", None),
                )
                add(
                    "putback2_combo_full_open_speed",
                    getattr(self, "putback2_combo_full_open_speed_spin", None),
                )
                add(
                    "putback2_after_combo_x_offset",
                    getattr(self, "putback2_after_combo_x_offset_spin", None),
                )
                add(
                    "putback2_after_combo_x_duration",
                    getattr(self, "putback2_after_combo_x_duration_spin", None),
                )
            else:
                add_many(
                    "post_gripper_movej_xyz",
                    getattr(self, "post_gripper_movej_xyz_spins", []),
                )
                add_many(
                    "post_gripper_movej_rpy",
                    getattr(self, "post_gripper_movej_rpy_spins", []),
                )
                add(
                    "post_gripper_movej_duration",
                    getattr(self, "post_gripper_movej_duration_spin", None),
                )
                add_many("debug_joint", getattr(self, "debug_joint_spins", []))
                add("flow_debug_duration", getattr(self, "flow_debug_duration_spin", None))
                add_many("turn_stage1_joint", getattr(self, "turn_stage1_joint_spins", []))
                add("flow_turn_duration", getattr(self, "flow_turn_duration_spin", None))
                add_many(
                    "takeout_specified_pose_xyz",
                    getattr(self, "takeout_specified_pose_xyz_spins", []),
                )
                add_many(
                    "takeout_specified_pose_rpy",
                    getattr(self, "takeout_specified_pose_rpy_spins", []),
                )
                add(
                    "takeout_specified_pose_duration",
                    getattr(self, "takeout_specified_pose_duration_spin", None),
                )
                add_many(
                    "takeout_step13_fine_tune_xyz",
                    getattr(self, "takeout_step13_fine_tune_xyz_spins", []),
                )
                add_many(
                    "takeout_step13_fine_tune_rpy",
                    getattr(self, "takeout_step13_fine_tune_rpy_spins", []),
                )
                add(
                    "takeout_step13_fine_tune_duration",
                    getattr(self, "takeout_step13_fine_tune_duration_spin", None),
                )
                add_many("final_joint", getattr(self, "final_joint_spins", []))
                add("flow_final_duration", getattr(self, "flow_final_duration_spin", None))
        else:
            add("gripper_open", getattr(self, "gripper_open_spin", None))
            add("gripper_close", getattr(self, "gripper_close_spin", None))
            add("gripper_effort", getattr(self, "gripper_effort_spin", None))
            add("gripper_start_effort", getattr(self, "gripper_start_effort_spin", None))
            add("gripper_start_boost", getattr(self, "gripper_start_boost_spin", None))
            add("gripper_close_speed", getattr(self, "gripper_close_speed_spin", None))
            add("gripper_open_speed", getattr(self, "gripper_open_speed_spin", None))
            add("gripper_kp", getattr(self, "gripper_kp_spin", None))
            add("gripper_kd", getattr(self, "gripper_kd_spin", None))
            add("gripper_timeout", getattr(self, "gripper_timeout_spin", None))
            add("gripper_target_tolerance", getattr(self, "gripper_target_tolerance_spin", None))
            add("gripper_stall_tolerance", getattr(self, "gripper_stall_tolerance_spin", None))
            add("gripper_stall_time", getattr(self, "gripper_stall_time_spin", None))
            add("gripper_min_monitor", getattr(self, "gripper_min_monitor_spin", None))
            add("gripper_hold_margin", getattr(self, "gripper_hold_margin_spin", None))
            add("gripper_command_lead", getattr(self, "gripper_command_lead_spin", None))
            add(
                "gripper_stall_lead_threshold",
                getattr(self, "gripper_stall_lead_threshold_spin", None),
            )
            add("gripper_step_interval", getattr(self, "gripper_step_interval_spin", None))
            add("putback_target_line", getattr(self, "putback_target_line_combo", None))
            add("putback_pick_ratio", getattr(self, "putback_pick_ratio_combo", None))
            add("tail_book_height", getattr(self, "tail_book_height_spin", None))
            add("tail_book_thickness", getattr(self, "tail_book_thickness_spin", None))
            add("tail_gripper_height", getattr(self, "tail_gripper_height_spin", None))
            add("tail_preplace_motion", getattr(self, "tail_preplace_motion_combo", None))
            add_many("tail_preplace_xyz", getattr(self, "tail_preplace_xyz_spins", []))
            add_many("tail_preplace_rpy", getattr(self, "tail_preplace_rpy_spins", []))
            add(
                "tail_preplace_movej_duration",
                getattr(self, "tail_preplace_movej_duration_spin", None),
            )
            add("tail_place_motion", getattr(self, "tail_place_motion_combo", None))
            add_many("tail_place_offset_xyz", getattr(self, "tail_place_offset_xyz_spins", []))
            add_many(
                "tail_place_rpy_offset",
                getattr(self, "tail_place_rpy_offset_spins", []),
            )
            add(
                "tail_target_movel_duration",
                getattr(self, "tail_target_movel_duration_spin", None),
            )
            add(
                "tail_fine_tune_motion",
                getattr(self, "tail_fine_tune_motion_combo", None),
            )
            add_many("tail_fine_tune_xyz", getattr(self, "tail_fine_tune_xyz_spins", []))
            add_many("tail_fine_tune_rpy", getattr(self, "tail_fine_tune_rpy_spins", []))
            add(
                "tail_fine_tune_movel_duration",
                getattr(self, "tail_fine_tune_movel_duration_spin", None),
            )
            add(
                "tail_gripper_slight_open",
                getattr(self, "tail_gripper_slight_open_spin", None),
            )
            add(
                "tail_gripper_slight_open_speed",
                getattr(self, "tail_gripper_slight_open_speed_spin", None),
            )
            add(
                "tail_gripper_slight_open_timeout",
                getattr(self, "tail_gripper_slight_open_timeout_spin", None),
            )
            add(
                "tail_gripper_slight_open_target_tolerance",
                getattr(self, "tail_gripper_slight_open_target_tolerance_spin", None),
            )
            add(
                "tail_gripper_slight_open_stall_tolerance",
                getattr(self, "tail_gripper_slight_open_stall_tolerance_spin", None),
            )
            add(
                "tail_gripper_slight_open_stall_time",
                getattr(self, "tail_gripper_slight_open_stall_time_spin", None),
            )
            add(
                "tail_gripper_slight_open_min_monitor",
                getattr(self, "tail_gripper_slight_open_min_monitor_spin", None),
            )
            add(
                "tail_gripper_slight_open_hold_margin",
                getattr(self, "tail_gripper_slight_open_hold_margin_spin", None),
            )
            add(
                "tail_gripper_slight_open_command_lead",
                getattr(self, "tail_gripper_slight_open_command_lead_spin", None),
            )
            add(
                "tail_gripper_slight_open_stall_lead_threshold",
                getattr(self, "tail_gripper_slight_open_stall_lead_threshold_spin", None),
            )
            add_many("tail_backoff_xyz", getattr(self, "tail_backoff_xyz_spins", []))
            add_many("tail_backoff_rpy", getattr(self, "tail_backoff_rpy_spins", []))
            add(
                "tail_backoff_movel_duration",
                getattr(self, "tail_backoff_movel_duration_spin", None),
            )
            add("tail_prepush_motion", getattr(self, "tail_prepush_motion_combo", None))
            add_many(
                "tail_prepush_offset_xyz",
                getattr(self, "tail_prepush_offset_xyz_spins", []),
            )
            add_many(
                "tail_prepush_rpy_offset",
                getattr(self, "tail_prepush_rpy_offset_spins", []),
            )
            add(
                "tail_prepush_movel_duration",
                getattr(self, "tail_prepush_movel_duration_spin", None),
            )
            add_many(
                "tail_push_offset_xyz",
                getattr(self, "tail_push_offset_xyz_spins", []),
            )
            add_many(
                "tail_push_rpy_offset",
                getattr(self, "tail_push_rpy_offset_spins", []),
            )
            add(
                "tail_push_movel_duration",
                getattr(self, "tail_push_movel_duration_spin", None),
            )
            add("tail_final_home", getattr(self, "tail_final_home_combo", None))
            add(
                "tail_final_home_duration",
                getattr(self, "tail_final_home_duration_spin", None),
            )
            add_many("putback_target_rpy", getattr(self, "putback_target_rpy_spins", []))
            add_many("putback_target_comp_xyz", getattr(self, "putback_target_comp_xyz_spins", []))
            add_many("putback_prepush_xyz", getattr(self, "putback_prepush_xyz_spins", []))
            add_many(
                "putback_prepush_rpy_offset",
                getattr(self, "putback_prepush_rpy_offset_spins", []),
            )
            add_many("putback_push_xyz", getattr(self, "putback_push_xyz_spins", []))
            add_many(
                "putback_push_rpy_offset",
                getattr(self, "putback_push_rpy_offset_spins", []),
            )
            add_many(
                "putback_push_out_xyz",
                getattr(self, "putback_push_out_xyz_spins", []),
            )
            add_many(
                "putback_push_out_rpy_offset",
                getattr(self, "putback_push_out_rpy_offset_spins", []),
            )
            add("putback_push_movel_duration", getattr(self, "putback_push_movel_duration_spin", None))
            add("putback_leave_push_y", getattr(self, "putback_leave_push_y_spin", None))
            add_many("putback_insert_rpy", getattr(self, "putback_insert_rpy_spins", []))
            add("putback_insert_prepose_x", getattr(self, "putback_insert_prepose_x_spin", None))
            add("putback_insert_prepose_y", getattr(self, "putback_insert_prepose_y_spin", None))
            add("putback_insert_x", getattr(self, "putback_insert_x_spin", None))
            add("putback_insert_y", getattr(self, "putback_insert_y_spin", None))
            add("putback_leave_insert_x", getattr(self, "putback_leave_insert_x_spin", None))
            add("putback_gripper_open", getattr(self, "putback_gripper_open_spin", None))
            add("flow_movel_duration", getattr(self, "flow_movel_duration_spin", None))
            add("flow_ik_duration", getattr(self, "flow_ik_duration_spin", None))
            add("flow_debug_duration", getattr(self, "flow_debug_duration_spin", None))
            add_many("debug_joint", getattr(self, "debug_joint_spins", []))

        return controls

    def _connect_workflow_default_controls(self):
        for widget in self._workflow_default_controls.values():
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                widget.valueChanged.connect(self._on_workflow_default_control_changed)
            elif isinstance(widget, QLineEdit):
                widget.editingFinished.connect(self._on_workflow_default_control_changed)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._on_workflow_default_control_changed)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._on_workflow_default_control_changed)

    def _apply_workflow_defaults(self, defaults: dict):
        if "gripper_effort" in defaults:
            try:
                if float(defaults["gripper_effort"]) < 0.25:
                    defaults = dict(defaults)
                    defaults["gripper_effort"] = 0.25
            except (TypeError, ValueError):
                pass
        if "gripper_start_effort" in defaults:
            try:
                if abs(float(defaults["gripper_start_effort"]) - 0.45) < 1e-9:
                    defaults = dict(defaults)
                    defaults["gripper_start_effort"] = self.DEFAULT_GRIPPER_START_EFFORT_NM
            except (TypeError, ValueError):
                pass
        if "gripper_start_boost" in defaults:
            try:
                if abs(float(defaults["gripper_start_boost"]) - 0.8) < 1e-9:
                    defaults = dict(defaults)
                    defaults["gripper_start_boost"] = self.DEFAULT_GRIPPER_START_BOOST_S
            except (TypeError, ValueError):
                pass
        gripper_default_values = {
            "gripper_open_speed": self.DEFAULT_GRIPPER_OPEN_SPEED_DEG_S,
            "gripper_timeout": self.GRIPPER_CLOSE_TIMEOUT_S,
            "gripper_target_tolerance": self.GRIPPER_CLOSE_TOLERANCE_DEG,
            "gripper_stall_tolerance": self.GRIPPER_CLOSE_STALL_TOLERANCE_DEG,
            "gripper_stall_time": self.GRIPPER_CLOSE_STALL_S,
            "gripper_min_monitor": self.GRIPPER_CLOSE_MIN_MONITOR_S,
            "gripper_hold_margin": self.DEFAULT_GRIPPER_HOLD_MARGIN_DEG,
            "gripper_command_lead": self.GRIPPER_CLOSE_COMMAND_LEAD_S,
            "gripper_stall_lead_threshold": self.GRIPPER_CLOSE_STALL_LEAD_THRESHOLD_DEG,
            "gripper_step_interval": self.GRIPPER_CLOSE_STEP_INTERVAL_S,
        }
        missing_gripper_defaults = {
            key: value for key, value in gripper_default_values.items() if key not in defaults
        }
        if missing_gripper_defaults:
            defaults = dict(defaults)
            defaults.update(missing_gripper_defaults)
        shared_gripper_defaults = self._load_shared_gripper_defaults()
        if shared_gripper_defaults:
            defaults = dict(defaults)
            defaults.update(shared_gripper_defaults)
        if (
            "putback_push_base_y" not in defaults
            and "putback_push_end_y" in defaults
        ):
            defaults = dict(defaults)
            defaults["putback_push_base_y"] = defaults["putback_push_end_y"]
        if (
            "putback_push_base_y" not in defaults
            and "putback_push_right" in defaults
        ):
            defaults = dict(defaults)
            defaults["putback_push_base_y"] = defaults["putback_push_right"]
        if "putback_push_out_xyz_2" not in defaults and "putback_push_base_y" in defaults:
            defaults = dict(defaults)
            defaults.setdefault("putback_push_out_xyz_1", 0.0)
            defaults["putback_push_out_xyz_2"] = defaults["putback_push_base_y"]
            defaults.setdefault("putback_push_out_xyz_3", 0.0)
        if "putback_prepush_xyz_1" not in defaults and "putback_prepush_x" in defaults:
            defaults = dict(defaults)
            defaults["putback_prepush_xyz_1"] = defaults["putback_prepush_x"]
            defaults.setdefault("putback_prepush_xyz_2", 0.0)
            defaults.setdefault("putback_prepush_xyz_3", 0.0)
        if (
            self._workflow_mode == "putback2"
            and "putback2_gripper_full_open" not in defaults
            and "putback2_gripper_open" in defaults
        ):
            defaults = dict(defaults)
            defaults["putback2_gripper_full_open"] = defaults["putback2_gripper_open"]
        if (
            self._workflow_mode == "putback2"
            and "putback2_gripper_full_open_speed" not in defaults
            and "putback2_gripper_open_speed" in defaults
        ):
            defaults = dict(defaults)
            defaults["putback2_gripper_full_open_speed"] = defaults[
                "putback2_gripper_open_speed"
            ]
        if self._workflow_mode == "tail_putback":
            tail_aliases = {
                "tail_reapproach_offset_xyz_1": "tail_prepush_offset_xyz_1",
                "tail_reapproach_offset_xyz_2": "tail_prepush_offset_xyz_2",
                "tail_reapproach_offset_xyz_3": "tail_prepush_offset_xyz_3",
                "tail_reapproach_rpy_offset_1": "tail_prepush_rpy_offset_1",
                "tail_reapproach_rpy_offset_2": "tail_prepush_rpy_offset_2",
                "tail_reapproach_rpy_offset_3": "tail_prepush_rpy_offset_3",
                "tail_reapproach_motion": "tail_prepush_motion",
                "tail_reapproach_movel_duration": "tail_prepush_movel_duration",
                "tail_target_comp_offset_xyz_1": "tail_push_offset_xyz_1",
                "tail_target_comp_offset_xyz_2": "tail_push_offset_xyz_2",
                "tail_target_comp_offset_xyz_3": "tail_push_offset_xyz_3",
                "tail_target_comp_rpy_offset_1": "tail_push_rpy_offset_1",
                "tail_target_comp_rpy_offset_2": "tail_push_rpy_offset_2",
                "tail_target_comp_rpy_offset_3": "tail_push_rpy_offset_3",
                "tail_target_comp_movel_duration": "tail_push_movel_duration",
                "tail_return_home": "tail_final_home",
                "tail_return_home_duration": "tail_final_home_duration",
                "tail_final_zero_duration": "tail_final_home_duration",
            }
            missing_aliases = {
                new_key: defaults[old_key]
                for old_key, new_key in tail_aliases.items()
                if old_key in defaults and new_key not in defaults
            }
            if missing_aliases:
                defaults = dict(defaults)
                defaults.update(missing_aliases)
        for key, value in defaults.items():
            widget = self._workflow_default_controls.get(key)
            if widget is None:
                continue
            blocker = QSignalBlocker(widget)
            try:
                self._set_workflow_control_value(widget, value)
            finally:
                del blocker

    def _begin_workflow_default_edit(self):
        if self._workflow_default_edit_enabled:
            return
        self._workflow_default_edit_snapshot = self._current_workflow_default_values()
        self._set_workflow_default_edit_enabled(True)

    def _confirm_workflow_default_edit(self):
        if not self._workflow_default_edit_enabled:
            return
        if self._save_current_workflow_defaults():
            self._workflow_default_edit_snapshot = {}
            self._set_workflow_default_edit_enabled(False)

    def _cancel_workflow_default_edit(self):
        if not self._workflow_default_edit_enabled:
            return
        self._loading_workflow_defaults = True
        try:
            self._apply_workflow_defaults(self._workflow_default_edit_snapshot)
        finally:
            self._loading_workflow_defaults = False
        self._workflow_default_edit_snapshot = {}
        self._on_workflow_target_changed()
        self._set_workflow_default_edit_enabled(False)

    def _set_workflow_default_edit_enabled(self, enabled: bool):
        self._workflow_default_edit_enabled = bool(enabled)
        self._update_workflow_default_edit_buttons()

    def _update_workflow_default_edit_buttons(self):
        editing = self._workflow_default_edit_enabled
        if hasattr(self, "flow_defaults_btn"):
            self.flow_defaults_btn.setText("修改默认值中" if editing else "修改默认值")
            self.flow_defaults_btn.setEnabled(not editing)
        if hasattr(self, "flow_defaults_confirm_btn"):
            self.flow_defaults_confirm_btn.setVisible(editing)
            self.flow_defaults_confirm_btn.setEnabled(editing)
        if hasattr(self, "flow_defaults_cancel_btn"):
            self.flow_defaults_cancel_btn.setVisible(editing)
            self.flow_defaults_cancel_btn.setEnabled(editing)

    def _on_workflow_default_control_changed(self, *_args):
        if self._loading_workflow_defaults:
            return

    def _current_workflow_default_values(self) -> dict[str, object]:
        return {
            key: self._workflow_control_value(widget)
            for key, widget in self._workflow_default_controls.items()
        }

    def _save_current_workflow_defaults(self) -> bool:
        values = self._current_workflow_default_values()
        try:
            _save_book_workflow_defaults(self._workflow_mode, values)
            self._workflow_defaults = dict(values)
        except Exception as exc:
            self.error_occurred.emit(f"保存流程默认值失败: {exc}")
            return False
        return True

    @staticmethod
    def _workflow_control_value(widget):
        if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
            return widget.value()
        if isinstance(widget, QLineEdit):
            return widget.text()
        if isinstance(widget, QComboBox):
            data = widget.currentData()
            return str(data) if data is not None else widget.currentText()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        return None

    @staticmethod
    def _set_workflow_control_value(widget, value):
        if isinstance(widget, QDoubleSpinBox):
            try:
                widget.setValue(float(value))
            except (TypeError, ValueError):
                return
        elif isinstance(widget, QSpinBox):
            try:
                widget.setValue(int(round(float(value))))
            except (TypeError, ValueError):
                return
        elif isinstance(widget, QLineEdit):
            widget.setText("" if value is None else str(value))
        elif isinstance(widget, QComboBox):
            text = "" if value is None else str(value)
            idx = widget.findData(text)
            if idx < 0:
                idx = widget.findText(text)
            if idx >= 0:
                widget.setCurrentIndex(idx)
        elif isinstance(widget, QCheckBox):
            if isinstance(value, str):
                widget.setChecked(value.strip().lower() in {"1", "true", "yes", "on"})
            else:
                widget.setChecked(bool(value))

    def _bind_workflow_target_refresh(self, spins: Sequence[QDoubleSpinBox]):
        for spin in spins:
            spin.valueChanged.connect(lambda *_args: self._on_workflow_target_changed())

    def _start_capture(self):
        if self._capture_worker is not None and self._capture_worker.isRunning():
            return False
        if self._depth_max_m <= self._depth_min_m:
            self._set_error(tr("pc.depth_range_error"))
            return False

        self.capture_btn.setEnabled(False)
        self.capture_btn.setText(tr("pc.capturing"))
        self.status_label.setText(tr("pc.capturing"))
        if hasattr(self, "flow_detect_btn"):
            self.flow_detect_btn.setEnabled(False)
        self._clear_selection()

        self._capture_worker = RealSenseCaptureWorker(
            serial=self._serial,
            width=self._width,
            height=self._height,
            fps=self._fps,
            warmup=self._warmup,
            timeout_ms=self._timeout_ms,
            align_depth_to_color=self._align_depth_to_color,
            depth_min_m=self._depth_min_m,
            depth_max_m=self._depth_max_m,
            stride=self._stride,
            include_color=self._include_color,
            depth_width=self._depth_width,
            depth_height=self._depth_height,
            parent=self,
        )
        self._capture_worker.capture_finished.connect(self._on_capture_finished)
        self._capture_worker.error_occurred.connect(self._on_capture_error)
        self._capture_worker.finished.connect(self._on_capture_thread_finished)
        self._capture_worker.start()
        return True

    def _on_capture_finished(self, frame, point_cloud):
        self._frame = frame
        self._point_cloud = point_cloud
        self._book_spine_pick = None
        self.rgb_preview_label.hide()
        self._selected_display_index = None
        self._selected_raw_index = None
        self._selected_camera_point_m = None
        self._selected_robot_target_raw_m = None
        self._target_robot_point_m = None
        self._tail_putback_step4_target_point_m = None
        self._display_cloud(point_cloud)
        self.pick_mode_btn.setChecked(True)
        self.status_label.setText(
            tr("pc.capture_done", n=point_cloud.size, frame=frame.frame_number)
        )
        self.log_message.emit(
            tr("pc.capture_done", n=point_cloud.size, frame=frame.frame_number)
        )
        self._update_move_button_state()
        if self._flow_active and self._flow_step_index == 0:
            self._advance_book_grasp_flow("点云采集完成")

    def _detect_book_target(self):
        try:
            if self._detect_worker is not None and self._detect_worker.isRunning():
                return False

            template = Path(
                self.template_combo.currentData() or str(DEFAULT_BOOK_TEMPLATE_PATH)
            )
            self._detect_worker = BookSpineDetectWorker(
                serial=self._serial,
                width=self._width,
                height=self._height,
                fps=self._fps,
                warmup=self._warmup,
                timeout_ms=self._timeout_ms,
                align_depth_to_color=self._align_depth_to_color,
                depth_min_m=self._depth_min_m,
                depth_max_m=self._depth_max_m,
                stride=self._stride,
                include_color=self._include_color,
                depth_width=self._depth_width,
                depth_height=self._depth_height,
                template_path=template,
                config=_make_book_match_config(),
                parent=self,
            )
            self._detect_worker.detection_finished.connect(self._on_book_detection_finished)
            self._detect_worker.status_message.connect(self._on_book_detection_status)
            self._detect_worker.error_occurred.connect(self._on_capture_error)
            self._detect_worker.finished.connect(self._on_book_detection_thread_finished)

            self.capture_btn.setEnabled(False)
            if hasattr(self, "flow_detect_btn"):
                self.flow_detect_btn.setEnabled(False)
            self.capture_btn.setText(tr("pc.capturing"))
            self.status_label.setText(tr("pc.capturing"))
            self.book_status_label.setText(
                f"识别中，模板={self.template_combo.currentText()}，未达标会自动重采集..."
            )
            self._clear_selection()
            self._detect_worker.start()
            return True
        except Exception as exc:
            self._on_capture_error(str(exc))
            return False

    def _on_capture_error(self, message: str):
        self._set_error(tr("pc.capture_error", msg=message))
        if self._flow_active and self._flow_waiting_kind in {"capture", "detect"}:
            self._flow_waiting_motion = False
            self._flow_waiting_kind = None
            self._flow_rollback_waiting = False
            current = min(self._flow_step_index + 1, len(self._book_grasp_steps()))
            self.flow_status_label.setText(
                f"第 {current} 步出错，可调整后重试：{message}"
            )
            self._update_flow_button_state()

    def _on_capture_thread_finished(self):
        self.capture_btn.setEnabled(True)
        if hasattr(self, "flow_detect_btn"):
            self.flow_detect_btn.setEnabled(True)
        self.capture_btn.setText(tr("pc.capture"))

    def _on_book_detection_status(self, message: str):
        self.book_status_label.setText(message)

    def _on_book_detection_finished(self, frame, point_cloud, pick):
        self._frame = frame
        self._point_cloud = point_cloud
        self._book_spine_pick = pick
        self.rgb_preview_label.hide()
        self._display_cloud(point_cloud)

        polygon = np.asarray(pick.polygon, dtype=float).reshape(-1, 2)
        u, v = _book_pick_point_from_polygon(
            polygon,
            self._workflow_mode,
            self._putback_target_line(),
            self._putback_pick_right_edge_ratio(),
        )
        camera_point = np.asarray(
            self._frame.point_at_pixel(u, v, max_depth_m=self._depth_max_m),
            dtype=float,
        )
        robot_point = camera_point_to_robot_target(camera_point)
        pixels = np.asarray([u, v], dtype=int)
        raw_index = self._nearest_point_index_from_pixel(pixels)
        self.book_status_label.setText(
            f"score={pick.match_confidence:.2f}  good={pick.good_count}  inliers={pick.inlier_count}"
        )
        self._show_book_preview(pick)
        self._apply_book_pick(raw_index, camera_point, robot_point, pixels, polygon)
        self.status_label.setText(tr("pc.book_point_selected"))
        if self._flow_active and self._flow_step_index == 1:
            message = (
                "书籍识别完成，底边点已自动选中"
                if self._workflow_mode == "tail_putback"
                else "书籍识别完成，目标点已自动选中"
            )
            self._advance_book_grasp_flow(message)

    def _refresh_book_pick_from_last_detection(self):
        if getattr(self, "_loading_workflow_defaults", False):
            return
        if self._book_spine_pick is None or self._frame is None or self._point_cloud is None:
            return
        try:
            polygon = np.asarray(self._book_spine_pick.polygon, dtype=float).reshape(-1, 2)
            u, v = _book_pick_point_from_polygon(
                polygon,
                self._workflow_mode,
                self._putback_target_line(),
                self._putback_pick_right_edge_ratio(),
            )
            camera_point = np.asarray(
                self._frame.point_at_pixel(u, v, max_depth_m=self._depth_max_m),
                dtype=float,
            )
            robot_point = camera_point_to_robot_target(camera_point)
            pixels = np.asarray([u, v], dtype=int)
            raw_index = self._nearest_point_index_from_pixel(pixels)
            self._show_book_preview(self._book_spine_pick)
            self._apply_book_pick(raw_index, camera_point, robot_point, pixels, polygon)
            self.status_label.setText(tr("pc.book_point_selected"))
        except Exception as exc:
            self.error_occurred.emit(f"刷新书籍目标点失败: {exc}")

    def _on_book_detection_thread_finished(self):
        self.capture_btn.setEnabled(True)
        if hasattr(self, "flow_detect_btn"):
            self.flow_detect_btn.setEnabled(True)
        self.capture_btn.setText(tr("pc.capture"))

    def _start_book_spine_segmentation(self):
        try:
            if self._segment_worker is not None and self._segment_worker.isRunning():
                return False
            if self._depth_max_m <= self._depth_min_m:
                self._set_error(tr("pc.depth_range_error"))
                return False

            self._segmentation_result = None
            self.rgb_preview_label.hide()
            self.image_detect_btn.setEnabled(False)
            self.image_detect_btn.setText(tr("pc.image_detecting"))
            self.image_status_label.setText("准备拍摄并分割书脊...")
            self._reset_image_recognition_labels()

            self._segment_worker = BookSpineSegmentWorker(
                serial=self._serial,
                width=min(self._width, BOOK_SPINE_SEGMENT_CAPTURE_WIDTH),
                height=min(self._height, BOOK_SPINE_SEGMENT_CAPTURE_HEIGHT),
                fps=self._fps,
                warmup=min(self._warmup, BOOK_SPINE_SEGMENT_WARMUP),
                timeout_ms=self._timeout_ms,
                align_depth_to_color=self._align_depth_to_color,
                depth_min_m=self._depth_min_m,
                depth_max_m=self._depth_max_m,
                stride=self._stride,
                include_color=self._include_color,
                depth_width=min(self._depth_width, BOOK_SPINE_SEGMENT_CAPTURE_DEPTH_WIDTH),
                depth_height=min(self._depth_height, BOOK_SPINE_SEGMENT_CAPTURE_DEPTH_HEIGHT),
                parent=self,
            )
            self._segment_worker.segmentation_finished.connect(
                self._on_book_spine_segmentation_finished
            )
            self._segment_worker.status_message.connect(
                self._on_book_spine_segmentation_status
            )
            self._segment_worker.error_occurred.connect(
                self._on_book_spine_segmentation_error
            )
            self._segment_worker.finished.connect(
                self._on_book_spine_segmentation_thread_finished
            )
            self._segment_worker.start()
            return True
        except Exception as exc:
            self._on_book_spine_segmentation_error(str(exc))
            return False

    def _reset_image_recognition_labels(self):
        if hasattr(self, "segment_preview_label"):
            self.segment_preview_label.setPixmap(QPixmap())
            self.segment_preview_label.setText("分割结果预览")
        if hasattr(self, "segment_count_label"):
            self.segment_count_label.setText("识别到书脊数量: --")
        if hasattr(self, "segment_gap_label"):
            self.segment_gap_label.setText("Step2 最大书缝: --")
        if hasattr(self, "segment_gap_line_label"):
            self.segment_gap_line_label.setText("中线像素: --")
        if hasattr(self, "segment_gap_pixel_label"):
            self.segment_gap_pixel_label.setText("中线中点像素: --")
        if hasattr(self, "segment_gap_camera_label"):
            self.segment_gap_camera_label.setText("中点点云坐标: --")
        if hasattr(self, "segment_gap_robot_label"):
            self.segment_gap_robot_label.setText("机械臂末端点坐标: --")
        if hasattr(self, "segment_output_label"):
            self.segment_output_label.setText("输出路径: --")

    def _on_book_spine_segmentation_status(self, message: str):
        if hasattr(self, "image_status_label"):
            self.image_status_label.setText(message)

    def _on_book_spine_segmentation_error(self, message: str):
        self._set_error(tr("pc.capture_error", msg=message))
        if hasattr(self, "image_status_label"):
            self.image_status_label.setText(f"图片识别失败: {message}")

    def _on_book_spine_segmentation_thread_finished(self):
        if hasattr(self, "image_detect_btn"):
            self.image_detect_btn.setEnabled(True)
            self.image_detect_btn.setText(tr("pc.image_detect"))

    def _on_book_spine_segmentation_finished(self, result: dict):
        frame = result.get("frame")
        self._frame = frame
        self._point_cloud = None
        self._book_spine_pick = None
        self._clear_selection()
        self._segmentation_result = dict(result or {})

        overlay_path = Path(result.get("overlay_path", ""))
        spine_count = int(result.get("spine_count", 0))
        gap_result = result.get("gap_result")
        self._show_segmentation_preview(overlay_path, gap_result=gap_result)
        if hasattr(self, "segment_count_label"):
            self.segment_count_label.setText(f"识别到书脊数量: {spine_count}")
        if isinstance(gap_result, BookGapResult):
            if hasattr(self, "segment_gap_label"):
                self.segment_gap_label.setText(
                    "Step2 最大书缝: "
                    f"书脊 {gap_result.left_index} 与 {gap_result.right_index} 之间，"
                    f"平均宽度 {gap_result.gap_width_px:.1f} px"
                )
            if hasattr(self, "segment_gap_line_label"):
                self.segment_gap_line_label.setText(
                    "中线像素: "
                    f"({gap_result.line_top_uv[0]}, {gap_result.line_top_uv[1]}) -> "
                    f"({gap_result.line_bottom_uv[0]}, {gap_result.line_bottom_uv[1]})"
                )
            if hasattr(self, "segment_gap_pixel_label"):
                self.segment_gap_pixel_label.setText(
                    f"中线中点像素: u={gap_result.midpoint_uv[0]}, v={gap_result.midpoint_uv[1]}"
                )
            if hasattr(self, "segment_gap_camera_label"):
                self.segment_gap_camera_label.setText(
                    f"中点点云坐标: {_format_vec_m(gap_result.camera_point_m)}"
                )
            if hasattr(self, "segment_gap_robot_label"):
                self.segment_gap_robot_label.setText(
                    f"机械臂末端点坐标: {_format_vec_cm(gap_result.robot_point_m)}"
                )
        else:
            gap_error = str(result.get("gap_error") or "未找到有效书缝")
            if hasattr(self, "segment_gap_label"):
                self.segment_gap_label.setText(f"Step2 最大书缝: 计算失败，{gap_error}")
            if hasattr(self, "segment_gap_line_label"):
                self.segment_gap_line_label.setText("中线像素: --")
            if hasattr(self, "segment_gap_pixel_label"):
                self.segment_gap_pixel_label.setText("中线中点像素: --")
            if hasattr(self, "segment_gap_camera_label"):
                self.segment_gap_camera_label.setText("中点点云坐标: --")
            if hasattr(self, "segment_gap_robot_label"):
                self.segment_gap_robot_label.setText("机械臂末端点坐标: --")
        if hasattr(self, "segment_output_label"):
            self.segment_output_label.setText(f"输出路径: {overlay_path}")
        if hasattr(self, "image_status_label"):
            if isinstance(gap_result, BookGapResult):
                self.image_status_label.setText("书脊分割和最大书缝计算完成，结果已显示在右侧。")
            else:
                self.image_status_label.setText("书脊分割完成，但最大书缝未计算成功。")
        self.status_label.setText("书脊分割完成")
        self.log_message.emit(
            f"书脊分割完成: {spine_count} 个, overlay={overlay_path}, gap={gap_result}"
        )
        self._update_move_button_state()

    def _display_cloud(self, point_cloud):
        self._ensure_plotter()
        if not HAS_PYVISTA or self._plotter is None:
            return

        points = np.asarray(point_cloud.points_xyz_m, dtype=np.float64)
        count = len(points)
        max_points = max(1, self._max_points)
        if count > max_points:
            indices = np.linspace(0, count - 1, max_points, dtype=np.int64)
        else:
            indices = np.arange(count, dtype=np.int64)

        display_points = points[indices].copy()
        if self._flip_view:
            display_points[:, 1] *= -1.0
            display_points[:, 2] *= -1.0

        self._display_indices = indices
        self._display_points = display_points
        colors = None
        if point_cloud.colors_rgb is not None:
            colors = np.asarray(point_cloud.colors_rgb, dtype=np.uint8)[indices]

        self._reset_scene()
        cloud_poly = pv.PolyData(display_points)
        cloud_poly["point_id"] = np.arange(len(display_points), dtype=np.int32)
        cloud_poly.verts = np.column_stack(
            (
                np.ones(len(display_points), dtype=np.int64),
                np.arange(len(display_points), dtype=np.int64),
            )
        ).ravel()
        point_size = float(self._point_size)
        if colors is not None and len(colors) == len(display_points):
            cloud_poly["rgb"] = colors
            self._cloud_actor = self._plotter.add_mesh(
                cloud_poly,
                scalars="rgb",
                rgb=True,
                point_size=point_size,
                render_points_as_spheres=True,
                name="realsense_cloud",
            )
        else:
            self._cloud_actor = self._plotter.add_mesh(
                cloud_poly,
                color="#8bd5ff",
                point_size=point_size,
                render_points_as_spheres=True,
                name="realsense_cloud",
            )

        if self._picker is not None and self._cloud_actor is not None:
            try:
                self._picker.InitializePickList()
                self._picker.AddPickList(self._cloud_actor)
                self._picker.PickFromListOn()
                self._picker.SetTolerance(0.01)
            except Exception:
                pass

        try:
            self._plotter.reset_camera()
            self._apply_initial_view()
            if (
                self._selected_display_index is not None
                and self._selected_display_index < len(self._display_points)
            ):
                self._add_selection_marker(
                    self._selected_display_index,
                    self._display_points[self._selected_display_index],
                )
            self._plotter.render()
        except Exception:
            pass

    def _camera_point_to_display_point(self, camera_point: Sequence[float]) -> np.ndarray:
        point = np.asarray(camera_point, dtype=float).reshape(3)
        if self._flip_view:
            point = point.copy()
            point[1] *= -1.0
            point[2] *= -1.0
        return point

    def _redisplay_current_cloud(self):
        if self._point_cloud is not None:
            self._display_cloud(self._point_cloud)
        self._update_pick_mode_state()

    def _reset_scene(self):
        if not HAS_PYVISTA or self._plotter is None:
            return
        try:
            self._plotter.clear()
            sc = SCENE_COLORS[ThemeManager.instance().theme]
            self._plotter.set_background(sc["bg_bottom"], top=sc["bg_top"])
            self._plotter.add_axes()
        except Exception:
            pass
        self._cloud_actor = None
        self._selected_actor = None

    def _install_event_filter(self):
        if not HAS_PYVISTA or self._plotter is None or self._filter_installed:
            return
        self._plotter.interactor.installEventFilter(self)
        self._filter_installed = True

    def eventFilter(self, obj, event):
        if obj is self._viewer_widget and event.type() == QEvent.Type.Resize:
            self._position_rgb_preview()
            return False

        if (
            self._is_workflow_header_widget(obj)
            and event.type() == QEvent.Type.Wheel
            and hasattr(self, "workflow_scroll_area")
        ):
            self._scroll_workflow_steps(event)
            return True

        pick_mode_btn = getattr(self, "pick_mode_btn", None)
        try:
            pick_mode_checked = bool(pick_mode_btn is not None and pick_mode_btn.isChecked())
        except RuntimeError:
            pick_mode_checked = False

        if (
            not HAS_PYVISTA
            or self._plotter is None
            or obj is not self._plotter.interactor
            or not pick_mode_checked
        ):
            return False

        if (
            event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
            and self._display_points is not None
        ):
            picked = self._pick_display_point(event)
            if picked is not None:
                display_index, display_point = picked
                self._select_display_point(display_index, display_point)
                return True
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.RightButton:
            return False
        return False

    def _is_workflow_header_widget(self, obj) -> bool:
        header = getattr(self, "workflow_header_group", None)
        if header is None:
            return False
        return obj is header or (isinstance(obj, QWidget) and header.isAncestorOf(obj))

    def _scroll_workflow_steps(self, event):
        delta = event.pixelDelta().y()
        if delta == 0:
            delta = event.angleDelta().y()
        if delta == 0:
            return
        scrollbar = self.workflow_scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.value() - delta)

    def _position_rgb_preview(self):
        if not hasattr(self, "rgb_preview_label") or self._viewer_widget is None:
            return
        margin = 14
        x = max(margin, self._viewer_widget.width() - self.rgb_preview_label.width() - margin)
        self.rgb_preview_label.move(x, margin)
        self.rgb_preview_label.raise_()

    def _show_book_preview(self, pick):
        if self._frame is None or not hasattr(self, "rgb_preview_label") or cv2 is None:
            return

        image_bgr = self._frame.color_bgr.copy()
        corners = np.asarray(pick.polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image_bgr, [corners], True, (0, 0, 255), 5, cv2.LINE_AA)
        u, v = _book_pick_point_from_polygon(
            pick.polygon,
            self._workflow_mode,
            self._putback_target_line(),
            self._putback_pick_right_edge_ratio(),
        )
        cv2.circle(image_bgr, (int(u), int(v)), 12, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image_bgr, (int(u), int(v)), 15, (0, 0, 255), 3, cv2.LINE_AA)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = image_rgb.shape
        bytes_per_line = channels * width
        qimage = QImage(
            image_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.rgb_preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.rgb_preview_label.setPixmap(pixmap)
        self.rgb_preview_label.show()
        self._position_rgb_preview()

    def _show_segmentation_preview(
        self,
        overlay_path: Path,
        *,
        gap_result: Optional[BookGapResult] = None,
    ):
        if not hasattr(self, "segment_preview_label"):
            return
        display_path = overlay_path
        if isinstance(gap_result, BookGapResult) and cv2 is not None:
            image_bgr = cv2.imread(str(overlay_path))
            if image_bgr is not None:
                cv2.line(
                    image_bgr,
                    gap_result.line_top_uv,
                    gap_result.line_bottom_uv,
                    (0, 0, 255),
                    4,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    image_bgr,
                    gap_result.midpoint_uv,
                    9,
                    (0, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    image_bgr,
                    gap_result.midpoint_uv,
                    12,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image_bgr,
                    "MAX GAP",
                    (
                        max(0, gap_result.midpoint_uv[0] + 12),
                        max(22, gap_result.midpoint_uv[1] - 12),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                annotated_path = overlay_path.with_name(
                    f"{overlay_path.stem}_max_gap{overlay_path.suffix}"
                )
                cv2.imwrite(str(annotated_path), image_bgr)
                display_path = annotated_path
                if isinstance(self._segmentation_result, dict):
                    self._segmentation_result["annotated_overlay_path"] = annotated_path
        pixmap = QPixmap(str(display_path))
        if pixmap.isNull():
            self.segment_preview_label.setText(f"无法加载分割结果图: {display_path}")
            return
        pixmap = pixmap.scaled(
            self.segment_preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.segment_preview_label.setPixmap(pixmap)

    def _show_tail_putback_target_preview(
        self,
        bottom_robot_point_m: np.ndarray,
        target_robot_point_m: np.ndarray,
    ):
        if self._frame is None or not hasattr(self, "rgb_preview_label") or cv2 is None:
            return

        image_bgr = self._frame.color_bgr.copy()
        pick = self._book_spine_pick
        if pick is not None and hasattr(pick, "polygon"):
            corners = np.asarray(pick.polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(image_bgr, [corners], True, (0, 0, 255), 5, cv2.LINE_AA)

        bottom_uv = self._project_robot_point_to_pixel(bottom_robot_point_m)
        target_uv = self._project_robot_point_to_pixel(target_robot_point_m)
        if bottom_uv is not None:
            cv2.circle(image_bgr, bottom_uv, 12, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image_bgr, bottom_uv, 15, (0, 0, 255), 3, cv2.LINE_AA)
        if target_uv is not None:
            cv2.circle(image_bgr, target_uv, 12, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(image_bgr, target_uv, 16, (255, 128, 0), 3, cv2.LINE_AA)
        if bottom_uv is not None and target_uv is not None:
            cv2.line(image_bgr, bottom_uv, target_uv, (255, 255, 0), 3, cv2.LINE_AA)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = image_rgb.shape
        bytes_per_line = channels * width
        qimage = QImage(
            image_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.rgb_preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.rgb_preview_label.setPixmap(pixmap)
        self.rgb_preview_label.show()
        self._position_rgb_preview()

    def _project_robot_point_to_pixel(
        self,
        robot_point_m: Sequence[float],
    ) -> Optional[tuple[int, int]]:
        if self._frame is None:
            return None
        x, y, z = robot_point_to_camera_target(robot_point_m)
        if not np.isfinite(z) or z <= 0.0:
            return None
        intr = self._frame.intrinsics
        u = int(round(float(x) * intr.fx / float(z) + intr.ppx))
        v = int(round(float(y) * intr.fy / float(z) + intr.ppy))
        height, width = self._frame.depth_m.shape
        if not (0 <= u < width and 0 <= v < height):
            return None
        return u, v

    def _pick_display_point(self, event) -> Optional[tuple[int, np.ndarray]]:
        if self._display_points is None or self._plotter is None:
            return None

        vx, vy = self._qt_to_vtk_coords(event)
        picked = self._pick_nearest_screen_point(vx, vy)
        if picked is not None:
            return picked, np.asarray(self._display_points[picked], dtype=float)

        if self._picker is None:
            return None

        try:
            hit = self._picker.Pick(vx, vy, 0, self._plotter.renderer)
            if not hit:
                self.status_label.setText(tr("pc.pick_miss"))
                return None

            pick_position = np.asarray(self._picker.GetPickPosition(), dtype=float)
            point_id = int(self._picker.GetPointId())
            if (
                self._display_indices is not None
                and 0 <= point_id < len(self._display_indices)
            ):
                display_point = self._picked_display_point(point_id, pick_position)
                return point_id, display_point
            world_point = pick_position
        except Exception as exc:
            logger.debug("点云选点失败: %s", exc)
            self.status_label.setText(tr("pc.pick_miss"))
            return None

        if self._display_indices is None:
            self.status_label.setText(tr("pc.pick_miss"))
            return None
        if world_point is not None and self._display_points is not None:
            distances = np.linalg.norm(self._display_points - world_point.reshape(1, 3), axis=1)
            display_index = int(np.argmin(distances))
            display_point = self._picked_display_point(display_index, world_point)
            return display_index, display_point

        try:
            pick_point = np.asarray(self._plotter.picked_point, dtype=float).reshape(3)
            if self._display_points is not None:
                distances = np.linalg.norm(self._display_points - pick_point.reshape(1, 3), axis=1)
                display_index = int(np.argmin(distances))
                display_point = self._picked_display_point(display_index, pick_point)
                return display_index, display_point
        except Exception:
            pass

        self.status_label.setText(tr("pc.pick_miss"))
        return None

    def _qt_to_vtk_coords(self, event) -> tuple[float, float]:
        x = float(event.position().x())
        y = float(self._plotter.interactor.height() - event.position().y() - 1)
        try:
            render_width, render_height = self._plotter.ren_win.GetSize()
            widget_width = max(1, self._plotter.interactor.width())
            widget_height = max(1, self._plotter.interactor.height())
            x *= float(render_width) / float(widget_width)
            y *= float(render_height) / float(widget_height)
        except Exception:
            pass
        return x, y

    def _picked_display_point(
        self,
        display_index: int,
        pick_position: np.ndarray,
    ) -> np.ndarray:
        if pick_position.shape == (3,) and np.all(np.isfinite(pick_position)):
            return pick_position.astype(float, copy=True)
        return np.asarray(self._display_points[display_index], dtype=float)

    def _pick_nearest_screen_point(self, x: float, y: float) -> Optional[int]:
        if self._display_points is None or len(self._display_points) == 0:
            return None

        try:
            renderer = self._plotter.renderer
            render_width, render_height = self._plotter.ren_win.GetSize()
            best_index = None
            best_dist_sq = float("inf")
            best_depth = float("inf")

            for index, point in enumerate(self._display_points):
                renderer.SetWorldPoint(
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    1.0,
                )
                renderer.WorldToDisplay()
                sx, sy, sz = renderer.GetDisplayPoint()
                if (
                    not np.isfinite(sx)
                    or not np.isfinite(sy)
                    or not np.isfinite(sz)
                    or sx < 0.0
                    or sx >= render_width
                    or sy < 0.0
                    or sy >= render_height
                    or sz < 0.0
                    or sz > 1.0
                ):
                    continue

                dist_sq = (float(sx) - x) ** 2 + (float(sy) - y) ** 2
                if dist_sq < best_dist_sq or (
                    math.isclose(dist_sq, best_dist_sq) and float(sz) < best_depth
                ):
                    best_index = index
                    best_dist_sq = dist_sq
                    best_depth = float(sz)

            if best_index is None:
                return None

            point_size = max(1.0, float(self._point_size))
            threshold_px = max(12.0, point_size * 5.0)
            if best_dist_sq > threshold_px * threshold_px:
                logger.debug(
                    "屏幕最近点超过阈值: distance=%.2f px threshold=%.2f px",
                    math.sqrt(best_dist_sq),
                    threshold_px,
                )
                return None
            return int(best_index)
        except Exception as exc:
            logger.debug("屏幕空间点云选点失败: %s", exc)
            return None

    def _select_display_point(self, display_index: int, display_point: np.ndarray):
        if self._point_cloud is None or self._display_indices is None:
            return

        raw_index = int(self._display_indices[display_index])
        camera_point = np.asarray(self._point_cloud.points_xyz_m[raw_index], dtype=float)
        self._selected_display_index = int(display_index)
        self._selected_display_point_m = np.asarray(display_point, dtype=float).reshape(3)
        self._selected_raw_index = raw_index
        self._selected_camera_point_m = camera_point
        self._selected_robot_target_raw_m = camera_point_to_robot_target(camera_point)
        pixels = np.asarray(self._point_cloud.pixels_uv[raw_index], dtype=int)

        if pixels.shape == (2,) and np.all(pixels >= 0):
            self.pixel_value.setText(f"u={int(pixels[0])}, v={int(pixels[1])}")
        else:
            self.pixel_value.setText("--")

        self._add_selection_marker(display_index, self._selected_display_point_m)
        self._recompute_target_from_selection()
        self.status_label.setText(tr("pc.point_selected"))

    def _nearest_point_index_from_pixel(self, pixel_uv: np.ndarray) -> int:
        if self._point_cloud is None:
            return -1
        pixels = np.asarray(self._point_cloud.pixels_uv, dtype=int)
        if pixels.size == 0:
            return -1
        diffs = pixels - pixel_uv.reshape(1, 2)
        distances = np.sum(diffs.astype(float) ** 2, axis=1)
        return int(np.argmin(distances))

    def _apply_book_pick(
        self,
        raw_index: int,
        camera_point: np.ndarray,
        robot_point: np.ndarray,
        pixels: np.ndarray,
        corners: np.ndarray,
    ):
        if self._point_cloud is None:
            return

        if raw_index < 0:
            raw_index = self._nearest_point_index_from_pixel(pixels)
        if raw_index >= 0 and self._display_indices is not None:
            display_candidates = np.where(self._display_indices == raw_index)[0]
            if len(display_candidates) > 0:
                display_index = int(display_candidates[0])
                self._select_display_point(
                    display_index,
                    self._display_points[display_index],
                )
                return

        self._selected_display_index = None
        self._selected_display_point_m = self._camera_point_to_display_point(camera_point)
        self._selected_raw_index = int(raw_index)
        self._selected_camera_point_m = np.asarray(camera_point, dtype=float).reshape(3)
        self._selected_robot_target_raw_m = np.asarray(robot_point, dtype=float).reshape(3)
        self.pixel_value.setText(f"u={int(pixels[0])}, v={int(pixels[1])}")
        self._add_book_marker(corners, self._selected_display_point_m)
        self._recompute_target_from_selection()
        self._update_move_button_state()

    def _add_book_marker(self, corners: np.ndarray, display_point: np.ndarray):
        if not HAS_PYVISTA or self._plotter is None:
            return
        try:
            if self._selected_actor is not None:
                self._plotter.remove_actor(self._selected_actor)
                self._selected_actor = None
            self._selected_actor = self._plotter.add_points(
                np.asarray(display_point, dtype=float).reshape(1, 3),
                color="#ffcc00",
                point_size=max(10.0, float(self._point_size) * 6.0),
                render_points_as_spheres=True,
                name="book_detect_point",
            )
            if corners is not None and len(corners) == 4:
                world = []
                for u, v in np.asarray(corners, dtype=float):
                    if self._frame is None:
                        continue
                    try:
                        world_point = self._frame.point_at_pixel(
                            int(round(u)),
                            int(round(v)),
                            max_depth_m=self._depth_max_m,
                        )
                        world.append(self._camera_point_to_display_point(world_point))
                    except Exception:
                        continue
                if len(world) >= 2:
                    self._plotter.add_lines(
                        np.asarray(world, dtype=float),
                        color="#ff8800",
                        width=3,
                        name="book_detect_box",
                    )
            self._plotter.render()
        except Exception:
            pass

    def _add_selection_marker(
        self,
        display_index: int,
        display_point: Optional[np.ndarray] = None,
    ):
        if not HAS_PYVISTA or self._plotter is None or self._display_points is None:
            return
        if display_point is None:
            display_point = self._display_points[display_index]
        point = np.asarray(display_point, dtype=float)
        if self._selected_actor is not None:
            try:
                self._plotter.remove_actor(self._selected_actor)
            except Exception:
                pass
            self._selected_actor = None

        try:
            marker_points = np.asarray(point, dtype=float).reshape(1, 3)
            marker_size = max(10.0, float(self._point_size) * 6.0)
            self._selected_actor = self._plotter.add_points(
                marker_points,
                color="#ffcc00",
                point_size=marker_size,
                render_points_as_spheres=True,
                name="selected_realsense_point",
            )
            self._plotter.render()
        except Exception:
            pass

    def _recompute_target_from_selection(self):
        if self._selected_robot_target_raw_m is None:
            return

        if self._workflow_mode == "tail_putback":
            bottom_xyz = np.asarray(self._selected_robot_target_raw_m, dtype=float).reshape(3)
            previous_bottom = self._tail_putback_bottom_point_m
            bottom_changed = previous_bottom is None or not np.allclose(
                np.asarray(previous_bottom, dtype=float).reshape(3),
                bottom_xyz,
                rtol=0.0,
                atol=1e-9,
            )
            self._tail_putback_bottom_point_m = bottom_xyz.copy()
            if bottom_changed:
                self._target_robot_point_m = None
                self._tail_putback_step4_target_point_m = None
            elif self._target_robot_point_m is not None:
                self._compute_tail_putback_target_point(advance_flow=False)
                return
            self.move_target_point_cm_value.setText(_format_vec_cm(bottom_xyz))
            self.target_point_cm_value.setText(_format_vec_cm(bottom_xyz))
            if hasattr(self, "flow_target_label"):
                self.flow_target_label.setText(f"底边点: {_format_vec_cm(bottom_xyz)}")
            self._update_move_button_state()
            return

        target_xyz = np.asarray(self._selected_robot_target_raw_m, dtype=float).reshape(3)
        target_xyz = target_xyz + np.asarray(
            self._current_target_compensation_cm(),
            dtype=float,
        ) / M_TO_CM
        self.move_target_point_cm_value.setText(
            _format_vec_cm(self._selected_robot_target_raw_m)
        )
        # EndPoseCtrl/MoveL already interpret this as a TCP target and apply the
        # configured TCP offset inside IK. Do not pre-convert it to the URDF
        # end-effector frame here, or the offset will be applied twice.
        self._target_robot_point_m = np.asarray(target_xyz, dtype=float).reshape(3)
        self.target_point_cm_value.setText(_format_vec_cm(target_xyz))
        self._update_move_button_state()

    def _apply_initial_view(self):
        if not HAS_PYVISTA or self._plotter is None:
            return
        try:
            self._plotter.view_xy()
            self._plotter.camera.zoom(1.2)
        except Exception:
            pass

    def _update_pick_mode_state(self):
        if not HAS_PYVISTA or self._plotter is None:
            return
        if not hasattr(self, "pick_mode_btn"):
            return
        try:
            if self._picker is not None:
                if self.pick_mode_btn.isChecked() and self._viewer_visible:
                    self._picker.PickFromListOn()
                else:
                    self._picker.PickFromListOff()
        except Exception as exc:
            logger.debug("更新点选模式失败: %s", exc)

    def set_tcp_offset(self, tcp_offset):
        source = [] if tcp_offset is None else list(tcp_offset)
        values = np.asarray(source, dtype=float)
        if values.shape != (6,):
            normalized = np.zeros(6, dtype=float)
            for idx in range(min(6, len(values))):
                normalized[idx] = float(values[idx])
            values = normalized
        self._tcp_offset = values
        if self._selected_robot_target_raw_m is not None:
            self._recompute_target_from_selection()

    def update_current_end_pose(self, end_pose):
        self._current_end_pose = end_pose
        if not self._rpy_initialized:
            self._set_rpy_from_pose(end_pose)

    def set_arm_enabled(self, enabled: bool):
        self._arm_enabled = bool(enabled)
        self._update_move_button_state()

    def _fill_current_rpy(self):
        if self._current_end_pose is None:
            self.status_label.setText(tr("pc.no_pose"))
            return
        self._set_rpy_from_pose(self._current_end_pose)
        self._rpy_initialized = True
        self.status_label.setText(tr("pc.rpy_loaded"))

    def _set_rpy_from_pose(self, pose):
        if pose is None:
            return
        values_deg = [
            math.degrees(float(pose.rx)),
            math.degrees(float(pose.ry)),
            math.degrees(float(pose.rz)),
        ]
        for spin, value in zip(self.rpy_spins, values_deg):
            blocker = QSignalBlocker(spin)
            spin.setValue(value)
            del blocker
        if self._selected_robot_target_raw_m is not None:
            self._recompute_target_from_selection()

    def _mark_rpy_initialized(self):
        self._rpy_initialized = True

    def _set_rpy_spins(self, values_deg: Sequence[float]):
        for spin, value in zip(self.rpy_spins, values_deg):
            blocker = QSignalBlocker(spin)
            spin.setValue(float(value))
            del blocker
        self._rpy_initialized = True
        if self._selected_robot_target_raw_m is not None:
            self._recompute_target_from_selection()

    @staticmethod
    def _spin_values(spins: Sequence[QDoubleSpinBox]) -> list[float]:
        return [float(spin.value()) for spin in spins]

    def _is_putback_workflow(self) -> bool:
        return _is_putback_workflow_mode(self._workflow_mode)

    def _is_takeout_based_workflow(self) -> bool:
        return _is_takeout_based_workflow_mode(self._workflow_mode)

    def _is_image_recognition_workflow(self) -> bool:
        return _is_image_recognition_workflow_mode(self._workflow_mode)

    def _putback_target_line(self) -> str:
        combo = getattr(self, "putback_target_line_combo", None)
        if combo is None:
            return "right"
        value = combo.currentData()
        if value is None:
            value = combo.currentText()
        line = str(value).lower()
        if line in {"left", "middle", "right"}:
            return line
        return "right"

    def _putback_pick_right_edge_ratio(self) -> float:
        combo = getattr(self, "putback_pick_ratio_combo", None)
        if combo is None:
            return 0.75
        value = combo.currentData()
        if value is None:
            value = combo.currentText()
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.75

    def _workflow_group_title(self) -> str:
        if self._workflow_mode == "image_recognition":
            return tr("pc.image_recognition_group")
        if self._workflow_mode == "tail_putback":
            return tr("pc.book_tail_putback_group")
        if self._workflow_mode == "putback":
            return tr("pc.book_putback_group")
        if self._workflow_mode == "putback2":
            return tr("pc.book_putback2_group")
        return tr("pc.book_takeout_group")

    def _current_workflow_target_rpy_deg(self) -> list[float]:
        if self._is_putback_workflow() and hasattr(self, "putback_target_rpy_spins"):
            return self._spin_values(self.putback_target_rpy_spins)
        if self._is_takeout_based_workflow() and hasattr(self, "grasp_rpy_spins"):
            return self._spin_values(self.grasp_rpy_spins)
        return self._spin_values(self.rpy_spins)

    def _current_target_compensation_cm(self) -> list[float]:
        if self._is_putback_workflow() and hasattr(self, "putback_target_comp_xyz_spins"):
            return self._spin_values(self.putback_target_comp_xyz_spins)
        if self._is_takeout_based_workflow() and hasattr(self, "target_comp_xyz_spins"):
            return self._spin_values(self.target_comp_xyz_spins)
        return [0.0, 0.0, 0.0]

    def _on_workflow_target_changed(self):
        if self._selected_robot_target_raw_m is not None:
            self._recompute_target_from_selection()

    def _on_rpy_changed(self):
        self._mark_rpy_initialized()
        if self._selected_robot_target_raw_m is not None:
            self._recompute_target_from_selection()

    def _on_confirm_move(self):
        if self._target_robot_point_m is None:
            self.status_label.setText(tr("pc.no_target"))
            return
        pose = [
            float(self._target_robot_point_m[0]),
            float(self._target_robot_point_m[1]),
            float(self._target_robot_point_m[2]),
            math.radians(self.rpy_spins[0].value()),
            math.radians(self.rpy_spins[1].value()),
            math.radians(self.rpy_spins[2].value()),
        ]
        duration = float(self.duration_spin.value())
        self.move_l_requested.emit(pose, duration)
        self.status_label.setText(tr("pc.movel_sent"))
        self.log_message.emit(tr("pc.movel_sent"))

    def _flow_log_prefix(self) -> str:
        if self._workflow_mode == "tail_putback":
            return "书籍末尾放回流程"
        if self._workflow_mode == "putback2":
            return "书籍放回2流程"
        return "书籍放回流程" if self._workflow_mode == "putback" else "书籍取出流程"

    def _book_grasp_steps(self) -> list[str]:
        if self._workflow_mode == "tail_putback":
            return [
                "点云识别/采集",
                "书籍识别，自动取右边线最底点",
                "解算底边点",
                "解算目标点",
                "目标点姿态确认",
                "MoveJ 到预备放书点",
                "运动到放置点",
                "MoveL 微调书本位置",
                "夹爪微微张开",
                "基于当前位姿后移",
                "夹爪直接闭合",
                "基于步骤11位姿运动到预备推书位",
                "基于步骤12位姿运动到推书位",
                "回到归零位置或图书调试位置",
            ]
        if self._is_putback_workflow():
            return [
                "点云识别/采集",
                "书籍识别，自动选中目标点",
                "解算目标点，并设置目标姿态",
                "IK + MoveJ 到预备推书点",
                "MoveL 到推书点",
                "MoveL 沿基坐标系 Y+ 推开书本",
                "MoveL 沿当前位姿 Y+ 离开推书位置",
                "IK + MoveJ 到插入预备位",
                "MoveL 到插入位",
                "打开夹爪到指定角度",
                "MoveL 沿当前位姿 X+ 离开插入位",
                "MoveJ 回到图书调试位",
            ]
        if self._workflow_mode == "putback2":
            return [
                "采集点云",
                "识别书籍",
                "解算目标点",
                "到达预备抓取位姿",
                "到达抓取位姿",
                "杆电机到夹取位",
                "微调抓取位姿",
                "夹爪带监测持续关闭",
                "将末端姿态改为默认 RPY",
                "沿基坐标系 X+ 移动",
                "夹爪开一点",
                "杆电机运动到 90 度",
                "机械臂沿基坐标系 Z- 移动",
                "夹爪全部张开",
                "机械臂沿基坐标系 X+ 移动",
                "夹爪闭合、Y+移动、等待、X-移动、夹爪全开",
                "机械臂沿基坐标系 X+ 移动",
            ]
        return [
            "采集点云",
            "识别书籍",
            "解算目标点",
            "到达预备抓取位姿",
            "到达抓取位姿",
            "杆电机到夹取位",
            "微调抓取位姿",
            "夹爪带监测持续关闭",
            "当前位姿偏移 IK + MoveJ",
            "机械臂 MoveJ 回到图书调试位置",
            "升降台从取书位移动到还书位",
            "机械臂慢速转身到预备放书位置",
            "IK + MoveJ 到指定位姿",
            "基于步骤13位姿 MoveL 微调",
            "夹爪打开",
            "机械臂 MoveJ 到最终放置构型",
        ]

    def _format_flow_steps_text(self) -> str:
        return "\n".join(
            f"{idx + 1}. {step}" for idx, step in enumerate(self._book_grasp_steps())
        )

    def _show_flow_steps_dialog(self):
        QMessageBox.information(
            self,
            self._workflow_group_title(),
            self._format_flow_steps_text(),
        )

    def _reset_book_grasp_flow(self):
        self._flow_active = False
        self._flow_auto_run = False
        self._flow_step_index = 0
        self._flow_waiting_motion = False
        self._flow_waiting_kind = None
        self._flow_rollback_waiting = False
        self._flow_rollback_entry = None
        self._flow_pose_history.clear()
        self._flow_approach_pose = None
        self._flow_last_pose = None
        self._flow_pending_pose = None
        self._tail_putback_step4_target_point_m = None
        self._rod_current_angle_deg = None
        self._rod_target_angle_deg = None
        self._clear_gripper_close_monitor()
        self._clear_manual_gripper_open()
        self.flow_status_label.setText(tr("pc.workflow_pending"))
        if hasattr(self, "flow_target_label"):
            self.flow_target_label.setText(tr("pc.workflow_target"))
        self._update_flow_button_state()

    def _start_auto_book_grasp_flow(self):
        if self._workflow_mode != "takeout" or self._flow_waiting_motion:
            return
        steps = self._book_grasp_steps()
        if self._flow_active and self._flow_step_index >= len(steps):
            self._reset_book_grasp_flow()
        self._flow_auto_run = True
        if not self._flow_active:
            self._flow_step_index = 0
        self._execute_next_book_grasp_step()

    def _schedule_auto_book_grasp_step(self):
        if (
            self._workflow_mode != "takeout"
            or not self._flow_auto_run
            or self._flow_waiting_motion
            or not self._flow_active
            or self._flow_step_index >= len(self._book_grasp_steps())
        ):
            return
        QTimer.singleShot(0, self._execute_next_book_grasp_step)

    def _request_workflow_stop(self):
        self._flow_auto_run = False
        self._flow_waiting_motion = False
        self._flow_waiting_kind = None
        self._flow_rollback_waiting = False
        self._flow_rollback_entry = None
        self._flow_pending_pose = None
        self._rod_target_angle_deg = None
        self._clear_gripper_close_monitor()
        self._clear_manual_gripper_open()
        self.flow_status_label.setText("流程急停已触发，已停止继续发送运动命令")
        self.log_message.emit(f"{self._flow_log_prefix()}流程急停：停止继续发送运动命令")
        self._update_flow_button_state()
        self.workflow_stop_requested.emit()

    def _execute_next_book_grasp_step(self):
        if self._flow_waiting_motion:
            return
        if not self._flow_active:
            self._flow_active = True
            self._flow_step_index = 0
        steps = self._book_grasp_steps()
        if self._flow_step_index >= len(steps):
            self._reset_book_grasp_flow()
            return

        step = self._flow_step_index
        total = len(steps)
        self.flow_status_label.setText(f"执行中 [{step + 1}/{total}] {steps[step]}")
        self.log_message.emit(
            f"{self._flow_log_prefix()} [{step + 1}/{total}] {steps[step]}"
        )

        if step == 0:
            if self._start_capture():
                self._begin_flow_wait("capture")
        elif step == 1:
            if self._detect_book_target():
                self._begin_flow_wait("detect")
        elif step == 2:
            if self._workflow_mode == "tail_putback":
                self._solve_tail_putback_bottom_point()
            else:
                self._solve_book_grasp_target()
        elif step == 3 and self._workflow_mode == "tail_putback":
            self._solve_tail_putback_target_point()
        elif step == 4 and self._workflow_mode == "tail_putback":
            self._advance_book_grasp_flow("目标姿态已确认")
        elif step == 5 and self._workflow_mode == "tail_putback":
            self._execute_tail_putback_preplace_step()
        elif step == 6 and self._workflow_mode == "tail_putback":
            self._execute_tail_putback_target_movel_step()
        elif step == 7 and self._workflow_mode == "tail_putback":
            self._execute_tail_putback_fine_tune_step()
        elif step == 8 and self._workflow_mode == "tail_putback":
            self._execute_tail_putback_slight_open_step()
        elif step == 9 and self._workflow_mode == "tail_putback":
            self._execute_tail_putback_backoff_step()
        elif step == 10 and self._workflow_mode == "tail_putback":
            self._execute_tail_putback_close_step()
        elif step == 11 and self._workflow_mode == "tail_putback":
            self._execute_tail_putback_reapproach_step()
        elif step == 12 and self._workflow_mode == "tail_putback":
            self._execute_tail_putback_target_comp_step()
        elif step == 13 and self._workflow_mode == "tail_putback":
            self._execute_tail_putback_final_home_step()
        elif self._is_putback_workflow():
            putback_step = step - 1 if self._workflow_mode == "tail_putback" else step
            self._execute_putback_motion_step(putback_step)
        else:
            self._execute_takeout_motion_step(step)
        if (
            self._flow_auto_run
            and not self._flow_waiting_motion
            and self._flow_step_index == step
        ):
            self._flow_auto_run = False
            self._update_flow_button_state()

    def _execute_single_takeout_step(self, step_index: int):
        if self._workflow_mode != "takeout" or self._flow_waiting_motion:
            return
        self._flow_auto_run = False
        steps = self._book_grasp_steps()
        if step_index < 0 or step_index >= len(steps):
            return
        self._flow_active = True
        self._flow_step_index = int(step_index)
        self._execute_next_book_grasp_step()

    def _execute_takeout_motion_step(self, step: int):
        if self._workflow_mode == "putback2":
            self._execute_putback2_motion_step(step)
            return
        if step == 3:
            self._execute_pregrasp_step()
        elif step == 4:
            self._execute_approach_step()
        elif step == 5:
            self._write_rod_and_wait(self.rod_grasp_spin.value())
        elif step == 6:
            self._execute_relative_target_step()
        elif step == 7:
            self._start_monitored_gripper_close()
        elif step == 8:
            self._execute_post_gripper_movej_offset_step()
        elif step == 9:
            self._execute_return_debug_step()
        elif step == 10:
            self._execute_lift_to_return_step()
        elif step == 11:
            self._execute_turn_stage1_step()
        elif step == 12:
            self._execute_takeout_specified_pose_step()
        elif step == 13:
            self._execute_takeout_step13_fine_tune_step()
        elif step == 14:
            self._execute_takeout_gripper_open_step()
        elif step == 15:
            self._execute_final_joint_step()

    def _execute_putback2_motion_step(self, step: int):
        if step == 3:
            self._execute_pregrasp_step()
        elif step == 4:
            self._execute_approach_step()
        elif step == 5:
            self._write_rod_and_wait(self.rod_grasp_spin.value())
        elif step == 6:
            self._execute_relative_target_step()
        elif step == 7:
            self._start_monitored_gripper_close()
        elif step == 8:
            self._execute_putback2_reset_rpy_step()
        elif step == 9:
            self._execute_putback2_x_offset_step()
        elif step == 10:
            self._start_putback2_gripper_open(partial=True)
        elif step == 11:
            self._write_rod_and_wait(self.putback2_rod_release_spin.value())
        elif step == 12:
            self._execute_putback2_z_offset_step()
        elif step == 13:
            self._start_putback2_gripper_open(partial=False)
        elif step == 14:
            self._execute_putback2_after_open_x1_step()
        elif step == 15:
            self._start_putback2_combo_step()
        elif step == 16:
            self._execute_putback2_after_combo_x_step()

    def _execute_putback_motion_step(self, step: int):
        if step == 3:
            xyz_offset = self._spin_values(getattr(self, "putback_prepush_xyz_spins", []))
            if len(xyz_offset) < 3:
                xyz_offset = list(self.DEFAULT_PUTBACK_PREPUSH_OFFSET_CM)
            rpy_offset = self._spin_values(
                getattr(self, "putback_prepush_rpy_offset_spins", [])
            )
            if len(rpy_offset) < 3:
                rpy_offset = list(self.DEFAULT_PUTBACK_PREPUSH_RPY_OFFSET_DEG)
            target_rpy = self._spin_values(self.putback_target_rpy_spins)
            pose = self._make_putback_target_pose(
                x_offset_cm=xyz_offset[0],
                y_offset_cm=xyz_offset[1],
                z_offset_cm=xyz_offset[2],
                rpy_deg=[
                    float(target_rpy[0]) + float(rpy_offset[0]),
                    float(target_rpy[1]) + float(rpy_offset[1]),
                    float(target_rpy[2]) + float(rpy_offset[2]),
                ],
            )
            self._send_blocking_end_pose(pose, "等待机械臂 IK + MoveJ 到预备推书点")
        elif step == 4:
            xyz_offset = self._spin_values(getattr(self, "putback_push_xyz_spins", []))
            if len(xyz_offset) < 3:
                xyz_offset = list(self.DEFAULT_PUTBACK_PUSH_OFFSET_CM)
            rpy_offset = self._spin_values(
                getattr(self, "putback_push_rpy_offset_spins", [])
            )
            if len(rpy_offset) < 3:
                rpy_offset = list(self.DEFAULT_PUTBACK_PUSH_RPY_OFFSET_DEG)
            target_rpy = self._spin_values(self.putback_target_rpy_spins)
            pose = self._make_putback_target_pose(
                x_offset_cm=xyz_offset[0],
                y_offset_cm=xyz_offset[1],
                z_offset_cm=xyz_offset[2],
                rpy_deg=[
                    float(target_rpy[0]) + float(rpy_offset[0]),
                    float(target_rpy[1]) + float(rpy_offset[1]),
                    float(target_rpy[2]) + float(rpy_offset[2]),
                ],
            )
            self._send_blocking_movel(pose, "等待机械臂 MoveL 到推书点")
        elif step == 5:
            xyz_offset = self._spin_values(getattr(self, "putback_push_out_xyz_spins", []))
            if len(xyz_offset) < 3:
                xyz_offset = list(self.DEFAULT_PUTBACK_PUSH_OUT_OFFSET_CM)
            rpy_offset = self._spin_values(
                getattr(self, "putback_push_out_rpy_offset_spins", [])
            )
            if len(rpy_offset) < 3:
                rpy_offset = list(self.DEFAULT_PUTBACK_PUSH_OUT_RPY_OFFSET_DEG)
            pose = self._make_pose_from_current_end_pose(
                x_offset_cm=xyz_offset[0],
                y_offset_cm=xyz_offset[1],
                z_offset_cm=xyz_offset[2],
                rpy_offset_deg=rpy_offset,
                local_axes=False,
                prefer_last_flow_pose=True,
            )
            if pose is None:
                self._set_error("暂无末端位姿，无法基于当前位姿推开书本")
                return
            self._send_blocking_movel(
                pose,
                "等待机械臂基于当前位姿 MoveL 推开书本",
                duration=float(self.putback_push_movel_duration_spin.value()),
            )
        elif step == 6:
            pose = self._make_pose_from_current_end_pose(
                y_offset_cm=self.putback_leave_push_y_spin.value(),
                local_axes=True,
                prefer_last_flow_pose=True,
            )
            if pose is None:
                self._set_error("暂无末端位姿，无法离开推书位置")
                return
            self._send_blocking_movel(pose, "等待机械臂沿当前位姿 Y+ 离开推书位置")
        elif step == 7:
            pose = self._make_putback_target_pose(
                x_offset_cm=self.putback_insert_prepose_x_spin.value(),
                y_offset_cm=self.putback_insert_prepose_y_spin.value(),
                z_offset_cm=0.0,
                rpy_deg=self._spin_values(self.putback_insert_rpy_spins),
            )
            self._send_blocking_end_pose(pose, "等待机械臂 IK + MoveJ 到插入预备位")
        elif step == 8:
            pose = self._make_putback_target_pose(
                x_offset_cm=self.putback_insert_x_spin.value(),
                y_offset_cm=self.putback_insert_y_spin.value(),
                z_offset_cm=0.0,
                rpy_deg=self._spin_values(self.putback_insert_rpy_spins),
            )
            self._send_blocking_movel(pose, "等待机械臂 MoveL 到插入位")
        elif step == 9:
            target_deg = (
                float(self.putback_gripper_open_spin.value())
                if hasattr(self, "putback_gripper_open_spin")
                else float(self.gripper_open_spin.value())
            )
            self._start_shared_gripper_open(
                waiting_kind="putback_gripper_open",
                waiting_text=f"等待后台夹爪缓慢打开到 {target_deg:.1f}°",
                target_deg=target_deg,
            )
        elif step == 10:
            pose = self._make_pose_from_current_end_pose(
                x_offset_cm=self.putback_leave_insert_x_spin.value(),
                local_axes=True,
                prefer_last_flow_pose=True,
            )
            if pose is None:
                self._set_error("暂无末端位姿，无法离开插入位")
                return
            self._send_blocking_movel(pose, "等待机械臂沿当前位姿 X+ 离开插入位")
        elif step == 11:
            self._execute_return_debug_step()

    def _solve_book_grasp_target(self):
        if self._target_robot_point_m is None:
            self._set_error("请先识别书籍或手动选中目标点")
            return
        self._recompute_target_from_selection()
        if self._selected_camera_point_m is None:
            self._set_error("缺少相机坐标，请重新识别书籍或手动选中目标点")
            return
        camera_target_m = np.asarray(self._selected_camera_point_m, dtype=float).reshape(3)
        target_text = (
            f"相机坐标: {_format_vec_cm(camera_target_m)}\n"
            f"机械臂坐标: {_format_vec_cm(self._target_robot_point_m)}"
        )
        self.flow_target_label.setText(
            f"目标点:\n{target_text}"
        )
        self._advance_book_grasp_flow(
            f"目标点已解算: {target_text.replace(chr(10), '；')}"
        )

    def _solve_tail_putback_bottom_point(self):
        if self._selected_robot_target_raw_m is None:
            self._set_error("请先识别书籍末尾底边点")
            return
        self._tail_putback_bottom_point_m = np.asarray(
            self._selected_robot_target_raw_m,
            dtype=float,
        ).reshape(3)
        self._target_robot_point_m = None
        self.move_target_point_cm_value.setText(
            _format_vec_cm(self._tail_putback_bottom_point_m)
        )
        self.target_point_cm_value.setText(
            _format_vec_cm(self._tail_putback_bottom_point_m)
        )
        self.flow_target_label.setText(
            f"底边点: {_format_vec_cm(self._tail_putback_bottom_point_m)}"
        )
        self._update_move_button_state()
        self._advance_book_grasp_flow(
            f"底边点已解算: {_format_vec_cm(self._tail_putback_bottom_point_m)}"
        )

    def _solve_tail_putback_target_point(self):
        self._compute_tail_putback_target_point(advance_flow=True)

    def _compute_tail_putback_target_point(self, advance_flow: bool):
        if self._tail_putback_bottom_point_m is None:
            if self._selected_robot_target_raw_m is None:
                self._set_error("请先解算底边点")
                return False
            self._tail_putback_bottom_point_m = np.asarray(
                self._selected_robot_target_raw_m,
                dtype=float,
            ).reshape(3)

        bottom = np.asarray(self._tail_putback_bottom_point_m, dtype=float).reshape(3)
        target = bottom.copy()
        target[1] += float(self.tail_book_thickness_spin.value()) / (2.0 * M_TO_CM)
        target[2] += (
            float(self.tail_book_height_spin.value())
            - float(self.tail_gripper_height_spin.value()) / 2.0
        ) / M_TO_CM
        target += np.asarray(
            self._current_target_compensation_cm(),
            dtype=float,
        ) / M_TO_CM
        self._target_robot_point_m = target
        if self._workflow_mode == "tail_putback":
            self._tail_putback_step4_target_point_m = target.copy()
        self.target_point_cm_value.setText(_format_vec_cm(target))
        self.flow_target_label.setText(
            f"目标点: {_format_vec_cm(target)}"
        )
        self._show_tail_putback_target_preview(bottom, target)
        self._update_move_button_state()
        if advance_flow:
            self._advance_book_grasp_flow(
                f"目标点已解算: {_format_vec_cm(target)}"
            )
        return True

    def _execute_tail_putback_preplace_step(self):
        if self._target_robot_point_m is None:
            self._compute_tail_putback_target_point(advance_flow=False)
        if self._target_robot_point_m is None:
            self._set_error("没有目标点，无法到达预备放书点。请先执行步骤4解算目标点")
            return
        xyz_offset = self._spin_values(self.tail_preplace_xyz_spins)
        rpy_offset = self._spin_values(self.tail_preplace_rpy_spins)
        base_rpy = self._spin_values(self.putback_target_rpy_spins)
        pose = self._make_target_pose_with_offset(
            x_offset_cm=xyz_offset[0],
            y_offset_cm=xyz_offset[1],
            z_offset_cm=xyz_offset[2],
            rpy_deg=[
                float(base_rpy[0]) + float(rpy_offset[0]),
                float(base_rpy[1]) + float(rpy_offset[1]),
                float(base_rpy[2]) + float(rpy_offset[2]),
            ],
        )
        duration = float(self.tail_preplace_movej_duration_spin.value())
        motion_mode = (
            str(self.tail_preplace_motion_combo.currentData())
            if hasattr(self, "tail_preplace_motion_combo")
            else self.DEFAULT_TAIL_PREPLACE_MOTION_MODE
        )
        if motion_mode == "movel":
            self._send_blocking_movel(
                pose,
                "等待机械臂 MoveL 到预备放书点",
                duration=duration,
            )
        else:
            self._send_blocking_end_pose(
                pose,
                "等待机械臂 IK + MoveJ 到预备放书点",
                duration=duration,
            )

    def _execute_tail_putback_target_movel_step(self):
        if self._target_robot_point_m is None:
            self._compute_tail_putback_target_point(advance_flow=False)
        if self._target_robot_point_m is None:
            self._set_error("没有目标点，无法运动到放置点。请先执行步骤4解算目标点")
            return
        xyz_offset = self._spin_values(getattr(self, "tail_place_offset_xyz_spins", []))
        if len(xyz_offset) < 3:
            xyz_offset = list(self.DEFAULT_TAIL_PLACE_OFFSET_CM)
        rpy_offset = self._spin_values(getattr(self, "tail_place_rpy_offset_spins", []))
        if len(rpy_offset) < 3:
            rpy_offset = list(self.DEFAULT_TAIL_PLACE_RPY_OFFSET_DEG)
        base_rpy = self._spin_values(self.putback_target_rpy_spins)
        pose = self._make_target_pose_with_offset(
            x_offset_cm=xyz_offset[0],
            y_offset_cm=xyz_offset[1],
            z_offset_cm=xyz_offset[2],
            rpy_deg=[
                float(base_rpy[0]) + float(rpy_offset[0]),
                float(base_rpy[1]) + float(rpy_offset[1]),
                float(base_rpy[2]) + float(rpy_offset[2]),
            ],
        )
        duration = (
            float(self.tail_target_movel_duration_spin.value())
            if hasattr(self, "tail_target_movel_duration_spin")
            else self._flow_movel_duration()
        )
        motion_mode = (
            str(self.tail_place_motion_combo.currentData())
            if hasattr(self, "tail_place_motion_combo")
            else "movel"
        )
        if motion_mode == "movej":
            self._send_blocking_end_pose(
                pose,
                "等待机械臂 MoveJ 到书籍末尾放回放置点",
                duration=duration,
            )
        else:
            self._send_blocking_movel(
                pose,
                "等待机械臂 MoveL 到书籍末尾放回放置点",
                duration=duration,
            )

    def _execute_tail_putback_fine_tune_step(self):
        xyz_offset = self._spin_values(getattr(self, "tail_fine_tune_xyz_spins", []))
        if len(xyz_offset) < 3:
            xyz_offset = list(self.DEFAULT_TAIL_FINE_TUNE_OFFSET_CM)
        rpy_offset = self._spin_values(getattr(self, "tail_fine_tune_rpy_spins", []))
        if len(rpy_offset) < 3:
            rpy_offset = list(self.DEFAULT_TAIL_FINE_TUNE_RPY_OFFSET_DEG)
        pose = self._make_pose_from_current_end_pose(
            x_offset_cm=xyz_offset[0],
            y_offset_cm=xyz_offset[1],
            z_offset_cm=xyz_offset[2],
            rpy_offset_deg=rpy_offset,
            local_axes=False,
            prefer_last_flow_pose=True,
        )
        if pose is None:
            self._set_error("暂无步骤7后的位姿反馈，无法微调书本位置")
            return
        duration = (
            float(self.tail_fine_tune_movel_duration_spin.value())
            if hasattr(self, "tail_fine_tune_movel_duration_spin")
            else self.DEFAULT_TAIL_FINE_TUNE_MOVEL_DURATION_S
        )
        motion_mode = (
            str(self.tail_fine_tune_motion_combo.currentData())
            if hasattr(self, "tail_fine_tune_motion_combo")
            else "movel"
        )
        if motion_mode == "movej":
            self._send_blocking_end_pose(
                pose,
                "等待机械臂 MoveJ 微调书本位置",
                duration=duration,
            )
        else:
            self._send_blocking_movel(
                pose,
                "等待机械臂 MoveL 微调书本位置",
                duration=duration,
            )

    def _execute_tail_putback_slight_open_step(self):
        delta_deg = (
            float(self.tail_gripper_slight_open_spin.value())
            if hasattr(self, "tail_gripper_slight_open_spin")
            else float(self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_DEG)
        )
        base_deg = (
            float(self._current_gripper_angle_deg)
            if self._current_gripper_angle_deg is not None
            else float(self.gripper_close_spin.value())
            if hasattr(self, "gripper_close_spin")
            else float(self.DEFAULT_GRIPPER_CLOSE_DEG)
        )
        open_deg = (
            float(self.gripper_open_spin.value())
            if hasattr(self, "gripper_open_spin")
            else float(self.DEFAULT_GRIPPER_OPEN_DEG)
        )
        open_direction = 1.0 if open_deg > base_deg else -1.0
        target_deg = max(-30.0, min(140.0, base_deg + open_direction * abs(delta_deg)))
        target_deg = self._clamp_gripper_target_toward_open(
            base_deg=base_deg,
            target_deg=target_deg,
            open_deg=open_deg,
        )
        self._start_shared_gripper_open(
            waiting_kind="tail_gripper_slight_open",
            waiting_text=f"等待夹爪从当前 {base_deg:.1f}° 微张 {abs(delta_deg):.1f}° 到 {target_deg:.1f}°",
            target_deg=target_deg,
            speed_deg_s=(
                float(self.tail_gripper_slight_open_speed_spin.value())
                if hasattr(self, "tail_gripper_slight_open_speed_spin")
                else float(self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_SPEED_DEG_S)
            ),
            monitor_params=self._tail_gripper_slight_open_monitor_params(),
        )

    def _execute_tail_putback_backoff_step(self):
        xyz_offset = self._spin_values(getattr(self, "tail_backoff_xyz_spins", []))
        if len(xyz_offset) < 3:
            xyz_offset = list(self.DEFAULT_TAIL_BACKOFF_OFFSET_CM)
        rpy_offset = self._spin_values(getattr(self, "tail_backoff_rpy_spins", []))
        if len(rpy_offset) < 3:
            rpy_offset = list(self.DEFAULT_TAIL_BACKOFF_RPY_OFFSET_DEG)
        pose = self._make_pose_from_current_end_pose(
            x_offset_cm=xyz_offset[0],
            y_offset_cm=xyz_offset[1],
            z_offset_cm=xyz_offset[2],
            rpy_offset_deg=rpy_offset,
            local_axes=False,
            prefer_last_flow_pose=True,
        )
        if pose is None:
            self._set_error("暂无当前末端位姿反馈，无法执行步骤10后移")
            return
        duration = (
            float(self.tail_backoff_movel_duration_spin.value())
            if hasattr(self, "tail_backoff_movel_duration_spin")
            else self.DEFAULT_TAIL_BACKOFF_MOVEL_DURATION_S
        )
        self._send_blocking_movel(
            pose,
            "等待机械臂基于当前位姿 MoveL 后移",
            duration=duration,
        )

    def _execute_tail_putback_close_step(self):
        self._refresh_shared_gripper_defaults()
        self._start_monitored_gripper_close(waiting_kind="tail_home_gripper_close")

    def _tail_pose_from_current_with_offsets(
        self,
        *,
        xyz_offset_spins: Sequence[QDoubleSpinBox],
        rpy_offset_spins: Sequence[QDoubleSpinBox],
        xyz_fallback: Sequence[float],
        rpy_fallback: Sequence[float],
        missing_message: str,
    ) -> Optional[list[float]]:
        xyz_offset = self._spin_values(xyz_offset_spins)
        if len(xyz_offset) < 3:
            xyz_offset = list(xyz_fallback)
        rpy_offset = self._spin_values(rpy_offset_spins)
        if len(rpy_offset) < 3:
            rpy_offset = list(rpy_fallback)
        pose = self._make_pose_from_current_end_pose(
            x_offset_cm=xyz_offset[0],
            y_offset_cm=xyz_offset[1],
            z_offset_cm=xyz_offset[2],
            rpy_offset_deg=rpy_offset,
            local_axes=False,
            prefer_last_flow_pose=True,
        )
        if pose is None:
            self._set_error(missing_message)
            return None
        return pose

    def _execute_tail_putback_reapproach_step(self):
        pose = self._tail_pose_from_current_with_offsets(
            xyz_offset_spins=getattr(self, "tail_prepush_offset_xyz_spins", []),
            rpy_offset_spins=getattr(self, "tail_prepush_rpy_offset_spins", []),
            xyz_fallback=self.DEFAULT_TAIL_PREPUSH_OFFSET_CM,
            rpy_fallback=self.DEFAULT_TAIL_PREPUSH_RPY_OFFSET_DEG,
            missing_message="暂无步骤11后的末端位姿反馈，无法运动到预备推书位",
        )
        if pose is None:
            return
        duration = (
            float(self.tail_prepush_movel_duration_spin.value())
            if hasattr(self, "tail_prepush_movel_duration_spin")
            else float(self.DEFAULT_TAIL_PREPUSH_MOVEL_DURATION_S)
        )
        motion_mode = (
            str(self.tail_prepush_motion_combo.currentData())
            if hasattr(self, "tail_prepush_motion_combo")
            else self.DEFAULT_TAIL_PREPUSH_MOTION_MODE
        )
        if motion_mode == "movej":
            self._send_blocking_end_pose(
                pose,
                "等待机械臂基于步骤11位姿 MoveJ 到预备推书位",
                duration=duration,
            )
        else:
            self._send_blocking_movel(
                pose,
                "等待机械臂基于步骤11位姿 MoveL 到预备推书位",
                duration=duration,
            )

    def _execute_tail_putback_target_comp_step(self):
        pose = self._tail_pose_from_current_with_offsets(
            xyz_offset_spins=getattr(self, "tail_push_offset_xyz_spins", []),
            rpy_offset_spins=getattr(self, "tail_push_rpy_offset_spins", []),
            xyz_fallback=self.DEFAULT_TAIL_PUSH_OFFSET_CM,
            rpy_fallback=self.DEFAULT_TAIL_PUSH_RPY_OFFSET_DEG,
            missing_message="暂无步骤12后的末端位姿反馈，无法运动到推书位",
        )
        if pose is None:
            return
        duration = (
            float(self.tail_push_movel_duration_spin.value())
            if hasattr(self, "tail_push_movel_duration_spin")
            else float(self.DEFAULT_TAIL_PUSH_MOVEL_DURATION_S)
        )
        self._send_blocking_movel(
            pose,
            "等待机械臂基于步骤12位姿 MoveL 到推书位",
            duration=duration,
        )

    def _execute_tail_putback_final_home_step(self):
        mode = (
            str(self.tail_final_home_combo.currentData())
            if hasattr(self, "tail_final_home_combo")
            else self.DEFAULT_TAIL_FINAL_HOME_MODE
        )
        duration = (
            float(self.tail_final_home_duration_spin.value())
            if hasattr(self, "tail_final_home_duration_spin")
            else float(self.DEFAULT_TAIL_FINAL_HOME_DURATION_S)
        )
        if mode == "debug":
            self._ensure_debug_pose_controls()
            joints = self._joint_spins_to_radians(self.debug_joint_spins)
            waiting_text = (
                "等待机械臂 MoveJ 回到图书调试位 "
                f"{self._format_joint_spins_deg(self.debug_joint_spins)}"
            )
        else:
            joints = self._joint_control_zero_joints_rad()
            waiting_text = f"等待机械臂 MoveJ 回到归零位置 {self._format_joint_values_rad_as_deg(joints)}"
        self._send_manual_header_movej(
            joints,
            duration,
            "tail_final_home",
            waiting_text,
        )

    def _execute_pregrasp_step(self):
        if self._target_robot_point_m is None:
            self._set_error("没有目标点，无法执行预抓取")
            return
        self._set_grasp_rpy()
        if self._pregrasp_prepare_tools_enabled():
            self._refresh_shared_gripper_defaults()
            self._send_gripper(self.gripper_open_spin.value(), 0.0)
            self._send_rod_zero_nonblocking()
        xyz_offset = self._spin_values(self.pregrasp_offset_xyz_spins)
        pose = self._make_target_pose_with_offset(
            x_offset_cm=xyz_offset[0],
            y_offset_cm=xyz_offset[1],
            z_offset_cm=xyz_offset[2],
            rpy_deg=self._spin_values(self.pregrasp_rpy_spins),
        )
        self._send_blocking_end_pose(pose, "等待机械臂 IK + MoveJ 到达预备抓取位姿")

    def _pregrasp_prepare_tools_enabled(self) -> bool:
        return bool(
            not hasattr(self, "pregrasp_prepare_tools_check")
            or self.pregrasp_prepare_tools_check.isChecked()
        )

    def _execute_approach_step(self):
        if self._target_robot_point_m is None:
            self._set_error("没有目标点，无法执行夹取移动")
            return
        self._set_grasp_rpy()
        pose = self._make_target_pose_with_offset(
            x_offset_cm=0.0,
            y_offset_cm=0.0,
            z_offset_cm=0.0,
            rpy_deg=self._spin_values(self.pregrasp_rpy_spins),
        )
        self._flow_approach_pose = [float(value) for value in pose]
        self._send_blocking_movel(pose, "等待机械臂到达抓取位姿")

    def _set_grasp_rpy(self):
        self._set_rpy_spins(self._spin_values(self.pregrasp_rpy_spins))

    def _execute_relative_target_step(self):
        if self._target_robot_point_m is None:
            self._set_error("没有目标点，无法执行微调抓取位姿")
            return
        xyz_offset = self._spin_values(self.relative_target_xyz_spins)
        rpy_deg = self._spin_values(self.relative_target_rpy_spins)
        pose = self._make_target_pose_with_offset(
            x_offset_cm=xyz_offset[0],
            y_offset_cm=xyz_offset[1],
            z_offset_cm=xyz_offset[2],
            rpy_deg=rpy_deg,
        )
        duration = (
            float(self.relative_target_duration_spin.value())
            if hasattr(self, "relative_target_duration_spin")
            else float(self.DEFAULT_TARGET_RELATIVE_DURATION_S)
        )
        self._send_blocking_movel(
            pose,
            "等待机械臂微调到抓取位姿",
            duration=duration,
        )

    def _execute_post_gripper_movej_offset_step(self):
        xyz_offset = self._spin_values(self.post_gripper_movej_xyz_spins)
        rpy_offset = self._spin_values(self.post_gripper_movej_rpy_spins)
        pose = self._make_pose_from_current_end_pose(
            x_offset_cm=xyz_offset[0],
            y_offset_cm=xyz_offset[1],
            z_offset_cm=xyz_offset[2],
            rpy_offset_deg=rpy_offset,
            prefer_last_flow_pose=True,
        )
        if pose is None:
            self._set_error("暂无末端位姿，无法执行当前位姿偏移 MoveJ")
            return
        self._send_blocking_end_pose(
            pose,
            (
                "等待机械臂从当前状态偏移 "
                f"XYZ=[{xyz_offset[0]:.2f}, {xyz_offset[1]:.2f}, {xyz_offset[2]:.2f}]cm "
                f"RPY=[{rpy_offset[0]:.2f}, {rpy_offset[1]:.2f}, {rpy_offset[2]:.2f}]° "
                "并 IK + MoveJ 到目标位姿"
            ),
            duration=float(self.post_gripper_movej_duration_spin.value()),
        )

    def _execute_putback2_reset_rpy_step(self):
        rpy_deg = self._spin_values(self.putback2_reset_rpy_spins)
        pose = self._make_pose_from_current_end_pose(
            rpy_deg=rpy_deg,
            prefer_last_flow_pose=True,
        )
        if pose is None:
            self._set_error("暂无末端位姿，无法执行姿态恢复")
            return
        self._send_blocking_end_pose(
            pose,
            (
                "等待机械臂保持当前位置并将末端姿态改为 "
                f"RPY=[{rpy_deg[0]:.2f}, {rpy_deg[1]:.2f}, {rpy_deg[2]:.2f}]°"
            ),
            duration=float(self.putback2_reset_rpy_duration_spin.value()),
        )

    def _execute_putback2_x_offset_step(self):
        x_offset = float(self.putback2_x_offset_spin.value())
        self._execute_putback2_base_x_offset(
            x_offset,
            float(self.putback2_x_move_duration_spin.value()),
        )

    def _execute_putback2_base_x_offset(self, x_offset: float, duration: float):
        pose = self._make_pose_from_current_end_pose(
            x_offset_cm=float(x_offset),
            local_axes=False,
            prefer_last_flow_pose=True,
        )
        if pose is None:
            self._set_error("暂无末端位姿，无法沿基坐标系 X+ 移动")
            return
        self._send_blocking_movel(
            pose,
            f"等待机械臂沿基坐标系 X MoveL 移动 {float(x_offset):.2f}cm",
            duration=float(duration),
        )

    def _execute_putback2_z_offset_step(self):
        z_offset = float(self.putback2_z_offset_spin.value())
        pose = self._make_pose_from_current_end_pose(
            z_offset_cm=z_offset,
            local_axes=False,
            prefer_last_flow_pose=True,
        )
        if pose is None:
            self._set_error("暂无末端位姿，无法沿基坐标系 Z 移动")
            return
        self._send_blocking_movel(
            pose,
            f"等待机械臂沿基坐标系 Z MoveL 移动 {z_offset:.2f}cm",
            duration=float(self.putback2_z_move_duration_spin.value()),
        )

    def _execute_putback2_after_open_x1_step(self):
        self._execute_putback2_base_x_offset(
            float(self.putback2_after_open_x1_offset_spin.value()),
            float(self.putback2_after_open_x1_duration_spin.value()),
        )

    def _execute_putback2_after_combo_x_step(self):
        self._execute_putback2_base_x_offset(
            float(self.putback2_after_combo_x_offset_spin.value()),
            float(self.putback2_after_combo_x_duration_spin.value()),
        )

    def _start_putback2_combo_step(self):
        self._refresh_shared_gripper_defaults()
        self._start_putback2_gripper_move(
            target_deg=(
                float(self.putback2_combo_close_spin.value())
                if hasattr(self, "putback2_combo_close_spin")
                else float(self.gripper_close_spin.value())
            ),
            speed_deg_s=(
                float(self.putback2_combo_close_speed_spin.value())
                if hasattr(self, "putback2_combo_close_speed_spin")
                else self._gripper_close_speed_deg_s()
            ),
            waiting_kind="putback2_combo_close",
            waiting_text=(
                "步骤16：等待夹爪闭合到 "
                f"{float(self.putback2_combo_close_spin.value()) if hasattr(self, 'putback2_combo_close_spin') else float(self.gripper_close_spin.value()):.1f}°"
            ),
            effort=float(self.gripper_effort_spin.value()),
        )

    def _execute_putback2_combo_y_move(self):
        y_offset = float(self.putback2_combo_y_offset_spin.value())
        pose = self._make_pose_from_current_end_pose(
            y_offset_cm=y_offset,
            local_axes=False,
            prefer_last_flow_pose=True,
        )
        if pose is None:
            self._set_error("暂无末端位姿，无法执行步骤16的 Y 移动")
            return
        self._send_blocking_movel(
            pose,
            f"步骤16：等待机械臂沿基坐标系 Y MoveL 移动 {y_offset:.2f}cm",
            duration=float(self.putback2_combo_y_duration_spin.value()),
        )
        self._flow_waiting_kind = "putback2_combo_y_move"

    def _begin_putback2_combo_wait(self):
        wait_s = max(0.0, float(self.putback2_combo_wait_spin.value()))
        self._flow_waiting_motion = True
        self._flow_waiting_kind = "putback2_combo_wait"
        self._flow_pending_pose = None
        self._update_flow_button_state()
        self.flow_status_label.setText(f"步骤16：等待 {wait_s:.1f}s 后执行 X 移动")
        QTimer.singleShot(int(wait_s * 1000), self._finish_putback2_combo_wait)

    def _finish_putback2_combo_wait(self):
        if (
            self._workflow_mode != "putback2"
            or not self._flow_active
            or not self._flow_waiting_motion
            or self._flow_waiting_kind != "putback2_combo_wait"
        ):
            return
        self._execute_putback2_combo_x_move()

    def _execute_putback2_combo_x_move(self):
        x_offset = float(self.putback2_combo_x_offset_spin.value())
        pose = self._make_pose_from_current_end_pose(
            x_offset_cm=x_offset,
            local_axes=False,
            prefer_last_flow_pose=True,
        )
        if pose is None:
            self._set_error("暂无末端位姿，无法执行步骤16的 X 移动")
            return
        self._send_blocking_movel(
            pose,
            f"步骤16：等待机械臂沿基坐标系 X MoveL 移动 {x_offset:.2f}cm",
            duration=float(self.putback2_combo_x_duration_spin.value()),
        )
        self._flow_waiting_kind = "putback2_combo_x_move"

    def _finish_putback2_combo_with_full_open(self):
        target_deg = (
            float(self.putback2_combo_full_open_spin.value())
            if hasattr(self, "putback2_combo_full_open_spin")
            else float(self.gripper_open_spin.value())
        )
        speed_deg_s = (
            float(self.putback2_combo_full_open_speed_spin.value())
            if hasattr(self, "putback2_combo_full_open_speed_spin")
            else self._gripper_open_speed_deg_s()
        )
        self._start_putback2_gripper_move(
            target_deg=target_deg,
            speed_deg_s=speed_deg_s,
            waiting_kind="putback2_combo_full_open",
            waiting_text=f"步骤16：等待夹爪全部张开到 {target_deg:.1f}°",
            effort=0.0,
        )

    def _execute_return_debug_step(self):
        joints = self._joint_spins_to_radians(self.debug_joint_spins)
        self._record_previous_flow_pose()
        self._flow_waiting_motion = True
        self._flow_waiting_kind = "move_j"
        self._flow_pending_pose = None
        self._update_flow_button_state()
        self.flow_status_label.setText(
            f"等待机械臂 MoveJ 回到图书调试构型 {self._format_joint_spins_deg(self.debug_joint_spins)}"
        )
        self.move_j_block_requested.emit(
            joints,
            float(self.flow_debug_duration_spin.value()),
        )

    def _move_to_debug_pose_from_header(self):
        if self._flow_waiting_motion:
            return
        if not self._arm_enabled:
            self._set_error("机械臂未使能，无法运动到图书调试位")
            return
        if not hasattr(self, "debug_joint_spins"):
            self._set_error("图书调试位关节参数尚未初始化")
            return

        joints = self._joint_spins_to_radians(self.debug_joint_spins)
        self._send_manual_header_movej(
            joints,
            self._manual_header_move_duration(),
            "debug_move",
            f"等待机械臂 MoveJ 到图书调试位 {self._format_joint_spins_deg(self.debug_joint_spins)}",
        )

    def _move_to_zero_pose_from_header(self):
        if self._flow_waiting_motion:
            return
        if not self._arm_enabled:
            self._set_error("机械臂未使能，无法归零")
            return
        joints = self._joint_control_zero_joints_rad()
        self._send_manual_header_movej(
            joints,
            self._manual_header_zero_duration(),
            "zero_move",
            f"等待机械臂 MoveJ 归零 {self._format_joint_values_rad_as_deg(joints)}",
        )

    def _close_gripper_from_header(self):
        if self._flow_waiting_motion:
            return
        if not self._arm_enabled:
            self._set_error("机械臂未使能，无法关闭夹爪")
            return
        self._refresh_shared_gripper_defaults()
        self._clear_manual_gripper_open()
        self._start_monitored_gripper_close(waiting_kind="manual_gripper_close")

    def _open_gripper_from_header(self):
        if self._flow_waiting_motion:
            return
        if not self._arm_enabled:
            self._set_error("机械臂未使能，无法打开夹爪")
            return
        self._refresh_shared_gripper_defaults()
        self._start_shared_gripper_open(
            waiting_kind="manual_gripper_open",
            waiting_text=(
                f"等待后台夹爪缓慢打开到 {float(self.gripper_open_spin.value()):.1f}°"
            ),
        )

    def _manual_header_move_duration(self) -> float:
        return float(self.HEADER_MOVE_DURATION_S)

    def set_header_zero_duration(self, duration_s: float):
        try:
            value = float(duration_s)
        except (TypeError, ValueError):
            return
        self._header_zero_duration_s = max(0.5, min(30.0, value))

    def _manual_header_zero_duration(self) -> float:
        return float(self._header_zero_duration_s)

    def _send_manual_header_movej(
        self,
        joints: list[float],
        duration: float,
        waiting_kind: str,
        waiting_status: str,
    ):
        self._record_previous_flow_pose()
        self._flow_waiting_motion = True
        self._flow_waiting_kind = waiting_kind
        self._flow_rollback_waiting = False
        self._flow_pending_pose = None
        self._update_flow_button_state()
        self.flow_status_label.setText(waiting_status)
        self.move_j_block_requested.emit(joints, duration)

    def _manual_gripper_open_target_angle_deg(self) -> float:
        if hasattr(self, "gripper_open_spin"):
            return float(self.gripper_open_spin.value())
        return float(self.DEFAULT_GRIPPER_OPEN_DEG)

    def _clear_manual_gripper_open(self):
        self._manual_gripper_open_target_deg = None

    def _execute_turn_stage1_step(self):
        self._send_slow_turn_movej(
            self.turn_stage1_joint_spins,
            f"等待机械臂慢速转身到预备放书位置 {self._format_joint_spins_deg(self.turn_stage1_joint_spins)}",
        )

    def _execute_takeout_specified_pose_step(self):
        xyz_cm = self._spin_values(self.takeout_specified_pose_xyz_spins)
        rpy_deg = self._spin_values(self.takeout_specified_pose_rpy_spins)
        pose = [
            float(xyz_cm[0]) / M_TO_CM,
            float(xyz_cm[1]) / M_TO_CM,
            float(xyz_cm[2]) / M_TO_CM,
            math.radians(float(rpy_deg[0])),
            math.radians(float(rpy_deg[1])),
            math.radians(float(rpy_deg[2])),
        ]
        self._send_blocking_end_pose(
            pose,
            (
                "等待机械臂 IK + MoveJ 到指定位姿 "
                f"XYZ=[{xyz_cm[0]:.2f}, {xyz_cm[1]:.2f}, {xyz_cm[2]:.2f}]cm "
                f"RPY=[{rpy_deg[0]:.2f}, {rpy_deg[1]:.2f}, {rpy_deg[2]:.2f}]°"
            ),
            duration=float(self.takeout_specified_pose_duration_spin.value()),
        )

    def _execute_takeout_step13_fine_tune_step(self):
        xyz_offset = self._spin_values(
            getattr(self, "takeout_step13_fine_tune_xyz_spins", [])
        )
        if len(xyz_offset) < 3:
            xyz_offset = list(self.DEFAULT_TAKEOUT_STEP13_FINE_TUNE_OFFSET_CM)
        rpy_offset = self._spin_values(
            getattr(self, "takeout_step13_fine_tune_rpy_spins", [])
        )
        if len(rpy_offset) < 3:
            rpy_offset = list(self.DEFAULT_TAKEOUT_STEP13_FINE_TUNE_RPY_OFFSET_DEG)
        pose = self._make_pose_from_current_end_pose(
            x_offset_cm=xyz_offset[0],
            y_offset_cm=xyz_offset[1],
            z_offset_cm=xyz_offset[2],
            rpy_offset_deg=rpy_offset,
            local_axes=False,
            prefer_last_flow_pose=True,
        )
        if pose is None:
            self._set_error("暂无步骤13后的末端位姿反馈，无法执行步骤14微调")
            return
        duration = (
            float(self.takeout_step13_fine_tune_duration_spin.value())
            if hasattr(self, "takeout_step13_fine_tune_duration_spin")
            else float(self.DEFAULT_TAKEOUT_STEP13_FINE_TUNE_DURATION_S)
        )
        self._send_blocking_movel(
            pose,
            (
                "等待机械臂基于步骤13位姿 MoveL 微调 "
                f"XYZ=[{xyz_offset[0]:.2f}, {xyz_offset[1]:.2f}, {xyz_offset[2]:.2f}]cm "
                f"RPY=[{rpy_offset[0]:.2f}, {rpy_offset[1]:.2f}, {rpy_offset[2]:.2f}]°"
            ),
            duration=duration,
        )

    def _execute_takeout_gripper_open_step(self):
        self._start_shared_gripper_open(
            waiting_kind="takeout_gripper_open",
            waiting_text=(
                "等待步骤15夹爪缓慢打开到 "
                f"{float(self.gripper_open_spin.value()):.1f}°"
            ),
        )

    def _execute_final_joint_step(self):
        self._send_joint_movej(
            self.final_joint_spins,
            float(self.flow_final_duration_spin.value()),
            f"等待机械臂 MoveJ 到最终构型 {self._format_joint_spins_deg(self.final_joint_spins)}",
        )

    def _lift_position_height_cm(self, position: str, settings: dict) -> float:
        return_offset = float(settings.get("return_offset_cm", 10.0))
        take_offset = float(settings.get("take_offset_cm", 10.0))
        if position == self.LIFT_POSITION_RETURN:
            return return_offset
        if position == self.LIFT_POSITION_TAKE:
            return return_offset + take_offset
        return 0.0

    def _current_takeout_lift_position(self) -> str:
        combo = getattr(self, "takeout_lift_current_combo", None)
        if combo is None:
            return self.DEFAULT_TAKEOUT_LIFT_POSITION
        position = combo.currentData()
        if position in {
            self.LIFT_POSITION_LOWEST,
            self.LIFT_POSITION_RETURN,
            self.LIFT_POSITION_TAKE,
        }:
            return str(position)
        return self.DEFAULT_TAKEOUT_LIFT_POSITION

    def _execute_lift_to_return_step(self):
        settings = load_lift_platform_defaults()
        current_position = self.LIFT_POSITION_TAKE
        target_position = self.LIFT_POSITION_RETURN
        delta_cm = (
            self._lift_position_height_cm(target_position, settings)
            - self._lift_position_height_cm(current_position, settings)
        )
        if abs(delta_cm) < 1e-9:
            self._advance_book_grasp_flow("升降台已在还书位")
            return
        self._flow_waiting_motion = True
        self._flow_waiting_kind = "lift"
        self._flow_pending_pose = None
        self._update_flow_button_state()
        self.flow_status_label.setText(f"等待升降台从取书位移动到还书位 ({delta_cm:.3f} cm)")
        self.lift_move_distance_requested.emit(
            delta_cm,
            int(settings.get("speed_rpm", DEFAULT_LIFT_SPEED_RPM)),
            int(settings.get("acceleration", DEFAULT_LIFT_ACCELERATION)),
            float(settings.get("pulses_per_cm", DEFAULT_LIFT_PULSES_PER_CM)),
            -1 if bool(settings.get("reverse_up_direction", False)) else 1,
        )

    def _send_slow_turn_movej(self, joint_spins: Sequence[QDoubleSpinBox], waiting_text: str):
        self._send_joint_movej(
            joint_spins,
            float(self.flow_turn_duration_spin.value()),
            waiting_text,
        )

    def _send_joint_movej(
        self,
        joint_spins: Sequence[QDoubleSpinBox],
        duration: float,
        waiting_text: str,
    ):
        joints = self._joint_spins_to_radians(joint_spins)
        self._record_previous_flow_pose()
        self._flow_waiting_motion = True
        self._flow_waiting_kind = "move_j"
        self._flow_pending_pose = None
        self._update_flow_button_state()
        self.flow_status_label.setText(waiting_text)
        self.move_j_block_requested.emit(joints, duration)

    def _pose_from_current_end_pose(self) -> Optional[list[float]]:
        pose = self._current_end_pose
        if pose is None:
            return None
        return [
            float(pose.x),
            float(pose.y),
            float(pose.z),
            float(pose.rx),
            float(pose.ry),
            float(pose.rz),
        ]

    @staticmethod
    def _joint_spins_to_radians(spins: Sequence[QDoubleSpinBox]) -> list[float]:
        return [math.radians(float(spin.value())) for spin in spins]

    @staticmethod
    def _joint_control_zero_joints_rad() -> list[float]:
        joints = []
        for joint_id in range(1, 7):
            lo, hi = DEFAULT_JOINT_LIMITS[joint_id]
            joints.append(0.0 if lo <= 0.0 <= hi else float(lo))
        return joints

    @staticmethod
    def _format_joint_spins_deg(spins: Sequence[QDoubleSpinBox]) -> str:
        values = [float(spin.value()) for spin in spins]
        return "[" + ", ".join(f"{value:.2f}" for value in values) + "]"

    @staticmethod
    def _format_joint_values_rad_as_deg(joints: Sequence[float]) -> str:
        values = [math.degrees(float(value)) for value in joints]
        return "[" + ", ".join(f"{value:.2f}" for value in values) + "]"

    def _make_putback_target_pose(
        self,
        x_offset_cm: float,
        y_offset_cm: float,
        z_offset_cm: float,
        rpy_deg: Sequence[float],
    ) -> list[float]:
        return self._make_target_pose_with_offset(
            x_offset_cm=float(x_offset_cm),
            y_offset_cm=float(y_offset_cm),
            z_offset_cm=float(z_offset_cm),
            rpy_deg=rpy_deg,
        )

    def _send_blocking_movel(
        self,
        pose: list[float],
        waiting_text: str,
        duration: Optional[float] = None,
    ):
        self._record_previous_flow_pose()
        self._flow_waiting_motion = True
        self._flow_waiting_kind = "move_l"
        self._flow_pending_pose = [float(value) for value in pose]
        self._update_flow_button_state()
        self.flow_status_label.setText(waiting_text)
        if duration is None:
            duration = self._flow_movel_duration()
        self.move_l_block_requested.emit(pose, float(duration))

    def _flow_movel_duration(self) -> float:
        if hasattr(self, "flow_movel_duration_spin"):
            return float(self.flow_movel_duration_spin.value())
        return float(self.DEFAULT_FLOW_MOVEL_DURATION_S)

    def _send_blocking_end_pose(
        self,
        pose: list[float],
        waiting_text: str,
        duration: Optional[float] = None,
    ):
        self._record_previous_flow_pose()
        self._flow_waiting_motion = True
        self._flow_waiting_kind = "end_pose"
        self._flow_pending_pose = [float(value) for value in pose]
        self._update_flow_button_state()
        self.flow_status_label.setText(waiting_text)
        if duration is None:
            duration = (
                float(self.flow_ik_duration_spin.value())
                if hasattr(self, "flow_ik_duration_spin")
                else float(self.flow_debug_duration_spin.value())
            )
        self.end_pose_block_requested.emit(
            [float(value) for value in pose],
            float(duration),
        )

    def _record_previous_flow_pose(self):
        current_pose = self._pose_from_current_end_pose()
        if current_pose is None:
            return
        step = self._flow_step_index
        steps = self._book_grasp_steps()
        label = steps[step - 1] if 0 < step <= len(steps) else "当前位置"
        self._flow_pose_history.append((step, label, current_pose))
        if len(self._flow_pose_history) > 16:
            self._flow_pose_history = self._flow_pose_history[-16:]

    def _rollback_to_previous_flow_pose(self):
        if self._flow_waiting_motion:
            return
        self._flow_auto_run = False
        if not self._flow_pose_history:
            self.flow_status_label.setText("没有可回退的位置")
            return
        entry = self._flow_pose_history.pop()
        step, label, pose = entry
        if not self._flow_active:
            self._flow_active = True
        self._flow_step_index = max(0, min(step, len(self._book_grasp_steps()) - 1))
        self._flow_waiting_motion = True
        self._flow_waiting_kind = "end_pose"
        self._flow_rollback_waiting = True
        self._flow_rollback_entry = entry
        self._flow_pending_pose = [float(value) for value in pose]
        self._update_flow_button_state()
        self.flow_status_label.setText(f"正在回退到上一步位置：{label}")
        self.log_message.emit(f"{self._flow_log_prefix()}回退到上一步位置(IK + MoveJ): {label}")
        self.end_pose_block_requested.emit(
            [float(value) for value in pose],
            self._flow_rollback_duration(),
        )

    def _return_to_previous_flow_step(self):
        if self._flow_waiting_motion:
            return
        self._flow_auto_run = False
        if not self._flow_active or self._flow_step_index <= 0:
            self.flow_status_label.setText("没有可返回的上一步")
            return

        steps = self._book_grasp_steps()
        self._flow_step_index = max(0, min(self._flow_step_index - 1, len(steps) - 1))
        self._flow_waiting_motion = False
        self._flow_waiting_kind = None
        self._flow_rollback_waiting = False
        self._flow_rollback_entry = None
        self._flow_pending_pose = None
        current_step = steps[self._flow_step_index]
        self.flow_status_label.setText(
            f"已返回上一步。请确认后执行 [{self._flow_step_index + 1}/{len(steps)}] {current_step}"
        )
        self.log_message.emit(
            f"{self._flow_log_prefix()}返回上一步: [{self._flow_step_index + 1}/{len(steps)}] {current_step}"
        )
        self._update_flow_button_state()

    def _flow_rollback_duration(self) -> float:
        if hasattr(self, "flow_ik_duration_spin"):
            return float(self.flow_ik_duration_spin.value())
        if hasattr(self, "flow_debug_duration_spin"):
            return float(self.flow_debug_duration_spin.value())
        return self._flow_movel_duration()

    def _send_gripper(self, angle_deg: float, effort: float):
        self.gripper_requested.emit(
            math.radians(float(angle_deg)),
            float(effort),
            float(self.gripper_kp_spin.value()) if effort > 0.0 and hasattr(self, "gripper_kp_spin") else 0.0,
            float(self.gripper_kd_spin.value()) if effort > 0.0 and hasattr(self, "gripper_kd_spin") else 0.0,
        )

    @staticmethod
    def _step_toward_angle(current_deg: float, target_deg: float, step_deg: float) -> float:
        if math.isclose(current_deg, target_deg, abs_tol=1e-9):
            return float(target_deg)
        direction = 1.0 if target_deg > current_deg else -1.0
        next_deg = current_deg + direction * float(step_deg)
        if direction > 0.0:
            return float(min(target_deg, next_deg))
        return float(max(target_deg, next_deg))

    def _maybe_advance_gripper_close_target(self, now_s: float):
        target_deg = self._gripper_close_target_deg
        if target_deg is None:
            return
        if now_s - self._gripper_close_last_cmd_s < self.GRIPPER_CLOSE_STEP_INTERVAL_S:
            return
        base_deg = self._gripper_close_last_cmd_deg
        if base_deg is None:
            base_deg = self._current_gripper_angle_deg
        if base_deg is None:
            return
        step_deg = self._gripper_close_step_deg()
        next_deg = self._step_toward_angle(base_deg, target_deg, step_deg)
        if math.isclose(next_deg, base_deg, abs_tol=1e-9):
            return
        self._send_gripper(next_deg, self._gripper_close_effort_nm)
        self._gripper_close_last_cmd_deg = next_deg
        self._gripper_close_last_cmd_s = now_s

    def _gripper_close_speed_deg_s(self) -> float:
        if hasattr(self, "gripper_close_speed_spin"):
            return max(0.1, float(self.gripper_close_speed_spin.value()))
        return self.DEFAULT_GRIPPER_CLOSE_SPEED_DEG_S

    def _gripper_open_speed_deg_s(self) -> float:
        if hasattr(self, "gripper_open_speed_spin"):
            return max(0.1, float(self.gripper_open_speed_spin.value()))
        return self.DEFAULT_GRIPPER_OPEN_SPEED_DEG_S

    def _gripper_monitor_params(self) -> dict:
        return {
            "timeout_s": float(self.gripper_timeout_spin.value())
            if hasattr(self, "gripper_timeout_spin")
            else float(self.GRIPPER_CLOSE_TIMEOUT_S),
            "monitor_period_s": 0.05,
            "target_tolerance": math.radians(
                float(self.gripper_target_tolerance_spin.value())
                if hasattr(self, "gripper_target_tolerance_spin")
                else float(self.GRIPPER_CLOSE_TOLERANCE_DEG)
            ),
            "stall_tolerance": math.radians(
                float(self.gripper_stall_tolerance_spin.value())
                if hasattr(self, "gripper_stall_tolerance_spin")
                else float(self.GRIPPER_CLOSE_STALL_TOLERANCE_DEG)
            ),
            "stall_time_s": float(self.gripper_stall_time_spin.value())
            if hasattr(self, "gripper_stall_time_spin")
            else float(self.GRIPPER_CLOSE_STALL_S),
            "min_monitor_s": float(self.gripper_min_monitor_spin.value())
            if hasattr(self, "gripper_min_monitor_spin")
            else float(self.GRIPPER_CLOSE_MIN_MONITOR_S),
            "hold_margin": math.radians(
                float(self.gripper_hold_margin_spin.value())
                if hasattr(self, "gripper_hold_margin_spin")
                else float(self.DEFAULT_GRIPPER_HOLD_MARGIN_DEG)
            ),
            "command_lead_s": float(self.gripper_command_lead_spin.value())
            if hasattr(self, "gripper_command_lead_spin")
            else float(self.GRIPPER_CLOSE_COMMAND_LEAD_S),
            "stall_lead_threshold_min": math.radians(
                float(self.gripper_stall_lead_threshold_spin.value())
                if hasattr(self, "gripper_stall_lead_threshold_spin")
                else float(self.GRIPPER_CLOSE_STALL_LEAD_THRESHOLD_DEG)
            ),
        }

    def _tail_gripper_slight_open_monitor_params(self) -> dict:
        def spin_value(name: str, fallback: float) -> float:
            spin = getattr(self, name, None)
            return float(spin.value()) if spin is not None else float(fallback)

        return {
            "timeout_s": spin_value(
                "tail_gripper_slight_open_timeout_spin",
                self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_TIMEOUT_S,
            ),
            "monitor_period_s": 0.05,
            "target_tolerance": math.radians(
                spin_value(
                    "tail_gripper_slight_open_target_tolerance_spin",
                    self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_TARGET_TOLERANCE_DEG,
                )
            ),
            "stall_tolerance": math.radians(
                spin_value(
                    "tail_gripper_slight_open_stall_tolerance_spin",
                    self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_STALL_TOLERANCE_DEG,
                )
            ),
            "stall_time_s": spin_value(
                "tail_gripper_slight_open_stall_time_spin",
                self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_STALL_TIME_S,
            ),
            "min_monitor_s": spin_value(
                "tail_gripper_slight_open_min_monitor_spin",
                self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_MIN_MONITOR_S,
            ),
            "hold_margin": math.radians(
                spin_value(
                    "tail_gripper_slight_open_hold_margin_spin",
                    self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_HOLD_MARGIN_DEG,
                )
            ),
            "command_lead_s": spin_value(
                "tail_gripper_slight_open_command_lead_spin",
                self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_COMMAND_LEAD_S,
            ),
            "stall_lead_threshold_min": math.radians(
                spin_value(
                    "tail_gripper_slight_open_stall_lead_threshold_spin",
                    self.DEFAULT_TAIL_GRIPPER_SLIGHT_OPEN_STALL_LEAD_THRESHOLD_DEG,
                )
            ),
        }

    @staticmethod
    def _clamp_gripper_target_toward_open(
        *,
        base_deg: float,
        target_deg: float,
        open_deg: float,
    ) -> float:
        if open_deg >= base_deg:
            return max(float(base_deg), min(float(open_deg), float(target_deg)))
        return min(float(base_deg), max(float(open_deg), float(target_deg)))

    def _gripper_close_step_deg(self) -> float:
        speed_deg_s = self._gripper_close_speed_deg_s()
        interval_s = (
            float(self.gripper_step_interval_spin.value())
            if hasattr(self, "gripper_step_interval_spin")
            else float(self.GRIPPER_CLOSE_STEP_INTERVAL_S)
        )
        return max(0.1, speed_deg_s * max(0.01, interval_s))

    def _start_monitored_gripper_close(self, waiting_kind: str = "gripper_close"):
        target_deg = float(self.gripper_close_spin.value())
        effort = float(self.gripper_effort_spin.value())
        self._flow_waiting_motion = True
        self._flow_waiting_kind = str(waiting_kind)
        self._flow_pending_pose = None
        self._gripper_close_target_deg = target_deg
        self._gripper_close_effort_nm = effort
        self._gripper_close_started_at_s = time.monotonic()
        self._gripper_close_last_cmd_s = 0.0
        self._gripper_close_last_cmd_deg = None
        self._gripper_close_last_angle_deg = None
        self._gripper_close_stable_since_s = None
        self._update_flow_button_state()
        self.flow_status_label.setText("等待后台夹爪监测闭合并保护锁定")
        monitor_params = self._gripper_monitor_params()
        self.gripper_close_monitor_requested.emit(
            {
                "target_angle": math.radians(target_deg),
                "close_speed": math.radians(self._gripper_close_speed_deg_s()),
                "effort": effort,
                "kp": float(self.gripper_kp_spin.value()) if hasattr(self, "gripper_kp_spin") else 0.0,
                "kd": float(self.gripper_kd_spin.value()) if hasattr(self, "gripper_kd_spin") else 0.0,
                **monitor_params,
                "start_effort": (
                    float(self.gripper_start_effort_spin.value())
                    if hasattr(self, "gripper_start_effort_spin")
                    else float(self.DEFAULT_GRIPPER_START_EFFORT_NM)
                ),
                "start_boost_s": (
                    float(self.gripper_start_boost_spin.value())
                    if hasattr(self, "gripper_start_boost_spin")
                    else float(self.DEFAULT_GRIPPER_START_BOOST_S)
                ),
            }
        )

    def _start_monitored_gripper_open(self, target_deg: float):
        self._start_shared_gripper_open(
            waiting_kind="manual_gripper_open",
            waiting_text=f"等待后台夹爪缓慢打开到 {float(target_deg):.1f}°",
            target_deg=float(target_deg),
        )

    def _start_shared_gripper_open(
        self,
        *,
        waiting_kind: str,
        waiting_text: str,
        target_deg: Optional[float] = None,
        speed_deg_s: Optional[float] = None,
        monitor_params: Optional[dict] = None,
    ):
        self._refresh_shared_gripper_defaults()
        target_deg = (
            float(self.gripper_open_spin.value())
            if target_deg is None
            else float(target_deg)
        )
        speed_deg_s = (
            self._gripper_open_speed_deg_s()
            if speed_deg_s is None
            else max(0.1, float(speed_deg_s))
        )
        self._flow_waiting_motion = True
        self._flow_waiting_kind = str(waiting_kind)
        self._flow_rollback_waiting = False
        self._flow_pending_pose = None
        self._manual_gripper_open_target_deg = target_deg
        self._gripper_close_target_deg = target_deg
        self._gripper_close_effort_nm = 0.0
        self._gripper_close_started_at_s = time.monotonic()
        self._gripper_close_last_cmd_s = 0.0
        self._gripper_close_last_cmd_deg = None
        self._gripper_close_last_angle_deg = None
        self._gripper_close_stable_since_s = None
        self._update_flow_button_state()
        self.flow_status_label.setText(waiting_text)
        monitor_params = self._gripper_monitor_params() if monitor_params is None else dict(monitor_params)
        self.gripper_close_monitor_requested.emit(
            {
                "target_angle": math.radians(target_deg),
                "close_speed": math.radians(speed_deg_s),
                "effort": 0.0,
                "kp": float(self.gripper_kp_spin.value()) if hasattr(self, "gripper_kp_spin") else 0.0,
                "kd": float(self.gripper_kd_spin.value()) if hasattr(self, "gripper_kd_spin") else 0.0,
                **monitor_params,
            }
        )

    def _start_putback2_gripper_open(self, *, partial: bool):
        self._refresh_shared_gripper_defaults()
        if partial:
            target_deg = (
                float(self.putback2_gripper_partial_open_spin.value())
                if hasattr(self, "putback2_gripper_partial_open_spin")
                else float(self.DEFAULT_PUTBACK2_GRIPPER_PARTIAL_OPEN_DEG)
            )
            speed_deg_s = (
                float(self.putback2_gripper_partial_open_speed_spin.value())
                if hasattr(self, "putback2_gripper_partial_open_speed_spin")
                else float(self.DEFAULT_PUTBACK2_GRIPPER_PARTIAL_OPEN_SPEED_DEG_S)
            )
            waiting_text = f"等待夹爪开一点到 {target_deg:.1f}°"
        else:
            target_deg = (
                float(self.putback2_gripper_full_open_spin.value())
                if hasattr(self, "putback2_gripper_full_open_spin")
                else float(self.DEFAULT_PUTBACK2_GRIPPER_FULL_OPEN_DEG)
            )
            speed_deg_s = (
                float(self.putback2_gripper_full_open_speed_spin.value())
                if hasattr(self, "putback2_gripper_full_open_speed_spin")
                else float(self.DEFAULT_PUTBACK2_GRIPPER_FULL_OPEN_SPEED_DEG_S)
            )
            waiting_text = f"等待夹爪全部张开到 {target_deg:.1f}°"
        self._start_putback2_gripper_move(
            target_deg=target_deg,
            speed_deg_s=speed_deg_s,
            waiting_kind="putback2_gripper_open",
            waiting_text=waiting_text,
            effort=0.0,
        )

    def _start_putback2_gripper_move(
        self,
        *,
        target_deg: float,
        speed_deg_s: float,
        waiting_kind: str,
        waiting_text: str,
        effort: float,
    ):
        target_deg = float(target_deg)
        speed_deg_s = max(0.1, float(speed_deg_s))
        effort = float(effort)
        self._flow_waiting_motion = True
        self._flow_waiting_kind = str(waiting_kind)
        self._flow_rollback_waiting = False
        self._flow_pending_pose = None
        self._manual_gripper_open_target_deg = target_deg
        self._gripper_close_target_deg = target_deg
        self._gripper_close_effort_nm = effort
        self._gripper_close_started_at_s = time.monotonic()
        self._gripper_close_last_cmd_s = 0.0
        self._gripper_close_last_cmd_deg = None
        self._gripper_close_last_angle_deg = None
        self._gripper_close_stable_since_s = None
        self._update_flow_button_state()
        self.flow_status_label.setText(waiting_text)
        monitor_params = self._gripper_monitor_params()
        self.gripper_close_monitor_requested.emit(
            {
                "target_angle": math.radians(target_deg),
                "close_speed": math.radians(speed_deg_s),
                "effort": effort,
                "kp": (
                    float(self.gripper_kp_spin.value())
                    if effort > 0.0 and hasattr(self, "gripper_kp_spin")
                    else 0.0
                ),
                "kd": (
                    float(self.gripper_kd_spin.value())
                    if effort > 0.0 and hasattr(self, "gripper_kd_spin")
                    else 0.0
                ),
                **monitor_params,
                "start_effort": (
                    float(self.gripper_start_effort_spin.value())
                    if effort > 0.0 and hasattr(self, "gripper_start_effort_spin")
                    else None
                ),
                "start_boost_s": (
                    float(self.gripper_start_boost_spin.value())
                    if effort > 0.0 and hasattr(self, "gripper_start_boost_spin")
                    else 0.0
                ),
            }
        )

    def update_joint_feedback(self, joint_states):
        try:
            positions = joint_states.to_list(include_gripper=True)
        except Exception:
            return
        if len(positions) >= 7:
            self._current_gripper_angle_deg = math.degrees(float(positions[6]))

    def notify_gripper_close_done(self, result):
        if self._flow_waiting_kind not in {
            "gripper_close",
            "manual_gripper_close",
            "manual_gripper_open",
            "takeout_gripper_open",
            "putback_gripper_open",
            "tail_gripper_slight_open",
            "tail_home_gripper_close",
            "putback2_gripper_open",
            "putback2_combo_close",
            "putback2_combo_full_open",
        }:
            return
        if not self._flow_waiting_motion:
            return
        manual_action = self._flow_waiting_kind in {"manual_gripper_close", "manual_gripper_open"}
        if not manual_action and not self._flow_active:
            return
        result = dict(result or {})
        reason = str(result.get("reason", "done"))
        final_deg = math.degrees(float(result.get("final_angle", 0.0)))
        command_deg = math.degrees(float(result.get("command_angle", result.get("final_angle", 0.0))))
        self._current_gripper_angle_deg = final_deg
        self._clear_gripper_close_monitor()
        self._clear_manual_gripper_open()
        if self._flow_waiting_kind in {
            "manual_gripper_open",
            "takeout_gripper_open",
            "putback_gripper_open",
            "tail_gripper_slight_open",
            "putback2_gripper_open",
            "putback2_combo_full_open",
        }:
            if reason == "target_reached":
                message = f"夹爪已缓慢张开到目标附近：{final_deg:.1f}°"
            elif reason == "stalled":
                message = f"夹爪张开检测到阻力/卡滞趋势，已锁定保持角度：{command_deg:.1f}°"
            elif reason == "timeout":
                message = f"夹爪张开监测超时，已锁定保持角度：{command_deg:.1f}°"
            else:
                message = f"夹爪缓慢张开完成，保持角度：{command_deg:.1f}°"
        elif reason == "target_reached":
            message = f"夹爪已关闭到目标附近：{final_deg:.1f}°"
        elif reason == "stalled":
            message = f"夹爪检测到夹持/卡滞趋势，已锁定保持角度：{command_deg:.1f}°"
        elif reason == "timeout":
            message = f"夹爪监测闭合超时，已锁定保持角度：{command_deg:.1f}°"
        else:
            message = f"夹爪监测闭合完成，保持角度：{command_deg:.1f}°"
        if manual_action:
            self._flow_waiting_motion = False
            self._flow_waiting_kind = None
            self._flow_rollback_waiting = False
            self._flow_pending_pose = None
            self.flow_status_label.setText(message)
            self._update_flow_button_state()
        elif self._flow_waiting_kind == "putback2_combo_close":
            self._execute_putback2_combo_y_move()
        else:
            self._advance_book_grasp_flow(message)

    def _clear_gripper_close_monitor(self):
        self._gripper_close_target_deg = None
        self._gripper_close_last_cmd_deg = None
        self._gripper_close_started_at_s = 0.0
        self._gripper_close_last_cmd_s = 0.0
        self._gripper_close_last_angle_deg = None
        self._gripper_close_stable_since_s = None
        self._gripper_close_effort_nm = 0.0

    def _ensure_rod_connected(self):
        if self._rod_connected:
            return
        port = self.rod_port_edit.text().strip() or self.DEFAULT_ROD_PORT
        self.rod_connect_requested.emit(
            port,
            self.DEFAULT_ROD_BAUD,
            self.DEFAULT_ROD_TIMEOUT_S,
        )

    def _write_rod_and_wait(self, angle_deg: float):
        self._ensure_rod_connected()
        self._rod_target_angle_deg = float(angle_deg)
        self._flow_waiting_motion = True
        self._flow_waiting_kind = "rod"
        self._flow_pending_pose = None
        self._update_flow_button_state()
        self.flow_status_label.setText("等待杆电机动作命令发送完成")
        self.rod_write_requested.emit(
            float(angle_deg),
            self.rod_speed_spin.value(),
            self.rod_acc_spin.value(),
            self.DEFAULT_ROD_TORQUE,
        )

    def _send_rod_zero_nonblocking(self):
        if not hasattr(self, "rod_speed_spin") or not hasattr(self, "rod_acc_spin"):
            return
        self._ensure_rod_connected()
        self._rod_target_angle_deg = 0.0
        self.rod_write_requested.emit(
            0.0,
            self.rod_speed_spin.value(),
            self.rod_acc_spin.value(),
            self.DEFAULT_ROD_TORQUE,
        )
        self.log_message.emit(f"{self._flow_log_prefix()}杆电机归零命令已发送")

    def _advance_book_grasp_flow(self, message: str):
        if not self._flow_active:
            return
        self._flow_waiting_motion = False
        self._flow_waiting_kind = None
        self._flow_step_index += 1
        if self._flow_step_index >= len(self._book_grasp_steps()):
            self._flow_auto_run = False
            self.flow_status_label.setText(f"流程完成：{message}")
            self.log_message.emit(f"{self._flow_log_prefix()}完成")
        else:
            next_step = self._book_grasp_steps()[self._flow_step_index]
            self.flow_status_label.setText(
                f"{message}。请确认后执行 [{self._flow_step_index + 1}/{len(self._book_grasp_steps())}] {next_step}"
            )
        self._update_flow_button_state()
        self._schedule_auto_book_grasp_step()

    def _begin_flow_wait(self, kind: str):
        self._flow_waiting_motion = True
        self._flow_waiting_kind = kind
        self._flow_pending_pose = None
        self._update_flow_button_state()

    def _promote_pending_flow_pose(self):
        if self._flow_pending_pose is None:
            return
        self._flow_last_pose = [float(value) for value in self._flow_pending_pose]
        self._flow_pending_pose = None

    def _finish_flow_pose_rollback(self) -> bool:
        if not (self._flow_active and self._flow_rollback_waiting):
            return False
        self._flow_waiting_motion = False
        self._flow_waiting_kind = None
        self._flow_rollback_waiting = False
        self._flow_rollback_entry = None
        self._flow_pending_pose = None
        next_step = self._book_grasp_steps()[self._flow_step_index]
        self.flow_status_label.setText(
            f"已回到上一步位置。请确认后执行 [{self._flow_step_index + 1}/{len(self._book_grasp_steps())}] {next_step}"
        )
        self._update_flow_button_state()
        return True

    def notify_move_l_done(self):
        if self._flow_waiting_kind not in {
            None,
            "move_l",
            "putback2_combo_y_move",
            "putback2_combo_x_move",
        }:
            return
        if self._flow_active and self._flow_waiting_motion:
            self._promote_pending_flow_pose()
        if (
            self._workflow_mode == "putback2"
            and self._flow_active
            and self._flow_waiting_motion
            and self._flow_waiting_kind == "putback2_combo_y_move"
        ):
            self._begin_putback2_combo_wait()
            return
        if (
            self._workflow_mode == "putback2"
            and self._flow_active
            and self._flow_waiting_motion
            and self._flow_waiting_kind == "putback2_combo_x_move"
        ):
            self._finish_putback2_combo_with_full_open()
            return
        if self._finish_flow_pose_rollback():
            return
        if self._flow_active and self._flow_waiting_motion:
            self._advance_book_grasp_flow("机械臂运动完成")

    def notify_move_j_done(self):
        if self._flow_waiting_kind == "tail_final_home":
            self._flow_pending_pose = None
            if self._flow_active and self._flow_waiting_motion:
                self._advance_book_grasp_flow("机械臂已回到目标回位位置")
            return
        if self._flow_waiting_kind in {"debug_move", "zero_move"}:
            was_zero_move = self._flow_waiting_kind == "zero_move"
            self._flow_waiting_motion = False
            self._flow_waiting_kind = None
            self._flow_pending_pose = None
            suffix = "，可继续当前流程" if self._flow_active else ""
            done_text = "已归零" if was_zero_move else "已运动到图书调试位"
            self.flow_status_label.setText(f"{done_text}{suffix}")
            self._update_flow_button_state()
            return
        if self._flow_waiting_kind != "move_j":
            return
        self._flow_pending_pose = None
        if self._flow_active and self._flow_waiting_motion:
            self._advance_book_grasp_flow("机械臂 MoveJ 运动完成")

    def notify_end_pose_done(self):
        if self._flow_waiting_kind != "end_pose":
            return
        if self._finish_flow_pose_rollback():
            return
        if self._flow_active and self._flow_waiting_motion:
            self._promote_pending_flow_pose()
        if self._flow_active and self._flow_waiting_motion:
            self._advance_book_grasp_flow("机械臂 IK + MoveJ 运动完成")

    def notify_lift_move_done(self):
        if self._flow_waiting_kind != "lift":
            return
        if self._flow_active and self._flow_waiting_motion:
            self._advance_book_grasp_flow("升降台已移动到还书位")

    def notify_rod_write_done(self):
        if (
            self._is_takeout_based_workflow()
            and self._flow_active
            and self._flow_waiting_motion
            and self._flow_waiting_kind == "rod"
        ):
            self.flow_status_label.setText("杆电机动作命令已发送，等待到位")

    def notify_rod_angle_updated(self, angle_deg: float):
        self._rod_current_angle_deg = float(angle_deg)
        if (
            self._is_takeout_based_workflow()
            and self._flow_active
            and self._flow_waiting_motion
            and self._flow_waiting_kind == "rod"
            and self._rod_target_angle_deg is not None
            and abs(self._rod_current_angle_deg - self._rod_target_angle_deg)
            <= self._rod_wait_tolerance_deg
        ):
            self._advance_book_grasp_flow(
                f"杆电机已到位：{self._rod_current_angle_deg:.1f}°"
            )

    def notify_flow_error(self, message: str):
        self._flow_auto_run = False
        rollback_failed = self._flow_rollback_waiting
        rollback_entry = self._flow_rollback_entry
        if self._flow_waiting_kind in {
            "debug_move",
            "zero_move",
            "manual_gripper_close",
            "manual_gripper_open",
            "lift",
            "tail_home_gripper_close",
            "tail_final_home",
        }:
            waiting_kind = self._flow_waiting_kind
            self._flow_waiting_motion = False
            self._flow_waiting_kind = None
            self._flow_rollback_waiting = False
            self._flow_rollback_entry = None
            self._flow_pending_pose = None
            self._clear_gripper_close_monitor()
            self._clear_manual_gripper_open()
            if waiting_kind == "zero_move":
                label = "归零"
            elif waiting_kind == "debug_move":
                label = "运动到图书调试位"
            elif waiting_kind == "manual_gripper_close":
                label = "夹爪关闭"
            elif waiting_kind == "tail_home_gripper_close":
                label = "步骤11夹爪直接闭合"
            elif waiting_kind == "tail_final_home":
                label = "步骤14回到目标位置"
            elif waiting_kind == "lift":
                label = "升降台移动到还书位"
            else:
                label = "夹爪打开"
            self.flow_status_label.setText(f"{label}失败：{message}")
            self._update_flow_button_state()
            return
        if not self._flow_active:
            return
        self._flow_waiting_motion = False
        self._flow_waiting_kind = None
        self._flow_rollback_waiting = False
        self._flow_rollback_entry = None
        self._flow_pending_pose = None
        self._rod_target_angle_deg = None
        self._clear_gripper_close_monitor()
        self._clear_manual_gripper_open()
        if rollback_failed:
            if rollback_entry is not None:
                self._flow_pose_history.append(rollback_entry)
            self.flow_status_label.setText(f"回到上一步位置失败，可调整后重试：{message}")
        else:
            current = min(self._flow_step_index + 1, len(self._book_grasp_steps()))
            self.flow_status_label.setText(f"第 {current} 步出错，可调整后重试：{message}")
        self._update_flow_button_state()

    def set_rod_connected(self, connected: bool):
        self._rod_connected = bool(connected)
        self._update_flow_button_state()

    def _update_flow_button_state(self):
        if not hasattr(self, "flow_next_btn"):
            return
        completed = self._flow_active and self._flow_step_index >= len(self._book_grasp_steps())
        can_step = self._arm_enabled and not self._flow_waiting_motion
        if hasattr(self, "flow_debug_move_btn"):
            self.flow_debug_move_btn.setEnabled(
                self._arm_enabled
                and not self._flow_waiting_motion
                and hasattr(self, "debug_joint_spins")
            )
        if hasattr(self, "flow_zero_move_btn"):
            self.flow_zero_move_btn.setEnabled(
                self._arm_enabled and not self._flow_waiting_motion
            )
        if hasattr(self, "flow_gripper_close_btn"):
            self.flow_gripper_close_btn.setEnabled(
                self._arm_enabled and not self._flow_waiting_motion
            )
        if hasattr(self, "flow_gripper_open_btn"):
            self.flow_gripper_open_btn.setEnabled(
                self._arm_enabled and not self._flow_waiting_motion
            )
        if hasattr(self, "flow_auto_run_btn"):
            auto_visible = self._workflow_mode == "takeout"
            self.flow_auto_run_btn.setVisible(auto_visible)
            self.flow_auto_run_btn.setEnabled(
                auto_visible and can_step and not completed and not self._flow_auto_run
            )
            self.flow_auto_run_btn.setText(
                "自动执行中" if self._flow_auto_run else "完整执行"
            )
        if hasattr(self, "flow_emergency_stop_btn"):
            estop_visible = self._workflow_mode == "takeout"
            self.flow_emergency_stop_btn.setVisible(estop_visible)
            self.flow_emergency_stop_btn.setEnabled(estop_visible)
        self.flow_next_btn.setEnabled(can_step and not self._flow_auto_run)
        if hasattr(self, "flow_back_btn"):
            can_back = (
                not self._flow_waiting_motion
                and self._flow_active
                and self._flow_step_index > 0
                and not completed
                and not self._flow_auto_run
            )
            self.flow_back_btn.setEnabled(can_back)
        if hasattr(self, "flow_back_pose_btn"):
            self.flow_back_pose_btn.setEnabled(
                self._arm_enabled
                and not self._flow_waiting_motion
                and bool(self._flow_pose_history)
                and not self._flow_auto_run
            )
        for button in getattr(self, "_takeout_step_buttons", []):
            button.setEnabled(can_step and not completed and not self._flow_auto_run)
        if completed:
            self.flow_next_btn.setText("流程完成，点击重置")
        elif not self._flow_active:
            self.flow_next_btn.setText(tr("pc.workflow_next"))
        else:
            self.flow_next_btn.setText(f"确认执行第 {self._flow_step_index + 1} 步")

    def _clear_selection(self, keep_result: bool = False):
        if not keep_result:
            self._selected_display_index = None
            self._selected_display_point_m = None
            self._selected_raw_index = None
            self._selected_camera_point_m = None
            self._selected_robot_target_raw_m = None
            self._target_robot_point_m = None
            self._tail_putback_bottom_point_m = None
            self._tail_putback_step4_target_point_m = None
            self._segmentation_result = None
            self.pixel_value.setText("--")
            self.move_target_point_cm_value.setText("--")
            self.target_point_cm_value.setText("--")
        if self._selected_actor is not None and self._plotter is not None:
            try:
                self._plotter.remove_actor(self._selected_actor)
                self._plotter.render()
            except Exception:
                pass
        self._selected_actor = None
        self._update_move_button_state()

    def _update_move_button_state(self):
        can_move = self._arm_enabled and self._target_robot_point_m is not None
        if hasattr(self, "move_btn"):
            self.move_btn.setEnabled(can_move)
        self._update_flow_button_state()

    def _set_error(self, message: str):
        self._flow_auto_run = False
        self.status_label.setText(message)
        self.error_occurred.emit(message)

    def apply_theme(self):
        if HAS_PYVISTA and self._plotter is not None:
            try:
                sc = SCENE_COLORS[ThemeManager.instance().theme]
                self._plotter.set_background(sc["bg_bottom"], top=sc["bg_top"])
                self._plotter.render()
            except Exception:
                pass
        elif HAS_PYVISTA and hasattr(self, "_plotter_placeholder") and self._plotter_placeholder is not None:
            try:
                sc = SCENE_COLORS[ThemeManager.instance().theme]
                self._plotter_placeholder.setStyleSheet(
                    f"color: {sc['subtext']}; font-size: 12px; padding: 12px;"
                )
            except Exception:
                pass

    def retranslate_ui(self):
        self.capture_group.setTitle(tr("pc.capture_group"))
        self.pick_mode_btn.setText(tr("pc.pick_mode"))
        self.capture_btn.setText(tr("pc.capture"))
        self.book_group.setTitle(tr("pc.book_group"))
        if hasattr(self, "image_recognition_group"):
            self.image_recognition_group.setTitle(tr("pc.image_recognition_group"))
        if hasattr(self, "image_detect_btn"):
            self.image_detect_btn.setText(tr("pc.image_detect"))
        if hasattr(self, "image_status_label") and self._segmentation_result is None:
            self.image_status_label.setText(tr("pc.image_ready"))
        if hasattr(self, "grasp_group"):
            self.grasp_group.setTitle(self._workflow_group_title())
        if hasattr(self, "template_combo"):
            for idx, (label, path) in enumerate(BOOK_TEMPLATE_OPTIONS):
                if idx < self.template_combo.count():
                    self.template_combo.setItemText(idx, label)
                    self.template_combo.setItemData(idx, str(path))
        if hasattr(self, "flow_detect_btn"):
            self.flow_detect_btn.setText(tr("pc.workflow_detect"))
        if hasattr(self, "flow_steps_btn"):
            self.flow_steps_btn.setText(tr("pc.workflow_steps"))
        if hasattr(self, "flow_auto_run_btn"):
            self.flow_auto_run_btn.setText(
                "自动执行中" if self._flow_auto_run else "完整执行"
            )
        if hasattr(self, "flow_emergency_stop_btn"):
            self.flow_emergency_stop_btn.setText("急停")
        if hasattr(self, "flow_debug_move_btn"):
            self.flow_debug_move_btn.setText("运动到图书调试位")
        if hasattr(self, "flow_zero_move_btn"):
            self.flow_zero_move_btn.setText("归零")
        if hasattr(self, "flow_gripper_close_btn"):
            self.flow_gripper_close_btn.setText("夹爪关闭")
        if hasattr(self, "flow_gripper_open_btn"):
            self.flow_gripper_open_btn.setText("夹爪打开")
        if hasattr(self, "flow_reset_btn"):
            self.flow_reset_btn.setText(tr("pc.workflow_reset"))
        if hasattr(self, "flow_back_btn"):
            self.flow_back_btn.setText(tr("pc.workflow_back_step"))
        if hasattr(self, "flow_back_pose_btn"):
            self.flow_back_pose_btn.setText(tr("pc.workflow_back_pose"))
        if hasattr(self, "flow_defaults_btn"):
            self._update_workflow_default_edit_buttons()
        if hasattr(self, "flow_next_btn"):
            self._update_flow_button_state()
        for button in getattr(self, "_takeout_step_buttons", []):
            button.setText("执行本步")
        if hasattr(self, "flow_status_label") and not self._flow_active:
            self.flow_status_label.setText(tr("pc.workflow_pending"))
        if hasattr(self, "flow_target_label") and self._selected_robot_target_raw_m is None:
            self.flow_target_label.setText(tr("pc.workflow_target"))
        if self._frame is None and hasattr(self, "book_status_label"):
            self.book_status_label.setText(tr("pc.book_ready"))

        self.result_group.setTitle(tr("pc.result_group"))
        self.pixel_label.setText(tr("pc.pixel"))
        self.robot_point_cm_label.setText(tr("pc.move_target_point_cm"))
        self.target_point_cm_label.setText(tr("pc.target_point_cm"))
        self.move_group.setTitle(tr("pc.move_group"))
        for label, key in zip(self.rpy_labels, ("pc.rx", "pc.ry", "pc.rz")):
            label.setText(tr(key))
        self.duration_label.setText(tr("pc.duration"))
        self.read_rpy_btn.setText(tr("pc.read_rpy"))
        self.move_btn.setText(tr("pc.confirm_move"))
        if hasattr(self, "no_plot_label"):
            self.no_plot_label.setText(tr("pc.no_pyvista"))
        if hasattr(self, "_plotter_placeholder") and self._plotter_placeholder is not None:
            self._plotter_placeholder.setText(tr("pc.viewer_loading"))

    def _make_target_pose_with_offset(
        self,
        x_offset_cm: float,
        y_offset_cm: Optional[float] = None,
        z_offset_cm: Optional[float] = None,
        rpy_deg: Optional[Sequence[float]] = None,
    ) -> list[float]:
        target = np.asarray(self._target_robot_point_m, dtype=float).reshape(3).copy()
        target[0] += float(x_offset_cm) / M_TO_CM
        if y_offset_cm is None:
            y_offset_cm = 0.0
        if z_offset_cm is None:
            z_offset_cm = 0.0
        target[1] += float(y_offset_cm) / M_TO_CM
        target[2] += float(z_offset_cm) / M_TO_CM
        if rpy_deg is None:
            rpy_deg = [spin.value() for spin in self.rpy_spins]
        return [
            float(target[0]),
            float(target[1]),
            float(target[2]),
            math.radians(float(rpy_deg[0])),
            math.radians(float(rpy_deg[1])),
            math.radians(float(rpy_deg[2])),
        ]

    def _make_pose_from_current_end_pose(
        self,
        x_offset_cm: float = 0.0,
        y_offset_cm: float = 0.0,
        z_offset_cm: float = 0.0,
        rpy_deg: Optional[Sequence[float]] = None,
        rpy_offset_deg: Optional[Sequence[float]] = None,
        local_axes: bool = False,
        prefer_last_flow_pose: bool = False,
    ) -> Optional[list[float]]:
        pose = (
            [float(value) for value in self._flow_last_pose]
            if prefer_last_flow_pose and self._flow_last_pose is not None
            else self._pose_from_current_end_pose()
        )
        if pose is None:
            return None
        offset_m = np.array(
            [float(x_offset_cm), float(y_offset_cm), float(z_offset_cm)],
            dtype=float,
        ) / M_TO_CM
        if local_axes:
            offset_m = _rpy_to_matrix(pose[3], pose[4], pose[5]) @ offset_m
        pose[0] += float(offset_m[0])
        pose[1] += float(offset_m[1])
        pose[2] += float(offset_m[2])
        if rpy_deg is not None and len(rpy_deg) >= 3:
            pose[3] = math.radians(float(rpy_deg[0]))
            pose[4] = math.radians(float(rpy_deg[1]))
            pose[5] = math.radians(float(rpy_deg[2]))
        elif rpy_offset_deg is not None and len(rpy_offset_deg) >= 3:
            pose[3] += math.radians(float(rpy_offset_deg[0]))
            pose[4] += math.radians(float(rpy_offset_deg[1]))
            pose[5] += math.radians(float(rpy_offset_deg[2]))
        return pose

    def cleanup(self):
        if HAS_PYVISTA and self._plotter is not None and self._filter_installed:
            try:
                self._plotter.interactor.removeEventFilter(self)
            except Exception:
                pass
            self._filter_installed = False
        if self._capture_worker is not None and self._capture_worker.isRunning():
            self._capture_worker.requestInterruption()
            self._capture_worker.wait(1500)
        if self._detect_worker is not None and self._detect_worker.isRunning():
            self._detect_worker.requestInterruption()
            self._detect_worker.wait(1500)
        if self._segment_worker is not None and self._segment_worker.isRunning():
            self._segment_worker.requestInterruption()
            self._segment_worker.wait(1500)


__all__ = [
    "RealSensePointPanel",
    "camera_point_to_robot_target",
]
