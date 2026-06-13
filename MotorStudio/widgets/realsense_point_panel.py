"""RealSense point-cloud picking page for MotorStudio."""

from __future__ import annotations

import logging
import math
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from PyQt6.QtCore import QEvent, QSignalBlocker, QThread, Qt, pyqtSignal
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
from MotorStudio.utils.style import SCENE_COLORS
from MotorStudio.utils.theme_manager import ThemeManager
from MotorStudio.utils.tcp_offset_store import get_tcp_offset_path

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


def _format_vec_m(values: Sequence[float]) -> str:
    vals = np.asarray(values, dtype=float).reshape(3)
    return f"X={vals[0]:.4f} m, Y={vals[1]:.4f} m, Z={vals[2]:.4f} m"


def _format_vec_cm(values: Sequence[float]) -> str:
    vals = np.asarray(values, dtype=float).reshape(3) * M_TO_CM
    return f"X={vals[0]:.2f} cm, Y={vals[1]:.2f} cm, Z={vals[2]:.2f} cm"


def _book_pick_point_from_polygon(
    polygon: np.ndarray,
    workflow_mode: str = "takeout",
) -> tuple[int, int]:
    points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(points) < 4:
        raise ValueError("书脊识别结果缺少四边形角点。")
    tl, tr, br, bl = points[:4]
    if str(workflow_mode).lower() in {"putback", "return", "put_back"}:
        point = tr * 0.25 + br * 0.75
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
    DEFAULT_ROD_GRASP_DEG = 115.0
    DEFAULT_PREGRASP_OFFSET_CM = (10.0, 0.0, 0.0)
    DEFAULT_GRIPPER_OPEN_DEG = 0.0
    DEFAULT_GRIPPER_CLOSE_DEG = 108.5
    DEFAULT_GRIPPER_HOLD_EFFORT_NM = 0.12
    DEFAULT_GRIPPER_KP = 18.0
    DEFAULT_GRIPPER_KD = 2.0
    DEFAULT_GRIPPER_CLOSE_SPEED_DEG_S = 16.7
    DEFAULT_GRASP_RPY_DEG = (75.0, 0.0, 90.0)
    DEFAULT_FLOW_MOVEL_DURATION_S = 2.0
    GRIPPER_CLOSE_TOLERANCE_DEG = 2.0
    GRIPPER_CLOSE_STALL_TOLERANCE_DEG = 0.3
    GRIPPER_CLOSE_STALL_S = 0.8
    GRIPPER_CLOSE_MIN_MONITOR_S = 0.8
    GRIPPER_CLOSE_TIMEOUT_S = 8.0
    GRIPPER_CLOSE_STEP_DEG = 3.0
    GRIPPER_CLOSE_STEP_INTERVAL_S = 0.18
    DEFAULT_PUTBACK_TARGET_RPY_DEG = (90.0, 0.0, 60.0)
    DEFAULT_PUTBACK_INSERT_RPY_DEG = (90.0, 0.0, 90.0)
    DEFAULT_PUTBACK_PREPUSH_X_OFFSET_CM = 10.0
    DEFAULT_PUTBACK_PUSH_Y_OFFSET_CM = 3.0
    DEFAULT_PUTBACK_LEAVE_PUSH_Y_OFFSET_CM = 10.0
    DEFAULT_PUTBACK_INSERT_PREPOSE_X_OFFSET_CM = 10.0
    DEFAULT_PUTBACK_INSERT_PREPOSE_Y_OFFSET_CM = 1.0
    DEFAULT_PUTBACK_INSERT_X_OFFSET_CM = -3.0
    DEFAULT_PUTBACK_INSERT_Y_OFFSET_CM = 1.0
    DEFAULT_PUTBACK_LEAVE_INSERT_X_OFFSET_CM = 10.0
    DEFAULT_PUTBACK_GRIPPER_OPEN_DEG = 30.0
    DEFAULT_DEBUG_JOINTS_DEG = (0.0, 35.0, -45.0, 0.0, 0.0, 0.0)
    DEFAULT_TURN_STAGE1_JOINTS_DEG = (-90.0, 35.0, -40.0, 0.0, 0.0, 0.0)
    DEFAULT_TARGET_RELATIVE_OFFSET_CM = (0.0, 0.0, 0.0)
    DEFAULT_TARGET_RELATIVE_RPY_DEG = (90.0, 0.0, 90.0)
    DEFAULT_POST_GRIPPER_MOVEJ_OFFSET_CM = (0.0, 0.0, 0.0)
    DEFAULT_POST_GRIPPER_MOVEJ_RPY_OFFSET_DEG = (0.0, 0.0, 0.0)
    DEFAULT_POST_GRIPPER_MOVEJ_DURATION_S = 2.0
    DEFAULT_FINAL_JOINTS_DEG = (-119.64, 67.09, 67.22, 2.23, 51.75, -14.65)
    DEFAULT_TURN_DURATION_S = 6.0
    DEFAULT_DEBUG_MOVE_DURATION_S = 3.0
    HEADER_MOVE_DURATION_S = 2.0
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
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None, workflow_mode: str = "takeout"):
        super().__init__(parent)
        self._workflow_mode = "putback" if str(workflow_mode).lower() in {"putback", "return", "put_back"} else "takeout"
        self._capture_worker: Optional[RealSenseCaptureWorker] = None
        self._detect_worker: Optional[BookSpineDetectWorker] = None
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
        self._book_spine_pick: Optional[object] = None
        self._tcp_offset = np.zeros(6, dtype=float)
        self._current_end_pose = None
        self._rpy_initialized = False
        self._arm_enabled = False
        self._rod_connected = False
        self._flow_active = False
        self._flow_step_index = 0
        self._flow_waiting_motion = False
        self._flow_waiting_kind: Optional[str] = None
        self._flow_rollback_waiting = False
        self._flow_pose_history: list[tuple[int, str, list[float]]] = []
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
        self._workflow_default_edit_enabled = False
        self._workflow_default_controls: dict[str, QWidget] = {}
        self._workflow_default_edit_snapshot: dict[str, object] = {}
        self._loading_workflow_defaults = False
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

    def _init_ui(self):
        self._viewer_widget = self._create_viewer_widget()
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

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
        self.flow_debug_move_btn = QPushButton("运动到调试位")
        self.flow_debug_move_btn.clicked.connect(self._move_to_debug_pose_from_header)
        self._compact_flow_button(self.flow_debug_move_btn)
        management_row.addWidget(self.flow_debug_move_btn)
        self.flow_zero_move_btn = QPushButton("归零")
        self.flow_zero_move_btn.clicked.connect(self._move_to_zero_pose_from_header)
        self._compact_flow_button(self.flow_zero_move_btn)
        management_row.addWidget(self.flow_zero_move_btn)
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
                "流程完成，点击重置",
            ],
        )
        btn_row.addWidget(self.flow_next_btn)
        self.flow_back_btn = QPushButton(tr("pc.workflow_back"))
        self.flow_back_btn.clicked.connect(self._rollback_to_previous_flow_pose)
        self._compact_flow_button(self.flow_back_btn)
        btn_row.addWidget(self.flow_back_btn)
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
        self.grasp_group = QGroupBox(
            tr("pc.book_putback_group")
            if self._workflow_mode == "putback"
            else tr("pc.book_takeout_group")
        )
        layout = QGridLayout(self.grasp_group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        row = 0
        if self._workflow_mode == "takeout":
            step1, step1_layout = self._create_step_group("步骤1：采集点云")
            layout.addWidget(step1, row, 0, 1, 5)
            step1_layout.addWidget(QLabel("点云采集由顶部流程按钮触发"), 0, 0, 1, 5)

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

            row += 1
            step3, step3_layout = self._create_step_group("步骤3：解算目标点")
            layout.addWidget(step3, row, 0, 1, 5)
            self.book_status_label = QLabel(tr("pc.book_ready"))
            self.book_status_label.setWordWrap(True)
            self._stabilize_flow_label(self.book_status_label)
            step3_layout.addWidget(self.book_status_label, 0, 0, 1, 5)
            self.flow_target_label = QLabel(tr("pc.workflow_target"))
            self.flow_target_label.setWordWrap(True)
            self._stabilize_flow_label(self.flow_target_label)
            step3_layout.addWidget(self.flow_target_label, 1, 0, 1, 5)
            step3_layout.addWidget(QLabel("目标点补偿XYZ:"), 2, 0)
            comp_row = QHBoxLayout()
            comp_row.setContentsMargins(0, 0, 0, 0)
            comp_row.setSpacing(4)
            self.target_comp_xyz_spins = []
            for value in self.DEFAULT_TARGET_COMPENSATION_CM:
                spin = self._make_float_spin(-50.0, 50.0, value, 0.5, " cm")
                self.target_comp_xyz_spins.append(spin)
                comp_row.addWidget(spin)
            comp_row.addStretch()
            step3_layout.addLayout(comp_row, 2, 1, 1, 4)
            self._bind_workflow_target_refresh(self.target_comp_xyz_spins)

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
                self._stabilize_flow_label(self.flow_target_label)
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
        for value in self.DEFAULT_GRASP_RPY_DEG:
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

        step5, step5_layout = self._create_step_group("步骤5：到达抓取位姿")
        layout.addWidget(step5, base_row + 1, 0, 1, 5)
        step5_layout.addWidget(QLabel("执行后自动记录抓取位姿"), 0, 0, 1, 5)

        step6, step6_layout = self._create_step_group("步骤6：杆电机到夹取位")
        layout.addWidget(step6, base_row + 2, 0, 1, 5)
        step6_layout.addWidget(QLabel("串口:"), 0, 0)
        self.rod_port_edit = QLineEdit(self.DEFAULT_ROD_PORT)
        self.rod_port_edit.setFixedWidth(150)
        step6_layout.addWidget(self.rod_port_edit, 0, 1)
        step6_layout.addWidget(QLabel("杆夹取位:"), 0, 2)
        self.rod_grasp_spin = self._make_float_spin(-180.0, 180.0, self.DEFAULT_ROD_GRASP_DEG, 1.0, "°")
        step6_layout.addWidget(self.rod_grasp_spin, 0, 3)
        step6_layout.addWidget(QLabel("速度:"), 1, 0)
        self.rod_speed_spin = self._make_int_spin(1, 10000, self.DEFAULT_ROD_SPEED, 100)
        step6_layout.addWidget(self.rod_speed_spin, 1, 1)
        step6_layout.addWidget(QLabel("加速度:"), 1, 2)
        self.rod_acc_spin = self._make_int_spin(1, 10000, self.DEFAULT_ROD_ACC, 10)
        step6_layout.addWidget(self.rod_acc_spin, 1, 3)
        step6_layout.setColumnStretch(4, 1)

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

        step8, step8_layout = self._create_step_group("步骤8：夹爪带监测持续关闭")
        layout.addWidget(step8, base_row + 4, 0, 1, 5)
        step8_layout.addWidget(QLabel("开/关/力矩:"), 0, 0)
        gripper_row = QHBoxLayout()
        gripper_row.setContentsMargins(0, 0, 0, 0)
        gripper_row.setSpacing(4)
        self.gripper_open_spin = self._make_float_spin(-30.0, 140.0, self.DEFAULT_GRIPPER_OPEN_DEG, 1.0, "°")
        self.gripper_close_spin = self._make_float_spin(-30.0, 140.0, self.DEFAULT_GRIPPER_CLOSE_DEG, 1.0, "°")
        self.gripper_effort_spin = self._make_float_spin(0.0, 5.0, self.DEFAULT_GRIPPER_HOLD_EFFORT_NM, 0.01, " Nm")
        self.gripper_close_speed_spin = self._make_float_spin(
            1.0,
            60.0,
            self.DEFAULT_GRIPPER_CLOSE_SPEED_DEG_S,
            0.5,
            "°/s",
        )
        gripper_row.addWidget(self.gripper_open_spin)
        gripper_row.addWidget(self.gripper_close_spin)
        gripper_row.addWidget(self.gripper_effort_spin)
        gripper_row.addStretch()
        step8_layout.addLayout(gripper_row, 0, 1, 1, 3)
        step8_layout.addWidget(QLabel("保持 Kp/Kd:"), 1, 0)
        gripper_gain_row = QHBoxLayout()
        gripper_gain_row.setContentsMargins(0, 0, 0, 0)
        gripper_gain_row.setSpacing(4)
        self.gripper_kp_spin = self._make_float_spin(0.0, 200.0, self.DEFAULT_GRIPPER_KP, 1.0, "")
        self.gripper_kd_spin = self._make_float_spin(0.0, 50.0, self.DEFAULT_GRIPPER_KD, 0.5, "")
        gripper_gain_row.addWidget(self.gripper_kp_spin)
        gripper_gain_row.addWidget(self.gripper_kd_spin)
        gripper_gain_row.addStretch()
        step8_layout.addLayout(gripper_gain_row, 1, 1, 1, 2)
        self._add_compact_value_row(
            step8_layout,
            2,
            "闭合速度:",
            self.gripper_close_speed_spin,
        )

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

        step10, step10_layout = self._create_step_group("步骤10：回到调试位")
        layout.addWidget(step10, base_row + 6, 0, 1, 5)
        self.debug_joint_spins = []
        for value in self.DEFAULT_DEBUG_JOINTS_DEG:
            spin = self._make_float_spin(-360.0, 360.0, value, 1.0, "°")
            self.debug_joint_spins.append(spin)
        self._add_indexed_joint_spins(step10_layout, 0, "目标关节:", self.debug_joint_spins)
        self.flow_debug_duration_spin = self._make_float_spin(3.0, 20.0, self.DEFAULT_DEBUG_MOVE_DURATION_S, 0.5, " s")
        self._add_compact_value_row(step10_layout, 2, "MoveJ时间:", self.flow_debug_duration_spin)

        step11, step11_layout = self._create_step_group("步骤11：慢速转身到-90°")
        layout.addWidget(step11, base_row + 7, 0, 1, 5)
        self.turn_stage1_joint_spins = []
        for value in self.DEFAULT_TURN_STAGE1_JOINTS_DEG:
            spin = self._make_float_spin(-360.0, 360.0, value, 1.0, "°")
            self.turn_stage1_joint_spins.append(spin)
        self._add_indexed_joint_spins(step11_layout, 0, "目标关节:", self.turn_stage1_joint_spins)
        self.flow_turn_duration_spin = self._make_float_spin(3.0, 20.0, self.DEFAULT_TURN_DURATION_S, 0.5, " s")
        self._add_compact_value_row(step11_layout, 2, "转身时间:", self.flow_turn_duration_spin)

        step12, step12_layout = self._create_step_group("步骤12：MoveJ到最终放置构型")
        layout.addWidget(step12, base_row + 8, 0, 1, 5)
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

        return base_row + 9

    def _populate_putback_controls(self, layout: QGridLayout, row: int) -> int:
        base_row = row
        step1, step1_layout = self._create_step_group("步骤1：采集点云")
        layout.addWidget(step1, base_row, 0, 1, 5)
        step1_layout.addWidget(QLabel("点云采集由顶部流程按钮触发"), 0, 0, 1, 5)

        step2, step2_layout = self._create_step_group("步骤2：识别书籍")
        layout.addWidget(step2, base_row + 1, 0, 1, 5)
        self.template_combo.setFixedWidth(150)
        self._add_left_value_row(step2_layout, 0, "识别模板:", self.template_combo)

        step3, step3_layout = self._create_step_group("步骤3：解算目标点")
        layout.addWidget(step3, base_row + 2, 0, 1, 5)
        self.book_status_label = QLabel(tr("pc.book_ready"))
        self.book_status_label.setWordWrap(True)
        self._stabilize_flow_label(self.book_status_label)
        step3_layout.addWidget(self.book_status_label, 0, 0, 1, 5)

        self.flow_target_label = QLabel(tr("pc.workflow_target"))
        self.flow_target_label.setWordWrap(True)
        self._stabilize_flow_label(self.flow_target_label)
        step3_layout.addWidget(self.flow_target_label, 1, 0, 1, 5)

        self.putback_target_rpy_spins = []
        for value in self.DEFAULT_PUTBACK_TARGET_RPY_DEG:
            spin = self._make_float_spin(-180.0, 180.0, value, 1.0, "°")
            self.putback_target_rpy_spins.append(spin)
        self._add_left_spins_row(
            step3_layout,
            2,
            "目标姿态Rx/Ry/Rz:",
            self.putback_target_rpy_spins,
        )
        self._bind_workflow_target_refresh(self.putback_target_rpy_spins)

        self.putback_target_comp_xyz_spins = []
        for value in (0.0, 0.0, 0.0):
            spin = self._make_float_spin(-50.0, 50.0, value, 0.5, " cm")
            self.putback_target_comp_xyz_spins.append(spin)
        self._add_left_spins_row(
            step3_layout,
            3,
            "目标补偿XYZ:",
            self.putback_target_comp_xyz_spins,
        )
        self._bind_workflow_target_refresh(self.putback_target_comp_xyz_spins)

        step4, step4_layout = self._create_step_group("步骤4：到达预备推书点")
        layout.addWidget(step4, base_row + 3, 0, 1, 5)
        self.putback_prepush_x_spin = self._make_float_spin(
            -50.0,
            50.0,
            self.DEFAULT_PUTBACK_PREPUSH_X_OFFSET_CM,
            1.0,
            " cm",
        )
        self._add_left_value_row(
            step4_layout,
            0,
            "预备偏移X:",
            self.putback_prepush_x_spin,
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
            1,
            "IK+MoveJ时间:",
            self.flow_ik_duration_spin,
        )

        step5, step5_layout = self._create_step_group("步骤5：MoveL到推书点")
        layout.addWidget(step5, base_row + 4, 0, 1, 5)
        self.flow_movel_duration_spin = self._make_float_spin(
            0.5,
            30.0,
            2.0,
            0.5,
            " s",
        )
        self._add_left_value_row(
            step5_layout,
            0,
            "MoveL时间:",
            self.flow_movel_duration_spin,
        )

        step6, step6_layout = self._create_step_group("步骤6：推开书本")
        layout.addWidget(step6, base_row + 5, 0, 1, 5)
        self.putback_push_y_spin = self._make_float_spin(
            -50.0,
            50.0,
            self.DEFAULT_PUTBACK_PUSH_Y_OFFSET_CM,
            0.5,
            " cm",
        )
        self._add_left_value_row(step6_layout, 0, "推书Y+:", self.putback_push_y_spin)
        self._add_left_value_row(step6_layout, 1, "MoveL时间:", "同步骤5")

        step7, step7_layout = self._create_step_group("步骤7：离开推书位置")
        layout.addWidget(step7, base_row + 6, 0, 1, 5)
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
        layout.addWidget(step8, base_row + 7, 0, 1, 5)
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
        layout.addWidget(step9, base_row + 8, 0, 1, 5)
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
        layout.addWidget(step10, base_row + 9, 0, 1, 5)
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
        layout.addWidget(step11, base_row + 10, 0, 1, 5)
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

        step12, step12_layout = self._create_step_group("步骤12：MoveJ回到调试位")
        layout.addWidget(step12, base_row + 11, 0, 1, 5)
        self.debug_joint_spins = []
        for value in self.DEFAULT_DEBUG_JOINTS_DEG:
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

        return base_row + 12

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

    def _stabilize_flow_label(self, label: QLabel):
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        label.setMaximumWidth(420)
        label.setFixedHeight(self._wrapped_label_height(label, 2))

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
        self._workflow_default_controls = self._collect_workflow_default_controls()
        self._loading_workflow_defaults = True
        try:
            self._apply_workflow_defaults(
                _load_book_workflow_defaults(self._workflow_mode)
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

        if self._workflow_mode == "takeout":
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
            add("gripper_open", getattr(self, "gripper_open_spin", None))
            add("gripper_close", getattr(self, "gripper_close_spin", None))
            add("gripper_effort", getattr(self, "gripper_effort_spin", None))
            add("gripper_close_speed", getattr(self, "gripper_close_speed_spin", None))
            add("gripper_kp", getattr(self, "gripper_kp_spin", None))
            add("gripper_kd", getattr(self, "gripper_kd_spin", None))
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
            add_many("final_joint", getattr(self, "final_joint_spins", []))
            add("flow_final_duration", getattr(self, "flow_final_duration_spin", None))
        else:
            add_many("putback_target_rpy", getattr(self, "putback_target_rpy_spins", []))
            add_many("putback_target_comp_xyz", getattr(self, "putback_target_comp_xyz_spins", []))
            add("putback_prepush_x", getattr(self, "putback_prepush_x_spin", None))
            add("putback_push_y", getattr(self, "putback_push_y_spin", None))
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
        u, v = _book_pick_point_from_polygon(polygon, self._workflow_mode)
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
            self._advance_book_grasp_flow("书籍识别完成，目标点已自动选中")

    def _on_book_detection_thread_finished(self):
        self.capture_btn.setEnabled(True)
        if hasattr(self, "flow_detect_btn"):
            self.flow_detect_btn.setEnabled(True)
        self.capture_btn.setText(tr("pc.capture"))

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
        u, v = _book_pick_point_from_polygon(pick.polygon, self._workflow_mode)
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

    def _current_workflow_target_rpy_deg(self) -> list[float]:
        if self._workflow_mode == "putback" and hasattr(self, "putback_target_rpy_spins"):
            return self._spin_values(self.putback_target_rpy_spins)
        if self._workflow_mode == "takeout" and hasattr(self, "grasp_rpy_spins"):
            return self._spin_values(self.grasp_rpy_spins)
        return self._spin_values(self.rpy_spins)

    def _current_target_compensation_cm(self) -> list[float]:
        if self._workflow_mode == "takeout" and hasattr(self, "target_comp_xyz_spins"):
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
        return "书籍放回流程" if self._workflow_mode == "putback" else "书籍取出流程"

    def _book_grasp_steps(self) -> list[str]:
        if self._workflow_mode == "putback":
            return [
                "点云识别/采集",
                "书籍识别，自动选中目标点",
                "解算目标点，并设置目标姿态",
                "IK + MoveJ 到预备推书点",
                "MoveL 到推书点",
                "MoveL 沿当前位姿 Y+ 推开书本",
                "MoveL 沿当前位姿 Y+ 离开推书位置",
                "IK + MoveJ 到插入预备位",
                "MoveL 到插入位",
                "打开夹爪到指定角度",
                "MoveL 沿当前位姿 X+ 离开插入位",
                "MoveJ 回到调试位",
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
            "机械臂 MoveJ 回到调试位置",
            "机械臂慢速转身到 -90 度构型",
            "机械臂 MoveJ 到最终放置构型",
        ]

    def _format_flow_steps_text(self) -> str:
        return "\n".join(
            f"{idx + 1}. {step}" for idx, step in enumerate(self._book_grasp_steps())
        )

    def _show_flow_steps_dialog(self):
        title = (
            tr("pc.book_putback_group")
            if self._workflow_mode == "putback"
            else tr("pc.book_takeout_group")
        )
        QMessageBox.information(
            self,
            title,
            self._format_flow_steps_text(),
        )

    def _reset_book_grasp_flow(self):
        self._flow_active = False
        self._flow_step_index = 0
        self._flow_waiting_motion = False
        self._flow_waiting_kind = None
        self._flow_rollback_waiting = False
        self._flow_pose_history.clear()
        self._flow_approach_pose = None
        self._flow_last_pose = None
        self._flow_pending_pose = None
        self._rod_current_angle_deg = None
        self._rod_target_angle_deg = None
        self._clear_gripper_close_monitor()
        self.flow_status_label.setText(tr("pc.workflow_pending"))
        if hasattr(self, "flow_target_label"):
            self.flow_target_label.setText(tr("pc.workflow_target"))
        self._update_flow_button_state()

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
            self._solve_book_grasp_target()
        elif self._workflow_mode == "putback":
            self._execute_putback_motion_step(step)
        else:
            self._execute_takeout_motion_step(step)

    def _execute_takeout_motion_step(self, step: int):
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
            self._execute_turn_stage1_step()
        elif step == 11:
            self._execute_final_joint_step()

    def _execute_putback_motion_step(self, step: int):
        if step == 3:
            pose = self._make_putback_target_pose(
                x_offset_cm=self.putback_prepush_x_spin.value(),
                y_offset_cm=0.0,
                z_offset_cm=0.0,
                rpy_deg=self._spin_values(self.putback_target_rpy_spins),
                include_compensation=False,
            )
            self._send_blocking_end_pose(pose, "等待机械臂 IK + MoveJ 到预备推书点")
        elif step == 4:
            pose = self._make_putback_target_pose(
                x_offset_cm=0.0,
                y_offset_cm=0.0,
                z_offset_cm=0.0,
                rpy_deg=self._spin_values(self.putback_target_rpy_spins),
                include_compensation=True,
            )
            self._send_blocking_movel(pose, "等待机械臂 MoveL 到推书点")
        elif step == 5:
            pose = self._make_pose_from_current_end_pose(
                y_offset_cm=self.putback_push_y_spin.value(),
                local_axes=True,
                prefer_last_flow_pose=True,
            )
            if pose is None:
                self._set_error("暂无末端位姿，无法沿 Y+ 推书")
                return
            self._send_blocking_movel(pose, "等待机械臂沿当前位姿 Y+ 推开书本")
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
                include_compensation=False,
            )
            self._send_blocking_end_pose(pose, "等待机械臂 IK + MoveJ 到插入预备位")
        elif step == 8:
            pose = self._make_putback_target_pose(
                x_offset_cm=self.putback_insert_x_spin.value(),
                y_offset_cm=self.putback_insert_y_spin.value(),
                z_offset_cm=0.0,
                rpy_deg=self._spin_values(self.putback_insert_rpy_spins),
                include_compensation=False,
            )
            self._send_blocking_movel(pose, "等待机械臂 MoveL 到插入位")
        elif step == 9:
            self._send_gripper(self.putback_gripper_open_spin.value(), 0.0)
            self._advance_book_grasp_flow("夹爪打开命令已发送")
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
        self.flow_target_label.setText(
            f"目标点: {_format_vec_cm(self._target_robot_point_m)}"
        )
        self._advance_book_grasp_flow(
            f"目标点已解算: {_format_vec_cm(self._target_robot_point_m)}"
        )

    def _execute_pregrasp_step(self):
        if self._target_robot_point_m is None:
            self._set_error("没有目标点，无法执行预抓取")
            return
        self._set_grasp_rpy()
        if self._pregrasp_prepare_tools_enabled():
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
        self._send_blocking_movel(pose, "等待机械臂微调到抓取位姿")

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

    def _execute_return_debug_step(self):
        joints = self._joint_spins_to_radians(self.debug_joint_spins)
        self._record_previous_flow_pose()
        self._flow_waiting_motion = True
        self._flow_waiting_kind = "move_j"
        self._flow_pending_pose = None
        self._update_flow_button_state()
        self.flow_status_label.setText(
            f"等待机械臂 MoveJ 回到调试构型 {self._format_joint_spins_deg(self.debug_joint_spins)}"
        )
        self.move_j_block_requested.emit(
            joints,
            float(self.flow_debug_duration_spin.value()),
        )

    def _move_to_debug_pose_from_header(self):
        if self._flow_waiting_motion:
            return
        if not self._arm_enabled:
            self._set_error("机械臂未使能，无法运动到调试位")
            return
        if not hasattr(self, "debug_joint_spins"):
            self._set_error("调试位关节参数尚未初始化")
            return

        joints = self._joint_spins_to_radians(self.debug_joint_spins)
        self._send_manual_header_movej(
            joints,
            self._manual_header_move_duration(),
            "debug_move",
            f"等待机械臂 MoveJ 到调试位 {self._format_joint_spins_deg(self.debug_joint_spins)}",
        )

    def _move_to_zero_pose_from_header(self):
        if self._flow_waiting_motion:
            return
        if not self._arm_enabled:
            self._set_error("机械臂未使能，无法归零")
            return
        self._send_manual_header_movej(
            [0.0] * len(self.DEFAULT_DEBUG_JOINTS_DEG),
            self._manual_header_move_duration(),
            "zero_move",
            "等待机械臂 MoveJ 归零 [0.00, 0.00, 0.00, 0.00, 0.00, 0.00]",
        )

    def _manual_header_move_duration(self) -> float:
        return float(self.HEADER_MOVE_DURATION_S)

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

    def _execute_turn_stage1_step(self):
        self._send_slow_turn_movej(
            self.turn_stage1_joint_spins,
            f"等待机械臂慢速转身到 {self._format_joint_spins_deg(self.turn_stage1_joint_spins)}",
        )

    def _execute_final_joint_step(self):
        self._send_joint_movej(
            self.final_joint_spins,
            float(self.flow_final_duration_spin.value()),
            f"等待机械臂 MoveJ 到最终构型 {self._format_joint_spins_deg(self.final_joint_spins)}",
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
    def _format_joint_spins_deg(spins: Sequence[QDoubleSpinBox]) -> str:
        values = [float(spin.value()) for spin in spins]
        return "[" + ", ".join(f"{value:.2f}" for value in values) + "]"

    def _make_putback_target_pose(
        self,
        x_offset_cm: float,
        y_offset_cm: float,
        z_offset_cm: float,
        rpy_deg: Sequence[float],
        include_compensation: bool,
    ) -> list[float]:
        comp = (
            self._spin_values(self.putback_target_comp_xyz_spins)
            if include_compensation
            else [0.0, 0.0, 0.0]
        )
        return self._make_target_pose_with_offset(
            x_offset_cm=float(x_offset_cm) + comp[0],
            y_offset_cm=float(y_offset_cm) + comp[1],
            z_offset_cm=float(z_offset_cm) + comp[2],
            rpy_deg=rpy_deg,
        )

    def _send_blocking_movel(self, pose: list[float], waiting_text: str):
        self._record_previous_flow_pose()
        self._flow_waiting_motion = True
        self._flow_waiting_kind = "move_l"
        self._flow_pending_pose = [float(value) for value in pose]
        self._update_flow_button_state()
        self.flow_status_label.setText(waiting_text)
        self.move_l_block_requested.emit(pose, self._flow_movel_duration())

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
        if not self._flow_pose_history:
            self.flow_status_label.setText("没有可回退的位置")
            return
        step, label, pose = self._flow_pose_history.pop()
        if not self._flow_active:
            self._flow_active = True
        self._flow_step_index = max(0, min(step, len(self._book_grasp_steps()) - 1))
        self._flow_waiting_motion = True
        self._flow_waiting_kind = "move_l"
        self._flow_rollback_waiting = True
        self._flow_pending_pose = [float(value) for value in pose]
        self._update_flow_button_state()
        self.flow_status_label.setText(f"正在回退到上一步位置：{label}")
        self.log_message.emit(f"{self._flow_log_prefix()}回退到上一步位置: {label}")
        self.move_l_block_requested.emit(
            [float(value) for value in pose],
            self._flow_movel_duration(),
        )

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

    def _gripper_close_step_deg(self) -> float:
        speed_deg_s = self._gripper_close_speed_deg_s()
        return max(0.1, speed_deg_s * self.GRIPPER_CLOSE_STEP_INTERVAL_S)

    def _start_monitored_gripper_close(self):
        target_deg = float(self.gripper_close_spin.value())
        effort = float(self.gripper_effort_spin.value())
        self._flow_waiting_motion = True
        self._flow_waiting_kind = "gripper_close"
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
        self.gripper_close_monitor_requested.emit(
            {
                "target_angle": math.radians(target_deg),
                "close_speed": math.radians(self._gripper_close_speed_deg_s()),
                "effort": effort,
                "kp": float(self.gripper_kp_spin.value()) if hasattr(self, "gripper_kp_spin") else 0.0,
                "kd": float(self.gripper_kd_spin.value()) if hasattr(self, "gripper_kd_spin") else 0.0,
                "timeout_s": float(self.GRIPPER_CLOSE_TIMEOUT_S),
                "monitor_period_s": 0.05,
                "target_tolerance": math.radians(self.GRIPPER_CLOSE_TOLERANCE_DEG),
                "stall_tolerance": math.radians(self.GRIPPER_CLOSE_STALL_TOLERANCE_DEG),
                "stall_time_s": float(self.GRIPPER_CLOSE_STALL_S),
                "min_monitor_s": float(self.GRIPPER_CLOSE_MIN_MONITOR_S),
                "hold_margin": math.radians(0.5),
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
        if self._flow_waiting_kind != "gripper_close":
            return
        if not self._flow_active or not self._flow_waiting_motion:
            return
        result = dict(result or {})
        reason = str(result.get("reason", "done"))
        final_deg = math.degrees(float(result.get("final_angle", 0.0)))
        command_deg = math.degrees(float(result.get("command_angle", result.get("final_angle", 0.0))))
        self._current_gripper_angle_deg = final_deg
        self._clear_gripper_close_monitor()
        if reason == "target_reached":
            message = f"夹爪已关闭到目标附近：{final_deg:.1f}°"
        elif reason == "stalled":
            message = f"夹爪检测到夹持/卡滞趋势，已锁定保持角度：{command_deg:.1f}°"
        elif reason == "timeout":
            message = f"夹爪监测闭合超时，已锁定保持角度：{command_deg:.1f}°"
        else:
            message = f"夹爪监测闭合完成，保持角度：{command_deg:.1f}°"
        self._advance_book_grasp_flow(message)

    def _monitor_gripper_close(self, current_deg: float):
        target_deg = self._gripper_close_target_deg
        if target_deg is None:
            return
        now = time.monotonic()
        elapsed = now - self._gripper_close_started_at_s
        if abs(current_deg - target_deg) <= self.GRIPPER_CLOSE_TOLERANCE_DEG:
            self._clear_gripper_close_monitor()
            self._advance_book_grasp_flow(f"夹爪已关闭到位：{current_deg:.1f}°")
            return

        self._maybe_advance_gripper_close_target(now)

        last_angle = self._gripper_close_last_angle_deg
        if last_angle is not None and abs(current_deg - last_angle) <= self.GRIPPER_CLOSE_STALL_TOLERANCE_DEG:
            if self._gripper_close_stable_since_s is None:
                self._gripper_close_stable_since_s = now
            elif (
                elapsed >= self.GRIPPER_CLOSE_MIN_MONITOR_S
                and now - self._gripper_close_stable_since_s >= self.GRIPPER_CLOSE_STALL_S
            ):
                self._clear_gripper_close_monitor()
                self._advance_book_grasp_flow(
                    f"夹爪持续关闭并检测到稳定夹持：{current_deg:.1f}°"
                )
                return
        else:
            self._gripper_close_stable_since_s = None
        self._gripper_close_last_angle_deg = current_deg

        if elapsed >= self.GRIPPER_CLOSE_TIMEOUT_S:
            self._clear_gripper_close_monitor()
            self._advance_book_grasp_flow(
                f"夹爪持续关闭监测完成，当前角度 {current_deg:.1f}°"
            )

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
            self.flow_status_label.setText(f"流程完成：{message}")
            self.log_message.emit(f"{self._flow_log_prefix()}完成")
        else:
            next_step = self._book_grasp_steps()[self._flow_step_index]
            self.flow_status_label.setText(
                f"{message}。请确认后执行 [{self._flow_step_index + 1}/{len(self._book_grasp_steps())}] {next_step}"
            )
        self._update_flow_button_state()

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

    def notify_move_l_done(self):
        if self._flow_waiting_kind not in {None, "move_l"}:
            return
        if self._flow_active and self._flow_waiting_motion:
            self._promote_pending_flow_pose()
        if self._flow_active and self._flow_rollback_waiting:
            self._flow_waiting_motion = False
            self._flow_waiting_kind = None
            self._flow_rollback_waiting = False
            next_step = self._book_grasp_steps()[self._flow_step_index]
            self.flow_status_label.setText(
                f"已回退到上一步位置。请确认后执行 [{self._flow_step_index + 1}/{len(self._book_grasp_steps())}] {next_step}"
            )
            self._update_flow_button_state()
            return
        if self._flow_active and self._flow_waiting_motion:
            self._advance_book_grasp_flow("机械臂运动完成")

    def notify_move_j_done(self):
        if self._flow_waiting_kind in {"debug_move", "zero_move"}:
            was_zero_move = self._flow_waiting_kind == "zero_move"
            self._flow_waiting_motion = False
            self._flow_waiting_kind = None
            self._flow_pending_pose = None
            suffix = "，可继续当前流程" if self._flow_active else ""
            done_text = "已归零" if was_zero_move else "已运动到调试位"
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
        if self._flow_active and self._flow_waiting_motion:
            self._promote_pending_flow_pose()
        if self._flow_active and self._flow_waiting_motion:
            self._advance_book_grasp_flow("机械臂 IK + MoveJ 运动完成")

    def notify_rod_write_done(self):
        if (
            self._workflow_mode == "takeout"
            and self._flow_active
            and self._flow_waiting_motion
            and self._flow_waiting_kind == "rod"
            and self._flow_step_index == 5
        ):
            self.flow_status_label.setText("杆电机夹取位命令已发送，等待到位")

    def notify_rod_angle_updated(self, angle_deg: float):
        self._rod_current_angle_deg = float(angle_deg)
        if (
            self._workflow_mode == "takeout"
            and self._flow_active
            and self._flow_waiting_motion
            and self._flow_waiting_kind == "rod"
            and self._flow_step_index == 5
            and self._rod_target_angle_deg is not None
            and abs(self._rod_current_angle_deg - self._rod_target_angle_deg)
            <= self._rod_wait_tolerance_deg
        ):
            self._advance_book_grasp_flow(
                f"杆电机已到位：{self._rod_current_angle_deg:.1f}°"
            )

    def notify_flow_error(self, message: str):
        if self._flow_waiting_kind in {"debug_move", "zero_move"}:
            was_zero_move = self._flow_waiting_kind == "zero_move"
            self._flow_waiting_motion = False
            self._flow_waiting_kind = None
            self._flow_rollback_waiting = False
            self._flow_pending_pose = None
            label = "归零" if was_zero_move else "运动到调试位"
            self.flow_status_label.setText(f"{label}失败：{message}")
            self._update_flow_button_state()
            return
        if not self._flow_active:
            return
        self._flow_waiting_motion = False
        self._flow_waiting_kind = None
        self._flow_rollback_waiting = False
        self._flow_pending_pose = None
        self._rod_target_angle_deg = None
        self._clear_gripper_close_monitor()
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
        self.flow_next_btn.setEnabled(can_step)
        if hasattr(self, "flow_back_btn"):
            self.flow_back_btn.setEnabled(
                self._arm_enabled
                and not self._flow_waiting_motion
                and bool(self._flow_pose_history)
            )
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
        if hasattr(self, "grasp_group"):
            self.grasp_group.setTitle(
                tr("pc.book_putback_group")
                if self._workflow_mode == "putback"
                else tr("pc.book_takeout_group")
            )
        for idx, (label, path) in enumerate(BOOK_TEMPLATE_OPTIONS):
            if idx < self.template_combo.count():
                self.template_combo.setItemText(idx, label)
                self.template_combo.setItemData(idx, str(path))
        if hasattr(self, "flow_detect_btn"):
            self.flow_detect_btn.setText(tr("pc.workflow_detect"))
        if hasattr(self, "flow_steps_btn"):
            self.flow_steps_btn.setText(tr("pc.workflow_steps"))
        if hasattr(self, "flow_debug_move_btn"):
            self.flow_debug_move_btn.setText("运动到调试位")
        if hasattr(self, "flow_zero_move_btn"):
            self.flow_zero_move_btn.setText("归零")
        if hasattr(self, "flow_reset_btn"):
            self.flow_reset_btn.setText(tr("pc.workflow_reset"))
        if hasattr(self, "flow_back_btn"):
            self.flow_back_btn.setText(tr("pc.workflow_back"))
        if hasattr(self, "flow_defaults_btn"):
            self._update_workflow_default_edit_buttons()
        if hasattr(self, "flow_next_btn"):
            self._update_flow_button_state()
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


__all__ = [
    "RealSensePointPanel",
    "camera_point_to_robot_target",
]
