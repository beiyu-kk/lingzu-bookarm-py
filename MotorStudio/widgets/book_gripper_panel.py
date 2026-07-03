"""Shared gripper settings for book workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from MotorStudio.utils.i18n import tr
from MotorStudio.widgets.realsense_point_panel import (
    BOOK_DEBUG_POSE_WORKFLOW_MODES,
    BOOK_GRIPPER_DEFAULT_KEYS,
    BOOK_GRIPPER_WORKFLOW_MODE,
    NoWheelDoubleSpinBox,
    RealSensePointPanel,
    load_book_workflow_defaults,
    save_book_workflow_defaults,
)


@dataclass(frozen=True)
class GripperSetting:
    key: str
    label: str
    minimum: float
    maximum: float
    default: float
    step: float
    suffix: str


GRIPPER_SETTINGS = (
    GripperSetting("gripper_open", "打开角度:", -30.0, 140.0, RealSensePointPanel.DEFAULT_GRIPPER_OPEN_DEG, 1.0, "°"),
    GripperSetting("gripper_close", "关闭角度:", -30.0, 140.0, RealSensePointPanel.DEFAULT_GRIPPER_CLOSE_DEG, 1.0, "°"),
    GripperSetting("gripper_close_speed", "关闭速度:", 1.0, 60.0, RealSensePointPanel.DEFAULT_GRIPPER_CLOSE_SPEED_DEG_S, 0.5, "°/s"),
    GripperSetting("gripper_open_speed", "打开速度:", 1.0, 60.0, RealSensePointPanel.DEFAULT_GRIPPER_OPEN_SPEED_DEG_S, 0.5, "°/s"),
    GripperSetting("gripper_effort", "保持力矩:", 0.0, 5.0, RealSensePointPanel.DEFAULT_GRIPPER_HOLD_EFFORT_NM, 0.01, " Nm"),
    GripperSetting("gripper_start_effort", "起步力矩:", 0.0, 5.0, RealSensePointPanel.DEFAULT_GRIPPER_START_EFFORT_NM, 0.01, " Nm"),
    GripperSetting("gripper_start_boost", "起步时间:", 0.0, 3.0, RealSensePointPanel.DEFAULT_GRIPPER_START_BOOST_S, 0.05, " s"),
    GripperSetting("gripper_kp", "Kp:", 0.0, 200.0, RealSensePointPanel.DEFAULT_GRIPPER_KP, 1.0, ""),
    GripperSetting("gripper_kd", "Kd:", 0.0, 50.0, RealSensePointPanel.DEFAULT_GRIPPER_KD, 0.5, ""),
    GripperSetting("gripper_timeout", "超时:", 1.0, 30.0, RealSensePointPanel.GRIPPER_CLOSE_TIMEOUT_S, 0.5, " s"),
    GripperSetting("gripper_target_tolerance", "到位容差:", 0.1, 10.0, RealSensePointPanel.GRIPPER_CLOSE_TOLERANCE_DEG, 0.1, "°"),
    GripperSetting("gripper_stall_tolerance", "卡滞容差:", 0.1, 10.0, RealSensePointPanel.GRIPPER_CLOSE_STALL_TOLERANCE_DEG, 0.1, "°"),
    GripperSetting("gripper_stall_time", "卡滞时间:", 0.1, 10.0, RealSensePointPanel.GRIPPER_CLOSE_STALL_S, 0.1, " s"),
    GripperSetting("gripper_min_monitor", "最短监测:", 0.0, 10.0, RealSensePointPanel.GRIPPER_CLOSE_MIN_MONITOR_S, 0.1, " s"),
    GripperSetting("gripper_hold_margin", "锁定余量:", 0.0, 10.0, RealSensePointPanel.DEFAULT_GRIPPER_HOLD_MARGIN_DEG, 0.1, "°"),
    GripperSetting("gripper_command_lead", "指令提前:", 0.0, 3.0, RealSensePointPanel.GRIPPER_CLOSE_COMMAND_LEAD_S, 0.05, " s"),
    GripperSetting("gripper_stall_lead_threshold", "卡滞阈值:", 0.0, 30.0, RealSensePointPanel.GRIPPER_CLOSE_STALL_LEAD_THRESHOLD_DEG, 0.5, "°"),
    GripperSetting("gripper_step_interval", "步进周期:", 0.02, 1.0, RealSensePointPanel.GRIPPER_CLOSE_STEP_INTERVAL_S, 0.01, " s"),
)

GRIPPER_KEYS = BOOK_GRIPPER_DEFAULT_KEYS


class BookGripperPanel(QWidget):
    """Edit gripper parameters shared by book workflow pages."""

    settings_saved = pyqtSignal(dict)
    gripper_test_requested = pyqtSignal(object)
    move_j_requested = pyqtSignal(list, float)
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._spins: dict[str, QDoubleSpinBox] = {}
        self._status_label = None
        self._init_ui()
        self.retranslate_ui()

    def _init_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer_layout.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.group = QGroupBox()
        grid = QGridLayout(self.group)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        values = self._load_values()
        rows_per_column = (len(GRIPPER_SETTINGS) + 1) // 2
        for idx, setting in enumerate(GRIPPER_SETTINGS):
            row = idx % rows_per_column
            col = (idx // rows_per_column) * 2
            label = QLabel(setting.label)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(label, row, col)
            spin = self._make_spin(setting, values.get(setting.key, setting.default))
            self._spins[setting.key] = spin
            grid.addWidget(spin, row, col + 1)
        grid.setColumnStretch(4, 1)
        layout.addWidget(self.group)

        btn_row = QHBoxLayout()
        self.restore_btn = QPushButton()
        self.restore_btn.clicked.connect(self._restore_builtin_defaults)
        btn_row.addWidget(self.restore_btn)

        self.save_btn = QPushButton()
        self.save_btn.setObjectName("enableBtn")
        self.save_btn.clicked.connect(self._save_settings)
        btn_row.addWidget(self.save_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        test_row = QHBoxLayout()
        self.open_btn = QPushButton()
        self.open_btn.clicked.connect(self._test_open_gripper)
        test_row.addWidget(self.open_btn)

        self.close_btn = QPushButton()
        self.close_btn.setObjectName("enableBtn")
        self.close_btn.clicked.connect(self._test_close_gripper)
        test_row.addWidget(self.close_btn)

        test_row.addStretch()
        layout.addLayout(test_row)

        motion_row = QHBoxLayout()
        self.zero_btn = QPushButton()
        self.zero_btn.clicked.connect(self._move_to_zero_pose)
        motion_row.addWidget(self.zero_btn)

        self.debug_pose_btn = QPushButton()
        self.debug_pose_btn.clicked.connect(self._move_to_debug_pose)
        motion_row.addWidget(self.debug_pose_btn)
        motion_row.addStretch()
        layout.addLayout(motion_row)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        layout.addStretch()

        scroll.setWidget(content)

    @staticmethod
    def _make_spin(setting: GripperSetting, value: float) -> QDoubleSpinBox:
        spin = NoWheelDoubleSpinBox()
        spin.setRange(setting.minimum, setting.maximum)
        spin.setDecimals(3 if abs(setting.step) < 0.1 else 2)
        spin.setSingleStep(setting.step)
        spin.setSuffix(setting.suffix)
        spin.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        try:
            spin.setValue(float(value))
        except (TypeError, ValueError):
            spin.setValue(float(setting.default))
        decimals = spin.decimals()
        samples = [
            f"{setting.minimum:.{decimals}f}{setting.suffix}",
            f"{setting.maximum:.{decimals}f}{setting.suffix}",
            f"{setting.default:.{decimals}f}{setting.suffix}",
        ]
        spin.setFixedWidth(RealSensePointPanel._spinbox_text_width(spin, samples, extra=34))
        return spin

    def retranslate_ui(self):
        self.group.setTitle(tr("book_gripper.group"))
        self.restore_btn.setText(tr("book_gripper.restore"))
        self.save_btn.setText(tr("book_gripper.save"))
        self.open_btn.setText(tr("book_gripper.open"))
        self.close_btn.setText(tr("book_gripper.close"))
        self.zero_btn.setText(tr("book_gripper.zero"))
        self.debug_pose_btn.setText(tr("book_gripper.debug_pose"))
        if self._status_label is not None and not self._status_label.text():
            self._status_label.setText(tr("book_gripper.ready"))

    def current_settings(self) -> dict[str, float]:
        return {key: float(spin.value()) for key, spin in self._spins.items()}

    def set_settings(self, values: dict):
        for key, value in values.items():
            spin = self._spins.get(key)
            if spin is None:
                continue
            try:
                spin.setValue(float(value))
            except (TypeError, ValueError):
                continue

    def _builtin_values(self) -> dict[str, float]:
        return {setting.key: float(setting.default) for setting in GRIPPER_SETTINGS}

    def _load_values(self) -> dict[str, float]:
        values = self._builtin_values()
        stored = load_book_workflow_defaults(BOOK_GRIPPER_WORKFLOW_MODE)
        if not any(key in stored for key in GRIPPER_KEYS):
            stored = load_book_workflow_defaults(BOOK_DEBUG_POSE_WORKFLOW_MODES[0])
        for key in GRIPPER_KEYS:
            if key not in stored:
                continue
            try:
                values[key] = float(stored[key])
            except (TypeError, ValueError):
                pass
        return values

    def _restore_builtin_defaults(self):
        self.set_settings(self._builtin_values())

    def _save_settings(self):
        values = self.current_settings()
        try:
            standalone_defaults = load_book_workflow_defaults(BOOK_GRIPPER_WORKFLOW_MODE)
            standalone_defaults.update(values)
            save_book_workflow_defaults(BOOK_GRIPPER_WORKFLOW_MODE, standalone_defaults)

            for mode in BOOK_DEBUG_POSE_WORKFLOW_MODES:
                defaults = load_book_workflow_defaults(mode)
                defaults.update(values)
                save_book_workflow_defaults(mode, defaults)
        except Exception as exc:
            message = tr("book_gripper.save_failed", error=str(exc))
            self._status_label.setText(message)
            self.error_occurred.emit(message)
            return

        message = tr("book_gripper.saved")
        self._status_label.setText(message)
        self.log_message.emit(message)
        self.settings_saved.emit(values)

    def _monitor_params(self, values: dict[str, float]) -> dict:
        return {
            "timeout_s": float(values["gripper_timeout"]),
            "monitor_period_s": 0.05,
            "target_tolerance": math.radians(float(values["gripper_target_tolerance"])),
            "stall_tolerance": math.radians(float(values["gripper_stall_tolerance"])),
            "stall_time_s": float(values["gripper_stall_time"]),
            "min_monitor_s": float(values["gripper_min_monitor"]),
            "hold_margin": math.radians(float(values["gripper_hold_margin"])),
            "command_lead_s": float(values["gripper_command_lead"]),
            "stall_lead_threshold_min": math.radians(
                float(values["gripper_stall_lead_threshold"])
            ),
        }

    def _test_open_gripper(self):
        values = self.current_settings()
        params = {
            "target_angle": math.radians(float(values["gripper_open"])),
            "close_speed": math.radians(float(values["gripper_open_speed"])),
            "effort": 0.0,
            "kp": float(values["gripper_kp"]),
            "kd": float(values["gripper_kd"]),
            **self._monitor_params(values),
        }
        self._status_label.setText(
            tr("book_gripper.open_running", angle=values["gripper_open"])
        )
        self.log_message.emit(
            tr("book_gripper.open_running", angle=values["gripper_open"])
        )
        self.gripper_test_requested.emit(params)

    def _test_close_gripper(self):
        values = self.current_settings()
        params = {
            "target_angle": math.radians(float(values["gripper_close"])),
            "close_speed": math.radians(float(values["gripper_close_speed"])),
            "effort": float(values["gripper_effort"]),
            "kp": float(values["gripper_kp"]),
            "kd": float(values["gripper_kd"]),
            **self._monitor_params(values),
            "start_effort": float(values["gripper_start_effort"]),
            "start_boost_s": float(values["gripper_start_boost"]),
        }
        self._status_label.setText(
            tr("book_gripper.close_running", angle=values["gripper_close"])
        )
        self.log_message.emit(
            tr("book_gripper.close_running", angle=values["gripper_close"])
        )
        self.gripper_test_requested.emit(params)

    def _move_to_zero_pose(self):
        joints = [0.0] * 6
        duration = float(RealSensePointPanel.HEADER_MOVE_DURATION_S)
        self._status_label.setText(tr("book_gripper.zero_running"))
        self.log_message.emit(tr("book_gripper.zero_running"))
        self.move_j_requested.emit(joints, duration)

    def _move_to_debug_pose(self):
        joints_deg = self._load_debug_pose_deg()
        joints = [math.radians(float(value)) for value in joints_deg[:6]]
        duration = float(RealSensePointPanel.DEFAULT_DEBUG_MOVE_DURATION_S)
        self._status_label.setText(
            tr(
                "book_gripper.debug_pose_running",
                pose=", ".join(f"{value:.1f}" for value in joints_deg[:6]),
            )
        )
        self.log_message.emit(
            tr(
                "book_gripper.debug_pose_running",
                pose=", ".join(f"{value:.1f}" for value in joints_deg[:6]),
            )
        )
        self.move_j_requested.emit(joints, duration)

    def _load_debug_pose_deg(self) -> list[float]:
        values = load_book_workflow_defaults(BOOK_DEBUG_POSE_WORKFLOW_MODES[0])
        pose = [float(value) for value in RealSensePointPanel.DEFAULT_DEBUG_JOINTS_DEG[:6]]
        for idx in range(6):
            key = f"debug_joint_{idx + 1}"
            if key not in values:
                continue
            try:
                pose[idx] = float(values[key])
            except (TypeError, ValueError):
                pass
        return pose

    def notify_gripper_test_done(self, result):
        if not isinstance(result, dict):
            return
        reason = str(result.get("reason", "done"))
        final_angle = math.degrees(float(result.get("final_angle", 0.0)))
        command_angle = math.degrees(float(result.get("command_angle", final_angle)))
        message = tr(
            "book_gripper.test_done",
            reason=reason,
            final=final_angle,
            command=command_angle,
        )
        self._status_label.setText(message)
        self.log_message.emit(message)

    def notify_move_done(self):
        message = tr("book_gripper.move_done")
        self._status_label.setText(message)
        self.log_message.emit(message)
