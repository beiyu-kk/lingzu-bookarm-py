"""夹爪控制面板"""

import math
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QDoubleSpinBox, QPushButton, QGroupBox,
)
from PyQt6.QtCore import pyqtSignal, Qt

from MotorStudio.utils.i18n import tr
from MotorStudio.utils.theme_manager import ThemeManager
from MotorStudio.utils.style import SCENE_COLORS


class GripperPanel(QWidget):
    """夹爪控制：角度滑块、全开/全关、设零"""

    gripper_command = pyqtSignal(float, float, float, float)  # angle rad, effort Nm, kp, kd
    set_zero_requested = pyqtSignal()

    OPEN_DEG = 0.0
    CLOSE_DEG = 108.5
    GRIP_HOLD_EFFORT_NM = 0.12
    GRIP_KP = 18.0
    GRIP_KD = 2.0
    ANGLE_MIN_DEG = -15.0
    ANGLE_MAX_DEG = 125.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self._open_deg = self.OPEN_DEG
        self._close_deg = self.CLOSE_DEG
        self._init_ui()

    def _scene(self):
        return SCENE_COLORS[ThemeManager.instance().theme]

    def _apply_feedback_label_styles(self):
        sc = self._scene()
        self.fb_label.setStyleSheet(
            f"color: {sc['accent']}; font-weight: bold;")
        self.torque_label.setStyleSheet(f"color: {sc['warning']};")

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.group = QGroupBox()
        g_layout = QVBoxLayout()

        angle_layout = QHBoxLayout()
        self._lbl_angle = QLabel()
        angle_layout.addWidget(self._lbl_angle)
        self.angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.angle_slider.setRange(int(self.ANGLE_MIN_DEG * 10), int(self.ANGLE_MAX_DEG * 10))
        self.angle_slider.setValue(0)
        self.angle_slider.valueChanged.connect(self._on_slider_changed)
        angle_layout.addWidget(self.angle_slider)

        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(self.ANGLE_MIN_DEG, self.ANGLE_MAX_DEG)
        self.angle_spin.setDecimals(1)
        self.angle_spin.setSuffix("°")
        self.angle_spin.setMinimumWidth(104)
        self.angle_spin.valueChanged.connect(self._on_spin_changed)
        angle_layout.addWidget(self.angle_spin)
        g_layout.addLayout(angle_layout)

        limit_layout = QHBoxLayout()
        self._lbl_open = QLabel()
        limit_layout.addWidget(self._lbl_open)
        self.open_spin = QDoubleSpinBox()
        self.open_spin.setRange(self.ANGLE_MIN_DEG, self.ANGLE_MAX_DEG)
        self.open_spin.setDecimals(1)
        self.open_spin.setSuffix("°")
        self.open_spin.setMinimumWidth(96)
        self.open_spin.setValue(self._open_deg)
        self.open_spin.valueChanged.connect(self._on_open_limit_changed)
        limit_layout.addWidget(self.open_spin)

        self.use_current_open_btn = QPushButton()
        self.use_current_open_btn.clicked.connect(self._use_current_as_open)
        limit_layout.addWidget(self.use_current_open_btn)

        self._lbl_close = QLabel()
        limit_layout.addWidget(self._lbl_close)
        self.close_spin = QDoubleSpinBox()
        self.close_spin.setRange(self.ANGLE_MIN_DEG, self.ANGLE_MAX_DEG)
        self.close_spin.setDecimals(1)
        self.close_spin.setSuffix("°")
        self.close_spin.setMinimumWidth(96)
        self.close_spin.setValue(self._close_deg)
        self.close_spin.valueChanged.connect(self._on_close_limit_changed)
        limit_layout.addWidget(self.close_spin)

        self.use_current_close_btn = QPushButton()
        self.use_current_close_btn.clicked.connect(self._use_current_as_close)
        limit_layout.addWidget(self.use_current_close_btn)
        limit_layout.addStretch()
        g_layout.addLayout(limit_layout)

        debug_layout = QHBoxLayout()
        self._lbl_step = QLabel()
        debug_layout.addWidget(self._lbl_step)
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.1, 20.0)
        self.step_spin.setDecimals(1)
        self.step_spin.setSingleStep(0.5)
        self.step_spin.setSuffix("°")
        self.step_spin.setMinimumWidth(90)
        self.step_spin.setValue(2.0)
        debug_layout.addWidget(self.step_spin)

        self.step_open_btn = QPushButton()
        self.step_open_btn.clicked.connect(lambda: self._jog(-self.step_spin.value()))
        debug_layout.addWidget(self.step_open_btn)

        self.step_close_btn = QPushButton()
        self.step_close_btn.clicked.connect(lambda: self._jog(self.step_spin.value()))
        debug_layout.addWidget(self.step_close_btn)

        self._lbl_effort = QLabel()
        debug_layout.addWidget(self._lbl_effort)
        self.effort_spin = QDoubleSpinBox()
        self.effort_spin.setRange(0.0, 1.0)
        self.effort_spin.setDecimals(3)
        self.effort_spin.setSingleStep(0.02)
        self.effort_spin.setSuffix(" Nm")
        self.effort_spin.setMinimumWidth(100)
        self.effort_spin.setValue(0.0)
        debug_layout.addWidget(self.effort_spin)
        debug_layout.addStretch()
        g_layout.addLayout(debug_layout)

        fb_layout = QHBoxLayout()
        self._lbl_actual = QLabel()
        fb_layout.addWidget(self._lbl_actual)
        self.fb_label = QLabel("0.0°")
        fb_layout.addWidget(self.fb_label)
        fb_layout.addSpacing(20)
        self._lbl_torque = QLabel()
        fb_layout.addWidget(self._lbl_torque)
        self.torque_label = QLabel("0.00 Nm")
        fb_layout.addWidget(self.torque_label)
        fb_layout.addStretch()
        g_layout.addLayout(fb_layout)

        btn_layout = QHBoxLayout()
        self.send_btn = QPushButton()
        self.send_btn.setObjectName("enableBtn")
        self.send_btn.clicked.connect(self._send_command)
        btn_layout.addWidget(self.send_btn)

        self.open_btn = QPushButton()
        self.open_btn.clicked.connect(lambda: self._set_angle(self._open_deg))
        btn_layout.addWidget(self.open_btn)

        self.close_btn = QPushButton()
        self.close_btn.clicked.connect(lambda: self._set_angle(self._close_deg))
        btn_layout.addWidget(self.close_btn)

        self.grip_btn = QPushButton()
        self.grip_btn.clicked.connect(self._grip_command)
        btn_layout.addWidget(self.grip_btn)

        self.zero_btn = QPushButton()
        self.zero_btn.clicked.connect(self.set_zero_requested.emit)
        btn_layout.addWidget(self.zero_btn)

        btn_layout.addStretch()
        g_layout.addLayout(btn_layout)

        self.group.setLayout(g_layout)
        layout.addWidget(self.group)
        layout.addStretch()

        self.retranslate_ui()

    def retranslate_ui(self):
        self.group.setTitle(tr("grip.group"))
        self._lbl_angle.setText(tr("grip.angle"))
        self._lbl_open.setText("开位:")
        self._lbl_close.setText("关位:")
        self.use_current_open_btn.setText("设为开位")
        self.use_current_close_btn.setText("设为关位")
        self._lbl_step.setText("点动:")
        self.step_open_btn.setText("开一点")
        self.step_close_btn.setText("关一点")
        self._lbl_effort.setText("力度:")
        self._lbl_actual.setText(tr("grip.actual"))
        self._lbl_torque.setText(tr("grip.torque"))
        self.send_btn.setText(tr("grip.send"))
        self.open_btn.setText(tr("grip.open"))
        self.close_btn.setText(tr("grip.close"))
        self.grip_btn.setText(tr("grip.grip"))
        self.zero_btn.setText(tr("grip.set_zero"))
        self._apply_feedback_label_styles()

    def _on_slider_changed(self, val):
        if self._updating:
            return
        self._updating = True
        self.angle_spin.setValue(val / 10.0)
        self._updating = False

    def _on_spin_changed(self, val):
        if self._updating:
            return
        self._updating = True
        self.angle_slider.setValue(int(val * 10))
        self._updating = False

    def _set_angle(self, deg):
        self._updating = True
        value = self._clamp_angle(deg)
        self.angle_spin.setValue(value)
        self.angle_slider.setValue(int(value * 10))
        self._updating = False
        self._send_command()

    def _send_command(self):
        angle_rad = math.radians(self.angle_spin.value())
        effort = float(self.effort_spin.value()) if hasattr(self, "effort_spin") else 0.0
        kp = self.GRIP_KP if effort > 0.0 else 0.0
        kd = self.GRIP_KD if effort > 0.0 else 0.0
        self.gripper_command.emit(angle_rad, effort, kp, kd)

    def _grip_command(self):
        self._updating = True
        self.angle_spin.setValue(self._close_deg)
        self.angle_slider.setValue(int(self._close_deg * 10))
        self._updating = False
        self.gripper_command.emit(
            math.radians(self._close_deg),
            self.GRIP_HOLD_EFFORT_NM,
            self.GRIP_KP,
            self.GRIP_KD,
        )

    def _clamp_angle(self, deg):
        return max(self.ANGLE_MIN_DEG, min(self.ANGLE_MAX_DEG, float(deg)))

    def _jog(self, delta_deg):
        self._set_angle(self.angle_spin.value() + float(delta_deg))

    def _on_open_limit_changed(self, value):
        self._open_deg = self._clamp_angle(value)

    def _on_close_limit_changed(self, value):
        self._close_deg = self._clamp_angle(value)

    def _use_current_as_open(self):
        self.open_spin.setValue(self.angle_spin.value())

    def _use_current_as_close(self):
        self.close_spin.setValue(self.angle_spin.value())

    def update_feedback(self, joint_states, effort_states=None):
        positions = joint_states.to_list(include_gripper=True)
        if len(positions) >= 7:
            deg = math.degrees(positions[6])
            self.fb_label.setText(f"{deg:.1f}°")
        if effort_states:
            torques = effort_states.to_list(include_gripper=True)
            if len(torques) >= 7:
                self.torque_label.setText(f"{torques[6]:.2f} Nm")
