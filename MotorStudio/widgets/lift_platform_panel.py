"""Lift platform control panel."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from el_a3_sdk import (
    DEFAULT_LIFT_ACCELERATION,
    DEFAULT_LIFT_BAUDRATE,
    DEFAULT_LIFT_PORT,
    DEFAULT_LIFT_PULSES_PER_CM,
    DEFAULT_LIFT_SLAVE_ID,
    DEFAULT_LIFT_SPEED_RPM,
)
from MotorStudio.utils.i18n import tr
from MotorStudio.utils.lift_platform_defaults import (
    load_lift_platform_defaults,
    save_lift_platform_defaults,
)
from MotorStudio.utils.style import SCENE_COLORS
from MotorStudio.utils.theme_manager import ThemeManager


class LiftPlatformPanel(QWidget):
    """Panel for the lift platform."""

    POSITION_LOWEST = "lowest"
    POSITION_RETURN = "return"
    POSITION_TAKE = "take"
    POSITION_TRANSITIONS = {
        POSITION_LOWEST: {POSITION_RETURN, POSITION_TAKE},
        POSITION_RETURN: {POSITION_LOWEST, POSITION_TAKE},
        POSITION_TAKE: {POSITION_LOWEST, POSITION_RETURN},
    }

    connect_requested = pyqtSignal(str, int, int, int, int, float)
    disconnect_requested = pyqtSignal()
    move_distance_requested = pyqtSignal(float, int, int, float, int)
    move_pulses_requested = pyqtSignal(int)
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._connected = False
        self._defaults = load_lift_platform_defaults()
        self._current_position = str(self._defaults.get("current_position", self.POSITION_LOWEST))
        self._init_ui()

    def _scene(self):
        return SCENE_COLORS[ThemeManager.instance().theme]

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.conn_group = QGroupBox()
        conn_layout = QVBoxLayout()

        conn_row = QHBoxLayout()
        self.port_label = QLabel()
        conn_row.addWidget(self.port_label)
        self.port_edit = QLineEdit(str(self._defaults["port"]))
        self.port_edit.setMinimumWidth(160)
        conn_row.addWidget(self.port_edit)
        conn_row.addStretch()
        self.connect_btn = QPushButton()
        self.connect_btn.setObjectName("connectBtn")
        self.connect_btn.clicked.connect(self._toggle_connection)
        conn_row.addWidget(self.connect_btn)
        conn_layout.addLayout(conn_row)

        params_row = QHBoxLayout()
        self.baud_label = QLabel()
        params_row.addWidget(self.baud_label)
        self.baud_spin = QSpinBox()
        self.baud_spin.setRange(1200, 4000000)
        self.baud_spin.setValue(int(self._defaults["baudrate"]))
        self.baud_spin.setMinimumWidth(100)
        params_row.addWidget(self.baud_spin)

        self.slave_label = QLabel()
        params_row.addWidget(self.slave_label)
        self.slave_spin = QSpinBox()
        self.slave_spin.setRange(1, 247)
        self.slave_spin.setValue(int(self._defaults["slave_id"]))
        self.slave_spin.setMinimumWidth(64)
        params_row.addWidget(self.slave_spin)

        self.timeout_label = QLabel()
        params_row.addWidget(self.timeout_label)
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.05, 2.0)
        self.timeout_spin.setDecimals(2)
        self.timeout_spin.setSingleStep(0.05)
        self.timeout_spin.setValue(float(self._defaults["timeout"]))
        self.timeout_spin.setSuffix(" s")
        self.timeout_spin.setMinimumWidth(88)
        params_row.addWidget(self.timeout_spin)
        params_row.addStretch()
        conn_layout.addLayout(params_row)

        self.status_label = QLabel()
        conn_layout.addWidget(self.status_label)
        self.conn_group.setLayout(conn_layout)
        layout.addWidget(self.conn_group)

        self.ctrl_group = QGroupBox()
        ctrl_layout = QVBoxLayout()

        motion_row = QHBoxLayout()
        self.distance_label = QLabel()
        motion_row.addWidget(self.distance_label)
        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(-1000.0, 1000.0)
        self.distance_spin.setDecimals(3)
        self.distance_spin.setSingleStep(1.0)
        self.distance_spin.setValue(float(self._defaults["distance_cm"]))
        self.distance_spin.setSuffix(" cm")
        self.distance_spin.setMinimumWidth(110)
        motion_row.addWidget(self.distance_spin)

        self.speed_label = QLabel()
        motion_row.addWidget(self.speed_label)
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 10000)
        self.speed_spin.setValue(int(self._defaults["speed_rpm"]))
        self.speed_spin.setMinimumWidth(86)
        motion_row.addWidget(self.speed_spin)

        self.acc_label = QLabel()
        motion_row.addWidget(self.acc_label)
        self.acc_spin = QSpinBox()
        self.acc_spin.setRange(1, 10000)
        self.acc_spin.setValue(int(self._defaults["acceleration"]))
        self.acc_spin.setMinimumWidth(86)
        motion_row.addWidget(self.acc_spin)
        motion_row.addStretch()
        ctrl_layout.addLayout(motion_row)

        calibration_row = QHBoxLayout()
        self.pulses_per_cm_label = QLabel()
        calibration_row.addWidget(self.pulses_per_cm_label)
        self.pulses_per_cm_spin = QDoubleSpinBox()
        self.pulses_per_cm_spin.setRange(1.0, 1000000.0)
        self.pulses_per_cm_spin.setDecimals(1)
        self.pulses_per_cm_spin.setSingleStep(100.0)
        self.pulses_per_cm_spin.setValue(float(self._defaults["pulses_per_cm"]))
        self.pulses_per_cm_spin.setMinimumWidth(120)
        calibration_row.addWidget(self.pulses_per_cm_spin)

        self.reverse_check = QCheckBox()
        self.reverse_check.setChecked(bool(self._defaults["reverse_up_direction"]))
        calibration_row.addWidget(self.reverse_check)
        calibration_row.addStretch()
        ctrl_layout.addLayout(calibration_row)

        state_group = QGroupBox()
        state_layout = QVBoxLayout()

        state_row = QHBoxLayout()
        self.current_position_label = QLabel()
        state_row.addWidget(self.current_position_label)
        self.current_position_combo = QComboBox()
        self.current_position_combo.addItem("", self.POSITION_LOWEST)
        self.current_position_combo.addItem("", self.POSITION_RETURN)
        self.current_position_combo.addItem("", self.POSITION_TAKE)
        self.current_position_combo.currentIndexChanged.connect(self._on_current_position_changed)
        state_row.addWidget(self.current_position_combo)
        state_row.addStretch()
        state_layout.addLayout(state_row)

        offset_row = QHBoxLayout()
        self.return_offset_label = QLabel()
        offset_row.addWidget(self.return_offset_label)
        self.return_offset_spin = QDoubleSpinBox()
        self.return_offset_spin.setRange(0.0, 1000.0)
        self.return_offset_spin.setDecimals(3)
        self.return_offset_spin.setSingleStep(1.0)
        self.return_offset_spin.setValue(float(self._defaults["return_offset_cm"]))
        self.return_offset_spin.setSuffix(" cm")
        self.return_offset_spin.setMinimumWidth(110)
        self.return_offset_spin.valueChanged.connect(lambda *_args: self._update_state_status())
        offset_row.addWidget(self.return_offset_spin)

        self.take_offset_label = QLabel()
        offset_row.addWidget(self.take_offset_label)
        self.take_offset_spin = QDoubleSpinBox()
        self.take_offset_spin.setRange(0.0, 1000.0)
        self.take_offset_spin.setDecimals(3)
        self.take_offset_spin.setSingleStep(1.0)
        self.take_offset_spin.setValue(float(self._defaults["take_offset_cm"]))
        self.take_offset_spin.setSuffix(" cm")
        self.take_offset_spin.setMinimumWidth(110)
        self.take_offset_spin.valueChanged.connect(lambda *_args: self._update_state_status())
        offset_row.addWidget(self.take_offset_spin)
        offset_row.addStretch()
        state_layout.addLayout(offset_row)

        target_row = QHBoxLayout()
        self.move_lowest_btn = QPushButton()
        self.move_lowest_btn.clicked.connect(lambda: self._move_to_position(self.POSITION_LOWEST))
        target_row.addWidget(self.move_lowest_btn)
        self.move_return_btn = QPushButton()
        self.move_return_btn.clicked.connect(lambda: self._move_to_position(self.POSITION_RETURN))
        target_row.addWidget(self.move_return_btn)
        self.move_take_btn = QPushButton()
        self.move_take_btn.setObjectName("enableBtn")
        self.move_take_btn.clicked.connect(lambda: self._move_to_position(self.POSITION_TAKE))
        target_row.addWidget(self.move_take_btn)
        target_row.addStretch()
        state_layout.addLayout(target_row)

        self.state_status_label = QLabel()
        state_layout.addWidget(self.state_status_label)

        state_defaults_row = QHBoxLayout()
        self.save_state_defaults_btn = QPushButton()
        self.save_state_defaults_btn.clicked.connect(self._save_state_defaults)
        state_defaults_row.addWidget(self.save_state_defaults_btn)
        state_defaults_row.addStretch()
        state_layout.addLayout(state_defaults_row)
        state_group.setLayout(state_layout)
        self.state_group = state_group

        pulses_row = QHBoxLayout()
        self.pulses_label = QLabel()
        pulses_row.addWidget(self.pulses_label)
        self.pulses_spin = QSpinBox()
        self.pulses_spin.setRange(-2147483647, 2147483647)
        self.pulses_spin.setValue(int(self._defaults["pulses"]))
        self.pulses_spin.setSingleStep(1000)
        self.pulses_spin.setMinimumWidth(140)
        pulses_row.addWidget(self.pulses_spin)
        pulses_row.addStretch()
        ctrl_layout.addLayout(pulses_row)

        btn_row = QHBoxLayout()
        self.move_down_btn = QPushButton()
        self.move_down_btn.clicked.connect(lambda: self._emit_signed_distance(-1.0))
        btn_row.addWidget(self.move_down_btn)

        self.move_up_btn = QPushButton()
        self.move_up_btn.setObjectName("enableBtn")
        self.move_up_btn.clicked.connect(lambda: self._emit_signed_distance(1.0))
        btn_row.addWidget(self.move_up_btn)

        self.pulses_btn = QPushButton()
        self.pulses_btn.clicked.connect(self._emit_pulses)
        btn_row.addWidget(self.pulses_btn)

        self.stop_btn = QPushButton()
        self.stop_btn.setObjectName("estopBtn")
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        ctrl_layout.addLayout(btn_row)

        defaults_row = QHBoxLayout()
        self.save_defaults_btn = QPushButton()
        self.save_defaults_btn.clicked.connect(self._save_defaults)
        defaults_row.addWidget(self.save_defaults_btn)
        defaults_row.addStretch()
        ctrl_layout.addLayout(defaults_row)

        self.ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(self.ctrl_group)
        layout.addWidget(self.state_group)
        layout.addStretch()

        self._set_current_position(self._current_position)
        self.set_connected(False)
        self.retranslate_ui()
        self._apply_status_style()

    def retranslate_ui(self):
        self.conn_group.setTitle(tr("lift.conn_group"))
        self.ctrl_group.setTitle(tr("lift.ctrl_group"))
        self.port_label.setText(tr("lift.port"))
        self.baud_label.setText(tr("lift.baud"))
        self.slave_label.setText(tr("lift.slave"))
        self.timeout_label.setText(tr("lift.timeout"))
        self.distance_label.setText(tr("lift.distance"))
        self.speed_label.setText(tr("lift.speed"))
        self.acc_label.setText(tr("lift.acc"))
        self.pulses_per_cm_label.setText(tr("lift.pulses_per_cm"))
        self.reverse_check.setText(tr("lift.reverse"))
        self.pulses_label.setText(tr("lift.pulses"))
        self.state_group.setTitle(tr("lift.state_group"))
        self.current_position_label.setText(tr("lift.current_position"))
        self.current_position_combo.setItemText(0, tr("lift.position_lowest"))
        self.current_position_combo.setItemText(1, tr("lift.position_return"))
        self.current_position_combo.setItemText(2, tr("lift.position_take"))
        self.return_offset_label.setText(tr("lift.return_offset"))
        self.take_offset_label.setText(tr("lift.take_offset"))
        self.move_lowest_btn.setText(tr("lift.move_lowest"))
        self.move_return_btn.setText(tr("lift.move_return"))
        self.move_take_btn.setText(tr("lift.move_take"))
        self.save_state_defaults_btn.setText(tr("lift.save_state_defaults"))
        self._update_state_status()
        self._update_state_buttons()
        self.move_down_btn.setText(tr("lift.move_down"))
        self.move_up_btn.setText(tr("lift.move_up"))
        self.pulses_btn.setText(tr("lift.move_pulses"))
        self.stop_btn.setText(tr("lift.stop"))
        self.save_defaults_btn.setText(tr("lift.save_defaults"))
        self.connect_btn.setText(tr("lift.disconnect") if self._connected else tr("lift.connect"))
        if not self._connected:
            self.status_label.setText(tr("lift.disconnected"))

    def set_connected(self, connected: bool):
        self._connected = bool(connected)
        self.connect_btn.setText(tr("lift.disconnect") if connected else tr("lift.connect"))
        for widget in (
            self.move_down_btn,
            self.move_up_btn,
            self.pulses_btn,
            self.stop_btn,
        ):
            widget.setEnabled(connected)
        for widget in (
            self.port_edit,
            self.baud_spin,
            self.slave_spin,
            self.timeout_spin,
        ):
            widget.setEnabled(not connected)
        self.status_label.setText(tr("lift.connected") if connected else tr("lift.disconnected"))
        self._apply_status_style()
        self._update_state_buttons()

    def set_error(self, message: str):
        self.status_label.setText(tr("lift.error", msg=message))
        sc = self._scene()
        self.status_label.setStyleSheet(f"color: {sc['warning']}; font-weight: bold;")

    def _apply_status_style(self):
        sc = self._scene()
        color = sc["success"] if self._connected else sc["subtext"]
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _current_settings(self):
        return {
            "port": self.port_edit.text().strip() or DEFAULT_LIFT_PORT,
            "baudrate": self.baud_spin.value(),
            "slave_id": self.slave_spin.value(),
            "timeout": self.timeout_spin.value(),
            "distance_cm": self.distance_spin.value(),
            "speed_rpm": self.speed_spin.value(),
            "acceleration": self.acc_spin.value(),
            "pulses_per_cm": self.pulses_per_cm_spin.value(),
            "reverse_up_direction": self.reverse_check.isChecked(),
            "pulses": self.pulses_spin.value(),
            "current_position": self._current_position,
            "return_offset_cm": self.return_offset_spin.value(),
            "take_offset_cm": self.take_offset_spin.value(),
        }

    def _current_state_settings(self):
        return {
            **self._defaults,
            "current_position": self._current_position,
            "return_offset_cm": self.return_offset_spin.value(),
            "take_offset_cm": self.take_offset_spin.value(),
        }

    def _save_defaults(self):
        try:
            path = save_lift_platform_defaults(self._current_settings())
        except Exception as exc:
            self.set_error(str(exc))
            return
        self.status_label.setText(tr("lift.defaults_saved", path=str(path)))
        self._apply_status_style()

    def _save_state_defaults(self):
        try:
            path = save_lift_platform_defaults(self._current_state_settings())
            self._defaults = load_lift_platform_defaults()
        except Exception as exc:
            self.set_error(str(exc))
            return
        self.state_status_label.setText(tr("lift.state_defaults_saved", path=str(path)))

    def _toggle_connection(self):
        if self._connected:
            self.disconnect_requested.emit()
            return
        port = self.port_edit.text().strip() or DEFAULT_LIFT_PORT
        self.connect_requested.emit(
            port,
            self.baud_spin.value(),
            self.slave_spin.value(),
            self.speed_spin.value(),
            self.acc_spin.value(),
            self.timeout_spin.value(),
        )

    def _emit_signed_distance(self, sign: float):
        distance = abs(float(self.distance_spin.value())) * sign
        self.move_distance_requested.emit(
            distance,
            self.speed_spin.value(),
            self.acc_spin.value(),
            self.pulses_per_cm_spin.value(),
            -1 if self.reverse_check.isChecked() else 1,
        )

    def _emit_pulses(self):
        self.move_pulses_requested.emit(self.pulses_spin.value())

    def _on_current_position_changed(self, *_args):
        self._current_position = self.current_position_combo.currentData() or self.POSITION_LOWEST
        self._update_state_status()

    def _position_height_cm(self, position: str) -> float:
        return_offset = float(self.return_offset_spin.value())
        take_offset = float(self.take_offset_spin.value())
        if position == self.POSITION_RETURN:
            return return_offset
        if position == self.POSITION_TAKE:
            return return_offset + take_offset
        return 0.0

    def _position_label(self, position: str) -> str:
        if position == self.POSITION_RETURN:
            return tr("lift.position_return")
        if position == self.POSITION_TAKE:
            return tr("lift.position_take")
        return tr("lift.position_lowest")

    def _update_state_status(self):
        if not hasattr(self, "state_status_label"):
            return
        current = self._current_position
        self.state_status_label.setText(
            tr(
                "lift.state_status",
                pos=self._position_label(current),
                height=self._position_height_cm(current),
            )
        )
        self._update_state_buttons()

    def _update_state_buttons(self):
        if not hasattr(self, "move_lowest_btn"):
            return
        allowed = self.POSITION_TRANSITIONS.get(self._current_position, set())
        button_map = {
            self.POSITION_LOWEST: self.move_lowest_btn,
            self.POSITION_RETURN: self.move_return_btn,
            self.POSITION_TAKE: self.move_take_btn,
        }
        for position, button in button_map.items():
            button.setEnabled(position in allowed)
            if position == self._current_position:
                button.setToolTip(tr("lift.current_position_tip"))
            elif position in allowed:
                button.setToolTip(
                    tr(
                        "lift.transition_tip",
                        src=self._position_label(self._current_position),
                        dst=self._position_label(position),
                        delta=self._position_height_cm(position)
                        - self._position_height_cm(self._current_position),
                    )
                )
            else:
                button.setToolTip("")

    def _set_current_position(self, position: str):
        for idx in range(self.current_position_combo.count()):
            if self.current_position_combo.itemData(idx) == position:
                self.current_position_combo.setCurrentIndex(idx)
                self._current_position = position
                self._update_state_status()
                return
        self._current_position = position
        self._update_state_status()

    def _move_to_position(self, target_position: str):
        if not self._connected:
            self.status_label.setText(tr("lift.not_connected"))
            self._apply_status_style()
            return
        current_position = self._current_position
        if target_position not in self.POSITION_TRANSITIONS.get(current_position, set()):
            self.status_label.setText(
                tr("lift.transition_not_allowed", pos=self._position_label(target_position))
            )
            self._update_state_buttons()
            return
        delta_cm = self._position_height_cm(target_position) - self._position_height_cm(current_position)
        if abs(delta_cm) < 1e-9:
            self.status_label.setText(
                tr("lift.already_at_position", pos=self._position_label(target_position))
            )
            self._update_state_status()
            return
        self.move_distance_requested.emit(
            delta_cm,
            self.speed_spin.value(),
            self.acc_spin.value(),
            self.pulses_per_cm_spin.value(),
            -1 if self.reverse_check.isChecked() else 1,
        )
        self._set_current_position(target_position)
        self.status_label.setText(
            tr(
                "lift.state_move_sent",
                src=self._position_label(current_position),
                dst=self._position_label(target_position),
                delta=delta_cm,
            )
        )
