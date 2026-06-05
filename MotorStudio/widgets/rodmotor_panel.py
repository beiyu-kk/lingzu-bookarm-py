"""rodmotor control panel."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
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

from el_a3_sdk import RodMotorClient
from MotorStudio.utils.i18n import tr
from MotorStudio.utils.style import SCENE_COLORS
from MotorStudio.utils.theme_manager import ThemeManager


class RodMotorPanel(QWidget):
    """Panel for the independent rodmotor."""

    connect_requested = pyqtSignal(str, int, float)
    disconnect_requested = pyqtSignal()
    read_requested = pyqtSignal()
    write_requested = pyqtSignal(float, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._connected = False
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

        self.port_edit = QLineEdit(RodMotorClient.DEFAULT_PORT)
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
        self.baud_spin.setRange(9600, 4000000)
        self.baud_spin.setSingleStep(115200)
        self.baud_spin.setValue(RodMotorClient.DEFAULT_BAUDRATE)
        self.baud_spin.setMinimumWidth(110)
        params_row.addWidget(self.baud_spin)

        self.timeout_label = QLabel()
        params_row.addWidget(self.timeout_label)
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.05, 2.0)
        self.timeout_spin.setDecimals(2)
        self.timeout_spin.setSingleStep(0.05)
        self.timeout_spin.setValue(0.3)
        self.timeout_spin.setSuffix(" s")
        self.timeout_spin.setMinimumWidth(96)
        params_row.addWidget(self.timeout_spin)
        params_row.addStretch()
        conn_layout.addLayout(params_row)

        self.status_label = QLabel()
        conn_layout.addWidget(self.status_label)
        self.conn_group.setLayout(conn_layout)
        layout.addWidget(self.conn_group)

        self.ctrl_group = QGroupBox()
        ctrl_layout = QVBoxLayout()

        fb_row = QHBoxLayout()
        self.actual_label = QLabel()
        fb_row.addWidget(self.actual_label)
        self.actual_value = QLabel("--.-°")
        fb_row.addWidget(self.actual_value)
        fb_row.addStretch()
        self.read_btn = QPushButton()
        self.read_btn.clicked.connect(self.read_requested.emit)
        fb_row.addWidget(self.read_btn)
        ctrl_layout.addLayout(fb_row)

        target_row = QHBoxLayout()
        self.target_label = QLabel()
        target_row.addWidget(self.target_label)
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(-3600.0, 3600.0)
        self.angle_spin.setDecimals(2)
        self.angle_spin.setSingleStep(5.0)
        self.angle_spin.setSuffix("°")
        self.angle_spin.setMinimumWidth(126)
        target_row.addWidget(self.angle_spin)

        self.speed_label = QLabel()
        target_row.addWidget(self.speed_label)
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 10000)
        self.speed_spin.setValue(1000)
        self.speed_spin.setMinimumWidth(88)
        target_row.addWidget(self.speed_spin)

        self.acc_label = QLabel()
        target_row.addWidget(self.acc_label)
        self.acc_spin = QSpinBox()
        self.acc_spin.setRange(1, 10000)
        self.acc_spin.setValue(50)
        self.acc_spin.setMinimumWidth(88)
        target_row.addWidget(self.acc_spin)
        target_row.addStretch()
        ctrl_layout.addLayout(target_row)

        btn_row = QHBoxLayout()
        self.sync_btn = QPushButton()
        self.sync_btn.clicked.connect(self._sync_target_to_actual)
        btn_row.addWidget(self.sync_btn)

        self.send_btn = QPushButton()
        self.send_btn.setObjectName("enableBtn")
        self.send_btn.clicked.connect(self._emit_write)
        btn_row.addWidget(self.send_btn)
        btn_row.addStretch()
        ctrl_layout.addLayout(btn_row)

        self.ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(self.ctrl_group)
        layout.addStretch()

        self.set_connected(False)
        self.retranslate_ui()
        self._apply_status_style()

    def retranslate_ui(self):
        self.conn_group.setTitle(tr("rod.conn_group"))
        self.ctrl_group.setTitle(tr("rod.ctrl_group"))
        self.port_label.setText(tr("rod.port"))
        self.baud_label.setText(tr("rod.baud"))
        self.timeout_label.setText(tr("rod.timeout"))
        self.actual_label.setText(tr("rod.actual"))
        self.target_label.setText(tr("rod.target"))
        self.speed_label.setText(tr("rod.speed"))
        self.acc_label.setText(tr("rod.acc"))
        self.read_btn.setText(tr("rod.read"))
        self.sync_btn.setText(tr("rod.sync"))
        self.send_btn.setText(tr("rod.send"))
        self.connect_btn.setText(tr("rod.disconnect") if self._connected else tr("rod.connect"))
        if not self._connected:
            self.status_label.setText(tr("rod.disconnected"))

    def set_connected(self, connected: bool):
        self._connected = bool(connected)
        self.connect_btn.setText(tr("rod.disconnect") if connected else tr("rod.connect"))
        self.read_btn.setEnabled(connected)
        self.send_btn.setEnabled(connected)
        self.sync_btn.setEnabled(connected)
        self.port_edit.setEnabled(not connected)
        self.baud_spin.setEnabled(not connected)
        self.timeout_spin.setEnabled(not connected)
        self.status_label.setText(tr("rod.connected") if connected else tr("rod.disconnected"))
        self._apply_status_style()

    def update_angle(self, angle_deg: float):
        self.actual_value.setText(f"{angle_deg:.2f}°")

    def set_error(self, message: str):
        self.status_label.setText(tr("rod.error", msg=message))
        sc = self._scene()
        self.status_label.setStyleSheet(f"color: {sc['warning']}; font-weight: bold;")

    def _apply_status_style(self):
        sc = self._scene()
        color = sc["success"] if self._connected else sc["subtext"]
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.actual_value.setStyleSheet(f"color: {sc['accent']}; font-weight: bold;")

    def _toggle_connection(self):
        if self._connected:
            self.disconnect_requested.emit()
            return
        port = self.port_edit.text().strip() or RodMotorClient.DEFAULT_PORT
        self.connect_requested.emit(
            port,
            self.baud_spin.value(),
            self.timeout_spin.value(),
        )

    def _emit_write(self):
        self.write_requested.emit(
            self.angle_spin.value(),
            self.speed_spin.value(),
            self.acc_spin.value(),
        )

    def _sync_target_to_actual(self):
        text = self.actual_value.text().replace("°", "").strip()
        try:
            self.angle_spin.setValue(float(text))
        except ValueError:
            pass
