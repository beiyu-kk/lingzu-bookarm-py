"""Per-joint PD gain settings panel."""

from __future__ import annotations

import json
import os
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from MotorStudio.utils.i18n import tr


def _config_dir() -> Path:
    if os.name == "nt":
        base = os.getenv("APPDATA") or str(Path.home())
        return Path(base) / "MotorStudio"
    return Path.home() / ".config" / "el_a3_sdk"


PD_DEFAULTS_PATH = _config_dir() / "motorstudio_joint_pd_defaults.json"


class MotorPDPanel(QWidget):
    """Runtime PD gain settings for arm joints 1-6."""

    joint_pd_requested = pyqtSignal(int, float, float)
    all_joint_pd_requested = pyqtSignal(list)
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    KP_MIN = 0.0
    KP_MAX = 500.0
    KD_MIN = 0.0
    KD_MAX = 5.0
    RECOMMENDED_KP = 80.0
    RECOMMENDED_KD = 4.0
    URDF_KP = 100.0
    URDF_KD = 4.0
    SOFT_KP = 60.0
    SOFT_KD = 3.5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._kp_spins: dict[int, QDoubleSpinBox] = {}
        self._kd_spins: dict[int, QDoubleSpinBox] = {}
        self._status_label = None
        self._init_ui()
        self.retranslate_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.info_group = QGroupBox()
        info_layout = QVBoxLayout(self.info_group)
        self.range_label = QLabel()
        self.range_label.setWordWrap(True)
        info_layout.addWidget(self.range_label)
        layout.addWidget(self.info_group)

        self.table_group = QGroupBox()
        grid = QGridLayout(self.table_group)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self._headers = [QLabel(), QLabel(), QLabel(), QLabel(), QLabel()]
        for col, label in enumerate(self._headers):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("font-weight: bold;")
            grid.addWidget(label, 0, col)

        values = self._load_values()
        for row, motor_id in enumerate(range(1, 7), start=1):
            motor_label = QLabel(f"M{motor_id}")
            motor_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(motor_label, row, 0)

            kp_spin = self._make_spin(self.KP_MIN, self.KP_MAX, values[motor_id][0], 1.0, 1)
            kd_spin = self._make_spin(self.KD_MIN, self.KD_MAX, values[motor_id][1], 0.1, 2)
            self._kp_spins[motor_id] = kp_spin
            self._kd_spins[motor_id] = kd_spin
            grid.addWidget(kp_spin, row, 1)
            grid.addWidget(kd_spin, row, 2)

            rec_label = QLabel(f"{self.RECOMMENDED_KP:.0f} / {self.RECOMMENDED_KD:.1f}")
            rec_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(rec_label, row, 3)

            apply_btn = QPushButton()
            apply_btn.clicked.connect(lambda _checked=False, mid=motor_id: self._apply_one(mid))
            grid.addWidget(apply_btn, row, 4)
            setattr(self, f"_apply_btn_{motor_id}", apply_btn)

        grid.setColumnStretch(5, 1)
        layout.addWidget(self.table_group)

        preset_row = QHBoxLayout()
        self.recommended_btn = QPushButton()
        self.recommended_btn.clicked.connect(
            lambda: self._set_all_values(self.RECOMMENDED_KP, self.RECOMMENDED_KD)
        )
        preset_row.addWidget(self.recommended_btn)

        self.soft_btn = QPushButton()
        self.soft_btn.clicked.connect(lambda: self._set_all_values(self.SOFT_KP, self.SOFT_KD))
        preset_row.addWidget(self.soft_btn)

        self.urdf_btn = QPushButton()
        self.urdf_btn.clicked.connect(lambda: self._set_all_values(self.URDF_KP, self.URDF_KD))
        preset_row.addWidget(self.urdf_btn)

        self.apply_all_btn = QPushButton()
        self.apply_all_btn.setObjectName("enableBtn")
        self.apply_all_btn.clicked.connect(self._apply_all)
        preset_row.addWidget(self.apply_all_btn)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        layout.addStretch()

    @staticmethod
    def _make_spin(minimum: float, maximum: float, value: float, step: float, decimals: int):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(float(value))
        spin.setMinimumWidth(100)
        spin.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        return spin

    def retranslate_ui(self):
        self.info_group.setTitle(tr("pd.info_group"))
        self.range_label.setText(
            tr(
                "pd.range_text",
                kp_min=self.KP_MIN,
                kp_max=self.KP_MAX,
                kd_min=self.KD_MIN,
                kd_max=self.KD_MAX,
                rec_kp=self.RECOMMENDED_KP,
                rec_kd=self.RECOMMENDED_KD,
            )
        )
        self.table_group.setTitle(tr("pd.table_group"))
        for label, text in zip(
            self._headers,
            (tr("pd.motor"), "Kp", "Kd", tr("pd.recommended"), tr("pd.action")),
        ):
            label.setText(text)
        for motor_id in range(1, 7):
            getattr(self, f"_apply_btn_{motor_id}").setText(tr("pd.apply_one"))
        self.recommended_btn.setText(tr("pd.use_recommended"))
        self.soft_btn.setText(tr("pd.use_soft"))
        self.urdf_btn.setText(tr("pd.use_urdf"))
        self.apply_all_btn.setText(tr("pd.apply_all"))
        if self._status_label is not None and not self._status_label.text():
            self._status_label.setText(tr("pd.ready"))

    def _load_values(self) -> dict[int, tuple[float, float]]:
        values = {
            motor_id: (self.RECOMMENDED_KP, self.RECOMMENDED_KD)
            for motor_id in range(1, 7)
        }
        try:
            with PD_DEFAULTS_PATH.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception:
            return values
        if not isinstance(payload, dict):
            return values
        for motor_id in range(1, 7):
            item = payload.get(str(motor_id), {})
            if not isinstance(item, dict):
                continue
            try:
                values[motor_id] = (
                    self._clamp(float(item.get("kp", values[motor_id][0])), self.KP_MIN, self.KP_MAX),
                    self._clamp(float(item.get("kd", values[motor_id][1])), self.KD_MIN, self.KD_MAX),
                )
            except (TypeError, ValueError):
                pass
        return values

    def _save_values(self):
        payload = {
            str(motor_id): {
                "kp": float(self._kp_spins[motor_id].value()),
                "kd": float(self._kd_spins[motor_id].value()),
            }
            for motor_id in range(1, 7)
        }
        PD_DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PD_DEFAULTS_PATH.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, float(value)))

    def _set_all_values(self, kp: float, kd: float):
        for motor_id in range(1, 7):
            self._kp_spins[motor_id].setValue(float(kp))
            self._kd_spins[motor_id].setValue(float(kd))

    def _current_values(self) -> list[tuple[int, float, float]]:
        return [
            (
                motor_id,
                float(self._kp_spins[motor_id].value()),
                float(self._kd_spins[motor_id].value()),
            )
            for motor_id in range(1, 7)
        ]

    def _apply_one(self, motor_id: int):
        kp = float(self._kp_spins[motor_id].value())
        kd = float(self._kd_spins[motor_id].value())
        try:
            self._save_values()
        except Exception as exc:
            message = tr("pd.save_failed", error=str(exc))
            self._status_label.setText(message)
            self.error_occurred.emit(message)
            return
        self.joint_pd_requested.emit(int(motor_id), kp, kd)
        message = tr("pd.applied_one", motor=motor_id, kp=kp, kd=kd)
        self._status_label.setText(message)
        self.log_message.emit(message)

    def _apply_all(self):
        values = self._current_values()
        try:
            self._save_values()
        except Exception as exc:
            message = tr("pd.save_failed", error=str(exc))
            self._status_label.setText(message)
            self.error_occurred.emit(message)
            return
        self.all_joint_pd_requested.emit(values)
        message = tr("pd.applied_all")
        self._status_label.setText(message)
        self.log_message.emit(message)
