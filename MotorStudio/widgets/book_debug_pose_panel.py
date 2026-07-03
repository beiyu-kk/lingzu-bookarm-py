"""Book workflow debug pose settings panel."""

from __future__ import annotations

from typing import Sequence

from PyQt6.QtCore import pyqtSignal
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
from MotorStudio.widgets.realsense_point_panel import (
    BOOK_DEBUG_POSE_WORKFLOW_MODES,
    RealSensePointPanel,
    load_book_debug_pose_deg,
    save_book_debug_pose_deg,
)


class BookDebugPosePanel(QWidget):
    """Edit the shared book workflow debug MoveJ pose."""

    pose_saved = pyqtSignal(list)
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._spins: list[QDoubleSpinBox] = []
        self._status_label = None
        self._init_ui()
        self.retranslate_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.group = QGroupBox()
        grid = QGridLayout(self.group)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        defaults = load_book_debug_pose_deg(
            BOOK_DEBUG_POSE_WORKFLOW_MODES[0],
            RealSensePointPanel.DEFAULT_DEBUG_JOINTS_DEG,
        )
        for idx in range(6):
            label = QLabel(f"J{idx + 1}")
            grid.addWidget(label, idx // 3, (idx % 3) * 2)

            spin = QDoubleSpinBox()
            spin.setRange(-360.0, 360.0)
            spin.setDecimals(2)
            spin.setSingleStep(1.0)
            spin.setSuffix("°")
            spin.setValue(float(defaults[idx]))
            spin.setMinimumWidth(100)
            self._spins.append(spin)
            grid.addWidget(spin, idx // 3, (idx % 3) * 2 + 1)

        layout.addWidget(self.group)

        btn_row = QHBoxLayout()
        self.restore_btn = QPushButton()
        self.restore_btn.clicked.connect(self._restore_builtin_defaults)
        btn_row.addWidget(self.restore_btn)

        self.save_btn = QPushButton()
        self.save_btn.setObjectName("enableBtn")
        self.save_btn.clicked.connect(self._save_pose)
        btn_row.addWidget(self.save_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        layout.addStretch()

    def retranslate_ui(self):
        self.group.setTitle(tr("book_debug_pose.group"))
        self.restore_btn.setText(tr("book_debug_pose.restore"))
        self.save_btn.setText(tr("book_debug_pose.save"))
        if self._status_label is not None and not self._status_label.text():
            self._status_label.setText(tr("book_debug_pose.ready"))

    def current_pose_deg(self) -> list[float]:
        return [float(spin.value()) for spin in self._spins]

    def set_pose_deg(self, joints_deg: Sequence[float]):
        for spin, value in zip(self._spins, list(joints_deg)[:6]):
            spin.setValue(float(value))

    def _restore_builtin_defaults(self):
        self.set_pose_deg(RealSensePointPanel.DEFAULT_DEBUG_JOINTS_DEG)

    def _save_pose(self):
        pose = self.current_pose_deg()
        try:
            for mode in BOOK_DEBUG_POSE_WORKFLOW_MODES:
                save_book_debug_pose_deg(mode, pose)
        except Exception as exc:
            message = tr("book_debug_pose.save_failed", error=str(exc))
            self._status_label.setText(message)
            self.error_occurred.emit(message)
            return

        message = tr("book_debug_pose.saved", pose=", ".join(f"{v:.2f}" for v in pose))
        self._status_label.setText(message)
        self.log_message.emit(message)
        self.pose_saved.emit(pose)
