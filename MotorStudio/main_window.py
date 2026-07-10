"""主窗口：QDockWidget 布局 + 信号连接"""

import time
import logging
from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QStackedWidget,
    QWidget, QTextEdit, QPushButton, QGridLayout, QVBoxLayout,
    QButtonGroup, QApplication,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from MotorStudio.backend.arm_worker import ArmWorker
from MotorStudio.backend.lift_platform_worker import LiftPlatformWorker
from MotorStudio.backend.rodmotor_worker import RodMotorWorker
from MotorStudio.widgets.toolbar_panel import ToolbarPanel
from MotorStudio.widgets.joint_control_panel import JointControlPanel
from MotorStudio.widgets.motor_pd_panel import MotorPDPanel
from MotorStudio.widgets.monitoring_window import MonitoringWindow
from MotorStudio.widgets.trajectory_panel import TrajectoryPanel
from MotorStudio.widgets.tcp_panel import TcpPanel
from MotorStudio.widgets.teaching_panel import TeachingPanel
from MotorStudio.widgets.diagnostics_panel import DiagnosticsPanel
from MotorStudio.widgets.book_debug_pose_panel import BookDebugPosePanel
from MotorStudio.widgets.book_gripper_panel import BookGripperPanel
from MotorStudio.widgets.gripper_panel import GripperPanel
from MotorStudio.widgets.lift_platform_panel import LiftPlatformPanel
from MotorStudio.widgets.rodmotor_panel import RodMotorPanel
from MotorStudio.widgets.gamepad_panel import GamepadPanel
from MotorStudio.widgets.realsense_point_panel import RealSensePointPanel
from MotorStudio.widgets.viewer_3d import Viewer3DPanel
from MotorStudio.utils.i18n import tr
from MotorStudio.utils.lift_platform_defaults import load_lift_platform_defaults
from MotorStudio.utils.theme_manager import ThemeManager

logger = logging.getLogger("MotorStudio")


class MultiRowPanelTabs(QWidget):
    """Grouped navigation for the right-side function panels."""

    currentChanged = pyqtSignal(int)
    categoryChanged = pyqtSignal(str)
    LOW_LEVEL_CATEGORY = "low_level"
    LIBRARY_CATEGORY = "library"

    def __init__(self, parent=None, columns: int = 5):
        super().__init__(parent)
        self._columns = max(1, int(columns))
        self._category_columns: dict[str, int] = {}
        self._buttons: list[QPushButton] = []
        self._pages: list[QWidget] = []
        self._categories: list[str] = []
        self._active_category = self.LOW_LEVEL_CATEGORY
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._nav_widget = QWidget(self)
        self._nav_layout = QGridLayout(self._nav_widget)
        self._nav_layout.setContentsMargins(4, 4, 4, 0)
        self._nav_layout.setHorizontalSpacing(4)
        self._nav_layout.setVerticalSpacing(4)
        layout.addWidget(self._nav_widget)

        self._stack = QStackedWidget(self)
        layout.addWidget(self._stack, 1)

    def addTab(
        self,
        widget: QWidget,
        label: str,
        category: str = LOW_LEVEL_CATEGORY,
    ) -> int:
        index = len(self._pages)
        self._pages.append(widget)
        self._categories.append(self._normalize_category(category))
        self._stack.addWidget(widget)

        button = QPushButton(label, self)
        button.setCheckable(True)
        button.setMinimumHeight(30)
        button.setObjectName("panelTabButton")
        self._buttons.append(button)
        self._button_group.addButton(button, index)
        button.clicked.connect(lambda _checked=False, i=index: self.setCurrentIndex(i))
        self._rebuild_nav_layout()

        if index == 0:
            button.setChecked(True)
            self._stack.setCurrentIndex(0)
        return index

    def indexOf(self, widget: QWidget) -> int:
        try:
            return self._pages.index(widget)
        except ValueError:
            return -1

    def currentIndex(self) -> int:
        return self._stack.currentIndex()

    def currentCategory(self) -> str:
        return self._active_category

    def setCurrentIndex(self, index: int):
        if index < 0 or index >= len(self._pages):
            return
        category = self._categories[index]
        if category != self._active_category:
            self._active_category = category
            self._rebuild_nav_layout()
            self.categoryChanged.emit(category)
        if index == self._stack.currentIndex():
            self._buttons[index].setChecked(True)
            return
        self._stack.setCurrentIndex(index)
        self._buttons[index].setChecked(True)
        self.currentChanged.emit(index)

    def setTabText(self, index: int, text: str):
        if 0 <= index < len(self._buttons):
            self._buttons[index].setText(text)

    def setCategoryColumns(self, category: str, columns: int):
        self._category_columns[self._normalize_category(category)] = max(1, int(columns))
        self._rebuild_nav_layout()

    def setCategory(self, category: str):
        category = self._normalize_category(category)
        changed = category != self._active_category
        if changed:
            self._active_category = category
            self._rebuild_nav_layout()
            self.categoryChanged.emit(category)
        current = self.currentIndex()
        if current < 0 or self._categories[current] != category:
            first_index = self._first_index_for_category(category)
            if first_index >= 0:
                self.setCurrentIndex(first_index)

    def _normalize_category(self, category: str) -> str:
        if category == self.LIBRARY_CATEGORY:
            return self.LIBRARY_CATEGORY
        return self.LOW_LEVEL_CATEGORY

    def _first_index_for_category(self, category: str) -> int:
        for index, item_category in enumerate(self._categories):
            if item_category == category:
                return index
        return -1

    def _rebuild_nav_layout(self):
        while self._nav_layout.count():
            self._nav_layout.takeAt(0)
        visible_index = 0
        columns = self._category_columns.get(self._active_category, self._columns)
        for index, button in enumerate(self._buttons):
            visible = self._categories[index] == self._active_category
            button.setVisible(visible)
            if not visible:
                continue
            row = visible_index // columns
            col = visible_index % columns
            self._nav_layout.addWidget(button, row, col)
            visible_index += 1
        for c in range(max(self._columns, columns)):
            self._nav_layout.setColumnStretch(c, 1)


class MainWindow(QMainWindow):
    """EL-A3 调试上位机主窗口"""

    UI_UPDATE_INTERVAL_S = 0.05  # 20 Hz UI refresh cap

    def __init__(self, urdf_path=None, mesh_dir=None, sim_mode=False):
        super().__init__()
        self.setWindowTitle(tr("win.title"))
        self.setMinimumSize(1280, 800)
        self.resize(1600, 960)

        self._urdf_path = urdf_path
        self._mesh_dir = mesh_dir
        self._sim_mode = sim_mode
        self._last_joint_states = None
        self._last_effort_states = None
        self._last_ui_update_time = 0.0

        self._init_worker()
        self._init_ui()
        self._connect_signals()

        tm = ThemeManager.instance()
        tm.language_changed.connect(lambda _: self._retranslate_ui())
        tm.theme_changed.connect(lambda _: self.viewer_3d.apply_theme())
        tm.theme_changed.connect(
            lambda _: self.monitoring_window.panel.apply_theme()
        )
        tm.theme_changed.connect(lambda _: self.diagnostics_panel.retranslate_ui())
        tm.theme_changed.connect(lambda _: self.gripper_panel.retranslate_ui())
        tm.theme_changed.connect(lambda _: self.lift_platform_panel.retranslate_ui())
        tm.theme_changed.connect(lambda _: self.rodmotor_panel.retranslate_ui())
        tm.theme_changed.connect(lambda _: self.gamepad_panel.retranslate_ui())
        tm.theme_changed.connect(lambda _: self.tcp_panel.retranslate_ui())
        tm.theme_changed.connect(lambda _: self.book_takeout_panel.apply_theme())
        tm.theme_changed.connect(lambda _: self.book_putback_panel.apply_theme())
        tm.theme_changed.connect(lambda _: self.book_putback2_panel.apply_theme())
        tm.theme_changed.connect(lambda _: self.book_tail_putback_panel.apply_theme())
        tm.theme_changed.connect(lambda _: self.image_recognition_panel.apply_theme())

        QTimer.singleShot(500, self._init_3d_model)
        if self._sim_mode:
            QTimer.singleShot(100, self._start_sim_mode)

    def _init_worker(self):
        self.worker = ArmWorker()
        self.worker.start()
        self.rodmotor_worker = RodMotorWorker()
        self.rodmotor_worker.start()
        self.lift_platform_worker = LiftPlatformWorker()
        self.lift_platform_worker.start()

    def _init_ui(self):
        # --- 顶部工具栏（单行固定高度） ---
        toolbar_widget = ToolbarPanel()
        self.toolbar = toolbar_widget
        self.toolbar_dock = QDockWidget(tr("win.toolbar"), self)
        self.toolbar_dock.setWidget(toolbar_widget)
        self.toolbar_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        empty_title = QWidget()
        empty_title.setFixedHeight(0)
        self.toolbar_dock.setTitleBarWidget(empty_title)
        self.toolbar_dock.setStyleSheet("QDockWidget { border: none; }")
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self.toolbar_dock)

        # --- 左侧：3D 可视化 / 点云视图 ---
        self.left_stack = QStackedWidget()
        self.viewer_3d = Viewer3DPanel(
            urdf_path=self._urdf_path,
            mesh_dir=self._mesh_dir,
        )
        self.left_stack.addWidget(self.viewer_3d)
        self.viewer_dock = QDockWidget(tr("win.viewer"), self)
        self.viewer_dock.setWidget(self.left_stack)
        self.viewer_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        _hide = QWidget(); _hide.setFixedHeight(0)
        self.viewer_dock.setTitleBarWidget(_hide)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.viewer_dock)

        # --- 右侧：功能面板导航 ---
        self.tabs = MultiRowPanelTabs(columns=5)
        self.tabs.setCategoryColumns(MultiRowPanelTabs.LIBRARY_CATEGORY, 4)

        self.joint_panel = JointControlPanel()
        self.tabs.addTab(self.joint_panel, tr("tab.joint"))

        self.motor_pd_panel = MotorPDPanel()
        self.tabs.addTab(self.motor_pd_panel, tr("tab.motor_pd"))

        self.trajectory_panel = TrajectoryPanel()
        self.tabs.addTab(self.trajectory_panel, tr("tab.trajectory"))

        self.tcp_panel = TcpPanel()
        self.tabs.addTab(self.tcp_panel, tr("tab.tcp"))
        self.viewer_3d.set_tcp_offset(self.tcp_panel.get_tcp_offset())

        self.lift_platform_panel = LiftPlatformPanel()
        self.tabs.addTab(
            self.lift_platform_panel,
            tr("tab.lift_platform"),
            category=MultiRowPanelTabs.LIBRARY_CATEGORY,
        )

        self.book_gripper_panel = BookGripperPanel()
        self.tabs.addTab(
            self.book_gripper_panel,
            tr("tab.book_gripper"),
            category=MultiRowPanelTabs.LIBRARY_CATEGORY,
        )

        self.book_debug_pose_panel = BookDebugPosePanel()
        self.tabs.addTab(
            self.book_debug_pose_panel,
            tr("tab.book_debug_pose"),
            category=MultiRowPanelTabs.LIBRARY_CATEGORY,
        )

        self.image_recognition_panel = RealSensePointPanel(workflow_mode="image_recognition")
        self.tabs.addTab(
            self.image_recognition_panel,
            tr("tab.image_recognition"),
            category=MultiRowPanelTabs.LIBRARY_CATEGORY,
        )
        self.image_recognition_viewer = self.image_recognition_panel.viewer_widget()
        self.left_stack.addWidget(self.image_recognition_viewer)

        self.book_takeout_panel = RealSensePointPanel(workflow_mode="takeout")
        self.tabs.addTab(
            self.book_takeout_panel,
            tr("tab.book_takeout"),
            category=MultiRowPanelTabs.LIBRARY_CATEGORY,
        )
        self.book_takeout_viewer = self.book_takeout_panel.viewer_widget()
        self.left_stack.addWidget(self.book_takeout_viewer)

        self.book_putback_panel = RealSensePointPanel(workflow_mode="putback")
        self.tabs.addTab(
            self.book_putback_panel,
            tr("tab.book_putback"),
            category=MultiRowPanelTabs.LIBRARY_CATEGORY,
        )
        self.book_putback_viewer = self.book_putback_panel.viewer_widget()
        self.left_stack.addWidget(self.book_putback_viewer)

        self.book_putback2_panel = RealSensePointPanel(workflow_mode="putback2")
        self.tabs.addTab(
            self.book_putback2_panel,
            tr("tab.book_putback2"),
            category=MultiRowPanelTabs.LIBRARY_CATEGORY,
        )
        self.book_putback2_viewer = self.book_putback2_panel.viewer_widget()
        self.left_stack.addWidget(self.book_putback2_viewer)

        self.book_tail_putback_panel = RealSensePointPanel(workflow_mode="tail_putback")
        self.tabs.addTab(
            self.book_tail_putback_panel,
            tr("tab.book_tail_putback"),
            category=MultiRowPanelTabs.LIBRARY_CATEGORY,
        )
        self.book_tail_putback_viewer = self.book_tail_putback_panel.viewer_widget()
        self.left_stack.addWidget(self.book_tail_putback_viewer)

        self.teaching_panel = TeachingPanel()
        self.tabs.addTab(self.teaching_panel, tr("tab.teaching"))

        self.diagnostics_panel = DiagnosticsPanel()
        self.tabs.addTab(self.diagnostics_panel, tr("tab.diagnostics"))

        self.gripper_panel = GripperPanel()
        self.tabs.addTab(self.gripper_panel, tr("tab.gripper"))

        self.rodmotor_panel = RodMotorPanel()
        self.tabs.addTab(self.rodmotor_panel, tr("tab.rodmotor"))

        self.gamepad_panel = GamepadPanel()
        self.tabs.addTab(self.gamepad_panel, tr("tab.gamepad"))

        self.tabs_dock = QDockWidget(tr("win.panels"), self)
        self.tabs_dock.setWidget(self.tabs)
        self.tabs_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        _hide2 = QWidget(); _hide2.setFixedHeight(0)
        self.tabs_dock.setTitleBarWidget(_hide2)
        self.tabs_dock.setMinimumWidth(440)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.tabs_dock)

        # --- 底部日志 ---
        self.log_console = QTextEdit()
        self.log_console.setObjectName("logConsole")
        self.log_console.setReadOnly(True)
        self.log_console.setFixedHeight(100)
        self.log_dock = QDockWidget(tr("win.log"), self)
        self.log_dock.setWidget(self.log_console)
        self.log_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        _hide3 = QWidget(); _hide3.setFixedHeight(0)
        self.log_dock.setTitleBarWidget(_hide3)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)

        # --- 实时监控弹出窗口（按需打开） ---
        self.monitoring_window = MonitoringWindow(self.worker.data_buffer, parent=self)

        self.statusBar().showMessage(tr("win.ready"))

        self._book_takeout_tab_index = self.tabs.indexOf(self.book_takeout_panel)
        self._book_putback_tab_index = self.tabs.indexOf(self.book_putback_panel)
        self._book_putback2_tab_index = self.tabs.indexOf(self.book_putback2_panel)
        self._book_tail_putback_tab_index = self.tabs.indexOf(self.book_tail_putback_panel)
        self._image_recognition_tab_index = self.tabs.indexOf(self.image_recognition_panel)
        self._sync_book_header_zero_duration(
            self.joint_panel.zero_duration_spin.value()
        )
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs.categoryChanged.connect(
            lambda category: self.toolbar.set_panel_category(category, emit=False)
        )
        self.tabs.setCategory(MultiRowPanelTabs.LIBRARY_CATEGORY)
        self.toolbar.set_panel_category(self.tabs.currentCategory(), emit=False)
        self._sync_left_view_for_tab(self.tabs.currentIndex())
        QTimer.singleShot(0, lambda: self._adjust_dock_sizes(self.viewer_dock, self.tabs_dock))

    def _connect_signals(self):
        tb = self.toolbar
        tb.connect_requested.connect(self._on_connect)
        tb.disconnect_requested.connect(lambda: self.worker.submit_command("disconnect"))
        tb.enable_requested.connect(lambda: self.worker.submit_command("enable"))
        tb.disable_requested.connect(lambda: self.worker.submit_command("disable"))
        tb.panel_category_requested.connect(self.tabs.setCategory)
        tb.emergency_stop_requested.connect(
            lambda: self.worker.submit_command("emergency_stop")
        )
        tb.open_monitor_requested.connect(self._open_monitoring)

        self.worker.connected_changed.connect(tb.set_connected)
        self.worker.enabled_changed.connect(tb.set_enabled)
        self.worker.enabled_changed.connect(self.joint_panel.set_enabled)
        self.worker.enabled_changed.connect(self.viewer_3d.set_enabled)
        self.worker.enabled_changed.connect(self.book_takeout_panel.set_arm_enabled)
        self.worker.enabled_changed.connect(self.book_putback_panel.set_arm_enabled)
        self.worker.enabled_changed.connect(self.book_putback2_panel.set_arm_enabled)
        self.worker.enabled_changed.connect(self.book_tail_putback_panel.set_arm_enabled)
        self.worker.enabled_changed.connect(self.image_recognition_panel.set_arm_enabled)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.log_message.connect(self._append_log)
        self.worker.can_fps_updated.connect(tb.set_fps)
        self.worker.can_fps_updated.connect(self.diagnostics_panel.update_can_stats)

        self.worker.joints_updated.connect(self._on_joints_updated)
        self.worker.efforts_updated.connect(self._on_efforts_updated)
        self.worker.end_pose_updated.connect(self.trajectory_panel.update_current_end_pose)
        self.worker.end_pose_updated.connect(self.book_takeout_panel.update_current_end_pose)
        self.worker.end_pose_updated.connect(self.book_putback_panel.update_current_end_pose)
        self.worker.end_pose_updated.connect(self.book_putback2_panel.update_current_end_pose)
        self.worker.end_pose_updated.connect(self.book_tail_putback_panel.update_current_end_pose)
        self.worker.end_pose_updated.connect(self.image_recognition_panel.update_current_end_pose)
        self.worker.end_pose_updated.connect(self.viewer_3d.update_tcp_point)
        self.worker.tcp_offset_updated.connect(self.tcp_panel.set_tcp_offset)
        self.worker.tcp_offset_updated.connect(self.viewer_3d.set_tcp_offset)
        self.worker.motor_feedback_updated.connect(
            self.diagnostics_panel.update_motor_feedback
        )

        self.joint_panel.joint_command.connect(
            lambda pos: self.worker.submit_command("joint_ctrl", pos)
        )
        self.joint_panel.go_zero_requested.connect(
            lambda pos, dur: self.worker.submit_command("move_j", pos, dur)
        )
        self.joint_panel.zero_duration_spin.valueChanged.connect(
            self._sync_book_header_zero_duration
        )
        self.motor_pd_panel.joint_pd_requested.connect(
            lambda motor_id, kp, kd: self.worker.submit_command("set_joint_pd", motor_id, kp, kd)
        )
        self.motor_pd_panel.all_joint_pd_requested.connect(
            lambda values: self.worker.submit_command("set_all_joint_pd", values)
        )
        self.motor_pd_panel.log_message.connect(self._append_log)
        self.motor_pd_panel.error_occurred.connect(self._on_error)

        tp = self.trajectory_panel
        tp.move_j_requested.connect(
            lambda pos, dur: self.worker.submit_command("move_j", pos, dur)
        )
        tp.move_l_requested.connect(
            lambda pose, dur: self.worker.submit_command("move_l", pose, dur)
        )
        tp.cancel_requested.connect(
            lambda: self.worker.submit_command("cancel_motion")
        )

        tcp = self.tcp_panel
        tcp.tcp_apply_requested.connect(
            lambda offset: self.worker.submit_command("set_tcp_offset", offset)
        )
        tcp.tcp_apply_requested.connect(self.viewer_3d.set_tcp_offset)
        tcp.tcp_apply_requested.connect(self.book_takeout_panel.set_tcp_offset)
        tcp.tcp_apply_requested.connect(self.book_putback_panel.set_tcp_offset)
        tcp.tcp_apply_requested.connect(self.book_putback2_panel.set_tcp_offset)
        tcp.tcp_apply_requested.connect(self.book_tail_putback_panel.set_tcp_offset)
        tcp.tcp_apply_requested.connect(self.image_recognition_panel.set_tcp_offset)
        tcp.tcp_save_requested.connect(
            lambda offset: self.worker.submit_command("save_tcp_offset", offset)
        )
        tcp.tcp_save_requested.connect(self.viewer_3d.set_tcp_offset)
        tcp.tcp_save_requested.connect(self.book_takeout_panel.set_tcp_offset)
        tcp.tcp_save_requested.connect(self.book_putback_panel.set_tcp_offset)
        tcp.tcp_save_requested.connect(self.book_putback2_panel.set_tcp_offset)
        tcp.tcp_save_requested.connect(self.book_tail_putback_panel.set_tcp_offset)
        tcp.tcp_save_requested.connect(self.image_recognition_panel.set_tcp_offset)
        tcp.tcp_restore_requested.connect(
            lambda: self.worker.submit_command("restore_tcp_offset")
        )
        tcp.tcp_restore_requested.connect(
            lambda: self.viewer_3d.set_tcp_offset([0.0] * 6)
        )
        tcp.tcp_restore_requested.connect(
            lambda: self.book_takeout_panel.set_tcp_offset([0.0] * 6)
        )
        tcp.tcp_restore_requested.connect(
            lambda: self.book_putback_panel.set_tcp_offset([0.0] * 6)
        )
        tcp.tcp_restore_requested.connect(
            lambda: self.book_putback2_panel.set_tcp_offset([0.0] * 6)
        )
        tcp.tcp_restore_requested.connect(
            lambda: self.book_tail_putback_panel.set_tcp_offset([0.0] * 6)
        )
        tcp.tcp_restore_requested.connect(
            lambda: self.image_recognition_panel.set_tcp_offset([0.0] * 6)
        )
        self.worker.tcp_offset_updated.connect(self.book_takeout_panel.set_tcp_offset)
        self.worker.tcp_offset_updated.connect(self.book_putback_panel.set_tcp_offset)
        self.worker.tcp_offset_updated.connect(self.book_putback2_panel.set_tcp_offset)
        self.worker.tcp_offset_updated.connect(self.book_tail_putback_panel.set_tcp_offset)
        self.worker.tcp_offset_updated.connect(self.image_recognition_panel.set_tcp_offset)

        self.book_debug_pose_panel.pose_saved.connect(self._on_book_debug_pose_saved)
        self.book_debug_pose_panel.log_message.connect(self._append_log)
        self.book_debug_pose_panel.error_occurred.connect(self._on_error)
        self.book_gripper_panel.settings_saved.connect(self._on_book_gripper_saved)
        self.book_gripper_panel.gripper_test_requested.connect(
            lambda params: self.worker.submit_command("gripper_close_monitor", **params)
        )
        self.book_gripper_panel.move_j_requested.connect(
            lambda joints, duration: self.worker.submit_command("move_j", joints, duration)
        )
        self.worker.gripper_close_done.connect(self.book_gripper_panel.notify_gripper_test_done)
        self.worker.move_j_done.connect(self.book_gripper_panel.notify_move_done)
        self.book_gripper_panel.log_message.connect(self._append_log)
        self.book_gripper_panel.error_occurred.connect(self._on_error)

        self._connect_point_workflow_panel(self.book_takeout_panel)
        self._connect_point_workflow_panel(self.book_putback_panel)
        self._connect_point_workflow_panel(self.book_putback2_panel)
        self._connect_point_workflow_panel(self.book_tail_putback_panel)
        self._connect_point_workflow_panel(self.image_recognition_panel)

        teach = self.teaching_panel
        teach.zero_torque_requested.connect(
            lambda en: self.worker.submit_command("zero_torque", en)
        )
        teach.zero_torque_gravity_requested.connect(
            lambda en: self.worker.submit_command("zero_torque_gravity", en)
        )
        teach.move_j_requested.connect(
            lambda pos, dur: self.worker.submit_command("move_j", pos, dur)
        )
        teach.trajectory_playback_requested.connect(
            lambda traj: self.worker.submit_command("play_recorded_trajectory", traj)
        )

        gp = self.gripper_panel
        gp.gripper_command.connect(
            lambda angle, effort=0.0, kp=0.0, kd=0.0: self.worker.submit_command("gripper_ctrl", angle, effort, kp, kd)
        )
        gp.set_zero_requested.connect(
            lambda: self.worker.submit_command("set_zero_position", 7)
        )

        lift = self.lift_platform_panel
        lift.connect_requested.connect(
            lambda port, baud, slave, speed, acc, timeout: self.lift_platform_worker.submit_command(
                "connect", port, baud, slave, speed, acc, timeout
            )
        )
        lift.disconnect_requested.connect(
            lambda: self.lift_platform_worker.submit_command("disconnect")
        )
        lift.move_distance_requested.connect(
            lambda distance, speed, acc, pulses_per_cm, up_direction: self.lift_platform_worker.submit_command(
                "move_distance", distance, speed, acc, pulses_per_cm, up_direction
            )
        )
        lift.move_pulses_requested.connect(
            lambda pulses: self.lift_platform_worker.submit_command("move_pulses", pulses)
        )
        lift.stop_requested.connect(
            lambda: self.lift_platform_worker.submit_command("stop")
        )
        self.lift_platform_worker.connected_changed.connect(lift.set_connected)
        self.lift_platform_worker.error_occurred.connect(lift.set_error)
        self.lift_platform_worker.error_occurred.connect(self._on_error)
        self.lift_platform_worker.log_message.connect(self._append_log)

        rod = self.rodmotor_panel
        rod.connect_requested.connect(
            lambda port, baud, timeout: self.rodmotor_worker.submit_command(
                "connect", port, baud, timeout
            )
        )
        rod.disconnect_requested.connect(
            lambda: self.rodmotor_worker.submit_command("disconnect")
        )
        rod.read_requested.connect(
            lambda: self.rodmotor_worker.submit_command("read_angle")
        )
        rod.write_requested.connect(
            lambda angle, spd, acc: self.rodmotor_worker.submit_command(
                "write_angle", angle, spd, acc
            )
        )
        self.rodmotor_worker.connected_changed.connect(rod.set_connected)
        self.rodmotor_worker.angle_updated.connect(rod.update_angle)
        self.rodmotor_worker.error_occurred.connect(rod.set_error)
        self.rodmotor_worker.error_occurred.connect(self._on_error)
        self.rodmotor_worker.log_message.connect(self._append_log)

        diag = self.diagnostics_panel
        diag.read_param_requested.connect(
            lambda mid, pidx: self.worker.submit_command("read_motor_param", mid, pidx)
        )
        diag.write_param_requested.connect(
            lambda mid, pidx, val: self.worker.submit_command(
                "write_motor_param", mid, pidx, val
            )
        )
        diag.set_zero_requested.connect(
            lambda m: self.worker.submit_command("set_zero_position", m)
        )
        diag.verify_zero_sta_requested.connect(
            lambda: self.worker.submit_command("verify_zero_sta")
        )
        diag.set_all_zero_sta_requested.connect(
            lambda: self.worker.submit_command("set_all_zero_sta")
        )
        self.worker.zero_sta_verified.connect(diag.update_zero_sta_result)
        diag.scan_motors_requested.connect(
            lambda: self.worker.submit_command("scan_motors")
        )
        self.worker.motor_scan_result.connect(diag.update_scan_result)

        tp.drag_mode_toggled.connect(self.viewer_3d.set_drag_mode)
        tp.sync_feedback_requested.connect(self.viewer_3d.sync_to_feedback)
        self.viewer_3d.drag_angles_changed.connect(tp.update_drag_angles)

        self.viewer_3d.home_position_requested.connect(
            lambda: self.worker.submit_command(
                "move_j",
                [0.0] * 6,
                self.joint_panel.zero_duration_spin.value(),
            )
        )

        # --- Calibration panel ---
        calib = self.teaching_panel.calibration_panel
        calib.move_j_requested.connect(
            lambda pos, dur, block: self.worker.submit_command(
                "move_j_block" if block else "move_j", pos, dur
            )
        )
        self.worker.move_j_done.connect(calib.notify_move_done)

        self.gamepad_panel.gamepad_log.connect(self._append_log)
        self.worker.connected_changed.connect(self._on_connected_for_gamepad)

    def _connect_point_workflow_panel(self, panel: RealSensePointPanel):
        panel.move_l_requested.connect(
            lambda pose, dur: self.worker.submit_command("move_l", pose, dur)
        )
        panel.move_l_block_requested.connect(
            lambda pose, dur: self.worker.submit_command("move_l_block", pose, dur)
        )
        panel.move_j_block_requested.connect(
            lambda joints, dur: self.worker.submit_command("move_j_block", joints, dur)
        )
        panel.end_pose_block_requested.connect(
            lambda pose, dur: self.worker.submit_command("end_pose_block", *pose, duration=dur)
        )
        panel.gripper_requested.connect(
            lambda angle, effort, kp, kd: self.worker.submit_command(
                "gripper_ctrl", angle, effort, kp, kd
            )
        )
        panel.gripper_close_monitor_requested.connect(
            lambda params: self.worker.submit_command("gripper_close_monitor", **params)
        )
        panel.rod_connect_requested.connect(
            lambda port, baud, timeout: self.rodmotor_worker.submit_command(
                "connect", port, baud, timeout
            )
        )
        panel.rod_write_requested.connect(
            lambda angle, spd, acc, torque: self.rodmotor_worker.submit_command(
                "write_angle", angle, spd, acc, torque
            )
        )
        panel.lift_move_distance_requested.connect(
            self._submit_workflow_lift_move_distance
        )
        panel.workflow_stop_requested.connect(self._submit_workflow_stop)
        self.worker.move_l_done.connect(panel.notify_move_l_done)
        self.worker.move_j_done.connect(panel.notify_move_j_done)
        self.worker.end_pose_done.connect(panel.notify_end_pose_done)
        self.worker.gripper_close_done.connect(panel.notify_gripper_close_done)
        self.rodmotor_worker.connected_changed.connect(panel.set_rod_connected)
        self.rodmotor_worker.angle_updated.connect(panel.notify_rod_angle_updated)
        self.rodmotor_worker.write_done.connect(panel.notify_rod_write_done)
        self.lift_platform_worker.move_done.connect(panel.notify_lift_move_done)
        self.worker.error_occurred.connect(panel.notify_flow_error)
        self.rodmotor_worker.error_occurred.connect(panel.notify_flow_error)
        self.lift_platform_worker.error_occurred.connect(panel.notify_flow_error)
        panel.log_message.connect(self._append_log)
        panel.error_occurred.connect(self._on_error)

    def _submit_workflow_stop(self):
        self.worker.submit_command("cancel_motion")
        self.lift_platform_worker.submit_command("stop")

    def _submit_workflow_lift_move_distance(
        self,
        distance: float,
        speed: int,
        acc: int,
        pulses_per_cm: float,
        up_direction: int,
    ):
        if not self.lift_platform_worker.is_connected:
            settings = load_lift_platform_defaults()
            self.lift_platform_worker.submit_command(
                "connect",
                str(settings["port"]),
                int(settings["baudrate"]),
                int(settings["slave_id"]),
                int(settings["speed_rpm"]),
                int(settings["acceleration"]),
                float(settings["timeout"]),
            )
            self._append_log("升降台未连接，已按默认参数自动连接")
        self.lift_platform_worker.submit_command(
            "move_distance",
            distance,
            speed,
            acc,
            pulses_per_cm,
            up_direction,
        )

    # ---- retranslate ----

    def _retranslate_ui(self):
        self.setWindowTitle(tr("win.title"))
        self.toolbar_dock.setWindowTitle(tr("win.toolbar"))
        self._sync_left_view_for_tab(self.tabs.currentIndex())
        self.tabs_dock.setWindowTitle(tr("win.panels"))
        self.log_dock.setWindowTitle(tr("win.log"))
        self.statusBar().showMessage(tr("win.ready"))

        for panel, key in (
            (self.joint_panel, "tab.joint"),
            (self.motor_pd_panel, "tab.motor_pd"),
            (self.trajectory_panel, "tab.trajectory"),
            (self.tcp_panel, "tab.tcp"),
            (self.lift_platform_panel, "tab.lift_platform"),
            (self.book_gripper_panel, "tab.book_gripper"),
            (self.book_debug_pose_panel, "tab.book_debug_pose"),
            (self.image_recognition_panel, "tab.image_recognition"),
            (self.book_takeout_panel, "tab.book_takeout"),
            (self.book_putback_panel, "tab.book_putback"),
            (self.book_putback2_panel, "tab.book_putback2"),
            (self.book_tail_putback_panel, "tab.book_tail_putback"),
            (self.teaching_panel, "tab.teaching"),
            (self.diagnostics_panel, "tab.diagnostics"),
            (self.gripper_panel, "tab.gripper"),
            (self.rodmotor_panel, "tab.rodmotor"),
            (self.gamepad_panel, "tab.gamepad"),
        ):
            self.tabs.setTabText(self.tabs.indexOf(panel), tr(key))

        for panel in (self.toolbar, self.joint_panel, self.motor_pd_panel,
                      self.trajectory_panel,
                      self.tcp_panel, self.book_takeout_panel, self.book_putback_panel,
                      self.book_putback2_panel, self.book_tail_putback_panel,
                      self.image_recognition_panel, self.book_debug_pose_panel,
                      self.book_gripper_panel, self.teaching_panel,
                      self.diagnostics_panel, self.gripper_panel,
                      self.lift_platform_panel, self.rodmotor_panel,
                      self.gamepad_panel, self.viewer_3d):
            if hasattr(panel, "retranslate_ui"):
                panel.retranslate_ui()
        self.monitoring_window.retranslate_ui()

    # ---- helpers ----

    def _adjust_dock_sizes(self, viewer_dock, tabs_dock, left_ratio=0.55):
        w = self.width()
        left_w = int(w * left_ratio)
        right_w = w - left_w
        self.resizeDocks(
            [viewer_dock, tabs_dock], [left_w, right_w], Qt.Orientation.Horizontal
        )

    def _open_monitoring(self):
        mw = self.monitoring_window
        if mw.isVisible():
            mw.raise_()
            mw.activateWindow()
        else:
            mw.show()
            mw.raise_()

    def _on_connected_for_gamepad(self, connected: bool):
        if connected and self.worker.arm is not None:
            self.gamepad_panel.set_arm(self.worker.arm)
        elif not connected:
            self.gamepad_panel.set_arm(None)

    def _on_connect(self, can_name: str, connect_kwargs: dict):
        self.worker.submit_command("connect", can_name, **connect_kwargs)

    def _sync_book_header_zero_duration(self, duration_s: float):
        for panel in (
            self.book_takeout_panel,
            self.book_putback_panel,
            self.book_putback2_panel,
            self.book_tail_putback_panel,
        ):
            panel.set_header_zero_duration(duration_s)

    def _on_book_debug_pose_saved(self, pose_deg: list):
        for panel in (
            self.book_takeout_panel,
            self.book_putback_panel,
            self.book_putback2_panel,
            self.book_tail_putback_panel,
        ):
            panel.set_book_debug_pose_deg(pose_deg)

    def _on_book_gripper_saved(self, values: dict):
        for panel in (
            self.book_takeout_panel,
            self.book_putback_panel,
            self.book_putback2_panel,
            self.book_tail_putback_panel,
        ):
            panel.set_book_gripper_defaults(values)

    def _on_tab_changed(self, index: int):
        self._sync_left_view_for_tab(index)

    def _sync_left_view_for_tab(self, index: int):
        if index == getattr(self, "_book_takeout_tab_index", -1):
            self.left_stack.setCurrentWidget(self.book_takeout_viewer)
            self.viewer_dock.setWindowTitle(tr("win.point_cloud_viewer"))
            self.book_takeout_panel.show_viewer()
            self.book_putback_panel.hide_viewer()
            self.book_putback2_panel.hide_viewer()
            self.book_tail_putback_panel.hide_viewer()
            self.image_recognition_panel.hide_viewer()
            self._adjust_dock_sizes(
                self.viewer_dock,
                self.tabs_dock,
                left_ratio=0.60,
            )
            return
        if index == getattr(self, "_book_putback_tab_index", -1):
            self.left_stack.setCurrentWidget(self.book_putback_viewer)
            self.viewer_dock.setWindowTitle(tr("win.point_cloud_viewer"))
            self.book_putback_panel.show_viewer()
            self.book_takeout_panel.hide_viewer()
            self.book_putback2_panel.hide_viewer()
            self.book_tail_putback_panel.hide_viewer()
            self.image_recognition_panel.hide_viewer()
            self._adjust_dock_sizes(
                self.viewer_dock,
                self.tabs_dock,
                left_ratio=0.60,
            )
            return
        if index == getattr(self, "_book_putback2_tab_index", -1):
            self.left_stack.setCurrentWidget(self.book_putback2_viewer)
            self.viewer_dock.setWindowTitle(tr("win.point_cloud_viewer"))
            self.book_putback2_panel.show_viewer()
            self.book_takeout_panel.hide_viewer()
            self.book_putback_panel.hide_viewer()
            self.book_tail_putback_panel.hide_viewer()
            self.image_recognition_panel.hide_viewer()
            self._adjust_dock_sizes(
                self.viewer_dock,
                self.tabs_dock,
                left_ratio=0.60,
            )
            return
        if index == getattr(self, "_book_tail_putback_tab_index", -1):
            self.left_stack.setCurrentWidget(self.book_tail_putback_viewer)
            self.viewer_dock.setWindowTitle(tr("win.point_cloud_viewer"))
            self.book_tail_putback_panel.show_viewer()
            self.book_takeout_panel.hide_viewer()
            self.book_putback_panel.hide_viewer()
            self.book_putback2_panel.hide_viewer()
            self.image_recognition_panel.hide_viewer()
            self._adjust_dock_sizes(
                self.viewer_dock,
                self.tabs_dock,
                left_ratio=0.60,
            )
            return
        if index == getattr(self, "_image_recognition_tab_index", -1):
            self.left_stack.setCurrentWidget(self.image_recognition_viewer)
            self.viewer_dock.setWindowTitle(tr("win.point_cloud_viewer"))
            self.image_recognition_panel.show_viewer()
            self.book_takeout_panel.hide_viewer()
            self.book_putback_panel.hide_viewer()
            self.book_putback2_panel.hide_viewer()
            self.book_tail_putback_panel.hide_viewer()
            self._adjust_dock_sizes(
                self.viewer_dock,
                self.tabs_dock,
                left_ratio=0.60,
            )
            return
        self.left_stack.setCurrentWidget(self.viewer_3d)
        self.viewer_dock.setWindowTitle(tr("win.viewer"))
        self.book_takeout_panel.hide_viewer()
        self.book_putback_panel.hide_viewer()
        self.book_putback2_panel.hide_viewer()
        self.book_tail_putback_panel.hide_viewer()
        self.image_recognition_panel.hide_viewer()
        self._adjust_dock_sizes(
            self.viewer_dock,
            self.tabs_dock,
            left_ratio=0.55,
        )

    def _start_sim_mode(self):
        self.worker.submit_command("connect", "sim", sim_mode=True)
        QTimer.singleShot(200, lambda: self.worker.submit_command("enable"))
        self._append_log("仿真模式已启动")

    def _on_joints_updated(self, joint_states):
        self._last_joint_states = joint_states
        self.teaching_panel.update_positions(joint_states)
        self.trajectory_panel.update_current_positions(joint_states)

        positions = joint_states.to_list(include_gripper=False)
        self.teaching_panel.calibration_panel.feed_positions(positions)
        self.book_takeout_panel.update_joint_feedback(joint_states)
        self.book_putback_panel.update_joint_feedback(joint_states)
        self.book_putback2_panel.update_joint_feedback(joint_states)
        self.book_tail_putback_panel.update_joint_feedback(joint_states)
        self.image_recognition_panel.update_joint_feedback(joint_states)

        now = time.monotonic()
        if now - self._last_ui_update_time < self.UI_UPDATE_INTERVAL_S:
            return
        self._last_ui_update_time = now

        self.joint_panel.update_feedback(joint_states)
        self.viewer_3d.update_joint_angles(joint_states)
        self.gripper_panel.update_feedback(joint_states, self._last_effort_states)
        self.diagnostics_panel.update_motor_states(
            joint_states, None, self._last_effort_states
        )

    def _on_efforts_updated(self, effort_states):
        self._last_effort_states = effort_states
        efforts = effort_states.to_list(include_gripper=False)
        self.teaching_panel.calibration_panel.feed_efforts(efforts)

    def _on_error(self, msg: str):
        self.toolbar.set_error(msg)
        self._append_log(tr("win.error", msg=msg))

    def _append_log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_console.append(f"[{timestamp}] {msg}")
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _init_3d_model(self):
        success = self.viewer_3d.initialize_model()
        if success:
            self._append_log(tr("win.model_ok"))
        else:
            self._append_log(tr("win.model_fail"))

    def closeEvent(self, event):
        event.accept()
        self._append_log(tr("win.closing"))
        self.gamepad_panel.cleanup()
        self.book_takeout_panel.cleanup()
        self.book_putback_panel.cleanup()
        self.book_putback2_panel.cleanup()
        self.book_tail_putback_panel.cleanup()
        self.image_recognition_panel.cleanup()
        if self.monitoring_window.isVisible():
            self.monitoring_window.close()
        self.worker.stop()
        self.rodmotor_worker.stop()
        self.lift_platform_worker.stop()
        app = QApplication.instance()
        if app is not None:
            QTimer.singleShot(0, app.quit)
        super().closeEvent(event)
