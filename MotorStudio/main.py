"""EL-A3 机械臂调试上位机 - 入口
启动代码：
    conda run --no-capture-output -n lingarm python -m MotorStudio.main
"""

import sys
import os
import argparse
import logging
import signal
from pathlib import Path


def _get_base_path() -> Path:
    """PyInstaller frozen 环境返回 _MEIPASS, 否则返回源码 SDK 根目录"""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent.parent


SDK_ROOT = _get_base_path()
REPO_ROOT = SDK_ROOT.parent

if not getattr(sys, "frozen", False):
    if str(SDK_ROOT) not in sys.path:
        sys.path.insert(0, str(SDK_ROOT))
    if str(SDK_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(SDK_ROOT.parent))

os.environ.setdefault("QT_API", "pyqt6")

if sys.platform.startswith("linux"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


def _default_mesh_dir() -> str:
    local_meshes = SDK_ROOT / "resources" / "meshes"
    if local_meshes.exists():
        return str(local_meshes)

    ros_candidates = [
        REPO_ROOT / "el_a3_ros" / "el_a3_description" / "meshes",
        SDK_ROOT.parent.parent.parent / "el_a3_ros" / "el_a3_description" / "meshes",
    ]
    for candidate in ros_candidates:
        if candidate.exists():
            return str(candidate)
    return str(local_meshes)


def _install_terminal_shutdown_handlers(app, window, timer_cls):
    logger = logging.getLogger("MotorStudio.main")
    shutting_down = {"active": False}

    def _request_shutdown(signum, _frame):
        if shutting_down["active"]:
            return
        shutting_down["active"] = True
        try:
            signal_name = signal.Signals(signum).name
        except Exception:
            signal_name = str(signum)
        logger.info("收到终端信号 %s，正在关闭 GUI", signal_name)
        timer_cls.singleShot(0, window.close)

    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    for sig in handled_signals:
        try:
            signal.signal(sig, _request_shutdown)
        except (OSError, RuntimeError, ValueError):
            pass

    signal_timer = timer_cls(app)
    signal_timer.setInterval(200)
    signal_timer.timeout.connect(lambda: None)
    signal_timer.start()
    app.aboutToQuit.connect(signal_timer.stop)
    return signal_timer


def main():
    default_urdf = str(SDK_ROOT / "resources" / "urdf" / "el_a3.urdf")
    default_meshes = _default_mesh_dir()

    parser = argparse.ArgumentParser(description="EL-A3 Robot Arm MotorStudio")
    parser.add_argument("--can", default="can0", help="CAN interface name (default: can0)")
    parser.add_argument("--sim", action="store_true", help="Simulation mode (no hardware)")
    parser.add_argument(
        "--urdf",
        default=default_urdf,
        help="URDF file path",
    )
    parser.add_argument(
        "--meshes",
        default=default_meshes,
        help="STL mesh directory",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("EL-A3 MotorStudio")

    from MotorStudio.utils.theme_manager import ThemeManager
    from MotorStudio.utils.style import THEMES

    tm = ThemeManager.instance()
    app.setStyleSheet(THEMES[tm.theme])
    tm.theme_changed.connect(lambda t: app.setStyleSheet(THEMES[t]))

    from MotorStudio.main_window import MainWindow
    window = MainWindow(
        urdf_path=args.urdf,
        mesh_dir=args.meshes,
        sim_mode=args.sim,
    )
    window.show()
    _signal_timer = _install_terminal_shutdown_handlers(app, window, QTimer)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
