#!/usr/bin/env python3
"""Standalone GUI for rodmotor debugging.
启动方式：
    python scripts/rodmotor_test/rodmotor_gui.py --port /dev/rodmotor --baudrate 921600
"""

import argparse
import logging
import os
import sys
from pathlib import Path


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _get_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("QT_API", "pyqt6")


def main():
    parser = argparse.ArgumentParser(description="Standalone rodmotor GUI")
    parser.add_argument("--port", default="/dev/rodmotor", help="rodmotor 串口")
    parser.add_argument("--baudrate", type=int, default=921600, help="串口波特率")
    parser.add_argument("--timeout", type=float, default=0.3, help="响应超时")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from PyQt6.QtWidgets import QApplication, QMainWindow

    from MotorStudio.backend.rodmotor_worker import RodMotorWorker
    from MotorStudio.utils.theme_manager import ThemeManager
    from MotorStudio.utils.style import THEMES
    from MotorStudio.widgets.rodmotor_panel import RodMotorPanel

    app = QApplication(sys.argv)
    app.setApplicationName("RodMotor Debugger")

    tm = ThemeManager.instance()
    app.setStyleSheet(THEMES[tm.theme])
    tm.theme_changed.connect(lambda t: app.setStyleSheet(THEMES[t]))

    window = QMainWindow()
    window.setWindowTitle("RodMotor Debugger")
    window.resize(760, 420)

    panel = RodMotorPanel()
    panel.port_edit.setText(args.port)
    panel.baud_spin.setValue(args.baudrate)
    panel.timeout_spin.setValue(args.timeout)

    worker = RodMotorWorker()
    worker.start()

    panel.connect_requested.connect(
        lambda port, baud, timeout: worker.submit_command("connect", port, baud, timeout)
    )
    panel.disconnect_requested.connect(lambda: worker.submit_command("disconnect"))
    panel.read_requested.connect(lambda: worker.submit_command("read_angle"))
    panel.write_requested.connect(
        lambda angle, spd, acc: worker.submit_command("write_angle", angle, spd, acc)
    )

    worker.connected_changed.connect(panel.set_connected)
    worker.angle_updated.connect(panel.update_angle)
    worker.error_occurred.connect(panel.set_error)

    window.setCentralWidget(panel)
    window.show()

    exit_code = app.exec()
    worker.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
