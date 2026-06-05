"""Background worker for the independent rodmotor serial device."""

import logging
import math
import time
from queue import Empty, Queue
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from el_a3_sdk import RodMotorClient

logger = logging.getLogger("MotorStudio.rodmotor_worker")


class RodMotorWorker(QThread):
    """Serial worker for rodmotor read/write operations."""

    connected_changed = pyqtSignal(bool)
    angle_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    log_message = pyqtSignal(str)
    write_done = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._client: Optional[RodMotorClient] = None
        self._cmd_queue: Queue = Queue()
        self._running = False
        self._connected = False
        self._poll_interval = 0.2
        self._last_poll = 0.0
        self._read_error_count = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def submit_command(self, cmd: str, *args, **kwargs):
        self._cmd_queue.put((cmd, args, kwargs))

    def run(self):
        self._running = True
        try:
            while self._running:
                try:
                    self._process_commands()
                    self._poll_angle()
                except Exception as exc:
                    logger.error("rodmotor worker error: %s", exc)
                    self.error_occurred.emit(str(exc))
                time.sleep(0.02)
        finally:
            self._do_disconnect()

    def stop(self):
        self._running = False
        if self.isRunning() and not self.wait(1500):
            logger.warning("rodmotor worker did not stop within timeout; terminating thread")
            self.terminate()
            self.wait(1000)
        self._do_disconnect()

    def _process_commands(self):
        while True:
            try:
                cmd, args, kwargs = self._cmd_queue.get_nowait()
            except Empty:
                break

            if cmd == "connect":
                self._do_connect(*args, **kwargs)
            elif cmd == "disconnect":
                self._do_disconnect()
            elif cmd == "read_angle":
                self._do_read_angle()
            elif cmd == "write_angle":
                self._do_write_angle(*args, **kwargs)

    def _do_connect(
        self,
        port: str = RodMotorClient.DEFAULT_PORT,
        baudrate: int = RodMotorClient.DEFAULT_BAUDRATE,
        timeout: float = 0.3,
    ):
        if self._connected:
            return
        self._client = RodMotorClient(port=port, baudrate=baudrate, timeout=timeout)
        if not self._client.connect():
            self._client = None
            raise ConnectionError(f"无法连接 rodmotor 串口: {port}")
        self._connected = True
        self._read_error_count = 0
        self.connected_changed.emit(True)
        self.log_message.emit(f"rodmotor 已连接: {port}")
        try:
            self._do_read_angle()
        except Exception as exc:
            self.log_message.emit(f"rodmotor 已连接，但首次读角度超时: {exc}")

    def _do_disconnect(self):
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._connected:
            self._connected = False
            self._read_error_count = 0
            self.connected_changed.emit(False)
            self.log_message.emit("rodmotor 已断开")

    def _do_read_angle(self):
        if not self._connected or self._client is None:
            return
        angle = self._client.read_angle()
        self._read_error_count = 0
        self.angle_updated.emit(float(angle))

    def _do_write_angle(
        self,
        angle_deg: float,
        spd: int = 1000,
        acc: int = 50,
        torque: Optional[float] = None,
    ):
        if not self._connected or self._client is None:
            raise RuntimeError("rodmotor 未连接")
        self._client.write_angle(
            angle_deg,
            spd=spd,
            acc=acc,
            torque=torque,
            wait_response=False,
        )
        if torque is None:
            self.log_message.emit(f"rodmotor 目标角度: {float(angle_deg):.2f}°")
        else:
            self.log_message.emit(
                f"rodmotor 目标角度: {float(angle_deg):.2f}° torque={float(torque):.2f}"
            )
        self._last_poll = 0.0
        self.write_done.emit()

    def _poll_angle(self):
        if not self._connected or self._client is None:
            return
        now = time.monotonic()
        if now - self._last_poll < self._poll_interval:
            return
        self._last_poll = now
        try:
            angle = self._client.read_angle()
            self._read_error_count = 0
            self.angle_updated.emit(float(angle))
        except Exception as exc:
            self._read_error_count += 1
            self._poll_interval = min(2.0, 0.2 * (2 ** min(self._read_error_count, 4)))
            if self._read_error_count in (1, 3, 8):
                self.log_message.emit(
                    f"rodmotor 读取角度超时，已降低轮询频率: {exc}"
                )
