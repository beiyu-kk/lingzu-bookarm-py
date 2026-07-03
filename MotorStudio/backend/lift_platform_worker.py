"""Background worker for the lift platform serial device."""

import logging
import time
from queue import Empty, Queue
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from el_a3_sdk import (
    DEFAULT_LIFT_ACCELERATION,
    DEFAULT_LIFT_BAUDRATE,
    DEFAULT_LIFT_PORT,
    DEFAULT_LIFT_PULSES_PER_CM,
    DEFAULT_LIFT_SLAVE_ID,
    DEFAULT_LIFT_SPEED_RPM,
    LiftPlatformActuator,
)

logger = logging.getLogger("MotorStudio.lift_platform_worker")


class LiftPlatformWorker(QThread):
    """Serial worker for lift platform connect and incremental move commands."""

    connected_changed = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    log_message = pyqtSignal(str)
    move_done = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._client: Optional[LiftPlatformActuator] = None
        self._cmd_queue: Queue = Queue()
        self._running = False
        self._connected = False

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
                except Exception as exc:
                    logger.error("lift platform worker error: %s", exc)
                    self.error_occurred.emit(str(exc))
                time.sleep(0.02)
        finally:
            self._do_disconnect()

    def stop(self):
        self._running = False
        if self.isRunning() and not self.wait(1500):
            logger.warning("lift platform worker did not stop within timeout; terminating thread")
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
            elif cmd == "move_distance":
                self._do_move_distance(*args, **kwargs)
            elif cmd == "move_pulses":
                self._do_move_pulses(*args, **kwargs)
            elif cmd == "stop":
                self._do_stop_motor()

    def _do_connect(
        self,
        port: str = DEFAULT_LIFT_PORT,
        baudrate: int = DEFAULT_LIFT_BAUDRATE,
        slave_id: int = DEFAULT_LIFT_SLAVE_ID,
        speed_rpm: int = DEFAULT_LIFT_SPEED_RPM,
        acceleration: int = DEFAULT_LIFT_ACCELERATION,
        timeout: float = 0.1,
    ):
        if self._connected:
            return
        self._client = LiftPlatformActuator(
            port=port,
            baudrate=baudrate,
            slave_id=slave_id,
            speed_rpm=speed_rpm,
            acceleration=acceleration,
            timeout=timeout,
        )
        self._client.open()
        self._connected = True
        self.connected_changed.emit(True)
        self.log_message.emit(f"升降台已连接: {port}")

    def _do_disconnect(self):
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._connected:
            self._connected = False
            self.connected_changed.emit(False)
            self.log_message.emit("升降台已断开")

    def _do_move_distance(
        self,
        distance_cm: float,
        speed_rpm: int = DEFAULT_LIFT_SPEED_RPM,
        acceleration: int = DEFAULT_LIFT_ACCELERATION,
        pulses_per_cm: float = DEFAULT_LIFT_PULSES_PER_CM,
        up_direction: int = 1,
    ):
        if not self._connected or self._client is None:
            raise RuntimeError("升降台未连接")
        if float(distance_cm) == 0.0:
            self.log_message.emit("升降台目标距离为 0，已跳过")
            return
        self._client.move_lift_distance_cm(
            float(distance_cm),
            speed_rpm=int(speed_rpm),
            acceleration=int(acceleration),
            pulses_per_cm=float(pulses_per_cm),
            up_direction=int(up_direction),
        )
        direction = "上升" if float(distance_cm) > 0 else "下降"
        pulses = int(round(float(distance_cm) * float(pulses_per_cm))) * (1 if int(up_direction) >= 0 else -1)
        self.log_message.emit(
            f"升降台{direction}: {float(distance_cm):.3f} cm, pulses={pulses}, speed={int(speed_rpm)} RPM"
        )
        self.move_done.emit()

    def _do_move_pulses(self, pulses: int):
        if not self._connected or self._client is None:
            raise RuntimeError("升降台未连接")
        self._client.move_incremental(int(pulses))
        self.log_message.emit(f"升降台增量脉冲: {int(pulses)}")
        self.move_done.emit()

    def _do_stop_motor(self):
        if not self._connected or self._client is None:
            return
        self._client.stop_motor()
        self.log_message.emit("升降台停止指令已发送")
