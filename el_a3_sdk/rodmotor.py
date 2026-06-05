"""rodmotor serial JSON driver."""

import json
import logging
import math
import threading
import time
from typing import Any, Dict, Optional

from el_a3_sdk.serial_utils import load_pyserial

logger = logging.getLogger("el_a3_sdk.rodmotor")


class RodMotorClient:
    """rodmotor serial client with degree-based API."""

    DEFAULT_PORT = "/dev/rodmotor"
    DEFAULT_BAUDRATE = 921600

    CMD_READ_ANGLE = 103
    CMD_WRITE_ANGLE = 104

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout: float = 0.2,
        write_timeout: float = 0.2,
        auto_connect: bool = False,
    ):
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.write_timeout = float(write_timeout)

        self._serial = None
        self._lock = threading.Lock()

        if auto_connect:
            self.connect()

    def connect(self) -> bool:
        if self.is_connected:
            return True

        try:
            serial = load_pyserial()
        except ImportError as exc:
            logger.error("rodmotor pyserial import failed: %s", exc)
            raise

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.write_timeout,
            )
            time.sleep(0.05)
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
            logger.info("rodmotor serial connected: %s @ %d", self.port, self.baudrate)
            return True
        except Exception as exc:
            logger.error("rodmotor serial connect failed [%s]: %s", self.port, exc)
            self._serial = None
            return False

    def close(self):
        if self._serial is not None:
            try:
                if self._serial.is_open:
                    self._serial.close()
            finally:
                self._serial = None

    disconnect = close

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def read_angle(self, timeout: Optional[float] = None) -> float:
        """Read current angle in degrees."""
        response = self.request(self.CMD_READ_ANGLE, timeout=timeout, require_response=True)
        value = self._pick_value(response, "rad", "angle", "deg", "position", "pos", "value")
        return math.degrees(float(value))

    def write_angle(
        self,
        angle_deg: float,
        spd: int = 1000,
        acc: int = 50,
        torque: Optional[float] = None,
        wait_response: bool = False,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Command angle in degrees."""
        angle_deg = float(angle_deg)
        payload = {"rad": math.radians(angle_deg), "spd": int(spd), "acc": int(acc)}
        if torque is not None:
            payload["torque"] = float(torque)
        return self.request(
            self.CMD_WRITE_ANGLE,
            payload,
            timeout=timeout,
            require_response=wait_response,
        )

    def request(
        self,
        command: int,
        payload: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
        require_response: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if not self.is_connected and not self.connect():
            raise ConnectionError(f"Cannot open rodmotor serial port: {self.port}")

        message: Dict[str, Any] = {"T": int(command)}
        if payload:
            message.update(payload)

        line = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"

        with self._lock:
            self._serial.write(line)
            self._serial.flush()
            if not require_response:
                return None
            return self._read_response(int(command), timeout)

    def _read_response(
        self,
        command: int,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + (self.timeout if timeout is None else float(timeout))
        last_decode_error = ""

        while time.monotonic() < deadline:
            raw = self._serial.readline()
            if not raw:
                continue
            try:
                data = json.loads(raw.decode("utf-8", errors="replace").strip())
            except json.JSONDecodeError as exc:
                last_decode_error = str(exc)
                logger.debug("rodmotor ignored non-JSON line: %r", raw)
                continue
            if not isinstance(data, dict):
                continue

            ok = bool(data.get("ok", data.get("success", True)))
            if not ok:
                err = str(data.get("error") or data.get("err") or f"rodmotor command {command} failed")
                raise RuntimeError(err)
            return data

        detail = f" Last decode error: {last_decode_error}" if last_decode_error else ""
        raise TimeoutError(f"rodmotor command {command} timed out on {self.port}.{detail}")

    @staticmethod
    def _pick_value(data: Dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in data:
                return data[key]
        raise KeyError(f"rodmotor response missing any of keys: {', '.join(keys)}")


__all__ = ["RodMotorClient"]
