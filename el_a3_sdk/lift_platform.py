"""Serial Modbus RTU control for the lift platform."""

from __future__ import annotations

import os
import struct
import time
from typing import Any, Optional, Union

from el_a3_sdk.serial_utils import load_pyserial

serial = load_pyserial()


DEFAULT_LIFT_PORT = "COM9" if os.name == "nt" else "/dev/lift_port"
DEFAULT_LIFT_BAUDRATE = 19200
DEFAULT_LIFT_SLAVE_ID = 1
DEFAULT_LIFT_SPEED_RPM = 200
DEFAULT_LIFT_ACCELERATION = 100
DEFAULT_LIFT_PULSES_PER_CM = 32000.0


class LiftPlatformActuator:
    """Control the lift platform driver through Modbus RTU over serial."""

    def __init__(
        self,
        *,
        port: str = DEFAULT_LIFT_PORT,
        baudrate: int = DEFAULT_LIFT_BAUDRATE,
        slave_id: int = DEFAULT_LIFT_SLAVE_ID,
        speed_rpm: int = DEFAULT_LIFT_SPEED_RPM,
        acceleration: int = DEFAULT_LIFT_ACCELERATION,
        timeout: float = 0.1,
        setup_on_open: bool = True,
        verbose: bool = False,
    ) -> None:
        self.port = port
        self.baudrate = int(baudrate)
        self.slave_id = int(slave_id)
        self.speed_rpm = int(speed_rpm)
        self.acceleration = int(acceleration)
        self.timeout = float(timeout)
        self.setup_on_open = bool(setup_on_open)
        self.verbose = bool(verbose)
        self.ser: Optional[Any] = None

    @property
    def connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def open(self) -> "LiftPlatformActuator":
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout,
            )
        except Exception:
            self.ser = None
            raise
        if self.setup_on_open:
            self.initial_setup()
        return self

    def close(self) -> None:
        """Close the serial port without changing motor output state."""
        if self.ser and self.ser.is_open:
            self.ser.close()

    def cleanup(self) -> None:
        """Stop the motor, disable driver output, and close the serial port."""
        if self.ser and self.ser.is_open:
            self.stop_motor()
            time.sleep(0.1)
            self.disable_output()
            self.ser.close()

    def initial_setup(self) -> None:
        """Enable Modbus control, configure motion profile, and enable output."""
        self.enable_modbus_control()
        time.sleep(0.05)
        self.set_acceleration(self.acceleration)
        time.sleep(0.05)
        self.set_speed(self.speed_rpm)
        time.sleep(0.05)
        self.enable_output()
        time.sleep(0.1)

    def enable_modbus_control(self) -> Optional[bytes]:
        return self._send_modbus_command(0x06, 0x0000, 1)

    def set_acceleration(self, acceleration: int) -> Optional[bytes]:
        self.acceleration = int(acceleration)
        return self._send_modbus_command(0x06, 0x0003, self.acceleration)

    def set_speed(self, speed_rpm: int) -> Optional[bytes]:
        self.speed_rpm = int(speed_rpm)
        return self._send_modbus_command(0x06, 0x0002, self.speed_rpm)

    def enable_output(self) -> Optional[bytes]:
        return self._send_modbus_command(0x06, 0x0001, 1)

    def disable_output(self) -> Optional[bytes]:
        return self._send_modbus_command(0x06, 0x0001, 0)

    def move_incremental(self, pulses: int) -> Optional[bytes]:
        """Move by a signed incremental pulse count."""
        return self._send_modbus_command(0x10, 0x000C, int(pulses), is_write_single=False)

    def move_lift_distance_cm(
        self,
        distance_cm: float,
        *,
        speed_rpm: Optional[int] = None,
        acceleration: Optional[int] = None,
        pulses_per_cm: float = DEFAULT_LIFT_PULSES_PER_CM,
        up_direction: int = 1,
    ) -> Optional[bytes]:
        """Move the lift by a signed distance in cm."""
        if speed_rpm is not None:
            self.set_speed(int(speed_rpm))
        if acceleration is not None:
            self.set_acceleration(int(acceleration))
        direction = 1 if int(up_direction) >= 0 else -1
        pulses = int(round(float(distance_cm) * float(pulses_per_cm))) * direction
        return self.move_incremental(pulses)

    def stop_motor(self) -> Optional[bytes]:
        """Send a zero incremental move command."""
        return self.move_incremental(0)

    def _send_modbus_command(
        self,
        function_code: int,
        register_address: int,
        data_value: Optional[int] = None,
        *,
        is_write_single: bool = True,
    ) -> Optional[bytes]:
        if not self.ser:
            return None

        if is_write_single and data_value is not None:
            frame = bytearray()
            frame.append(self.slave_id)
            frame.append(function_code)
            frame.extend(register_address.to_bytes(2, byteorder="big"))
            frame.extend(int(data_value).to_bytes(2, byteorder="big"))
        elif not is_write_single and data_value is not None and function_code == 0x10:
            frame = bytearray()
            frame.append(self.slave_id)
            frame.append(0x10)
            frame.extend(struct.pack(">H", 0x000C))
            frame.extend(struct.pack(">H", 0x0002))
            frame.append(0x04)
            value = int(data_value)
            if value < 0:
                value = 0xFFFFFFFF + value + 1
            pu_8_15 = (value >> 8) & 0xFF
            pu_0_7 = value & 0xFF
            pu_24_31 = (value >> 24) & 0xFF
            pu_16_23 = (value >> 16) & 0xFF
            frame.append(pu_8_15)
            frame.append(pu_0_7)
            frame.append(pu_24_31)
            frame.append(pu_16_23)
        else:
            return None

        frame.extend(self._crc16(frame))
        if self.verbose:
            print(f"TX: {frame.hex(' ')}")
        self.ser.write(frame)
        time.sleep(0.01)
        response = self.ser.read(8)
        if self.verbose:
            print(f"RX: {response.hex(' ') if response else '<no response>'}")
        return response

    @staticmethod
    def _crc16(data: Union[bytes, bytearray]) -> bytes:
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc = crc >> 1
        return crc.to_bytes(2, byteorder="little")
