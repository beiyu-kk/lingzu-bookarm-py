#!/usr/bin/env python3
"""Lift platform Modbus frame tests."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from el_a3_sdk.lift_platform import LiftPlatformActuator


class FakeSerial:
    def __init__(self):
        self.is_open = True
        self.writes = []

    def write(self, data):
        self.writes.append(bytes(data))

    def read(self, size):
        return b""


def test_crc16_known_modbus_frame():
    actuator = LiftPlatformActuator()

    assert actuator._crc16(bytes.fromhex("01 06 00 00 00 01")) == bytes.fromhex("48 0a")


def test_move_incremental_positive_pulses_frame():
    actuator = LiftPlatformActuator(slave_id=1)
    fake = FakeSerial()
    actuator.ser = fake

    actuator.move_incremental(32000)

    assert fake.writes[-1].hex(" ") == "01 10 00 0c 00 02 04 7d 00 00 00 eb 96"


def test_move_incremental_negative_pulses_frame():
    actuator = LiftPlatformActuator(slave_id=1)
    fake = FakeSerial()
    actuator.ser = fake

    actuator.move_incremental(-32000)

    assert fake.writes[-1].hex(" ") == "01 10 00 0c 00 02 04 83 00 ff ff db ce"


def test_move_distance_uses_calibration_and_direction():
    actuator = LiftPlatformActuator(slave_id=1)
    fake = FakeSerial()
    actuator.ser = fake

    actuator.move_lift_distance_cm(2.5, speed_rpm=300, acceleration=120, pulses_per_cm=1000, up_direction=-1)

    assert fake.writes[-3].hex(" ") == "01 06 00 02 01 2c 28 47"
    assert fake.writes[-2].hex(" ") == "01 06 00 03 00 78 79 e8"
    assert fake.writes[-1].hex(" ") == "01 10 00 0c 00 02 04 f6 3c ff ff 01 ce"
