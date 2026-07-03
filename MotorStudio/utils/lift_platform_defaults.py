"""Lift platform default settings persistence."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

from el_a3_sdk import (
    DEFAULT_LIFT_ACCELERATION,
    DEFAULT_LIFT_BAUDRATE,
    DEFAULT_LIFT_PORT,
    DEFAULT_LIFT_PULSES_PER_CM,
    DEFAULT_LIFT_SLAVE_ID,
    DEFAULT_LIFT_SPEED_RPM,
)


CONFIG_VERSION = 1

DEFAULT_LIFT_PLATFORM_SETTINGS: Dict[str, Any] = {
    "port": DEFAULT_LIFT_PORT,
    "baudrate": DEFAULT_LIFT_BAUDRATE,
    "slave_id": DEFAULT_LIFT_SLAVE_ID,
    "timeout": 0.1,
    "distance_cm": 1.0,
    "speed_rpm": DEFAULT_LIFT_SPEED_RPM,
    "acceleration": DEFAULT_LIFT_ACCELERATION,
    "pulses_per_cm": DEFAULT_LIFT_PULSES_PER_CM,
    "reverse_up_direction": False,
    "pulses": 0,
    "current_position": "lowest",
    "return_offset_cm": 10.0,
    "take_offset_cm": 10.0,
}


def _config_dir() -> Path:
    if sys.platform.startswith("win"):
        base = os.getenv("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(base) / "el_a3_sdk"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "el_a3_sdk"

    base = os.getenv("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "el_a3_sdk"
    return Path.home() / ".config" / "el_a3_sdk"


def get_lift_platform_defaults_path() -> Path:
    return _config_dir() / "motorstudio_lift_platform_defaults.json"


def _coerce_float(value: Any, fallback: float, *, min_value: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if min_value is not None and parsed < min_value:
        return fallback
    return parsed


def _coerce_int(
    value: Any,
    fallback: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    if min_value is not None and parsed < min_value:
        return fallback
    if max_value is not None and parsed > max_value:
        return fallback
    return parsed


def normalize_lift_platform_settings(values: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    settings = dict(DEFAULT_LIFT_PLATFORM_SETTINGS)
    if not values:
        return settings

    port = str(values.get("port", settings["port"])).strip()
    settings["port"] = port or DEFAULT_LIFT_PORT
    settings["baudrate"] = _coerce_int(values.get("baudrate"), settings["baudrate"], min_value=1200)
    settings["slave_id"] = _coerce_int(values.get("slave_id"), settings["slave_id"], min_value=1, max_value=247)
    settings["timeout"] = _coerce_float(values.get("timeout"), settings["timeout"], min_value=0.05)
    settings["distance_cm"] = _coerce_float(values.get("distance_cm"), settings["distance_cm"])
    settings["speed_rpm"] = _coerce_int(values.get("speed_rpm"), settings["speed_rpm"], min_value=1)
    settings["acceleration"] = _coerce_int(values.get("acceleration"), settings["acceleration"], min_value=1)
    settings["pulses_per_cm"] = _coerce_float(
        values.get("pulses_per_cm"),
        settings["pulses_per_cm"],
        min_value=1.0,
    )
    settings["reverse_up_direction"] = bool(values.get("reverse_up_direction", settings["reverse_up_direction"]))
    settings["pulses"] = _coerce_int(values.get("pulses"), settings["pulses"])
    current_position = str(values.get("current_position", settings["current_position"])).strip()
    if current_position not in {"lowest", "return", "take"}:
        current_position = "lowest"
    settings["current_position"] = current_position
    settings["return_offset_cm"] = _coerce_float(
        values.get("return_offset_cm"),
        settings["return_offset_cm"],
        min_value=0.0,
    )
    settings["take_offset_cm"] = _coerce_float(
        values.get("take_offset_cm"),
        settings["take_offset_cm"],
        min_value=0.0,
    )
    return settings


def load_lift_platform_defaults() -> Dict[str, Any]:
    path = get_lift_platform_defaults_path()
    if not path.exists():
        return normalize_lift_platform_settings()

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return normalize_lift_platform_settings()

    values = payload.get("settings", payload) if isinstance(payload, dict) else {}
    if not isinstance(values, dict):
        values = {}
    return normalize_lift_platform_settings(values)


def save_lift_platform_defaults(settings: Mapping[str, Any]) -> Path:
    path = get_lift_platform_defaults_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CONFIG_VERSION,
        "settings": normalize_lift_platform_settings(settings),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
