#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RULE_SRC="${REPO_ROOT}/resources/udev/99-lift-platform.rules"
RULE_DST="/etc/udev/rules.d/99-lift-platform.rules"

if [[ ! -f "${RULE_SRC}" ]]; then
  echo "udev rule not found: ${RULE_SRC}" >&2
  exit 1
fi

sudo cp "${RULE_SRC}" "${RULE_DST}"
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "Installed ${RULE_DST}"
echo "Replug lift platform USB serial adapter, then check: ls -l /dev/lift_port"
echo "If the alias does not appear, verify the USB path matches resources/udev/99-lift-platform.rules"
