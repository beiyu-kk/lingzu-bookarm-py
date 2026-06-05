#!/usr/bin/env python3
"""
rodmotor 角度功能测试

覆盖 RodMotorClient 的公开功能:
  - connect / close / is_connected / context manager
  - read_angle
  - write_angle

说明:
  - 对外接口统一使用“角度（度）”
  - 底层协议仍会转成固件需要的弧度字段

用法:
  python3 scripts/rodmotor_test/test_rodmotor_all.py
  python3 scripts/rodmotor_test/test_rodmotor_all.py --write-angle 90
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from el_a3_sdk import RodMotorClient

PASS = 0
FAIL = 0
SKIP = 0


def log_pass(msg):
    global PASS
    PASS += 1
    print(f"  [PASS] {msg}")


def log_fail(msg):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {msg}")


def log_skip(msg):
    global SKIP
    SKIP += 1
    print(f"  [SKIP] {msg}")


def main():
    parser = argparse.ArgumentParser(description="rodmotor 角度功能测试")
    parser.add_argument("--port", default="/dev/rodmotor", help="rodmotor 固定串口")
    parser.add_argument("--baudrate", type=int, default=921600, help="串口波特率")
    parser.add_argument("--timeout", type=float, default=0.3, help="响应超时")
    parser.add_argument("--write-angle", type=float, default=None,
                        help="写入角度（度，不在 SDK 层限制范围）")
    parser.add_argument("--speed", type=int, default=1000, help="写入速度参数")
    parser.add_argument("--acc", type=int, default=50, help="写入加速度参数")
    args = parser.parse_args()

    print("=" * 60)
    print(f" rodmotor 角度测试 ({args.port})")
    print("=" * 60)

    try:
        with RodMotorClient(
            port=args.port,
            baudrate=args.baudrate,
            timeout=args.timeout,
            auto_connect=True,
        ) as rod:
            print("\n  --- Connect ---")
            if rod.is_connected:
                log_pass("连接成功")
            else:
                log_fail("连接失败")

            print("\n  --- Read Angle ---")
            try:
                angle = rod.read_angle(timeout=args.timeout)
                log_pass(f"read_angle 成功: {angle:.3f}°")
            except Exception as e:
                log_fail(f"read_angle 异常: {e}")
                angle = None

            print("\n  --- Write Angle ---")
            if args.write_angle is None:
                log_skip("未指定 --write-angle，跳过写入")
            else:
                try:
                    resp = rod.write_angle(
                        args.write_angle,
                        spd=args.speed,
                        acc=args.acc,
                        wait_response=True,
                        timeout=args.timeout,
                    )
                    if resp is not None:
                        log_pass(f"write_angle 成功: {args.write_angle:.3f}°")
                    else:
                        log_fail("write_angle 未返回响应")
                except Exception as e:
                    log_fail(f"write_angle 异常: {e}")

            print("\n  --- Close ---")
            rod.close()
            if not rod.is_connected:
                log_pass("close 成功")
            else:
                log_fail("close 后仍连接")

        log_pass("context manager 正常退出")

    except Exception as e:
        log_fail(f"初始化或连接阶段异常: {e}")

    print(f"\n{'=' * 60}")
    print(f" 结果: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print(f"{'=' * 60}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
