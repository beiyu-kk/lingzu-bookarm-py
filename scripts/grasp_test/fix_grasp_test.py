"""机械臂 + 杆电机联动顺序测试。

启动方式:
  1. 进入项目根目录::

       cd /home/boreas/project/lingzu_arm/EDULITE_A3/el_a3_sdk

  2. 确保机械臂已上电、CAN 已开启、rodmotor 已接到固定串口。
  3. 在 ``lingarm`` 环境中运行::

       conda run --no-capture-output -n lingarm \
         python scripts/grasp_test/fix_grasp_test.py

  4. 如果需要显式指定接口，可以这样启动::

       conda run --no-capture-output -n lingarm \
         python scripts/grasp_test/fix_grasp_test.py \
         --can can0 --rod-port /dev/rodmotor

默认参数:
  - CAN: ``can0``
  - rodmotor 串口: ``/dev/rodmotor``
  - rodmotor 波特率: ``921600``
  - MoveJ 时长: ``4.0 s``
  - MoveL 时长: ``4.5 s``
  - 出错时默认保持机械臂使能，不自动断力矩

执行流程:
  1. 连接机械臂与杆电机
  2. 机械臂 MoveJ 从当前构型到全零构型
  3. 杆电机同时到零位
  4. 夹爪缓慢打开
  5. 机械臂 MoveJ 到调试位
  6. 等待用户回车
  7. 机械臂 MoveL 到指定末端位姿
  8. 等待用户回车
  9. 夹爪缓慢关闭
  10. 机械臂 MoveL 回调试位

单位说明:
  - 机械臂关节: rad
  - 末端位置: m
  - 末端姿态: rad
  - 本脚本输入的 MoveL 目标位置用 cm，运行时自动换算成 m
  - 本脚本输入的 MoveL 姿态数值按“度”理解，运行时自动换算成 rad
  - 杆电机接口使用角度（度）
"""

import argparse
import math
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from el_a3_sdk.data_types import ArmEndPose
from el_a3_sdk import ELA3Interface, LogLevel, RodMotorClient
from el_a3_sdk.motion import interpolate_pose, smooth_time_scale

M_TO_CM = 100.0
CM_TO_M = 0.01
DEG_TO_RAD = math.pi / 180.0

DEFAULT_DEBUG_JOINTS_DEG = [0.0, 35.0, -35.0, 0.0, -20.0, 0.0]


def log(msg: str):
    print(msg, flush=True)


def prompt(msg: str):
    input(f"\n{msg} 按回车继续...")


def deg_list_to_rad(values):
    return [float(v) * DEG_TO_RAD for v in values]


def cm_to_m(values):
    return [float(v) * CM_TO_M for v in values]


def parse_pose_args(args):
    position_cm = [args.target_x, args.target_y, args.target_z]
    orientation_deg = [args.target_rx, args.target_ry, args.target_rz]
    return cm_to_m(position_cm), deg_list_to_rad(orientation_deg)


def make_pose(position_m, orientation_rad):
    return ArmEndPose(
        x=position_m[0],
        y=position_m[1],
        z=position_m[2],
        rx=orientation_rad[0],
        ry=orientation_rad[1],
        rz=orientation_rad[2],
    )


def wait_motion(arm, timeout):
    if not arm.wait_for_motion(timeout=timeout):
        raise TimeoutError("机械臂运动等待超时")


def get_current_arm_joints(arm):
    js = arm.GetArmJointMsgs()
    if js is not None and js.timestamp > 0:
        return js.to_list(include_gripper=False)
    if hasattr(arm, "_read_feedback_positions"):
        return arm._read_feedback_positions()
    return [0.0] * 6


def hold_current_position(arm):
    """取消轨迹，但保持控制循环和电机使能，在当前位置继续发保持指令。"""
    try:
        arm.cancel_motion()
    except Exception:
        pass

    q_hold = get_current_arm_joints(arm)
    try:
        arm.JointCtrl(*q_hold)
    except Exception:
        pass
    log(f"已取消当前轨迹并保持当前位置: {[round(math.degrees(v), 2) for v in q_hold]}")
    return q_hold


def check_movel_reachable(arm, target_pose: ArmEndPose, duration: float, n_waypoints: int = 50):
    """
    使用和 MoveL 相同的 IK 采样逻辑做预检查。
    这里只计算，不下发任何运动指令。
    """
    kin = arm._get_kinematics()
    if kin is None:
        return False, "Pinocchio/运动学初始化失败，不能执行 MoveL"

    current_q = get_current_arm_joints(arm)
    start_pose = kin.forward_kinematics(current_q)
    control_period = getattr(arm, "_control_period", 0.005)
    control_running = getattr(arm, "control_loop_running", False)
    duration = max(float(duration), control_period)
    sample_dt = max(control_period * 2.0, 0.01) if control_running else 0.01
    sample_count = max(int(n_waypoints), int(math.ceil(duration / sample_dt)), 2)
    sample_count = min(sample_count, 1200)

    q_prev = current_q
    ik_tol = 3e-4
    for i in range(1, sample_count + 1):
        tau = i / sample_count
        wp = interpolate_pose(start_pose, target_pose, smooth_time_scale(tau))
        q_sol = None
        q_seed = list(q_prev)
        err_norm = float("inf")

        q_next, err_norm = kin.ik_step(
            wp,
            q_seed,
            damping=8e-3,
            max_step=0.25,
            max_iter=8,
            converge_eps=ik_tol,
        )
        if q_next is not None:
            q_seed = q_next
            if err_norm <= ik_tol:
                q_sol = q_next

        if q_sol is None:
            q_fallback = kin.inverse_kinematics(
                wp,
                q_init=q_seed,
                max_iter=120,
                eps=ik_tol,
            )
            if q_fallback is not None:
                q_sol = q_fallback

        if q_sol is None:
            return (
                False,
                "MoveL 目标路径不可达，"
                f"IK 失败点 {i}/{sample_count}, "
                f"误差 {err_norm:.4f}, "
                f"失败位姿=({wp.x:.3f}, {wp.y:.3f}, {wp.z:.3f}, "
                f"{math.degrees(wp.rx):.2f}, {math.degrees(wp.ry):.2f}, {math.degrees(wp.rz):.2f})",
            )
        q_prev = q_sol

    return True, f"MoveL IK 预检查通过，共 {sample_count} 个采样点"


def safe_movel(arm, target_pose: ArmEndPose, duration: float, timeout: float, label: str):
    ok, msg = check_movel_reachable(arm, target_pose, duration)
    if not ok:
        log(f"\n[SAFE] {label} 未执行: {msg}")
        hold_current_position(arm)
        return False
    log(f"[SAFE] {msg}")

    if not arm.MoveL(target_pose, duration=duration, block=False):
        log(f"\n[SAFE] {label} 下发失败，保持当前位置")
        hold_current_position(arm)
        return False

    if not arm.wait_for_motion(timeout=timeout):
        log(f"\n[SAFE] {label} 等待超时，取消轨迹并保持当前位置")
        hold_current_position(arm)
        return False

    return True


def wait_for_operator_recovery(arm, debug_pose: Optional[ArmEndPose], args, reason: str) -> bool:
    log("\n!!! 当前流程已进入安全保持状态。")
    log(f"!!! 原因: {reason}")
    log("!!! 机械臂保持使能，控制循环继续运行，脚本不会自动断力矩。")
    log("!!! 注意: 在这里按 Ctrl+C 不会退出；必须输入 d 并二次确认后才会失能退出。")

    hold_current_position(arm)
    while True:
        log("\n可选操作:")
        log("  h / 回车 : 继续保持当前姿态")
        log("  r        : 尝试 MoveL 回到调试位")
        log("  d        : 确认安全后，失能并退出")
        try:
            choice = input("请输入操作: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            log("\n已拦截退出请求，机械臂继续保持当前位置。")
            hold_current_position(arm)
            continue

        if choice in ("", "h"):
            hold_current_position(arm)
            continue

        if choice == "r":
            if debug_pose is None:
                log("还没有记录调试位，不能自动回调试位。")
                continue
            if safe_movel(
                arm,
                debug_pose,
                duration=args.movel_time,
                timeout=args.motion_timeout,
                label="回调试位",
            ):
                log("已回到调试位，继续保持使能。")
            continue

        if choice == "d":
            try:
                confirm = input("确认要失能机械臂并退出？输入 yes: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                log("\n已取消失能，继续保持。")
                hold_current_position(arm)
                continue
            if confirm == "yes":
                return True
            log("已取消失能，继续保持。")
            continue

        log("无效输入。")


def quintic_time_scale(tau: float) -> float:
    tau = max(0.0, min(1.0, float(tau)))
    return tau * tau * tau * (10.0 + tau * (-15.0 + 6.0 * tau))


def get_gripper_angle(arm, fallback_rad: float) -> float:
    try:
        joints = arm.GetArmJointMsgs().to_list(include_gripper=True)
        if len(joints) >= 7:
            return float(joints[6])
    except Exception:
        pass
    return float(fallback_rad)


def move_gripper_slow(
    arm,
    target_rad: float,
    duration: float,
    effort: float = 0.0,
    kp: float = 18.0,
    kd: float = 2.0,
    start_rad: Optional[float] = None,
):
    start = get_gripper_angle(arm, target_rad) if start_rad is None else float(start_rad)
    duration = max(float(duration), 0.1)
    steps = max(int(duration / 0.05), 2)
    sleep_s = duration / steps

    for i in range(1, steps + 1):
        s = quintic_time_scale(i / steps)
        angle = start + s * (target_rad - start)
        if not arm.GripperCtrl(
            gripper_angle=angle,
            gripper_effort=effort,
            gripper_enable=True,
            kp=kp,
            kd=kd,
        ):
            raise RuntimeError("夹爪控制失败")
        time.sleep(sleep_s)


def main():
    parser = argparse.ArgumentParser(description="机械臂 + 杆电机联动顺序测试")
    parser.add_argument("--can", default="can0", help="机械臂 CAN 接口名")
    parser.add_argument("--rod-port", default="/dev/rodmotor", help="rodmotor 串口")
    parser.add_argument("--rod-baudrate", type=int, default=921600, help="rodmotor 波特率")
    parser.add_argument("--motion-timeout", type=float, default=120.0, help="单步运动等待超时")
    parser.add_argument("--movej-time", type=float, default=4.0, help="MoveJ 默认时长")
    parser.add_argument("--movel-time", type=float, default=4.5, help="MoveL 默认时长")
    parser.add_argument("--gripper-open-deg", type=float, default=-1.2, help="夹爪打开角度")
    parser.add_argument("--gripper-close-deg", type=float, default=108.5, help="夹爪关闭角度")
    parser.add_argument("--gripper-open-time", type=float, default=2.0, help="夹爪打开耗时")
    parser.add_argument("--gripper-close-time", type=float, default=2.0, help="夹爪关闭耗时")
    parser.add_argument(
        "--disable-on-error",
        action="store_true",
        help="异常退出时也自动停止控制循环并失能机械臂（默认不自动失能）",
    )
    parser.add_argument(
        "--keep-enabled-on-success",
        action="store_true",
        help="流程正常完成后也保持机械臂使能（默认正常完成会清理失能）",
    )
    parser.add_argument("--target-x", type=float, default=-50.0, help="MoveL 目标 X (cm)")
    parser.add_argument("--target-y", type=float, default=-5.98, help="MoveL 目标 Y (cm)")
    parser.add_argument("--target-z", type=float, default=24.0, help="MoveL 目标 Z (cm)")
    parser.add_argument("--target-rx", type=float, default=60.0, help="MoveL 目标 Rx (deg)")
    parser.add_argument("--target-ry", type=float, default=-0.12, help="MoveL 目标 Ry (deg)")
    parser.add_argument("--target-rz", type=float, default=90.35, help="MoveL 目标 Rz (deg)")
    parser.add_argument(
        "--debug-joints-deg",
        nargs=6,
        type=float,
        default=DEFAULT_DEBUG_JOINTS_DEG,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="调试位关节角（度）",
    )
    parser.add_argument(
        "--rod-zero-angle",
        type=float,
        default=0.0,
        help="rodmotor 零位角度（度）",
    )
    args = parser.parse_args()

    target_pose_m, target_rpy_rad = parse_pose_args(args)
    target_pose = make_pose(target_pose_m, target_rpy_rad)
    debug_joints = deg_list_to_rad(args.debug_joints_deg)

    log("=" * 70)
    log(" 机械臂 + 杆电机联动顺序测试")
    log("=" * 70)
    log(f"机械臂 CAN: {args.can}")
    log(f"rodmotor: {args.rod_port} @ {args.rod_baudrate}")
    log(
        "MoveL 目标: "
        f"({args.target_x:.2f}, {args.target_y:.2f}, {args.target_z:.2f}) cm, "
        f"({args.target_rx:.2f}, {args.target_ry:.2f}, {args.target_rz:.2f}) deg"
    )

    arm = ELA3Interface(
        can_name=args.can,
        logger_level=LogLevel.INFO,
        gravity_feedforward_ratio=1.0,
    )
    rod = RodMotorClient(
        port=args.rod_port,
        baudrate=args.rod_baudrate,
        timeout=0.3,
        auto_connect=False,
    )

    completed = False
    cleanup_allowed = False
    debug_pose = None
    try:
        log("\n[1/10] 连接机械臂")
        if not arm.ConnectPort():
            raise RuntimeError("机械臂连接失败")
        if not arm.EnableArm():
            raise RuntimeError("机械臂使能失败")
        arm.start_control_loop(rate_hz=200.0)
        time.sleep(0.2)
        log("机械臂已连接并进入控制循环")

        log("\n[2/10] 连接 rodmotor")
        if not rod.connect():
            raise RuntimeError(f"rodmotor 连接失败: {args.rod_port}")
        log("rodmotor 已连接")

        current_joint = arm.GetArmJointMsgs().to_list(include_gripper=False)
        log(f"当前关节: {[round(math.degrees(v), 2) for v in current_joint]}")

        log("\n[3/10] 机械臂 MoveJ 到全零构型，同时 rodmotor 到零位")
        arm_ok = arm.MoveJ([0.0] * 6, duration=args.movej_time, block=False)
        if not arm_ok:
            raise RuntimeError("机械臂 MoveJ 到零位失败")
        rod.write_angle(args.rod_zero_angle, wait_response=False)
        log("rodmotor 零位命令已下发")
        wait_motion(arm, args.motion_timeout)
        log("机械臂已到零构型")

        log("\n[4/10] 夹爪缓慢打开")
        move_gripper_slow(
            arm,
            math.radians(args.gripper_open_deg),
            args.gripper_open_time,
            effort=0.0,
        )
        log("夹爪打开完成")

        log("\n[5/10] 机械臂 MoveJ 到调试位")
        if not arm.MoveJ(debug_joints, duration=args.movej_time, block=False):
            raise RuntimeError("机械臂 MoveJ 到调试位失败")
        wait_motion(arm, args.motion_timeout)
        debug_pose = arm.GetArmEndPoseMsgs()
        log("机械臂已到调试位")

        prompt("[6/10] 请确认当前状态后")

        log("\n[7/10] 机械臂 MoveL 到目标位姿")
        if not safe_movel(
            arm,
            target_pose,
            duration=args.movel_time,
            timeout=args.motion_timeout,
            label="MoveL 到目标位姿",
        ):
            wait_for_operator_recovery(
                arm,
                debug_pose,
                args,
                "MoveL 到目标位姿失败或不可达",
            )
            cleanup_allowed = True
            return
        log("机械臂已到目标位姿")

        prompt("[8/10] 目标位姿确认后")

        log("\n[9/10] 夹爪缓慢关闭")
        move_gripper_slow(
            arm,
            math.radians(args.gripper_close_deg),
            args.gripper_close_time,
            effort=0.12,
        )
        log("夹爪关闭完成")

        log("\n[10/10] 机械臂 MoveL 回调试位")
        if not safe_movel(
            arm,
            debug_pose,
            duration=args.movel_time,
            timeout=args.motion_timeout,
            label="MoveL 回调试位",
        ):
            wait_for_operator_recovery(
                arm,
                debug_pose,
                args,
                "MoveL 回调试位失败或不可达",
            )
            cleanup_allowed = True
            return
        log("机械臂已回到调试位")

        log("\n流程完成")
        completed = True
        cleanup_allowed = not args.keep_enabled_on_success

    except KeyboardInterrupt:
        log("\n用户中断")
        if args.disable_on_error:
            cleanup_allowed = True
        else:
            wait_for_operator_recovery(arm, debug_pose, args, "用户中断")
            cleanup_allowed = True
    except Exception as exc:
        log(f"\n流程异常: {exc}")
        if args.disable_on_error:
            cleanup_allowed = True
        else:
            wait_for_operator_recovery(arm, debug_pose, args, str(exc))
            cleanup_allowed = True
    finally:
        if cleanup_allowed:
            try:
                arm.cancel_motion()
            except Exception:
                pass
            try:
                arm.stop_control_loop()
            except Exception:
                pass
            try:
                arm.DisableArm()
            except Exception:
                pass
            try:
                arm.DisconnectPort()
            except Exception:
                pass
        else:
            log("机械臂保持当前使能状态，未自动 stop_control_loop/DisableArm/DisconnectPort")
        try:
            rod.close()
        except Exception:
            pass

        if cleanup_allowed:
            log("资源已释放")
        elif completed:
            log("流程完成，按参数要求保持机械臂使能")


if __name__ == "__main__":
    main()
