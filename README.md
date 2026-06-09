# EL-A3 Python SDK

> 7-DOF 桌面机械臂纯 Python SDK，基于 Direct CAN 通信，支持多臂管理、Pinocchio 动力学、S-curve 轨迹规划。

---

## 文档

| 语言 | 文件 |
|------|------|
| 中文 | [README_zh.md](README_zh.md) |
| English | [README_en.md](README_en.md) |

---

## 快速开始

```bash
# 创建 conda 环境
conda create -n lingarm python=3.12
conda activate lingarm

# 安装 SDK（开发模式）
pip install -e .

# 基础安装已包含 RealSense / Open3D / OpenCV 视觉依赖

# 如需运动学/动力学功能
pip install -e ".[dynamics]"

# 如需 MotorStudio GUI（PyQt6 + PyVista/VTK 3D 可视化）
sudo apt install -y libgl1 libglx-mesa0 libgl1-mesa-dri libxrender1 libxcb-xinerama0 libxcb-cursor0
pip install -e ".[gui]"

# 如需使用独立杆电机，请安装固定串口别名 /dev/rodmotor
sudo bash scripts/rodmotor_test/install_rodmotor_udev.sh
# 重新插拔杆电机后确认别名存在
ls -l /dev/rodmotor

# 配置 CAN 接口
sudo bash scripts/setup_can.sh can0 1000000

# 运行示例
python3 demo/control_loop_demo.py
python3 demo/zero_torque_mode.py --gravity

# 启动 MotorStudio 上位机
lingzu-bookarm-debugger
```

---

## 项目结构

| 目录/文件 | 说明 |
|-----------|------|
| `el_a3_sdk/` | Python 包核心代码 |
| `demo/` | 示例脚本 |
| `docs/` | SDK API 协议文档 |
| `resources/` | URDF、Meshes、惯性参数配置 |
| `scripts/` | CAN 配置、测试脚本 |
| `scripts/camera_test/` | RealSense 点云可视化与选点脚本 |
| `pyproject.toml` | pip 安装配置 |

---

## 依赖

- **必需**: `numpy`, `pyyaml`, `pyserial`, `pyrealsense2`, `open3d`, `opencv-python`
- **可选**: `pin` (Pinocchio) - 运动学/动力学
- **视觉**: `pyrealsense2`, `open3d`, `opencv-python` — RealSense RGB-D、点云、书脊识别
- **MotorStudio GUI**: `pyqt6`, `pyqtgraph`, `pyvista`, `pyvistaqt`, `vtk` — GUI + 3D URDF/点云可视化

独立杆电机通过固定串口别名 `/dev/rodmotor` 连接。安装项目环境时请运行 `sudo bash scripts/rodmotor_test/install_rodmotor_udev.sh` 安装 udev 规则；如果更换 USB 转串口设备后别名没有出现，请根据 `ls -l /dev/serial/by-id/` 中的新设备信息更新 `resources/udev/99-rodmotor.rules`，再重新运行安装脚本并重新插拔设备。

---

## 示例

| 示例 | 说明 |
|------|------|
| `control_loop_demo.py` | 200Hz 控制循环、MoveJ、JointCtrl |
| `xbox_control.py` | Xbox 手柄笛卡尔控制；D-pad 上/下小步打开/关闭夹爪，按住连续平滑动作 |
| `zero_torque_mode.py` | 零力矩拖动模式 |
| `dynamics_demo.py` | 重力补偿、雅可比、质量矩阵 |
| `trajectory_demo.py` | S-curve 和样条轨迹规划 |
| `cartesian_control_demo.py` | 笛卡尔空间控制 |
| `motion_control.py` | 路径点运动控制 |
| `waypoint_loop_real.py` | 路径点循环测试 |
| `read_joints.py` | 关节状态读取 |
