# 杆电机固件

这是一个独立的 ESP32 杆电机固件，用于控制第五关节外杆。

## 支持命令

| T | 功能 |
|---:|---|
| 101 | 读取 raw 位置 |
| 102 | 控制 raw 位置 |
| 103 | 读取弧度位置 |
| 104 | 控制弧度位置 |
| 105 | 设置 PID |
| 106 | 重置 PID |
| 107 | 修改 ID |
| 108 | 设置当前位置为中位 |

默认杆电机 ID 为 `16`。执行 `T=107` 后，固件会自动更新当前运行时 ID。

## 编译与烧录

rodmotor 设备固定使用串口别名 `/dev/rodmotor`。

当前硬件识别信息：

- USB 芯片：Silicon Labs CP2102
- `idVendor`: `10c4`
- `idProduct`: `ea60`
- `serial`: `0001`

安装固定串口规则：

```bash
sudo cp resources/udev/99-rodmotor.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

重新插拔 rodmotor 后确认：

```bash
ls -l /dev/rodmotor
```

```powershell
pio run
pio run -t upload --upload-port /dev/rodmotor
pio device monitor -p /dev/rodmotor -b 921600
```

## 串口协议

每行发送一个 JSON：

```json
{"T":101}
{"T":102,"pos":2047,"spd":1000,"acc":50}
{"T":104,"rad":1.57,"spd":1000,"acc":50}
{"T":105,"p":16,"i":0,"d":0}
{"T":107,"raw":16,"new":17}
{"T":108}
```

`id` 对 `101-106` 和 `108` 都是可选的，默认使用当前运行时杆电机 ID。  
`T=107` 支持 `raw`、`old` 或 `id` 指定当前 ID。

## Python SDK 用法

```python
from el_a3_sdk import RodMotorClient

rod = RodMotorClient()
rod.connect()

angle = rod.read_angle()
rod.write_angle(180.0, spd=1000, acc=50)

rod.close()
```

`write_angle()` 接收单位为度，SDK 不再额外限制角度范围；实际可达范围以电机机构与固件限位为准。
