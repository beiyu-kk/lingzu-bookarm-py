# 书脊识别推理轻量包

这个文件夹只保留在新电脑上运行 YOLO 书脊分割推理所需的文件，不包含训练集、训练脚本、日志或 WebUI。

## 文件说明

- `infer.py`：命令行推理脚本。
- `weights/best.pt`：已经训练好的书脊分割模型权重。
- `requirements.txt`：推理所需的最小 Python 依赖。
- `samples/test.jpeg`：用于快速测试的样例图片。

## 环境安装

进入本目录：

```bash
cd book_spine_inference_light
```

创建并激活 Python 虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
```

如果是 Windows，激活命令改为：

```powershell
.venv\Scripts\activate
```

安装 CPU 版 PyTorch 和推理依赖：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

上面的 CPU 版安装方式兼容性最好，适合大多数新电脑。  
如果新电脑有 NVIDIA 显卡并且想用 GPU 推理，请先按 PyTorch 官网说明安装匹配 CUDA 版本的 `torch` 和 `torchvision`，再执行：

```bash
pip install -r requirements.txt
```

## 运行推理

测试样例图片：

```bash
python infer.py --source samples/test.jpeg
```

推理单张图片：

```bash
python infer.py --source /path/to/image.jpg --output runs
```

推理整个图片文件夹：

```bash
python infer.py --source /path/to/images --output runs
```

常用参数示例：

```bash
python infer.py --source /path/to/images --conf 0.2 --imgsz 1024 --max-det 600
```

参数含义：

- `--source`：输入图片或图片文件夹，必填。
- `--output`：结果输出目录，默认是当前目录下的 `runs`。
- `--conf`：置信度阈值，数值越低检出越多，默认 `0.25`。
- `--imgsz`：推理图片尺寸，默认 `1024`。
- `--max-det`：单张图片最多检测数量，默认 `600`。

## 输出结果

每张输入图片都会在输出目录下生成一个同名文件夹，里面包含：

- `*_overlay.jpg`：在原图上绘制书脊分割结果的预览图。
- `masked/*.png`：每个书脊的透明背景裁剪图。
- `rectified/*.png`：每个书脊旋转扶正后的裁剪图。

## 快速验证

安装完成后执行：

```bash
python infer.py --source samples/test.jpeg --output runs/test
```

如果看到类似下面的输出，就说明推理环境正常：

```text
samples/test.jpeg: 18 spines -> runs/test/test/test_overlay.jpg
Done. Segmented 18 book spines from 1 image(s).
```
