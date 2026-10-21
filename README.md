# 知叶 · 面向农科学习者的作物叶片智能识别与辅助学习系统

知叶是一个面向农科学习者的作物叶片智能识别与辅助学习系统：用 PyTorch 训练 A/B/C/D 四个轻量分类模型，通过 Flask + 原生 HTML/CSS/JS 提供摄像头实时识别、图片上传识别、Grad-CAM 可解释性热力图和实验数据看板。

## 功能

- **摄像头实时识别**：浏览器打开摄像头，按设定间隔抓取中心正方形画面并返回 Top-k 类别与置信度；上一帧未返回时不会重复发送，切换到后台时暂停。
- **上传图片识别**：拖拽或选择 JPG/PNG 图片（单张不超过 10 MB），显示 Top-k 类别、中文名、学名、置信度与推理耗时。
- **三个模型组、四个模型**：可在「标本（5）／标本（32）／实景（5）」之间切换，也可在组内切换 A/B/C/D。切换模型或模型组时会保留当前图片并自动重新识别。
- **Grad-CAM 可解释性**：对当前图片生成模型关注区域的热力图叠加图，A/B/C/D × 三个模型组均可用；开关打开或关闭都会重新请求当前图片。
- **数据看板**：扫描 `pic/` 下的 6 个实验目录，用本地 ECharts 渲染交互式折线图/柱状图，数值通过鼠标悬停的 tooltip 查看。
- **系统操作（演示）**：「添加数据」「模型训练」按钮只显示演示提示和假进度条，不会上传数据、不会启动训练、不会写入 `runs/`。

## 模型

| 模型 | 结构 | 用途 |
|---|---|---|
| A | MobileNetV3-Small | 轻量基线 |
| B | MobileNetV2 | 对比模型 |
| C | LeafNet-MSFF（MobileNetV3-Small + 多尺度特征融合） | 验证 MSFF 增益 |
| D | LeafNet-MSFF+CBAM（MobileNetV3-Small + MSFF + CBAM） | 主模型 |
| E | MobileNetV3-Small+CBAM | 仅用于消融实验，网页不加载 |

五个类别为紫荆、海桐、香樟、桂花、紫薇（Flavia 编号 4/12/15/17/20）；「标本（32）」使用 Flavia 全部 32 类。

## 目录结构

```text
DeepLearning/
├─ webapp/                 # Flask 入口、模型服务、Grad-CAM、看板数据与前端页面
│  ├─ app.py               # 启动入口与 JSON API
│  ├─ model_service.py     # 扫描、校验并加载三个模型组的 A/B/C/D
│  ├─ gradcam.py           # Grad-CAM 热力图
│  ├─ gallery_service.py   # pic/ 看板数据
│  ├─ chart_data.py        # 从 runs/ 与 pic/ 生成 ECharts 图表规格
│  ├─ templates/index.html
│  └─ static/              # CSS、JS 与本地 echarts.min.js（无 CDN）
├─ src/                    # 数据管道、模型定义、训练与工具函数
├─ tools/                  # 数据划分、缓存、评测与图表生成脚本
├─ data/                   # 原始数据、注册表与划分清单
├─ runs/                   # 训练输出与模型权重（生成的，不入库）
├─ pic/                    # 实验图表与看板用 CSV
└─ results/                # 报告截图等产物
```

所有命令都在**项目根目录**执行，`src/` 与 `webapp/` 以 Python 模块方式运行。

## 环境与依赖

- Python **3.10**（已在 3.10.21 上验证）
- 仅需 CPU，不要求 NVIDIA GPU
- 已验证版本：Flask 3.1.3、PyTorch 2.5.1、torchvision 0.20.1、Pillow 11.1.0、NumPy 2.0.1、pandas 2.3.3

安装 CPU 版 PyTorch（版本需配套）：

```bash
python -m pip install "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cpu
```

安装网页运行、数据处理与训练所需的其他包：

```bash
python -m pip install "flask==3.1.3" "pillow==11.1.0" "numpy==2.0.1" "pandas==2.3.3" "scikit-learn==1.7.2" "matplotlib==3.10.9"
```

> 如果使用 Conda 环境，请先激活环境（例如 `conda activate torch_env`），后续命令里的 `python` 都指该环境中的解释器。
>
> 首次训练会自动下载 ImageNet 预训练的 MobileNetV3-Small 与 MobileNetV2 权重，需要能访问 `download.pytorch.org`。

## 数据准备

### 原始数据不在仓库中

GitHub 仓库**不包含**原始图片和训练产物。克隆后需要自行放置：

| 路径 | 内容 | 是否入库 |
|---|---|---|
| `data/Leaves/` | Flavia 叶片图片，共 1798 张 JPG（32 类，文件名如 `1001.jpg`） | 否 |
| `data/leaf/` | 实景视频抽帧，按 `cercis_test/`、`osmanthus_test_hard/` 等子目录存放 | 否 |
| `data/cache/` | 训练用的 320px 图片缓存 | 否 |
| `runs/` | 训练输出与 `best.pt` 权重 | 否 |

仓库**包含**以下元数据，无需重新生成：

- `data/class_map.json`：Flavia 32 类映射（唯一类别映射来源）
- `data/class_registry.json`：五类注册表
- `data/class_registry_all32.json`：32 类注册表
- `data/splits/5class/`：五类白底划分（178/58/58，共 294 张）
- `data/splits/all32/`：32 类白底划分（1082/358/358，共 1798 张）
- `data/splits/live5/`：实景五类划分（301/79/52，视频级划分）
- `data/splits/live_all/`、`data/splits/live_all32/`：实景清单

`data/Leaves/` 中的图片应直接放在该目录下（不要额外套一层文件夹），因为划分 CSV 中的 `filename` 是相对 `data/Leaves/` 的路径；`data/leaf/` 同理，CSV 中的路径形如 `cercis_test/leaf01_000000.jpg`。

### 可选：生成 320px 缓存

缓存能显著加快 CPU 训练；没有缓存时训练会直接读取原图，速度较慢但结果不受影响。

```bash
python tools/prepare_cache.py --src data/Leaves --dst data/cache/leaf_320 --long-edge 320
python tools/prepare_cache.py --src data/Leaves --dst data/cache/leaf_320_all32 --long-edge 320
```

### 可选：重新生成划分

一般情况下仓库里的划分清单已经够用，不需要执行。若要重建：

```bash
python tools/build_dataset.py --leaves data/Leaves --classes 4,12,15,17,20 --subset 5class --out data/splits --registry data/class_registry.json --seed 42 --materialize
python tools/build_dataset.py --leaves data/Leaves --classes all --subset all32 --out data/splits --registry data/class_registry_all32.json --seed 42 --materialize
python tools/build_live_dataset.py --leaf-dir data/leaf --mode video-split --registry data/class_registry.json --out data/splits/live5
```

## 模型权重准备

网页启动时会扫描 `runs/`，按固定目录名查找 A/B/C/D 的 `best.pt`：

```text
runs/
├─ spec5/                         # 标本（5）：白底五类
│  ├─ A_mnv3s_seed42/best.pt
│  ├─ B_mnv2_seed42/best.pt
│  ├─ C_leafnet_msff_seed42/best.pt
│  └─ D_leafnet_msff_cbam_seed42/best.pt
├─ spec32/                        # 标本（32）：白底 32 类
│  ├─ A_mnv3s_seed42/best.pt
│  ├─ B_mnv2_seed42/best.pt
│  ├─ C_leafnet_msff_seed42/best.pt
│  └─ D_leafnet_msff_cbam_seed42/best.pt
└─ live5/                         # 实景（5）：实景五类
   ├─ A_mnv3s_seed42/best.pt
   ├─ B_mnv2_seed42/best.pt
   ├─ C_leafnet_msff_seed42/best.pt
   └─ D_leafnet_msff_cbam_seed42/best.pt
```

说明：

- 目录名与模型名必须完全一致；`seed42` 来自训练时的 `--seed 42`。
- `best.pt` 是必需文件；同目录的 `test_metrics.json` 可选，用于在页面上显示测试准确率等指标。
- `E_mnv3s_cbam_seed42` 只用于消融实验，网页不会加载。
- 启动时会把 checkpoint 里的 `class_keys`、`flavia_labels`、`num_classes` 与对应注册表校验，不一致时该模型标记为不可用。

### 没有权重时如何训练

三个模型组分别训练，命令如下（均从项目根目录执行，使用 A/B/C/D 四个模型）：

**1. 标本（5）：白底五类**

```bash
python -m src.train --models A B C D --data data/splits/5class --registry data/class_registry.json --num-classes 5 --cache data/cache/leaf_320 --epochs 40 --batch-size 32 --patience 20 --output runs/spec5 --num-workers 0
```

**2. 标本（32）：白底 32 类**

```bash
python -m src.train --models A B C D --data data/splits/all32 --registry data/class_registry_all32.json --num-classes 32 --cache data/cache/leaf_320_all32 --epochs 40 --batch-size 32 --patience 20 --output runs/spec32 --num-workers 0
```

**3. 实景（5）：实景五类**

```bash
python -m src.train --models A B C D --data data/splits/live5 --image-root data/leaf --registry data/class_registry.json --num-classes 5 --epochs 40 --patience 20 --output runs/live5 --num-workers 0
```

每条命令会在 `--output` 下生成 `<模型名>_seed42/` 目录，包含 `best.pt`、`last.pt`、`history.csv`、`test_metrics.json`、`train.log` 等文件。训练完成后即可启动网页。

> 训练默认使用 CPU。Windows 上建议保留 `--num-workers 0`，避免多进程 DataLoader 报错；`--cache` 指向的缓存目录不存在时会自动回退读取原图。

## 启动网页

```bash
python -m webapp.app --checkpoints runs --host 127.0.0.1 --port 5000
```

浏览器打开：

```text
http://127.0.0.1:5000
```

参数说明：

- `--checkpoints runs`：权重根目录，默认就是 `runs`
- `--host 127.0.0.1`：本机监听；不要改成局域网 IP 后再用摄像头（见下方说明）
- `--port 5000`：端口被占用时可改成 `5001` 等

### 摄像头权限注意

- 必须通过 `http://127.0.0.1:5000` 或 `http://localhost:5000` 打开页面。浏览器只在这类**安全上下文**中允许 `getUserMedia`；用 `http://<局域网IP>:5000` 打开时摄像头会被拒绝。
- 首次使用需在浏览器弹窗中允许相机权限。
- Windows 需在「设置 → 隐私和安全性 → 相机」中允许浏览器访问相机，并确认没有其他程序占用摄像头。
- 即使拒绝摄像头权限，**上传图片识别仍然可用**。

## 使用说明

### 识别页

1. 选择模型组：**标本（5）／标本（32）／实景（5）**。
2. 选择模型：**A/B/C/D**，按钮上会显示该模型在对应测试集上的准确率（有 `test_metrics.json` 时）。
3. 选择识别方式：
   - **摄像头**：点击开启，页面显示正方形取景框，把叶片放在框内；点击「拍照」可冻结当前帧。
   - **上传图片**：拖拽或点击选择 JPG/PNG，自动开始识别。
4. 查看结果：Top-3（五类）或 Top-5（32 类）类别、中文名/学名、置信度、推理耗时；低置信度时会给出提示。
5. 打开 **Grad-CAM** 开关，可查看模型关注区域的热力图；关闭后恢复普通图片。
6. 切换模型或模型组时，当前图片/拍照快照会保留，并自动用新模型重新识别，无需重新选择图片。

### 数据看板

顶部导航进入「数据看板」，可切换以下 6 个目录：

| 目录 | 内容 |
|---|---|
| `spec-train-test` | 五类白底训练曲线与测试指标 |
| `spec-train-test-all` | 32 类白底训练曲线与测试指标 |
| `spec-train-live-test` | 五类白底模型在普通/困难实景帧上的指标 |
| `spec-train-live-test-all` | 32 类白底模型在实景帧上的指标 |
| `live-train-test` | 实景五类训练曲线与实景测试指标 |
| `melt-test` | 消融实验结果与模块增益 |

图表全部为 ECharts 交互图；把鼠标悬停在柱子/折线上即可查看数值。页面不展示原始图片网格、docx 或下载链接。

### 系统操作页

「添加数据」「模型训练」为演示按钮：前者弹出“未接入真实数据接口”，后者显示 2–3 秒假进度条并提示“未启动真实训练”。它们不会写入任何文件。

## 常见问题

**打开页面后模型全部不可用 / 识别返回 503**

检查 `runs/<组>/<模型名>_seed42/best.pt` 是否存在，目录名是否与上文字母大小写、下划线完全一致。可访问 `http://127.0.0.1:5000/api/models` 查看每个模型的 `available` 与 `error` 字段。若 `error` 提示 `class_keys` 或 `num_classes` 与注册表不一致，说明权重与注册表不匹配，应使用对应数据重新训练。

**只训练了部分模型组，网页还能用吗？**

可以。缺少权重的模型会显示为不可用，其他已加载的模型和模型组仍可正常使用；对应的数据看板可能只显示部分图表。

**端口 5000 被占用**

换一个端口启动：

```bash
python -m webapp.app --checkpoints runs --host 127.0.0.1 --port 5001
```

然后打开 `http://127.0.0.1:5001`。

**训练时报找不到图片（`FileNotFoundError`）**

确认原始数据已放在正确位置：白底数据在 `data/Leaves/` 下且文件名与 CSV 的 `filename` 一致；实景数据在 `data/leaf/` 下且保留 `cercis_test/` 等子目录。仓库本身不包含这些图片。

**Windows 上训练卡住或 DataLoader 报错**

使用 `--num-workers 0`（上文三条命令已包含）。网页推理使用 Flask 进程内的 CPU 推理，不涉及 DataLoader worker。

**摄像头打不开**

确认地址是 `127.0.0.1` 或 `localhost`、浏览器已允许相机、Windows 相机隐私权限已开启、没有其他程序占用摄像头。仍不可用时直接使用上传图片。

**数据看板显示“暂无可绘制图表”**

该目录依赖的 CSV 或 `runs/` 下的训练产物缺失。`runs/` 不入库，克隆仓库后如果没有训练或没有拷贝权重，训练曲线和部分测试柱状图会缺失；`pic/` 中的 CSV 已在仓库中。

## 数据、权重与离线说明

- `.gitignore` 已排除 `data/Leaves/`、`data/leaf/`、`data/cache/`、`runs/`、`*.pt`、`*.pth` 等大文件，因此原始数据、训练缓存和模型权重**不在 GitHub 仓库中**，需要按上文自行准备。
- 仓库包含类别映射、注册表、划分 CSV、`pic/` 图表数据与报告截图，克隆后即可查看数据看板中的已有 CSV 图表。
- 前端不引用任何 CDN：CSS、JavaScript 与 ECharts 都在 `webapp/static/` 本地加载，断网也能正常打开页面（模型推理同样在本地 CPU 完成）。

## 类别对照

- `data/class_map.json`：Flavia 32 类映射，程序使用的唯一类别映射来源。
- `data/Flavia类别对照表.md`：32 类的中文名、学名、编号区间与图片张数，便于人工核对。
