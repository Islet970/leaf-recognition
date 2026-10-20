# 校园植物叶片识别 —— 可执行实施方案

## Context（背景与现状）

本项目要完成四件事：**（1）以 §13 的 32 类映射生成类别注册表与数据清单（分类校验已完成）；（2）按 Flavia 编号划出 5 类训练/验证/测试集；（3）训练 A/B/C/D 四个可对比模型并产出报告所需图表；（4）做一个支持摄像头实时识别、上传图片、模型切换和 Grad-CAM 的网页。** 后续再扩到全部 32 类。

现状核实结果（2026-10-19）：

- [data/Leaves/](data/Leaves/) 是 **Flavia 公开数据集的扁平副本**：1798 张 1600×1200 JPEG，文件名 1001–3621，**没有类别子文件夹**，类别由编号区间决定。
- **类别映射以本计划 §13 的 32 类表为准**：Flavia 编号 1–12、14–33（无 13），编号区间、学名、英文名、中文名均已核对。项目内**没有也不需要 `cato.txt`**；后续脚本、注册表、网页一律不再依赖它，旧文档中的“cato label”统一理解为 Flavia 原始编号。
- **分类校验已完成**：32/32 编号一致、32/32 区间一致，无需重新编码；结果已落盘为 [data/Flavia类别对照表.md](data/Flavia类别对照表.md)（32 类全表、1798 张、五类子集划分表）。
- **实景视频帧已抽取（当前版本）**：`data/leaf/<class>_test/` 共 **432 张**普通实景帧（83/96/100/86/67）；`data/leaf/<class>_test_hard/` 的 227 张困难帧**仅用于实景鲁棒性测试**（§15.9，不参与训练/主测试；全量困难帧每类 45/39/50/40/53，比 leaf04 子集更均衡）。文件名前缀 `leaf01`–`leaf05` 表示 5 段视频，可按视频划分 train/val/test。`data/test_preview/` 有预览与抽帧统计。旧版 791 张清单已被当前数据取代。
- [Desktop/Leaves/](../Leaves/) 下的 5 个学名文件夹是 Flavia 白底扫描图，可与 `data/Leaves` 对应区间互相核对；训练与评测仍以 `data/Leaves` 为准。
- 报告模板要求：A/B/C/D 四模型对比、训练 loss/acc 曲线、测试 loss/acc、至少 3 个样本对比、消融实验，以及支持模型切换/摄像头/上传的网页。
- 本机 **无 NVIDIA GPU**：Ryzen 7 7840HS（8C/16T）、15.2 GB RAM；`torch_env` 为 CPU 版 PyTorch 2.5.1；缺 scikit-learn、matplotlib、pandas、flask、tqdm。
- 现有 [tools/extract_frames.py](tools/extract_frames.py)、[tools/extract_test_frames.py](tools/extract_test_frames.py) 是已完成的数据准备脚本，保留不动。

已确认的决策：

| 项目 | 决策 |
|---|---|
| 类别标准 | **以本计划 §13 的 32 类映射为准**（Flavia 编号 + 文件编号区间）；不依赖 `cato.txt` |
| 当前训练 | 5 类子集：Flavia 编号 **4/12/15/17/20**（紫荆、海桐、香樟、桂花、紫薇），共 294 张 |
| 后续扩展 | 全 32 类（编号 1–12、14–33），1798 张；通过 `--classes` 配置切换，不改代码 |
| 划分 | 按类分层 6:2:2，固定 seed，**seed 按 Flavia 编号生成**，扩类时已有类别的划分不变 |
| 基础网络 | MobileNetV3-Small（ImageNet 预训练），分类头 `num_classes` 可配置（5 或 32） |
| 实景照片 | **已有 432 张普通实景帧**（`data/leaf/*_test/`）；另有 227 张困难帧仅用于鲁棒性测试（§15.9）；5 类可按视频划分实景训练集；`data/campus/<key>/` 继续预留其他物种拍摄 |
| 最终模型名 | **LeafNet-MSC**（Multi-Scale Fusion + CBAM，基于 MobileNetV3-Small） |
| 现有交付物 | `data/Flavia类别对照表.md`（已完成） |
| 待新增交付物 | `data/class_map.json`（32 类映射，供程序读取）、`data/zh_names.json`、`data/class_registry.json`、`data/class_registry_all32.json`、5 类与 32 类两套训练清单、实景测试清单 |

---

## 1. 数据集与划分

### 1.1 Flavia 编号映射与派生注册表

**权威映射**是本文 §13 的 32 类表，已同步为 [data/Flavia类别对照表.md](data/Flavia类别对照表.md)。程序读取 `data/class_map.json`（由 §13 一次性生成并提交），不再解析任何 `cato.txt`：

```json
{
  "source": "plans.md §13",
  "classes": [
    {"flavia_label": 4, "key": "cercis_chinensis", "zh_name": "紫荆",
     "scientific": "Cercis chinensis", "common": "Chinese redbud", "range": [1123, 1194]},
    {"flavia_label": 12, "key": "pittosporum_tobira", "zh_name": "海桐",
     "scientific": "Pittosporum tobira (Thunb.) Ait. f.", "common": "Japanese cheesewood", "range": [2051, 2113]}
  ]
}
```

`tools/build_dataset.py` 的字段含义：

- `flavia_label`：Flavia 原始编号（1–12、14–33，**不连续**，不能直接当 PyTorch 标签）；
- `scientific`、`common`、`range`：来自 §13 映射；
- `key`：由学名生成的小写 slug（如 `cercis_chinensis`），在注册表中保持稳定；第 25 类手动覆盖为 `toona_sinensis`（原表拼写 `Tonna`，备注保留差异）；
- `zh_name`：来自 [公开数据集类别.md](公开数据集类别.md)，固化到 `data/zh_names.json`；
- `py_label`：当前训练子集内的连续 0..N-1，映射关系写入注册表。

`--classes` 支持三种写法：

```bash
--classes 4,12,15,17,20          # 当前五类，按 Flavia 编号
--classes cercis_chinensis,...   # 按 key
--classes all                    # 全部 32 类
```

`data/class_registry.json` 是当前五类训练、评测、网页的类别来源；后续全类另写 `data/class_registry_all32.json`。

```json
{
  "source": "data/class_map.json",
  "seed": 42,
  "classes": [
    {"flavia_label": 4, "py_label": 0, "key": "cercis_chinensis",
     "zh_name": "紫荆", "scientific": "Cercis chinensis",
     "common": "Chinese redbud", "range": [1123, 1194],
     "image_count": 67, "subset": "5class"}
  ]
}
```

当前五类的映射：

| Flavia 编号 | py label | key | 中文名 | 学名 | 区间 | 张数 |
|---:|---:|---|---|---|---|---:|
| 4 | 0 | cercis_chinensis | 紫荆 | Cercis chinensis | 1123-1194 | 67 |
| 12 | 1 | pittosporum_tobira | 海桐 | Pittosporum tobira | 2051-2113 | 55 |
| 15 | 2 | cinnamomum_camphora | 香樟 | Cinnamomum camphora | 2166-2230 | 61 |
| 17 | 3 | osmanthus_fragrans | 桂花 | Osmanthus fragrans | 2291-2346 | 54 |
| 20 | 4 | lagerstroemia_indica | 紫薇 | Lagerstroemia indica | 2486-2546 | 57 |

> 训练后不要再改 `--classes` 的顺序；checkpoint 里保存 `class_keys` 和 `flavia_labels`，网页加载时校验，不一致就拒绝加载。
>
> `py_label` 顺序就是 `--classes` 的书写顺序。若报告想按“桂花、香樟、海桐、紫薇、紫荆”的顺序展示，**在第一次训练前**把命令改成 `--classes 17,15,12,20,4` 即可；一旦开始训练就固定下来。

### 1.2 `Flavia类别对照表.md`（已完成）

已于 2026-10-19 生成 [data/Flavia类别对照表.md](data/Flavia类别对照表.md)（32 类、1798 张、五类子集划分表），分类校验工作到此完成。后续仅在映射或数据变更时用 `tools/make_class_table.py` 重新生成，内容包括：

- 中文名来源：从 [公开数据集类别.md](公开数据集类别.md) 提取为 `data/zh_names.json`（`flavia_label → 中文名`）；注册表和 MD 之后都读这个 JSON，不每次解析 Markdown；
- 校验摘要：32/32 编号一致、32/32 区间一致、无需重新编码；
- 32 类全表：`Flavia 编号 | key | 中文名 | 学名 | 英文名 | 编号区间 | 实际张数 | 备注`；
- 五类训练子集表（编号、key、中文名、区间、张数、6:2:2 划分）；
- 备注列标出：第 25 类原表 `Tonna sinensis` / 正确 `Toona sinensis`；第 6 类大小写差异；学名带作者缩写；
- 后续扩类说明：如何用 `--classes all`、如何补新图。

> 这份 MD 是“给人看”的对照表；`class_registry.json` 是“给程序用”的注册表。两者都由 §13 映射派生，§13 是唯一权威来源。

### 1.3 `tools/build_dataset.py`

职责：读取 `data/class_map.json` → 生成注册表 → 按所选类别分配图片 → 分层划分 → 泄漏检查 → 生成清单/目录。

关键函数：
- `load_class_map(path)`：读取 §13 派生的 32 类映射；校验编号唯一、range 合法且互不重叠。
- `build_registry(class_rows, zh_table, leaves_dir, selected)`：合并中文名、生成 key、统计实际张数、分配 `py_label`，写 `data/class_registry.json`。
- `select_classes(registry, spec)`：支持 `all`、逗号分隔 Flavia 编号、逗号分隔 key；所选类别不在映射中时报错。
- `assign_by_ranges(leaves_dir, selected)`：只扫描 `data/Leaves/*.jpg`，要求 4 位数字文件名；落在选中区间的文件分配 `flavia_label`/`py_label`，不在任何选中区间的忽略，一个文件落入多个区间则报错。
- `stratified_split(records, seed=42)`：每类按编号排序，`rng = random.Random(42 + flavia_label)` 打乱；`n_val=round(0.2n)`、`n_test=round(0.2n)`，其余 train。**用 Flavia 编号而不是类内序号做种子，扩类时已有类别的划分保持不变。**
- `leakage_check(...)`：三个 split 文件名两两不相交、并集等于选中图片总数；SHA-256 查重（同内容跨类直接报错）；可选 dHash 近重复报告。
- `write_manifest(...)`：`data/splits/<subset>/{train,val,test,all}.csv`，列 `filename,image_id,flavia_label,py_label,key,zh_name,scientific,common,split,sha256`。
- `--materialize`：硬链接（失败则复制）生成 `data/dataset/<subset>/{train,val,test}/<key>/*.jpg`，仅供人工检查；训练仍以 CSV 为准。
- `--classes`：`4,12,15,17,20` / key 列表 / `all`；默认五类。
- `--real-dir`：扫描 `data/leaf/<key>_test/*.jpg`，按目录映射生成只读实景测试清单（`--subset real`），不做任何划分。

五类划分（seed 按 Flavia 编号）：

| Flavia 编号 | key | 中文名 | 总数 | train | val | test |
|---:|---|---|---:|---:|---:|---:|
| 4 | cercis_chinensis | 紫荆 | 67 | 41 | 13 | 13 |
| 12 | pittosporum_tobira | 海桐 | 55 | 33 | 11 | 11 |
| 15 | cinnamomum_camphora | 香樟 | 61 | 37 | 12 | 12 |
| 17 | osmanthus_fragrans | 桂花 | 54 | 32 | 11 | 11 |
| 20 | lagerstroemia_indica | 紫薇 | 57 | 35 | 11 | 11 |
| **合计** | | | **294** | **178** | **58** | **58** |

全 32 类（同一 seed 规则，已按实际文件统计）：

| 项目 | 数值 |
|---|---:|
| 类别数 | 32（label 1–12、14–33） |
| 图片 | 1798 |
| train / val / test | **1082 / 358 / 358** |
| 每类张数 | 46–73 |

输出：
- `data/zh_names.json`：`flavia_label → 中文名`，从公开表提取一次，后续复用。
- `data/class_registry.json`：当前五类的唯一类别来源（`flavia_label`、`py_label`、`key`、中文名、学名、英文名、range、image_count、subset）。
- `data/class_registry_all32.json`：后续全类注册表，单独存放，**不覆盖五类注册表**。
- `data/splits/5class/{train,val,test,all}.csv`：当前五类。
- `data/splits/all32/{train,val,test,all}.csv`：后续全类；两套清单互不覆盖。
- `data/splits/<subset>/split_stats.json`：计数、哈希、seed、校验状态。
- 可选 `--materialize`：用 NTFS 硬链接（失败则复制）生成 `data/dataset/{train,val,test}/<key>/*.jpg`，方便人工检查；**训练仍以 CSV 清单为准**，避免目录被手动改动后不可复现。

> `data/Leaves` 中其余 1504 张（其他 27 类）不参与**当前五类**训练，原文件保持不动；以后扩到全部 32 类只需 `--classes all` 重跑 `build_dataset.py`；只有新增 Flavia 之外的物种才走 §6 的校园实景流程。

### 1.4 数据预览与缓存

- `tools/make_dataset_preview.py`：生成 `results/figures/fig2_1_dataset_samples.png`（5 类各 1 张，带中文学名），用于报告 2.1。
- `tools/prepare_cache.py`：把选中的图片（五类 294 张；全类 1798 张）EXIF 校正、长边缩到 320、JPEG q=95，输出 `data/cache/leaf_320/`，避免每轮重复解码 1600×1200 原图。缓存文件名沿用原编号。

### 1.5 实景测试帧与视频级划分（`data/leaf`，当前版本）

现状（2026-10-20 更新）：

- `data/leaf/<class>_test/`：**432 张**普通实景帧（camphor 83 / cercis 96 / lagerstroemia 100 / osmanthus 86 / pittosporum 67）。
- `data/leaf/<class>_test_hard/` 的 227 张困难帧**仅用于实景鲁棒性测试**（§15.9），不参与 train/val/主测试；全量困难帧每类 45/39/50/40/53，比只用 leaf04 的 35 张更均衡。
- 文件名前缀 `leaf01`–`leaf05` 表示 5 段视频。旧版 791 张清单已被当前数据取代；`data/splits/real/all.csv` 保留但不再使用。

目录 → 类别映射：

| 目录前缀 | 中文名 | Flavia 编号 | key |
|---|---|---:|---|
| camphor | 香樟 | 15 | cinnamomum_camphora |
| cercis | 紫荆 | 4 | cercis_chinensis |
| lagerstroemia | 紫薇 | 20 | lagerstroemia_indica |
| osmanthus | 桂花 | 17 | osmanthus_fragrans |
| pittosporum | 海桐 | 12 | pittosporum_tobira |

视频级划分（用于实景训练 L5，防止同一段视频的相近帧泄漏）：

| split | 视频 | 普通 `_test` |
|---|---|---:|
| train | leaf01 / leaf02 / leaf03 | 301 |
| val | leaf05 | 79 |
| test | leaf04 | 52 |

清单生成（T5.1）：

- `data/splits/live_all/test.csv`：E1（W5）与 E3b（W32）用**全量普通实景**（432），5 类标签；
- `data/splits/live5/{train,val,test}.csv`：E2（L5）视频级划分（301 / 79 / 52）；
- `data/splits/live_all32/test.csv`：E3b 用 32 类注册表的 `py_label`（5 类映射到 0–31）；
- CSV 列沿用 `filename,image_id,flavia_label,py_label,key,zh_name,scientific,common,split,sha256`；`filename` 为相对 `data/leaf` 的路径（如 `camphor_test/leaf04_000010.jpg`）。
- **红线**：同一视频不得跨 split；生成后做 SHA-256 跨 split 查重与 dHash 近重复报告；实景帧不得混入 Flavia 白底 train/val/test；困难帧仅用于鲁棒性测试（§15.9），不参与主测试/训练。

---

## 2. 模型设计与消融

所有模型输入 **128×128**、ImageNet 初始化，分类头宽度保持一致；`num_classes` 由注册表决定（当前 5，后续 32），保证两套实验的对比公平。

| 运行 | 模型 | 组成 | 参数量 | Grad-CAM 目标层（128 输入） |
|---|---|---|---:|---|
| A | MobileNetV3-Small 基线 | torchvision backbone + 576→1024→num_classes | 1.52M（32 类 1.55M） | `features[12][0]`，576×4×4 |
| B | MobileNetV2 对比模型 | torchvision backbone + 1280→num_classes | 2.23M（32 类 2.26M） | `features[18][0]`，1280×4×4 |
| C | LeafNet-MS | A + M1（多尺度融合） | ≈1.81M | MSFF 最后一层 1×1 Conv，576×8×8 |
| D | **LeafNet-MSC（最终）** | A + M1 + M2（CBAM） | ≈1.85M | MSFF 最后一层 1×1 Conv，576×8×8 |
| E | 消融用 | A + M2（仅 CBAM） | ≈1.56M | `features[12][0]`，576×4×4 |

> A/B 参数量已用 torchvision 实测：5 类时 MobileNetV3-Small = 1,522,981、MobileNetV2 = 2,230,277；32 类时分别为 1,550,656、2,264,864；C/D/E 为按模块估算，落地时以 `sum(p.numel())` 实测为准。

> B 选 MobileNetV2 而不是 ResNet18：它是 V3 的直接前代，同为倒残差/深度可分离结构，参数 2.23M、CPU 速度可接受；ResNet18 有 11.7M 参数，与“轻量化”主题不符，也更慢。

**改进动机（可直接写进报告 3.2 开头）**：
1. MobileNetV3-Small 最后一层只有 **4×4**，叶缘锯齿、叶脉等细粒度特征在多次下采样后被削弱；
2. 内置 SE 只做通道重标定，不建模“叶片在画面中的哪个位置”，空间判别力不足；
3. Flavia 全是白底扫描，模型可能把“白底”当捷径，换到摄像头场景会掉点。
M1 用 8×8 的多尺度融合补细节，M2 用 CBAM 的空间分支补位置，强数据增强 + 实景评测用于暴露和缓解背景捷径。

### 2.1 M1：轻量多尺度特征融合（MSFF）

`LeafNetV3` 在 forward 中把 backbone 分成两段（已用真实模型核对形状）：
- `low = features[:9](x)`：3×128×128 → 48×8×8
- `high = features[9:](low)`：48×8×8 → 576×4×4

融合流程：
1. `low` 48×8×8 → 1×1 Conv-BN-Hardswish → 96×8×8
2. `high` 576×4×4 → 1×1 Conv-BN-Hardswish → 96×4×4 → 双线性上采样到 96×8×8
3. concat 192×8×8 → 3×3 Conv-BN-Hardswish → 1×1 Conv-BN → 576×8×8
4. 与上采样后的原始 `high`（576×8×8）残差相加

残差的作用：保留预训练高层语义，同时注入 8×8 的叶缘/叶脉细节。M1 约 +0.28M 参数。文献对应 FPN（Lin et al., 2017）、SPP（He et al., 2014）、ASPP（Chen et al., 2017）。

### 2.2 M2：CBAM 通道-空间注意力

**注意**：MobileNetV3-Small 的倒残差块内部已经内置 SE（本配置部分启用），因此报告里不能把“通道注意力”当作全新创新。M2 的差异点是：
- 同时使用 average-pool 和 max-pool 两路通道描述符（SE 只有 average）；
- **增加空间注意力分支**（channel-wise avg/max → concat 2×8×8 → 7×7 Conv → sigmoid）；
- 放在 MSFF 融合之后，而不是塞回 backbone 内部。

通道分支 576→36→576 共享 MLP，空间分支 7×7 Conv；合计约 +0.042M 参数。文献对应 CBAM（Woo et al., 2018）、SE（Hu et al., 2018）。

### 2.3 最终模型 D 的数据流

```
3×128×128
→ features[:9] → 48×8×8
→ features[9:] → 576×4×4
→ MSFF → 576×8×8
→ CBAM → 576×8×8
→ GAP → 576
→ FC 1024 → Hardswish → Dropout 0.2 → FC num_classes → softmax
```

### 2.4 实现要点

- `src/models.py`：`build_model(model_id, num_classes, pretrained=True)`；`num_classes` 从 `data/class_registry.json` 读取（5 或 32）；类 `MSFF`、`CBAM`、`LeafNetV3`。
- 分类头替换（`num_classes` 由参数传入，不写死）：
  - MobileNetV3-Small：`classifier[3] = nn.Linear(1024, num_classes)`
  - MobileNetV2：`classifier[1] = nn.Linear(1280, num_classes)`
- C/D/E 以 torchvision backbone 为母体：**推荐保留 `self.features = backbone.features`，在 forward 里用 `low = self.features[:9](x)`、`high = self.features[9:](low)`**，这样 state_dict 前缀仍是 `features.*`，`strict=False` 加载时 missing keys 只会是新增模块和分类头；如果拆成独立的 `self.stem`/`self.tail`，必须做 `features.i.* → stem/tail` 的键重映射，否则会把整个 backbone 都当成 missing。
- 加载后断言：unexpected keys 为空；missing keys 只允许出现 MSFF/CBAM/分类头；抽查 `features[12][0].weight` 与 torchvision 原模型一致。
- 每个模型实现 `get_gradcam_layer()` 并返回明确的 Conv2d；Grad-CAM 前做一次 shape 断言（A/E 576×4×4，B 1280×4×4，C/D 576×8×8）。
- `tools/make_diagrams.py` 生成 `fig3_1_leafnet_architecture.png`、`fig3_2_msff_cbam.png`（matplotlib 方块+箭头即可，不依赖 graphviz）。

---

## 3. 训练流程

### 3.1 依赖安装

```bash
"C:/Anaconda/envs/torch_env/python.exe" -m pip install scikit-learn matplotlib pandas flask tqdm
```

首次运行会自动下载 MobileNetV3-Small（约 10 MB）和 MobileNetV2（约 14 MB）预训练权重；已确认可以访问 `download.pytorch.org`。

### 3.2 数据变换

```python
# 训练
RandomResizedCrop(128, scale=(0.65, 1.0), ratio=(0.8, 1.25))
RandomHorizontalFlip()
RandomVerticalFlip(p=0.3)
RandomRotation(20)
ColorJitter(0.2, 0.2, 0.15, 0.03)
ToTensor()
Normalize(ImageNet mean/std)
RandomErasing(p=0.1)

# 验证/测试/网页推理（三者必须完全一致）
Resize(146) → CenterCrop(128) → ToTensor() → Normalize()
```

> `src/data.py` 提供唯一的 `build_eval_transform()`，训练、评测、Flask 都调用它，避免网页预处理和验证集不一致。

### 3.3 超参数（`src/train.py` 的 `TrainConfig`）

| 参数 | 值 |
|---|---|
| 输入 / batch | 128×128 / train 32、eval 64 |
| 损失 | CrossEntropyLoss(label_smoothing=0.1) |
| 优化器 | AdamW；backbone lr=1e-4，新模块/分类头 lr=1e-3；weight_decay=0.02 |
| 调度 | 3 epoch 线性 warmup + cosine 衰减到 1e-5 |
| 轮数 | 40，early stopping patience=20（监控 val macro-F1，平局看 val loss；首轮 patience=10 时 D 在 warmup 后 F1 回落被过早停止，五模型统一改为 20） |
| 其他 | dropout 0.2、梯度裁剪 5.0、seed=42 |
| CPU | `torch.set_num_threads(8)`，DataLoader `num_workers=4`（Windows 异常时回退 0） |
| 混合精度 | bf16 开关默认关闭（实测本机不快） |

类别数量：五类 54–67；全类 46–73，仍属基本平衡，不加 class weight / sampler。

### 3.4 日志与检查点

每个 run 一个目录：

```text
runs/spec5/{A_mnv3s,B_mnv2,C_leafnet_msff,D_leafnet_msff_cbam,E_mnv3s_cbam}_seed42/
├─ best.pt           # model_state, model_id, class_keys, flavia_labels, num_classes, input_size, epoch, val_acc, val_f1, config
├─ last.pt
├─ history.csv       # epoch,lr,train_loss,train_acc,val_loss,val_acc,val_f1,epoch_sec
├─ history.json
├─ config.json
├─ test_metrics.json
└─ train.log

runs/spec32/<run_id>/  # 后续全类，目录结构相同
```

`class_keys`、`flavia_labels`、`num_classes` 必须与对应注册表（`class_registry.json` 或 `class_registry_all32.json`）一致，网页加载时校验。

### 3.5 预计耗时

有 320px 缓存时：
- **五类**：单模型约 2–5 分钟（40 epoch），A/B/C/D/E 合计 **15–30 分钟**；无缓存最坏 40–60 分钟。
- **全 32 类**：图片约 6 倍（1798 张），单模型约 8–15 分钟，A/B/C/D/E 合计 **45–90 分钟**；无缓存最坏 2–3 小时。

实际每轮耗时写入 `results/tables/training_summary.csv`，报告 4.2 可直接引用。

### 3.6 训练命令

```bash
cd /c/Users/Voidkongbai/Desktop/DeepLearning
# 1) 读取 §13 映射、生成注册表和五类清单（对照表已生成，无需 --write-md）
"C:/Anaconda/envs/torch_env/python.exe" tools/build_dataset.py --leaves data/Leaves --classes 4,12,15,17,20 --subset 5class --out data/splits --registry data/class_registry.json --seed 42 --materialize
# 2) 320px 缓存
"C:/Anaconda/envs/torch_env/python.exe" tools/prepare_cache.py --src data/Leaves --dst data/cache/leaf_320 --long-edge 320
# 3) 五类训练
"C:/Anaconda/envs/torch_env/python.exe" -m src.train --models A B C D E --data data/splits/5class --registry data/class_registry.json --num-classes 5 --cache data/cache/leaf_320 --epochs 40 --batch-size 32 --seed 42
# 4) 实景清单（T5.1；普通 432 用于主测试，困难 227 用于鲁棒性测试 §15.9）
"C:/Anaconda/envs/torch_env/python.exe" tools/build_live_dataset.py --leaf-dir data/leaf --mode all --registry data/class_registry.json --out data/splits/live_all
"C:/Anaconda/envs/torch_env/python.exe" tools/build_live_dataset.py --leaf-dir data/leaf --mode video-split --registry data/class_registry.json --out data/splits/live5
```

后续全 32 类沿用同一流程，只改三个参数：

```bash
"C:/Anaconda/envs/torch_env/python.exe" tools/build_dataset.py --leaves data/Leaves --classes all --subset all32 --out data/splits --registry data/class_registry_all32.json --seed 42 --materialize
"C:/Anaconda/envs/torch_env/python.exe" -m src.train --models A B C D E --data data/splits/all32 --registry data/class_registry_all32.json --num-classes 32 --cache data/cache/leaf_320 --epochs 40 --batch-size 32 --seed 42
```

> 约定：数据处理类脚本放 `tools/`，训练/评测/推理代码放 `src/`，网页放 `webapp/`。`src/` 和 `webapp/` 都要有 `__init__.py`，命令从项目根目录执行。

---

## 4. 评测与报告图表

### 4.1 指标

`src/evaluate.py` 在 test 上只跑一次，计算：
- accuracy、macro precision/recall/F1、test CE loss；
- top-3（五类）/ top-5（32 类）accuracy；
- 每类 recall、混淆矩阵（5×5 或 32×32）；
- 参数量、模型文件大小、batch-1 推理耗时；
- accuracy 的 Wilson 95% 置信区间：五类 test 58 张，1 张 = 1.72 个百分点；32 类 test 358 张，1 张 = 0.28 个百分点，报告里写清楚。

输出：
- `results/tables/table4_1_test_metrics.csv`
- `results/tables/training_summary.csv`
- `results/tables/table4_2_ablation.csv`
- `results/predictions/test_predictions.csv`（每个模型 58 行，含 5 类概率）

### 4.2 报告图表清单

| 文件 | 对应报告位置 | 内容 |
|---|---|---|
| `fig2_1_dataset_samples.png` | 2.1 | 5 类样本各 1 张 |
| `fig3_1_leafnet_architecture.png` | 3.1/3.3 | 最终模型结构图 |
| `fig3_2_msff_cbam.png` | 3.2 | M1/M2 模块图 |
| `fig4_1_train_loss_ABCD.png` | 4.3 | A/B/C/D 训练 loss 同图（细线原始 + 粗线 EMA） |
| `fig4_2_train_acc_ABCD.png` | 4.3 | A/B/C/D 训练准确率同图 |
| `fig4_3_test_loss_bar.png` | 4.3 | 测试 loss 柱状图 |
| `fig4_4_test_acc_bar.png` | 4.3 | 测试 accuracy 柱状图（随机线：五类 20%，全类 3.1%） |
| `fig4_5_confusion_matrix_ABCD.png` | 4.3 | 四个混淆矩阵 |
| `fig4_6_sample_predictions.png` | 4.3 | 3 个样本 × 4 个模型，真实/预测/置信度，绿/红边框 |
| `fig4_7_ablation.png` | 4.4 | A/E/C/D 测试准确率与模块增益 |
| `fig4_8_gradcam_examples.png` | 4.3/5.2 | Grad-CAM 示例（可选） |
| `fig4_9_robustness.png` | 4.3/5.2 | 压力测试对比（强烈建议，见下） |

> A/B/C/D 四模型对比图由 `tools/make_comparison_plots.py` 生成（每次训练完成后必须重绘）：五类白底图存 `pic/spec-train-test/`，32 类白底图存 `pic/spec-train-test-all/`；实景实验图分别存 `pic/spec-train-live-test/`、`pic/live-train-test/`、`pic/spec-train-live-test-all/`（E3c 不执行）。文件名为 `fig4_1_train_loss_ABCD.png`、`fig4_2_train_acc_ABCD.png`、`fig4_3_test_loss_bar.png`、`fig4_4_test_acc_bar.png`、`fig4_6_sample_predictions_ABCD.png`。

**样本选择规则**（`src/report_figures.py`，不要硬编码文件名）：
1. A 错、D 对的样本（若有）；
2. D 置信度最高的正确样本；
3. D 置信度最低的样本；
4. 没有上述情况时，取预测熵最高的 3 张。

### 4.3 为什么必须加压力测试

Flavia 是白底扫描图，5 类差异明显，A/B/C/D 很可能都接近 100%，消融差异会被 58 张测试集的噪声淹没。因此增加 `tools/make_stress_test.py`，只在 **test split** 上生成扰动版本（不参与训练）：
- 高斯模糊（σ=1.5/3）
- 低亮度/低对比度
- JPEG q=30
- 旋转 ±15°
- 随机遮挡 10–20% 面积

`src/evaluate.py --split stress` 输出 `table4_3_robustness.csv`，`src/report_figures.py` 据此生成 `fig4_9_robustness.png`。这组结果既能区分模型，也正好对应报告里“跨背景泛化/鲁棒性”的分析；如果 A/B/C/D 在干净测试集上都是 100%，报告 4.3/4.4 就应以压力测试和跨域测试作为主要证据，而不是夸大零点几个百分点。

**诊断补充（2026-10-19，五类训练后）**：
- 五模型 val loss 持续下降（末轮约 0.10–0.13），val acc 始终 100%，train acc 约 99% → **没有经典过拟合**迹象；问题不是训练轮数过多。
- 仅用图片四角背景 RGB 的最近质心分类器在 test 上达到 **63.8%**；16×16 灰度形状达到 **93.1%**；第 17 类（桂花）背景 RGB `(236.5, 244.1, 232.9)` 明显偏绿，其余类接近纯白 → **背景/采集批次与类别存在相关性**，CNN 可利用这类捷径。
- dHash 近邻：test 图在 train 中 18/58 有距离 ≤6 的近邻（≤4 为 6/58），最近邻像素 MAD 约 8.8–9.6，未发现完全重复图；存在同种叶片的相似性，但不足以解释全部高分。
- 因此后续必须用 **432 张普通实景帧**（§1.5）和压力测试判断泛化；32 类训练只能增加数据多样性，不能单独消除背景捷径。

### 4.4 中文字体

`src/report_figures.py` 设置：

```python
matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False
```

Windows 字体路径 `C:\Windows\Fonts\msyh.ttc`、`simhei.ttf` 已确认存在。

### 4.5 评测命令

```bash
"C:/Anaconda/envs/torch_env/python.exe" -m src.evaluate --runs runs --split test
"C:/Anaconda/envs/torch_env/python.exe" -m src.evaluate --runs runs --split stress
"C:/Anaconda/envs/torch_env/python.exe" -m src.report_figures --runs runs --out results
```

---

## 5. 摄像头识别网页（报告 4.5）

### 5.1 技术栈

| 层 | 技术选型 | 用途 |
|---|---|---|
| 后端框架 | Python 3.10 + Flask 3.1 + Werkzeug | 路由、Jinja2 模板、静态文件、JSON API |
| 模型推理 | PyTorch 2.5.1（CPU）+ torchvision 0.20.1 | A/B/C/D 推理；E 仅消融，不进网页 |
| 图像处理 | Pillow 11 + NumPy 2 | EXIF 校正、RGB、中心裁剪、ImageNet 归一化 |
| 可解释性 | PyTorch forward/backward hook | Grad-CAM 热力图 |
| 数据格式 | JSON（注册表）+ CSV（指标/划分） | 模型元数据、看板图表数据 |
| 页面模板 | Jinja2 + HTML5 | `templates/index.html` |
| 样式 | 原生 CSS3（Grid/Flex + CSS 变量），不用 UI 框架 | 叶绿色主题、900px 以下单栏 |
| 交互 | 原生 JavaScript ES6 + Fetch API | 上传、摄像头轮询、模型切换、演示按钮 |
| 摄像头 | `navigator.mediaDevices.getUserMedia` + Canvas | 实时预览、中心裁正方形抓帧 |
| 图表 | Apache ECharts 5.x（本地 `static/vendor/echarts.min.js`） | pic 数据看板动态图表 |
| 运行方式 | `python -m webapp.app`，无前端构建、无 CDN | 离线可用 |

### 5.2 文件与架构

```
webapp/
├─ __init__.py
├─ app.py              # Flask 入口、路由、参数校验、演示按钮 API
├─ model_service.py    # ModelRegistry：发现/加载/缓存 A/B/C/D × 三个模型组
├─ gradcam.py          # Grad-CAM：forward/backward hook、热力图叠加、PNG base64
├─ gallery_service.py  # 扫描 pic/、解析 CSV、组装看板图表数据
├─ chart_data.py       # 从 runs/ 与 pic/ 生成 ECharts 图表规格
├─ templates/index.html
└─ static/
   ├─ css/style.css
   ├─ js/app.js        # 识别页：摄像头/上传/模型切换后重识别/Grad-CAM
   ├─ js/gallery.js    # 数据看板：目录分类、交互式折线图/柱状图
   └─ vendor/echarts.min.js
```

`ModelRegistry`：
- 启动时读取 `data/class_registry.json` 与 `data/class_registry_all32.json`（存在时）；
- 扫描 `runs/spec5`、`runs/spec32`、`runs/live5` 下 A/B/C/D 的 `best.pt`，只加载 A/B/C/D（E 仅消融）；
- 三个模型组固定为 **标本（5）**（spec5）、**标本（32）**（spec32）、**实景（5）**（live5），组内模型保持 A/B/C/D 一致；
- 校验 checkpoint 的 `class_keys` / `flavia_labels` / `num_classes` 与注册表一致；
- 每个模型保存 `model_id → (model, transform, metadata)`；CPU 推理；
- 模型组切换：按组自动切换 Top-3/Top-5、类别图例与测试指标；**切换模型或模型组时保留当前图片与预览，并自动用新模型重新识别**。

### 5.3 功能需求

| 功能 | 行为 | 说明 |
|---|---|---|
| 摄像头实时识别 | 打开摄像头，700 ms 轮询一帧，返回 Top-k + 置信度 + 可选 Grad-CAM | 上一帧未返回不发下一帧；切后台暂停 |
| 上传图片识别 | 拖拽/选择图片，调用 `/api/predict` | JPG/PNG，限制 10 MB |
| 模型切换 | A/B/C/D pills；三个模型组：**标本（5）／标本（32）／实景（5）** | 显示各模型测试 accuracy；切换后不清除当前图片，自动重新识别 |
| Grad-CAM | 对当前图片返回热力图叠加图；开关切换时重新请求 | 后端 hook 实现；A/B/C/D × 三个模型组均可用 |
| 添加数据（演示） | 点击弹窗/Toast：“演示功能，未接入真实数据接口” | 不写文件、不调用真实上传/训练 API |
| 模型训练（演示） | 点击弹窗 + 2–3 秒假进度条，结束提示“演示功能” | 不启动训练进程、不写 `runs/` |
| pic 数据看板 | 按 `pic/<子目录>` 分类展示**交互式 ECharts 折线图/柱状图** | 不显示图片网格、文档或下载链接；数值通过 tooltip 查看 |

> 本轮范围：Grad-CAM 必须真实可用（不再显示“暂不可用”）；数据看板只保留交互图，静态 PNG、docx 与所有下载入口一律不展示。“用测试样本试一下”仍为后续可选项，未实现。

### 5.4 API

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/` | 返回页面 |
| GET | `/api/models` | 模型 id、显示名、参数量、测试 accuracy |
| GET | `/api/health` | 已加载模型、设备、版本 |
| POST | `/api/predict` | multipart：`image`、`model`、`topk`（默认 3，32 类可传 5）、`gradcam=1` |
| POST | `/api/predict_frame` | JSON：`{model, image_data_url, gradcam}`，供摄像头轮询 |
| POST | `/api/predict_all` | 同一张图返回 A/B/C/D 四组结果，方便做报告 4.3 的样本对比图 |
| GET | `/api/gallery` | 扫描 `pic/` 一级目录，返回每个目录的 `charts[]`（折线/柱状图规格）与文件元数据；前端只用 `charts` |
| GET | `/pic/<path>` | 安全地提供 pic 下的静态文件（保留兼容；数据看板不再展示或链接图片/docx/CSV 下载） |
| POST | `/api/demo/add-data` | 演示：返回 `{demo:true, message:...}`，不写文件、不接真实上传 |
| POST | `/api/demo/train` | 演示：返回 `{demo:true, message:...}`，不启动训练、不写 `runs/` |

返回字段：`model`、`topk[{key,zh_name,scientific_name,prob}]`、`latency_ms`、`gradcam_available`、`gradcam_png_base64`（`gradcam=1` 时必须为非空 PNG base64）。

预处理必须调用 `src/data.py::build_eval_transform()`；输入先 `ImageOps.exif_transpose`，转 RGB，再中心裁正方形。**Flask 多线程下用一把全局锁包住 forward/backward**，Grad-CAM 的 hook 必须在 `finally` 中移除。
> Grad-CAM 实现：`webapp/gradcam.py` 通过 `model.get_gradcam_layer()` 注册 forward hook，取目标类别 logit 反传，按 channel 平均梯度加权、ReLU、归一化，再缩放到输入尺寸并用伪彩色叠加到原图；输出 PNG base64。A/E 为 576×4×4，B 为 1280×4×4，C/D 为 576×8×8，统一上采样。

### 5.5 摄像头流程

- 页面按钮触发 `navigator.mediaDevices.getUserMedia({video:{width:640,height:480,facingMode:'environment'}, audio:false})`；
- 视频上叠一个**正方形取景框**，提示把叶片放中间；发送前从 canvas 中心裁正方形，再交给统一 transform；
- 实时模式每 700 ms 抓一帧，`POST /api/predict_frame`；**上一帧未返回不发下一帧**，停止/切后台（`document.hidden`）时暂停；
- “拍照”按钮冻结当前帧，带 `gradcam=1` 再请求一次；
- 摄像头被拒绝时显示明确提示，上传功能仍然可用。

**权限限制**：必须用 `http://127.0.0.1:5000` 或 `http://localhost:5000` 打开；`http://<局域网IP>:5000` 在浏览器里不是安全上下文，`getUserMedia` 会被拒绝。还要确认 Windows 相机隐私权限已开启、没有其他程序占用摄像头。

### 5.6 界面布局

- 顶部导航：**识别 / 数据看板 / 系统操作**；移动端折叠为汉堡菜单。
- 品牌与标题：项目名统一为 **知叶**；小字为 **面向农学生的叶片识别与学习平台**；页面 `<title>` 为“知叶 · 农学生叶片识别与学习平台”。
- 识别页：
  - 标题：“知叶 · 叶片识别与学习平台”；说明文字体现“帮助农学生认识校园植物叶片”；
  - 左侧：摄像头预览 / 上传图片两个 Tab，拖拽上传、预览、正方形取景框；
  - 右侧：模型切换 pill（A/B/C/D，显示各自测试 accuracy；模型组固定为 **标本（5）／标本（32）／实景（5）**）、Top-k 置信度条（五类 Top-3，32 类 Top-5）、中文名+学名、推理耗时、Grad-CAM 开关（真实可用）、低置信度警告；
  - **切换模型或模型组时不清除当前上传图片/拍照快照，自动用新模型重新识别**；Grad-CAM 开关变化时也对当前图片重新请求；
  - 底部：类别图例。五类直接显示 5 个；32 类用可滚动/可搜索列表，避免页面过长；
  - “用测试样本试一下”一行仍为后续可选项，本轮不实现。
- 系统操作页：
  - “添加数据”按钮：点击直接弹出演示提示“演示功能，未接入真实数据接口”，不写文件；
  - “模型训练”按钮：弹出确认框，显示 2–3 秒假进度条，结束提示“演示功能，未启动真实训练”；
  - 两个按钮都只调用 `/api/demo/*`，不执行任何真实上传或训练；
- 报告模板里的“手写字体测试”与叶片识别无关，用**上传图片**替代，报告 4.5 里写明即可；
- 配色叶绿色系，卡片式两栏布局，900 px 以下单栏；**不引用 CDN**，CSS/JS/ECharts 全部本地，保证离线可用。

### 5.7 pic 数据看板（ECharts）

后端 `gallery_service.py` + `chart_data.py`：
- 扫描 `pic/` 一级子目录（当前包括 `spec-train-test`、`spec-train-test-all`、`spec-train-live-test`、`spec-train-live-test-all`、`live-train-test`、`melt-test`）；
- 忽略 `~$*` 等 Office 临时文件与隐藏文件；
- `/api/gallery` 为每个目录返回 `charts[]` 图表规格；文件元数据仅保留兼容，不再供前端展示或下载；
- 图表规格统一为：
  `{id, title, type: "line"|"bar"|"groupedBar", xName, yName, unit: "percent"|"value", x: [...], series: [{key, name, data: [...]}], showDataLabels: bool}`；
- 数据缺失或 CSV 解析失败时该目录返回空 `charts`，前端显示“暂无可绘制图表”，不生成假数据；
- 不修改 `pic/` 下任何原始图片、CSV 或报告文件。

图表目录（只列可绘制数据，不包含样本预测图）：

| `pic/` 目录 | 折线图 | 柱状图 |
|---|---|---|
| `spec-train-test` | 训练损失–轮次、训练准确率–轮次（`runs/spec5/*/history.csv`） | 测试准确率、测试 CE 损失（`runs/spec5/*/test_metrics.json`） |
| `spec-train-test-all` | 训练损失–轮次、训练准确率–轮次（`runs/spec32/*/history.csv`） | 测试准确率、测试 CE 损失（`runs/spec32/*/test_metrics.json`） |
| `spec-train-live-test` | — | 普通实景准确率/损失、困难帧准确率/损失（`metrics_test.csv`、`metrics_test_hard.csv`） |
| `spec-train-live-test-all` | — | 普通实景准确率/损失、困难帧准确率/损失（同上） |
| `live-train-test` | 训练损失–轮次、训练准确率–轮次（`runs/live5/*/history.csv`） | 普通实景准确率/损失、困难帧准确率/损失（两个 CSV） |
| `melt-test` | — | 主测试准确率/损失、困难帧准确率/损失、主测试模块增益、困难帧模块增益（`ablation_metrics.csv` + `raw/metrics_robust_*.csv`） |

前端 `gallery.js`：
- 目录 tab 的数量改为该目录的**图表数量**；目录内只渲染交互图，**不渲染图片网格、灯箱、文档列表、CSV 下载或“下载原图”入口**；
- 所有折线图/柱状图统一规则：
  - 模型横坐标一律用短 ID：普通对比为 **A/B/C/D**，消融为 **A/E/C/D**，不再出现 `A_mnv3s` 等全称；
  - 横坐标标签强制水平摆放（`axisLabel.rotate = 0`、`interval = 0`）；
  - 图例固定右上角，显示模型全称，字号缩小到 10：
    `A MobileNetV3-Small`、`B MobileNetV2`、`C LeafNet-MSFF`、`D LeafNet-MSFF+CBAM`、`E MobileNetV3-Small+CBAM`；
  - 单系列测试柱状图按模型拆成 4 个系列（每个系列只有自己位置有数值），使右上角图例能显示模型全称；
  - `melt-test` 的所有柱状图**关闭柱顶数据标签**，数值只通过鼠标悬停 tooltip 查看；
  - tooltip 统一显示百分比（accuracy）或数值（CE 损失），并在图例/坐标轴上保持 A/B/C/D 顺序；
- 图表使用本地 Apache ECharts 5.x，窗口 resize 自适应；空目录显示“暂无可绘制图表”；
- 目录与图表顺序稳定：目录按名称排序，图表按“训练曲线 → 测试柱状图 → 困难帧 → 增益”的固定顺序。

### 5.8 报告截图

- `results/screenshots/recognition.png`：识别页（三个模型组、模型切换、摄像头占位、类别图例）；
- `results/screenshots/upload_result.png`：上传 `1123.jpg` 的识别结果（紫荆 92.5%，Top-3）；
- `results/screenshots/gradcam.png`：Grad-CAM 热力图叠加效果；
- `results/screenshots/switch_model.png`：同一张图片切换模型后自动重新识别；
- `results/screenshots/gallery.png`：pic 数据看板（6 个目录、全交互图表、右上角图例）；
- `results/screenshots/melt_test.png`：melt-test 图表（无柱顶标签、悬停 tooltip、横坐标水平）；
- `results/screenshots/system_actions.png`：添加数据/模型训练演示按钮页；
- `results/screenshots/demo_toast.png`：点击“添加数据”后的演示提示；
- 摄像头实拍截图（webcam_01–04）需要真实摄像头，在无头验收中未生成，报告阶段可人工补拍。

### 5.9 启动命令

```bash
"C:/Anaconda/envs/torch_env/python.exe" -m webapp.app --checkpoints runs --host 127.0.0.1 --port 5000
```

浏览器打开 `http://127.0.0.1:5000`。

---

## 6. 校园实景照片流程（预留，不阻塞主线）

目录：`data/campus/<key>/*.jpg`，5 个 key 与注册表一致（`cercis_chinensis`、`pittosporum_tobira`、`cinnamomum_camphora`、`osmanthus_fragrans`、`lagerstroemia_indica`）。建议每类至少 10–20 张，至少覆盖 2 个拍摄地点。

> **现状**：`data/leaf` 已有 432 张普通实景帧（另有 227 张困难帧用于鲁棒性测试）；5 类视频级划分与实景训练见 §1.5 与 §15。`data/campus/<key>/` 仅用于后续补拍 Flavia 之外的物种；不得把测试帧直接当训练集。

- `tools/import_campus.py`：EXIF 校正、转 RGB、长边缩到 1024、JPEG q=90、去元数据；SHA-256 + dHash 去重；Laplacian 方差 < 30 的模糊图剔除；生成 contact sheet 供人工确认。
- 划分：`data/splits/campus_support.csv` / `campus_test.csv`，每类 70/30 且 support 不超过 10 张；**绝不与 Flavia 的 train/val/test 混合**。
- `src/finetune_campus.py`：从 D 的 `best.pt` 初始化，backbone lr=5e-5、head lr=5e-4、batch 16、20 epoch、early stopping 5；同时评测 campus holdout 和 Flavia test，监控是否灾难性遗忘。
- `results/tables/campus_cross_domain.csv`：行 = A zero-shot、D zero-shot、D fine-tuned；列 = campus accuracy、campus macro-F1、Flavia test accuracy。

**没有实景数据时**：所有 campus 脚本打印 “campus data not available; skipped” 并以 0 退出，图表生成自动忽略该部分。网页和报告仍可用 Flavia 测试图 + 实时摄像头完成演示。

---

## 7. 目录结构与执行顺序

```
DeepLearning/
├─ plans.md                   # 本计划（唯一权威映射见 §13）
├─ AGENTS.md / WORKLOG.md     # 协作规范 / 统一工作日志
├─ data/
│  ├─ Leaves/                 # 原始 Flavia，只读
│  ├─ leaf/                   # 实景视频抽帧（5 类 *_test），只作测试
│  ├─ test_preview/           # 抽帧预览、extract_stats.json、frame_quality.csv
│  ├─ Flavia类别对照表.md      # 32 类校验/中文名/张数，给人看（已完成）
│  ├─ class_map.json          # 32 类映射（§13 派生），程序唯一映射来源
│  ├─ class_registry.json     # 五类，从 class_map 派生
│  ├─ class_registry_all32.json # 后续全类（单独存放，不覆盖五类）
│  ├─ zh_names.json           # flavia_label → 中文名（从公开表提取一次）
│  ├─ splits/
│  │  ├─ 5class/              # 五类白底 train/val/test/all.csv（178/58/58）
│  │  ├─ all32/               # 32 类白底（1082/358/358）
│  │  ├─ live_all/            # E1/E3b：全量普通实景（432）
│  │  ├─ live5/               # E2：实景视频级划分（301/79/52）
│  │  └─ live_all32/          # E3b：32 类标签的实景清单
│  ├─ dataset/                # --materialize 生成，人工检查用
│  ├─ cache/leaf_320/         # 训练缓存
│  └─ campus/<key>/           # 后续实地拍摄预留
├─ src/
│  ├─ __init__.py
│  ├─ data.py                 # Dataset + build_eval_transform
│  ├─ models.py               # A/B/C/D/E，num_classes 可配置
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ gradcam.py
│  ├─ report_figures.py
│  ├─ finetune_campus.py
│  └─ utils.py                # seed、日志、checkpoint
├─ tools/
│  ├─ build_dataset.py
│  ├─ build_live_dataset.py   # 实景清单与视频级划分（T5.1）
│  ├─ evaluate_live.py        # 实景指标 + 两张组合柱状图（E1/E3b）
│  ├─ make_class_table.py     # 生成 Flavia类别对照表.md
│  ├─ prepare_cache.py
│  ├─ make_dataset_preview.py
│  ├─ make_stress_test.py
│  ├─ make_diagrams.py
│  ├─ import_campus.py
│  ├─ extract_frames.py       # 已存在，保留不动
│  ├─ extract_test_frames.py  # 已存在，保留不动
│  └─ verify_project.py
├─ webapp/                    # 见 5.1–5.9
├─ runs/{spec5,live5,spec32,live32}/<run_id>/
├─ pic/
│  ├─ spec-train-test/        # W5 白底训练标准 5 图（已完成）
│  ├─ spec-train-test-all/    # E3a：W32 白底训练标准图
│  ├─ spec-train-live-test/    # E1：W5 → 普通实景（432）
│  ├─ spec-train-live-test-all/ # E3b：W32 → 实景 5 类子集
│  ├─ live-train-test/     # E2：L5 实景训练 → 实景测试
│  └─ live-train-test-all/ # E3c：用户确认不执行（仅预留）
└─ results/{figures,tables,predictions,screenshots}/
```

执行顺序（建议按里程碑验收）：

1. **环境**：安装 5 个缺失包；跑 `python -c "import sklearn, matplotlib, pandas, flask, tqdm"` 验证。
2. **数据**：生成 `data/class_map.json` → `build_dataset.py` → 检查 294/178/58/58、无交集 → `make_dataset_preview.py` 看 5 类图。（分类校验已完成，无需再做）
3. **实景清单（T5.1）**：`build_live_dataset.py` 生成 live_all / live5 / live_all32（只用普通 `_test`）；检查 432、301/79/52、432，且无跨 split 泄漏。
4. **模型冒烟**：A 训练 1 epoch，确认 loss 有限、checkpoint 可加载、val acc > 20%。（已完成）
5. **五类白底全量训练（W5）**：A/B/C/D/E，40 epoch（已完成）。
6. **E1**：W5 → 普通 `_test`（432），输出 `pic/spec-train-live-test/` 一张组合柱状图。
7. **E2（L5）**：实景训练 A/B/C/D（leaf01-03 train / leaf05 val / leaf04 test），输出 `pic/live-train-test/` 标准图。
8. **E3a（W32）**：生成 all32 数据/缓存，训练 A/B/C/D，白底 test 标准图 → `pic/spec-train-test-all/`。
9. **E3b**：W32 → 实景 5 类子集，输出 `pic/spec-train-live-test-all/` 两张组合柱状图。
10. **E3c（L32）**：用户确认其它种类实景不做演示，**不执行**；目录 `pic/live-train-test-all/` 仅保留不使用。
11. **网页与报告素材**：网页、截图、压力测试、报告图表。

---

## 8. 验证清单

> 分类校验（编号 / 区间 / 对照表 / 实景抽帧）已于 2026-10-19 完成，对应条目已勾选；其余条目随执行进度勾选。

- [x] Flavia 32 类的编号与区间已核对：32/32 一致，无需重新编码 — 2026-10-19
- [x] `data/Flavia类别对照表.md` 已生成且 32 行齐全（1798 张） — 2026-10-19
- [x] `data/Leaves` 的 1798 张图片可按编号区间唯一归属 32 类；五类 4/12/15/17/20 共 294 张 — 2026-10-19
- [x] `data/leaf` 实景帧已抽取并去重：432 张普通实景帧；227 张困难帧仅用于鲁棒性测试 — 2026-10-20（更新）
- [ ] `data/class_map.json` 与 §13、`data/Flavia类别对照表.md` 逐项一致（32 类、编号、区间、名称）。
- [ ] 五类清单恰好 294 张，train/val/test = 178/58/58，三集合两两不相交且并集完整。
- [ ] `class_registry.json` 的 `flavia_label/key/zh_name/range` 与 `class_map.json` + 公开表一致；五类子集 `py_label` 为 0–4。
- [x] checkpoint 内 `class_keys`、`flavia_labels`、`num_classes` 与注册表一致 — 2026-10-19
- [x] `--classes all` 生成 32 类清单：总数 1798，train/val/test = 1082/358/358 — 2026-10-19
- [x] 实景清单生成：live_all（432 + hard 227）、live5（301/79/52 + hard 227）、live_all32（432 + hard 227）；hard 仅用于鲁棒性测试 — 2026-10-20
- [x] 每个模型 forward 输出 `[1, num_classes]`（五类 `[1,5]`，全类 `[1,32]`）；参数量与 §2 表格一致；预训练权重确实加载（missing keys 仅新模块/头）— 2026-10-19
- [x] A 冒烟训练 1 epoch 正常；全量训练每个 run 有 `best.pt`、`history.csv`、`train.log` — 2026-10-19
- [ ] test 预测 CSV 五类每个模型 58 行、全类每个模型 358 行，各类概率和为 1。
- [x] E1 实景评测每个模型 432 行；E2 每个模型 52 行；E3b 432 行（32 类标签）；A/B/C/D 顺序一致 — 2026-10-19
- [x] E1 / E2 / E3a / E3b 主测试图表全部生成到指定目录；E3c 不执行（用户确认）；hard 鲁棒性图见 §15.9 — 2026-10-20
- [ ] 报告所需图表全部生成，中文无方框，A/B/C/D 顺序一致。
- [x] 网页能加载 A/B/C/D；上传、拍照、实时、模型切换、Top-3 均可用 — 2026-10-20
- [ ] 识别页三个模型组 **标本（5）／标本（32）／实景（5）** 均可加载；切换模型/模型组保留当前图片并自动重识别。
- [ ] Grad-CAM 可视化：上传、拍照、实时均返回热力图，A/B/C/D × 三个模型组均可用。
- [ ] 数据看板只显示交互式折线图/柱状图：无图片网格、无文档/CSV/原图下载；横坐标统一 A/B/C/D、图例右上角全称小字、melt-test 仅 tooltip 显示数值。
- [ ] 摄像头轮询不重叠；切后台暂停；拒绝权限时有提示且上传仍可用。
- [ ] 4 张摄像头截图 + 5 张分类上传截图已保存到 `results/screenshots/`。
- [ ] `tools/verify_project.py` 对文件、划分、checkpoint、指标、图表、截图逐项输出 PASS/FAIL。
- [ ] `data/campus/` 为空时，campus 脚本以 0 退出且不产生空图表。

---

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| 只有 CPU | 128×128、MobileNet 系、320px 缓存、8 线程、batch 32；不引入 ResNet18 |
| 测试集小（五类 58 张 / 全类 358 张） | 报告 Wilson 95% CI、错误张数、macro-F1、每类 recall；差 1–2 张不解读为显著 |
| Flavia 上接近 100% | 不夸大零点几个百分点；用压力测试 + 实景跨域表作为主要证据 |
| 白底扫描 → 摄像头域差异大 | 强裁剪/旋转/颜色增强 + RandomErasing；Grad-CAM 检查；有实景后微调 D |
| 过拟合 | dropout 0.2、weight_decay 0.02、label smoothing、early stopping；看 train-val gap |
| B 可能反超 D | 如实报告；必要时 A/D 各跑 3 个 seed 报 mean±std，不挑 seed |
| Windows DataLoader 卡死 | `if __name__ == "__main__"` 保护；`num_workers=0` 回退；UTF-8 路径 |
| 摄像头打不开 | 只用 localhost；检查 Windows 相机权限/占用；保留上传入口 |
| 重复图泄漏 | SHA-256 + dHash；跨类重复直接中止 |
| Grad-CAM 挂错层 | `get_gradcam_layer()` 集中注册 + 每次 shape 断言；hook 在 finally 移除 |
| Flavia 编号不连续（缺 13） | 不直接把编号当 PyTorch 标签；统一用 registry 的 `py_label` 映射 |
| 扩类后旧 checkpoint 标签错位 | checkpoint 存 `class_keys`/`flavia_labels`/`num_classes`，网页和评测加载时校验 |
| 32 类混淆矩阵太密 | 无标注热图 + `table4_4_top_confusions.csv`；报告正文仍用五类结果 |
| 训练后改 `--classes` 顺序 | checkpoint 存 `class_keys`，网页加载时校验并拒绝不一致 |
| 实景帧与视频源高度相关 | 只做整体跨域测试，不随机拆分、不进入训练；报告单列说明 |

---

## 10. 与报告章节的对应

| 报告章节 | 本方案产出 |
|---|---|
| 2.1 数据集设计 | Flavia 5 类 294 张；1600×1200 JPEG；6:2:2 分层划分；`fig2_1_dataset_samples.png`；32 类映射与编号区间校验（`data/Flavia类别对照表.md`） |
| 3.1 基础网络 | MobileNetV3-Small 结构；`fig3_1_leafnet_architecture.png` |
| 3.2 改进 | 3.2.1 MSFF、3.2.2 CBAM；3.2.3 可写轻量分类头 + 差分学习率 + 强增强（模板的三小节是示例，两网络模块+训练策略足够） |
| 3.3 最终模型 | LeafNet-MSC 结构图与数据流 |
| 4.1 环境 | Windows 11、Ryzen 7 7840HS、15.2GB、CPU PyTorch 2.5.1、Python 3.10.21 |
| 4.2 训练过程 | §3.3 参数表 + `training_summary.csv` 的每轮耗时/曲线 |
| 4.3 训练结果 | `fig4_1`–`fig4_6`、测试 loss/acc 表、`table4_5_real_scene.csv`（432 张普通实景；E1/E2/E3b） |
| 4.4 消融 | A / E(+CBAM) / C(+MSFF) / D(两者) + `fig4_7`，并给出交互项 `(D−C)−(E−A)` |
| 4.5 可视化网站 | §5 网页 + 摄像头/上传/模型切换/Grad-CAM + pic 数据看板（ECharts）+ 添加数据/训练演示按钮 + 截图 |
| 5.1/5.2 收获与展望 | 域差异、小测试集、压力测试、实景微调；后续扩到 32 类的方案与初步结果 |
| 附录 | `src/models.py`、`src/train.py`、`webapp/app.py`、`webapp/static/js/app.js` 等 |

> 附录代码按模板要求用**浅色背景深色文本**截图，建议 VS Code 浅色主题；每张代码图保留文件名标签，便于对应正文。

---

## 11. 预计工期

| 阶段 | 预计 |
|---|---|
| 分类校验 + 生成 `data/Flavia类别对照表.md` | 0（已完成） |
| 依赖 + 五类数据构建 + 预览 | 30–45 分钟 |
| 实景清单 + 跨域评测 | 20–30 分钟 |
| 模型/训练/评测代码与调试 | 3–5 小时 |
| 五类 5 个模型训练 | 15–30 分钟（CPU） |
| 图表与消融 | 30–60 分钟 |
| 网页开发与调试 | 2–4 小时 |
| 截图与报告素材 | 1 小时 |
| 实景照片微调（有数据时） | 1–2 小时 |
| 全 32 类扩展训练 + 评测（后续） | 1.5–3 小时 |

最短可交付路径：**数据 → A 冒烟 → A/B/C/D 训练 → 图表 → 网页上传/摄像头**；实景微调与压力测试可以在主链跑通后追加，不影响报告主体。

---

## 12. 后续扩展到全部 32 类（当前不执行）

五类报告和网页验收通过后，再启动这一步；代码不需要重写，只换数据清单和 `num_classes`。

### 12.1 数据

```bash
"C:/Anaconda/envs/torch_env/python.exe" tools/build_dataset.py --leaves data/Leaves --classes all --subset all32 --out data/splits --registry data/class_registry_all32.json --seed 42 --materialize
```

- 类别：Flavia 编号 1–12、14–33，共 32 类；
- 图片：1798 张（每类 46–73 张）；
- 划分：**train/val/test = 1082 / 358 / 358**（6:2:2，按类分层）；
- 因为划分种子是 `42 + flavia_label`，**五类的 train/val/test 与 all32 中对应类别完全一致**，两份结果可以直接对比；五类 test 是 all32 test 的子集；
- 生成 `data/splits/all32/` 和 `data/class_registry_all32.json`，**不覆盖**五类的注册表和清单。

### 12.2 模型与训练

```bash
"C:/Anaconda/envs/torch_env/python.exe" -m src.train --models A B C D E --data data/splits/all32 --registry data/class_registry_all32.json --num-classes 32 --cache data/cache/leaf_320 --epochs 40 --batch-size 32 --seed 42
```

- A/B/C/D/E 结构与五类完全相同，只把分类头换成 `num_classes=32`；
- 参数量：A≈1.55M、B≈2.26M，C/D/E 相应增加约 2.8 万；
- 超参数、增强、损失、优化器、early stopping 全部沿用 §3.3；
- 预计 CPU 训练 **45–90 分钟**（A/B/C/D/E 合计，有缓存）；
- 可选：用五类 D 的 backbone 初始化（`--init runs/spec5/D_.../best.pt --strict False`），分类头重新初始化；为了与五类实验公平对比，推荐默认仍用 ImageNet 预训练并从零微调，把“五类初始化”作为一组附加实验。

### 12.3 评测与报告

```bash
"C:/Anaconda/envs/torch_env/python.exe" -m src.evaluate --runs runs/spec32 --registry data/class_registry_all32.json --split test
"C:/Anaconda/envs/torch_env/python.exe" -m src.report_figures --runs runs/spec32 --out results/all32
```

- 指标：top-1、top-5、macro-F1、每类 recall；test 358 张，1 张 = 0.28 个百分点；
- 混淆矩阵 32×32 会很密：`fig4_5_confusion_matrix_all32.png` 用无数字标注的热图，另出 `table4_4_top_confusions.csv`（最容易混淆的 10 对类别）；
- 额外做一次 **子集评测**：把 all32 模型在五类 test 子集上的表现，与五类专用模型对比，写进“扩类后的精度变化”；**按 `flavia_label` 对齐，不能按 `py_label`**（两套标签顺序不同）；
- 当前报告主体仍用五类结果；all32 的结果放在 5.2“项目展望”或附录，避免冲淡 4.3/4.4 的主线。

### 12.4 网页

- `ModelRegistry` 同时加载 `class_registry.json` 和 `class_registry_all32.json`；
- 页面增加“五类 / 全部 32 类”模型组切换；
- 32 类模式：Top-5、可滚动/可搜索图例、随机 8 张测试样本；
- 五类模式的截图和报告素材保持不变。

### 12.5 继续增加 Flavia 之外的种类

- **仍是 Flavia 编号体系**：在 `data/class_map.json` 与本文 §13 追加新条目（新编号、学名、英文名、区间），把图片按编号放进 `data/Leaves`，重跑 `build_dataset.py --classes all`；
- **校园实景新物种**（没有 Flavia 编号）：不要硬塞进 Flavia 的 filename 区间；在 `data/campus/<key>/` 下建类别，给注册表加 `source: campus` 条目，用独立的 `campus_support.csv` / `campus_test.csv` 微调分类头；与 Flavia 32 类模型分开报告；
- 每次扩类都保留旧注册表/旧 checkpoint，网页按 `class_keys` 校验，避免新旧标签错位。

---

## 13. 附录：Flavia 32 类映射（权威来源，已同步到 `data/Flavia类别对照表.md`）

> 本表现已作为项目唯一的类别映射来源：`data/class_map.json`、`data/class_registry.json`、`data/splits/*` 与网页都从这里派生。学名沿用原 Flavia 类别表记录；中文名来自公开表；“备注”标出历史差异。实际张数已按 `data/Leaves` 统计。

| Flavia 编号 | key | 中文名 | 学名 | 英文名 | 区间 | 张数 | 备注 |
|---:|---|---|---|---|---|---:|---|
| 1 | phyllostachys_edulis | 毛竹 | Phyllostachys edulis (Carr.) Houz. | pubescent bamboo | 1001-1059 | 57 | |
| 2 | aesculus_chinensis | 七叶树 | Aesculus chinensis | Chinese horse chestnut | 1060-1122 | 57 | |
| 3 | berberis_anhweiensis | 安徽小檗 | Berberis anhweiensis Ahrendt | Anhui Barberry | 1552-1616 | 62 | |
| 4 | cercis_chinensis | 紫荆 | Cercis chinensis | Chinese redbud | 1123-1194 | 67 | 五类子集 |
| 5 | indigofera_tinctoria | 木蓝 | Indigofera tinctoria L. | true indigo | 1195-1267 | 71 | |
| 6 | acer_palmatum | 鸡爪槭 | Acer Palmatum | Japanese maple | 1268-1323 | 54 | 学名大写 P（按原表保留） |
| 7 | phoebe_nanmu | 楠木 | Phoebe nanmu (Oliv.) Gamble | Nanmu | 1324-1385 | 58 | |
| 8 | kalopanax_septemlobus | 刺楸 | Kalopanax septemlobus (Thunb. ex A.Murr.) Koidz. | castor aralia | 1386-1437 | 51 | |
| 9 | cinnamomum_japonicum | 天竺桂 | Cinnamomum japonicum Sieb. | Chinese cinnamon | 1497-1551 | 53 | |
| 10 | koelreuteria_paniculata | 栾树 | Koelreuteria paniculata Laxm. | goldenrain tree | 1438-1496 | 55 | |
| 11 | ilex_macrocarpa | 大果冬青 | Ilex macrocarpa Oliv. | Big-fruited Holly | 2001-2050 | 46 | |
| 12 | pittosporum_tobira | 海桐 | Pittosporum tobira (Thunb.) Ait. f. | Japanese cheesewood | 2051-2113 | 55 | 五类子集 |
| 14 | chimonanthus_praecox | 蜡梅 | Chimonanthus praecox L. | wintersweet | 2114-2165 | 50 | 无 label 13 |
| 15 | cinnamomum_camphora | 香樟 | Cinnamomum camphora (L.) J. Presl | camphortree | 2166-2230 | 61 | 五类子集 |
| 16 | viburnum_awabuki | 日本珊瑚树 | Viburnum awabuki K.Koch | Japan Arrowwood | 2231-2290 | 57 | |
| 17 | osmanthus_fragrans | 桂花 | Osmanthus fragrans Lour. | sweet osmanthus | 2291-2346 | 54 | 五类子集 |
| 18 | cedrus_deodara | 雪松 | Cedrus deodara (Roxb.) G. Don | deodar | 2347-2423 | 73 | |
| 19 | ginkgo_biloba | 银杏 | Ginkgo biloba L. | ginkgo, maidenhair tree | 2424-2485 | 61 | 英文名比公开表多 |
| 20 | lagerstroemia_indica | 紫薇 | Lagerstroemia indica (L.) Pers. | Crape myrtle, Crepe myrtle | 2486-2546 | 57 | 五类子集；英文名多 Crepe myrtle |
| 21 | nerium_oleander | 夹竹桃 | Nerium oleander L. | oleander | 2547-2612 | 62 | |
| 22 | podocarpus_macrophyllus | 罗汉松 | Podocarpus macrophyllus (Thunb.) Sweet | yew plum pine | 2616-2675 | 57 | |
| 23 | prunus_serrulata | 日本晚樱 | Prunus serrulata Lindl. var. lannesiana auct. | Japanese Flowering Cherry | 3001-3055 | 53 | |
| 24 | ligustrum_lucidum | 女贞 | Ligustrum lucidum Ait. f. | Glossy Privet | 3056-3110 | 53 | |
| 25 | toona_sinensis | 香椿 | Tonna sinensis M. Roem. | Chinese Toon | 3111-3175 | 63 | 原表拼写 Tonna；正确 Toona；key 用正确拼写 |
| 26 | prunus_persica | 桃 | Prunus persica (L.) Batsch | peach | 3176-3229 | 49 | |
| 27 | manglietia_fordiana | 木莲 | Manglietia fordiana Oliv. | Ford Woodlotus | 3230-3281 | 47 | |
| 28 | acer_buergerianum | 三角枫 | Acer buergerianum Miq. | trident maple | 3282-3334 | 47 | |
| 29 | mahonia_bealei | 阔叶十大功劳 | Mahonia bealei (Fortune) Carr. | Beale's barberry | 3335-3389 | 54 | |
| 30 | magnolia_grandiflora | 广玉兰 | Magnolia grandiflora L. | southern magnolia | 3390-3446 | 53 | |
| 31 | populus_canadensis | 加杨 | Populus ×canadensis Moench | Canadian poplar | 3447-3510 | 59 | 学名用 × |
| 32 | liriodendron_chinense | 鹅掌楸 | Liriodendron chinense (Hemsl.) Sarg. | Chinese tulip tree | 3511-3563 | 51 | |
| 33 | citrus_reticulata | 柑橘 | Citrus reticulata Blanco | tangerine | 3566-3621 | 51 | |

**校验结论（已完成）**：32/32 编号一致；32/32 区间一致；无需按公开表重新编码。差异仅为学名作者缩写/大小写、英文名同义词和第 25 类的 `Tonna/Toona` 拼写。编号和区间保持本表不变；展示表保留原表写法并在备注注明。**本表是项目唯一权威映射，不再依赖任何外部 cato 文件。**

---

## 14. 执行进度（逐步勾选）

> 规则见 [`AGENTS.md`](AGENTS.md)：验证通过才勾选；代码、勾选、日志放在同一次提交中。

### 14.1 环境与数据

- [x] T1.1 安装缺失依赖并通过导入检查（scikit-learn / matplotlib / pandas / flask / tqdm）— 2026-10-19
- [x] T1.2 生成 `data/class_map.json` 与 `data/zh_names.json`（32 类映射）— 2026-10-19
- [x] T1.3 实现 `tools/build_dataset.py`，生成五类注册表与划分清单（294 / 178 / 58 / 58）— 2026-10-19
- [x] T1.4 旧版实景清单 `data/splits/real/all.csv`（791 行）已生成 — 2026-10-19（当前实景数据已更新为 432 + 227，见 T5.1）
- [x] T1.5a 生成 320px 训练缓存（294 张）— 2026-10-19
- [ ] T1.5b 生成数据预览图 `fig2_1_dataset_samples.png`

### 14.2 训练代码

- [x] T2.1 实现 `src/utils.py` + `src/data.py`（数据集、变换、DataLoader）— 2026-10-19
- [x] T2.2 实现 `src/models.py`（A/B/C/D/E，MSFF + CBAM）— 2026-10-19
- [x] T2.3 实现 `src/train.py`（训练循环、日志、checkpoint、测试指标）— 2026-10-19
- [x] T2.4 独立代码审查通过 — 2026-10-19（1 条重要 + 3 条建议已修复）
- [x] T2.5 A 模型 1 epoch 冒烟训练通过（val acc > 20%）— 2026-10-19（val acc 39.66%）

### 14.3 全量训练

- [x] T3.1 五类 A/B/C/D/E 40 epoch 全量训练完成 — 2026-10-19（patience=20）
- [x] T3.2 每个 run 的 `best.pt` / `history.csv` / `train.log` / `test_metrics.json` 校验通过 — 2026-10-19

### 14.4 评测与报告图表

- [x] T4.1 生成 A/B/C/D 四模型对比图到 `pic/`：训练 loss/acc 折线、测试 acc/loss 柱状、至少 3 样本 × 4 模型识别对比 — 2026-10-19

### 14.5 实景与全类扩展实验（新增计划，尚未执行）

- [x] T5.1 生成实景清单（live_all / live5 / live_all32）并做视频级泄漏检查 — 2026-10-19
- [x] T5.2 E1：W5 五类白底模型 → 普通 `_test`（432），输出 `pic/spec-train-live-test/` 一张组合柱状图 — 2026-10-19
- [x] T5.3 E2：L5 实景训练（leaf01-03 / leaf05 / leaf04，A/B/C/D）→ 标准图，输出 `pic/live-train-test/` — 2026-10-19
- [x] T5.4 生成 all32 数据/注册表/缓存（1798 → 1082/358/358）— 2026-10-19
- [x] T5.5 E3a：W32 训练（A/B/C/D）→ 白底 test 标准图，输出 `pic/spec-train-test-all/` — 2026-10-19
- [x] T5.6 E3b：W32 → 实景 5 类子集，输出 `pic/spec-train-live-test-all/` 两张组合柱状图 — 2026-10-19
- [x] T5.7 E3c 不执行（用户确认：其它种类实景不做演示）
- [x] T5.8 全部产物校验、WORKLOG 记录与 git 提交 — 2026-10-19
- [x] T5.9 取消困难帧主测试：移除 hard 清单/图/CSV（历史步骤）— 2026-10-20
- [x] T5.10 恢复全部困难帧（227）作为鲁棒性测试集，完成 R1/R2/R3 评测与图表 — 2026-10-20

### 14.6 消融实验与报告（新增）

- [x] T6.1 补齐消融模型 E（`runs/live5`、`runs/spec32`），采集 A/E/C/D 指标 — 2026-10-20
- [x] T6.2 生成消融图表与 Word 报告到 `pic/melt-test/` — 2026-10-20
- [x] T6.3 验证图表/报告并提交 — 2026-10-20

> 输出：`pic/melt-test/ablation_main_accuracy.png`、`ablation_robustness_accuracy.png`、`ablation_gains.png`、`ablation_metrics.csv`、`消融实验报告.docx`（2 页）。

### 14.7 网页设计与实现（2026-10-20 已完成）

- [x] T7.1 Flask 骨架、静态资源与本地 ECharts vendor — 2026-10-20
- [x] T7.2 ModelRegistry（A/B/C/D，五类/32 类）与推理 API（Grad-CAM 字段预留）— 2026-10-20
- [x] T7.3 摄像头实时识别、上传识别、模型切换 — 2026-10-20
- [x] T7.4 添加数据/模型训练演示按钮与提示 — 2026-10-20
- [x] T7.5 pic 数据看板：按目录分类、图片灯箱、CSV ECharts — 2026-10-20
- [x] T7.6 页面验收、API 端到端测试、截图与提交 — 2026-10-20

> 输出：`webapp/` 全部文件；`results/screenshots/` 5 张截图。T7 轮次中 Grad-CAM 与“测试样例行”未实现；Grad-CAM 在 §14.8 的 T8.4–T8.5 补齐。

### 14.8 知叶网页二轮修改（2026-10-20 新增，待执行）

> 目标：按用户第二轮反馈修正识别页、品牌、Grad-CAM 与数据看板。全部完成后重新生成 `results/screenshots/` 并提交。

- [ ] T8.1 后端增加第三个模型组 **实景（5）**（`runs/live5`），三个组统一命名 **标本（5）／标本（32）／实景（5）**（验收：`/api/models` 返回 3 组、每组 A/B/C/D 共 12 个模型，`available=true`；`/api/health` 的 `num_models=12`；组内 `class_keys` 与对应注册表一致）
- [ ] T8.2 品牌文案改为 **知叶**，小字改为 **面向农学生的叶片识别与学习平台**；同步 `<title>`、页头、识别页标题、页脚与 `app.py` 描述（验收：全站不再出现旧名“校园叶片识别与可解释性可视化系统”或“叶知”；页面标题/品牌/小字与计划一致）
- [ ] T8.3 识别页切换模型或模型组时保留当前图片/快照并自动重新识别（验收：上传图片得到结果后点击 B/C/D 或切换模型组，预览不消失、结果自动更新为新模型、无需重新选择图片；拍照快照同理；无重复请求/竞态）
- [ ] T8.4 实现 Grad-CAM 后端 `webapp/gradcam.py` 并接入 `model_service.predict()`（验收：A/B/C/D × 三个模型组在 `gradcam=1` 时均返回非空 `gradcam_png_base64`，可被 Pillow 解码为 ≥128×128 PNG；`gradcam_available=true`；hook 在 `finally` 移除，连续 20 次调用无 hook 泄漏/显存增长）
- [ ] T8.5 打通 Grad-CAM 前端显示与 API 链路（验收：`/api/predict` 与 `/api/predict_frame` 的 `gradcam=1` 均返回可显示热力图；开关关闭隐藏、开启后对当前图片重新请求并显示；上传、拍照、实时均可用；截图 `gradcam.png` 非空）
- [ ] T8.6 看板图表数据层：新增 `webapp/chart_data.py`，从 `runs/spec5|spec32|live5` 的 `history.csv`/`test_metrics.json` 与 `pic/` CSV 生成 ECharts 规格，`/api/gallery` 按目录返回 `charts[]`（验收：六个目录图表数为 `4/4/4/4/6/6`；折线点数=40，系列顺序 A/B/C/D；柱状数值与 CSV/JSON 逐项一致；缺失数据返回空 charts，不造数）
- [ ] T8.7 数据看板改为纯交互图：移除图片网格、灯箱、文档列表、CSV/原图下载入口；目录 tab 数量改为图表数（验收：页面 DOM 不存在图片灯箱/下载链接/图片网格；每个目录只出现 ECharts canvas 与标题；空目录提示“暂无可绘制图表”；控制台无报错）
- [ ] T8.8 图表样式统一：横坐标模型名统一 A/B/C/D（消融 A/E/C/D）且水平；图例固定右上角、显示模型全称、字号 10；`melt-test` 柱状图关闭柱顶标签、仅 tooltip 悬停查看（验收：ECharts option 中 `xAxis.data`、`axisLabel.rotate=0`、`legend.right`、`legend.textStyle.fontSize=10`、`legend.data` 为全称、melt 系列 `label.show=false` 全部断言通过）
- [ ] T8.9 二轮验收：后端 test client + Playwright + 截图 + 计划勾选 + WORKLOG + git 提交（验收：T8.1–T8.8 验收项全部 PASS；新增/更新截图 7 张；无失败项才允许提交）

> 改动边界：只改 `webapp/` 与 `results/screenshots/`，不重训模型、不改 `data/class_map.json`/注册表/划分、不删除 `pic/` 原始文件、不改 `runs/` 权重。

---

## 15. 实景与全类扩展实验计划（2026-10-19 新增）

> 本节只定义计划与验收，当前不启动训练。执行顺序见 §15.5。

### 15.1 两个量级的模型组

| 量级 | 类别 | 白底数据 | 实景数据 | 四个对比模型 |
|---|---|---|---|---|
| **S（5 类）** | Flavia 4/12/15/17/20 | `data/Leaves` 294 张（178/58/58） | `data/leaf` 432 张普通实景（困难帧不参与） | A / B / C / D |
| **A（32 类）** | Flavia 1–12、14–33 | `data/Leaves` 1798 张（1082/358/358） | 当前只有 5 类实景，其余 27 类待补 | A / B / C / D |

- **每个量级固定四个模型 A/B/C/D**；E（仅 CBAM）只用于消融，不进入四模型对比图。
- 同一量级内，白底训练与实景训练使用完全相同的模型结构、超参、seed=42、40 epoch、patience=20，保证可比。
- 当前状态：W5（`runs/spec5`）的 A/B/C/D/E 已训练，A/B/C/D 为四模型组；E1 尚未执行；L5、W32 待执行；L32 不执行。
- 四模型组在每个实验中都必须保持一致；不能因为某次结果不好而替换模型或单独调参。

### 15.2 数据与划分

**实景数据（当前）**：

| 数据集 | 总数 | camphor | cercis | lagerstroemia | osmanthus | pittosporum |
|---|---:|---:|---:|---:|---:|---:|
| `*_test`（普通） | 432 | 83 | 96 | 100 | 86 | 67 |

> `*_test_hard`（227 张）**仅用于鲁棒性测试**（§15.9），不参与 train/val/主测试；全量困难帧每类 45/39/50/40/53，比 leaf04 子集更均衡。

**视频级划分（E2/L5）**：

| split | 视频 | 普通 `_test` |
|---|---|---:|
| train | leaf01 / leaf02 / leaf03 | 301 |
| val | leaf05 | 79 |
| test | leaf04 | 52 |

- 同一视频不得跨 split；生成清单后做 SHA-256 跨 split 查重与 dHash 近重复报告。
- 困难帧生成 `hard_test.csv`（全部 227 张），仅用于 §15.9 鲁棒性测试，不进入 train/val。

**清单**：

| 清单 | 用途 | 规模 |
|---|---|---:|
| `data/splits/live_all/test.csv` | E1、E3b：全量普通实景 | 432 |
| `data/splits/live5/{train,val,test}.csv` | E2：视频级划分 | 301 / 79 / 52 |
| `data/splits/live_all32/test.csv` | E3b：32 类标签 | 432 |
| `data/splits/live_all/hard_test.csv` | R1/R2：全量困难帧（5 类） | 227 |
| `data/splits/live5/hard_test.csv` | R2：全量困难帧（5 类） | 227 |
| `data/splits/live_all32/hard_test.csv` | R3：全量困难帧（32 类标签） | 227 |
| `data/splits/all32/{train,val,test,all}.csv` | E3a：32 类白底 | 1082 / 358 / 358 |

- 实景 CSV 的 `filename` 是相对 `data/leaf` 的路径（如 `camphor_test/leaf04_000010.jpg`）。
- `live_all32` 使用 `data/class_registry_all32.json` 的 `py_label`，5 个实景类映射到 32 类中的对应编号。
- 旧 `data/splits/real/all.csv`（791 行）保留但不再使用。

### 15.3 实验矩阵

| 编号 | 量级 | 训练 | 测试 | 模型目录 | 图片目录 |
|---|---|---|---|---|---|
| **E1** | S | W5 白底（已有） | `_test` 432 | `runs/spec5` | `pic/spec-train-live-test/` |
| **E2** | S | L5 实景（301 train） | `_test` leaf04 52 | `runs/live5` | `pic/live-train-test/` |
| **E3a** | A | W32 白底（1082 train） | Leaves test 358 | `runs/spec32` | `pic/spec-train-test-all/` |
| **E3b** | A | W32 白底 | 实景 5 类子集 432 | `runs/spec32` | `pic/spec-train-live-test-all/` |
| ~~E3c~~ | A | —（用户确认其它种类实景不做演示） | — | — | — |

- **E1** 不重新训练，只做推理评测。
- **E2** 每个模型训练 40 epoch，产物与 W5 相同：`best.pt/last.pt/history.*/config.json/test_metrics.json/train.log`。
- **E3b** 的 32 类模型只在 5 个有实景数据的类别上评测；其他 27 类的预测一律算错。报告写清“实景覆盖 5/32 类”，不能声称 32 类实景泛化。
- **E3c 不执行**：用户确认其它种类实景不做演示；32 类模型只做白底测试（E3a）和 5 类实景子集测试（E3b）。
- **困难帧**：227 张全量用于 §15.9 鲁棒性测试，不参与主测试/训练；它们来自 leaf01–05，与实景训练视频有重叠，结果按压力测试解读，不能当独立验证集。

### 15.4 输出与图表命名

**W5 白底标准 5 图（已完成）**：`pic/spec-train-test/`（当前 W5 的 5 张标准图已归档在此）。

**E3a 白底标准 5 图**：`pic/spec-train-test-all/`；当前目录中的同名图是占位副本，W32 训练后必须重绘覆盖。

**E1 / E3b（一张图含两个子图：测试准确率柱状 + 测试 loss 柱状；四模型同图）**：

- `pic/spec-train-live-test/test_ABCD.png`（普通 `_test` 432）
- `pic/spec-train-live-test-all/test_ABCD.png`（普通 `_test` 432，32 类标签）
- 同时输出 `metrics_test.csv`，列为 `model,accuracy,macro_f1,ce_loss,n`。
- 困难帧鲁棒性图见 §15.9。

**E2（AGENTS.md §7 标准 5 张）**：

- `pic/live-train-test/fig4_1_train_loss_ABCD.png`
- `pic/live-train-test/fig4_2_train_acc_ABCD.png`
- `pic/live-train-test/fig4_3_test_loss_bar.png`
- `pic/live-train-test/fig4_4_test_acc_bar.png`
- `pic/live-train-test/fig4_6_sample_predictions_ABCD.png`
- `pic/live-train-test/metrics_test.csv`
- 困难帧鲁棒性图见 §15.9。

**E3a（标准 5 张，白底测试）**：

- `pic/spec-train-test-all/fig4_1_train_loss_ABCD.png`
- `pic/spec-train-test-all/fig4_2_train_acc_ABCD.png`
- `pic/spec-train-test-all/fig4_3_test_loss_bar.png`
- `pic/spec-train-test-all/fig4_4_test_acc_bar.png`
- `pic/spec-train-test-all/fig4_6_sample_predictions_ABCD.png`

**E3c（不执行）**：用户确认其它种类实景不做演示；`pic/live-train-test-all/` 目录保留但留空。

### 15.5 执行步骤与命令

**T5.1 实景清单**：

```bash
"C:/Anaconda/envs/torch_env/python.exe" tools/build_live_dataset.py \
  --leaf-dir data/leaf --mode all --registry data/class_registry.json --out data/splits/live_all
"C:/Anaconda/envs/torch_env/python.exe" tools/build_live_dataset.py \
  --leaf-dir data/leaf --mode video-split --registry data/class_registry.json --out data/splits/live5
"C:/Anaconda/envs/torch_env/python.exe" tools/build_live_dataset.py \
  --leaf-dir data/leaf --mode all --registry data/class_registry_all32.json --out data/splits/live_all32
```

验收：普通实景 432、301/79/52、432；hard 227（每类 45/39/50/40/53）；无同一视频跨 split；SHA-256/dHash 检查通过。

**T5.2 E1（W5 → 实景）**：

```bash
"C:/Anaconda/envs/torch_env/python.exe" tools/evaluate_live.py \
  --runs runs/spec5 --models A B C D --registry data/class_registry.json \
  --manifest data/splits/live_all/test.csv --image-root data/leaf \
  --out pic/spec-train-live-test --tag test
```

**T5.3 E2（L5 实景训练 + 标准图）**：

```bash
"C:/Anaconda/envs/torch_env/python.exe" -m src.train \
  --models A B C D --data data/splits/live5 --image-root data/leaf \
  --registry data/class_registry.json --num-classes 5 \
  --epochs 40 --batch-size 32 --eval-batch-size 64 --seed 42 --patience 20 \
  --output runs/live5 --num-workers 0 --threads 8
"C:/Anaconda/envs/torch_env/python.exe" tools/make_comparison_plots.py \
  --runs runs/live5 --models A B C D --data data/splits/live5 \
  --image-root data/leaf --registry data/class_registry.json --out pic/live-train-test --dpi 160
```

**T5.4 生成 all32 数据与缓存**：

```bash
"C:/Anaconda/envs/torch_env/python.exe" tools/build_dataset.py \
  --leaves data/Leaves --classes all --subset all32 --out data/splits \
  --registry data/class_registry_all32.json --seed 42 --materialize
"C:/Anaconda/envs/torch_env/python.exe" tools/prepare_cache.py \
  --src data/Leaves --dst data/cache/leaf_320_all32 --long-edge 320 \
  --manifest data/splits/all32/all.csv
```

**T5.5 E3a（W32 白底训练 + 白底标准图）**：

```bash
"C:/Anaconda/envs/torch_env/python.exe" -m src.train \
  --models A B C D --data data/splits/all32 --registry data/class_registry_all32.json \
  --num-classes 32 --cache data/cache/leaf_320_all32 \
  --epochs 40 --batch-size 32 --eval-batch-size 64 --seed 42 --patience 20 \
  --output runs/spec32 --num-workers 0 --threads 8
"C:/Anaconda/envs/torch_env/python.exe" tools/make_comparison_plots.py \
  --runs runs/spec32 --models A B C D --data data/splits/all32 --image-root data/Leaves \
  --registry data/class_registry_all32.json --out pic/spec-train-test-all --dpi 160
```

**T5.6 E3b（W32 → 实景 5 类子集）**：

```bash
"C:/Anaconda/envs/torch_env/python.exe" tools/evaluate_live.py \
  --runs runs/spec32 --models A B C D --registry data/class_registry_all32.json \
  --manifest data/splits/live_all32/test.csv --image-root data/leaf \
  --out pic/spec-train-live-test-all --tag test
```

**T5.7 E3c（不执行）**：用户确认其它种类实景不做演示，跳过 L32 训练；不生成 `runs/live32`，不写 `pic/live-train-test-all/`。

**T5.8 验收**：按 §15.8 逐项检查，更新 `plans.md` 勾选，写日志并提交。

**T5.10 实景鲁棒性测试**：见 §15.9.5；使用全部 227 张困难帧，测试 R1/R2/R3 并输出图表。

### 15.6 需要新增/修改的工具

| 工具 | 改动 | 用途 |
|---|---|---|
| `tools/build_live_dataset.py` | 新增 | 解析 `_test`（普通 432）与 `_test_hard`（全部 227，仅写 `hard_test.csv`），按 5 类或 32 类注册表生成清单；`--mode all` / `--mode video-split`；SHA-256/dHash 泄漏检查 |
| `tools/evaluate_live.py` | 新增 | 加载 A/B/C/D checkpoint，在普通或困难清单上计算 accuracy/macro-F1/CE，画组合柱状图，输出 metrics CSV |
| `tools/make_comparison_plots.py` | 修改 | 增加 `--image-root`，支持实景样本图；默认路径保持兼容 |
| `src/train.py` | 不改 | 已支持 `--image-root`、`--data`、`--output`、`--num-classes` |
| `src/data.py` | 不改 | 已支持相对路径 `filename` 与 `image_root` |
| `tools/build_dataset.py` / `prepare_cache.py` | 不改 | 复用现有 all32 与缓存逻辑 |

### 15.7 风险与依赖

- **32 类实景不做**：用户确认其它种类实景不做演示；E3c 不执行，不构成风险。不得用白底图冒充实景，也不得把 5 类结果说成 32 类泛化。
- **困难帧只做鲁棒性测试**：227 张全量、每类 45/39/50/40/53；它们来自 leaf01–05，与实景训练视频（leaf01-03）有重叠，结果反映“困难条件下的鲁棒性”，不能当作独立验证集或泛化证据。
- **实景训练集小且不均衡**：301 张、最多 76 / 最少 43；报告每类 recall，必要时加 class weight（但四个模型必须一致）。
- **白底捷径**：E1/E3b 的实景结果才是跨域泛化的主要证据；白底 test 接近饱和不能说明模型学到了叶片形状。
- **视频级泄漏**：务必按 leaf01-03 / 05 / 04 划分，禁止随机拆帧；再做 SHA-256/dHash 检查。
- **CPU 时间**：L5 约 5–10 分钟；W32 约 45–90 分钟；使用 `--num-workers 0`，避免 Windows 多进程劣化。
- **四模型公平性**：A/B/C/D 必须同结构、同超参、同 seed、同 epoch；E 不参与四模型对比。

### 15.8 验收标准

- [x] T5.1 三个实景清单生成，计数与 §15.2 一致，无跨 split 泄漏；
- [x] E1：`pic/spec-train-live-test/` 下 `test_ABCD.png` 存在且非空，四模型 acc/loss 数值正确；
- [x] E2：`runs/live5` 四个模型各有 7 个产物；`pic/live-train-test/` 下 5 张标准图存在且非空；样本网格 3×4；
- [x] E3a：`runs/spec32` 四个模型各有 7 个产物；`pic/spec-train-test-all/` 下 5 张标准图存在且非空；
- [x] E3b：`pic/spec-train-live-test-all/` 下 `test_ABCD.png` 存在且非空；报告明确“实景覆盖 5/32 类”；
- [x] E3c：不执行（用户确认其它种类实景不做演示）；
- [x] 困难帧仅用于鲁棒性测试；R1/R2/R3 鲁棒性图见 §15.9；
- [x] 所有图的中文正常、四模型顺序一致、数值与 metrics CSV 对得上；
- [x] `plans.md` 勾选、`WORKLOG.md` 记录、git 提交完成。

---

## 15.9 实景鲁棒性测试（全部困难帧 227，2026-10-20 新增）

### 15.9.1 目的与数据

- 用 **全部困难帧**（`data/leaf/*_test_hard/`，227 张）测试模型在困难条件下的鲁棒性。
- 类别分布：camphor 45 / cercis 39 / lagerstroemia 50 / osmanthus 40 / pittosporum 53，比只用 leaf04 的 35 张更均衡。
- 困难帧**仅用于测试**，不进入 train/val；它们来自 leaf01–05，其中 leaf01–03 与实景训练视频重叠，结果按压力测试解读，**不能当作独立验证集或泛化证据**。
- 主测试仍是普通 `_test` 432 张（§15.2）；鲁棒性测试单独报告，不与主测试混合。

### 15.9.2 实验矩阵

| 编号 | 模型组 | 训练数据 | 困难测试 | 图片目录 |
|---|---|---|---|---|
| **R1** | W5（A/B/C/D） | 5 类白底 294 张 | 全部 227 张（5 类标签） | `pic/spec-train-live-test/` |
| **R2** | L5（A/B/C/D） | 5 类实景 301/79/52 | 全部 227 张（5 类标签） | `pic/live-train-test/` |
| **R3** | W32（A/B/C/D） | 32 类白底 1798 张 | 全部 227 张（32 类标签，覆盖 5/32 类） | `pic/spec-train-live-test-all/` |

### 15.9.3 清单

- `data/splits/live_all/hard_test.csv`：5 类标签，227 行；
- `data/splits/live5/hard_test.csv`：5 类标签，227 行；
- `data/splits/live_all32/hard_test.csv`：32 类标签，227 行；
- 所有行 `split=hard_test`，`filename` 为相对 `data/leaf` 的路径（如 `camphor_test_hard/leaf01_000010.jpg`）。

### 15.9.4 输出

- `pic/spec-train-live-test/test_hard_ABCD.png` + `metrics_test_hard.csv`
- `pic/live-train-test/test_hard_ABCD.png` + `metrics_test_hard.csv`
- `pic/spec-train-live-test-all/test_hard_ABCD.png` + `metrics_test_hard.csv`
- 每张图：左“测试准确率”柱状、右“测试 loss”柱状，四模型同图，中文标签，dpi 160。

### 15.9.5 命令

```bash
# 生成 hard 清单（T5.10）
"C:/Anaconda/envs/torch_env/python.exe" tools/build_live_dataset.py \
  --leaf-dir data/leaf --mode all --registry data/class_registry.json --out data/splits/live_all
"C:/Anaconda/envs/torch_env/python.exe" tools/build_live_dataset.py \
  --leaf-dir data/leaf --mode video-split --registry data/class_registry.json --out data/splits/live5
"C:/Anaconda/envs/torch_env/python.exe" tools/build_live_dataset.py \
  --leaf-dir data/leaf --mode all --registry data/class_registry_all32.json --out data/splits/live_all32

# R1：W5 → 全部困难帧
"C:/Anaconda/envs/torch_env/python.exe" tools/evaluate_live.py \
  --runs runs/spec5 --models A B C D --registry data/class_registry.json \
  --manifest data/splits/live_all/hard_test.csv --image-root data/leaf \
  --out pic/spec-train-live-test --tag test_hard
# R2：L5 → 全部困难帧
"C:/Anaconda/envs/torch_env/python.exe" tools/evaluate_live.py \
  --runs runs/live5 --models A B C D --registry data/class_registry.json \
  --manifest data/splits/live5/hard_test.csv --image-root data/leaf \
  --out pic/live-train-test --tag test_hard
# R3：W32 → 全部困难帧（32 类标签）
"C:/Anaconda/envs/torch_env/python.exe" tools/evaluate_live.py \
  --runs runs/spec32 --models A B C D --registry data/class_registry_all32.json \
  --manifest data/splits/live_all32/hard_test.csv --image-root data/leaf \
  --out pic/spec-train-live-test-all --tag test_hard
```

### 15.9.6 验收

- [x] 三套 `hard_test.csv` 各 227 行，每类 45/39/50/40/53，全部 `split=hard_test`；
- [x] R1/R2/R3 各自生成 `test_hard_ABCD.png` 与 `metrics_test_hard.csv`，四模型数值齐全；
- [x] R1/R2 使用 5 类标签；R3 使用 32 类标签，并说明实景覆盖 5/32 类；
- [x] 报告中明确困难帧与训练视频（leaf01-03）有重叠，只作鲁棒性/压力测试；
- [x] `plans.md` 勾选 T5.10、`WORKLOG.md` 记录、git 提交完成。
