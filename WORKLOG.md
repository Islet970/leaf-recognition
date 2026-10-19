# WORKLOG — 统一工作日志

> 全项目唯一的日志文件，按时间顺序追加，不覆盖历史。格式见 [`AGENTS.md`](AGENTS.md) §6。

---

## [2026-10-19 12:45] T0 初始化代理协作规范与仓库

- 执行代理: 主代理
- 目标: 建立主代理/子代理分工、计划勾选、日志与提交规范
- 改动:
  - 新增 `AGENTS.md`：角色分工、子代理调用规范（opus + medium）、标准工作循环、勾选/日志/提交规则、红线与中断恢复
  - 新增 `CLAUDE.md`：导入 `AGENTS.md`，确保每次会话自动加载规范
  - 新增 `.gitignore`：排除数据集、缓存、权重、临时文件
  - 新增 `WORKLOG.md`：本日志文件
- 命令与结果:
  - `ls` / `du -sh data/*` → `data/Leaves` 约 1013 MB、`data/leaf` 约 17 MB，均不应入库
  - `git init` → 初始化仓库
  - `git status` → 仅纳入源码、文档与小体积配置
- 验证结论: 通过（规范文件已落盘；仓库已初始化）
- 提交: `chore(agents): 初始化多代理协作规范与工作流`（哈希见 `git log --grep="T0"`）
- 遗留问题 / 下一步: 按 `plans.md` 执行顺序，从「环境依赖安装」开始；首个待办步骤为安装 5 个缺失包并验证导入

---

## [2026-10-19 13:10] T0.1 按实际状态修订 `plans.md`

- 执行代理: 主代理（计划修订属于方案协调职责）
- 目标: 让计划与仓库现状一致：移除不存在的 `cato.txt` 依赖，登记已完成的分类校验，纳入已有实景帧
- 改动:
  - `plans.md`：Context、§1.1–1.5、§3.4、§3.6、§5.1、§6、§7、§8、§9、§10、§11、§12、§13 全面更新
  - 新增 `data/class_map.json` 为程序唯一映射来源（由 §13 派生，待生成）
  - 字段 `cato_label` 统一改为 `flavia_label`；`cato_labels` 改为 `flavia_labels`
  - 新增 §1.5 实景测试帧：`data/leaf/*_test/` 共 791 张，只做跨域评测，不进训练
  - §8 勾选已完成的分类校验与实景抽帧条目
  - `data/Flavia类别对照表.md`：表头与来源说明同步为 `plans.md` §13，数据不变
- 命令与结果:
  - 解析 `plans.md` §13 → 32 行、张数合计 1798；与 `data/Flavia类别对照表.md` 逐行比对 → mismatch: none；区间无重叠
  - 统计 `data/leaf/*/*.jpg` → 160/163/155/145/168，合计 791
- 验证结论: 通过
- 提交: `docs(plan): 按实际状态修订类别映射与实景数据说明`（哈希见 `git log --grep="T0.1"`）
- 遗留问题 / 下一步: 按执行顺序先做环境依赖安装；随后生成 `data/class_map.json` 并实现 `tools/build_dataset.py`

---

## [2026-10-19 13:25] T1.1 安装训练依赖

- 执行代理: 主代理（环境操作）
- 目标: 补齐训练/评测/网页所需依赖
- 改动: 无仓库文件改动；`plans.md` 新增 §14 执行进度并勾选 T1.1
- 命令与结果:
  - `C:/Anaconda/envs/torch_env/python.exe -m pip install scikit-learn matplotlib pandas flask tqdm` → 成功安装 scikit-learn 1.7.2、matplotlib 3.10.9、pandas 2.3.3、flask 3.1.3、tqdm 4.70.1
  - 导入检查脚本 → 5 个包全部 OK；torch 2.5.1、torchvision 0.20.1、CUDA 不可用（CPU 训练）
- 验证结论: 通过
- 提交: `chore(env): 安装训练依赖并建立执行进度`（哈希见 `git log --grep="T1.1"`）
- 遗留问题 / 下一步: T1.2 生成 32 类映射 JSON；T2.1/T2.2 并行实现数据与模型代码

---

## [2026-10-19 13:40] T1.2 生成 32 类映射与中文名清单

- 执行代理: 子代理（编码，opus / medium）+ 主代理复核
- 目标: 把 plans.md §13 的 32 类映射固化为程序可读 JSON
- 改动:
  - 新增 `tools/make_class_map.py`：解析 plans.md §13，并与 `data/Flavia类别对照表.md` 交叉核对
  - 新增 `data/class_map.json`：32 类，字段 `flavia_label/key/zh_name/scientific/common/range`
  - 新增 `data/zh_names.json`：`flavia_label → 中文名`
- 命令与结果:
  - `python -X utf8 tools/make_class_map.py` → 32 类；编号 1–12/14–33；区间互不重叠；与对照表 0 差异；张数合计 1798
  - 主代理独立重读 JSON → 32 条、编号升序、range 均为两个 int、中文名键齐全
- 验证结论: 通过
- 提交: `feat(data): 生成 32 类映射与中文名清单`
- 遗留问题 / 下一步: T1.3 构建五类划分

---

## [2026-10-19 13:42] T2.2 实现 A/B/C/D/E 模型

- 执行代理: 子代理（编码，opus / medium）+ 主代理复核
- 目标: 按 plans §2 实现五个可对比模型与 Grad-CAM 目标层
- 改动:
  - 新增 `src/__init__.py`
  - 新增 `src/models.py`：`MODEL_IDS`、`MODEL_NAMES`、`build_model()`、`MSFF`、`CBAM`、`LeafNetV3`、`get_gradcam_layer()`
  - A=MobileNetV3-Small、B=MobileNetV2、C=A+MSFF、D=A+MSFF+CBAM、E=A+CBAM
- 命令与结果:
  - 子代理：五模型 forward `(2,5)`；A/B 预训练权重下载成功；unexpected keys 全为 0，missing 仅为新增模块/分类头
  - 主代理独立复核：A 1,522,981 / B 2,230,277 / C 1,805,797 / D 1,847,367 / E 1,564,551；Grad-CAM 层输出 A/E 576×4×4、B 1280×4×4、C/D 576×8×8
- 验证结论: 通过（参数与计划表一致；代码审查 T2.4 待做）
- 提交: `feat(models): 实现 A/B/C/D/E 五模型与 Grad-CAM 层`
- 遗留问题 / 下一步: T2.4 独立代码审查；T2.3 训练脚本

---

## [2026-10-19 13:50] T2.1 实现数据管道

- 执行代理: 子代理（编码，opus / medium）+ 主代理复核
- 目标: 数据集、训练/评测变换、DataLoader、通用工具
- 改动:
  - 新增 `src/utils.py`：`set_seed`、`get_logger`、`AverageMeter`、`save/load_checkpoint`、`macro_f1`、`count_parameters`
  - 新增 `src/data.py`：`LeafDataset`、`build_train_transform`、`build_eval_transform`、`build_dataloaders`
- 命令与结果:
  - 子代理临时目录自测：batch `(4,3,128,128)`、train/eval transform 均 `(3,128,128)`、缺列报错、缓存分支正常
  - 主代理用真实清单复核：`build_dataloaders('data/splits/5class','data/Leaves')` → train 178 / val 58 / test 58，batch `(8,3,128,128)` float32，标签 0–4
  - `py_compile src/utils.py src/data.py` 通过
- 验证结论: 通过
- 提交: `feat(data): 实现数据集与 DataLoader 管道`
- 遗留问题 / 下一步: T2.3 训练脚本；T2.4 代码审查

---

## [2026-10-19 13:52] T1.3 + T1.4 构建五类划分与实景清单

- 执行代理: 子代理（编码，opus / medium）+ 主代理复核
- 目标: 生成五类注册表、train/val/test 清单与 791 行实景测试清单
- 改动:
  - 新增 `tools/build_dataset.py`
  - 新增 `data/class_registry.json`（五类，py_label 0–4）
  - 新增 `data/splits/5class/{train,val,test,all}.csv` 与 `split_stats.json`
  - 新增 `data/splits/real/all.csv`（791 行，split=test）
  - `data/dataset/5class/` 物化 294 个硬链接（已被 .gitignore 忽略，不入库）
- 命令与结果:
  - `build_dataset.py --classes 4,12,15,17,20 --subset 5class --seed 42 --materialize` → 294 → 178/58/58
  - `build_dataset.py --real-dir data/leaf --subset real` → 791 → 163/168/160/145/155
  - 主代理独立复核：三个 split 两两不相交、并集等于 all.csv；各类 41/33/37/32/35（train）、13/11/12/11/11（val/test）；注册表 image_count 67/55/61/54/57；实景 791 行全部 split=test、filename 唯一且为相对路径
  - SHA-256 跨类重复 0，unique = 294；实景 unique = 791
- 验证结论: 通过
- 提交: `feat(data): 构建五类划分与实景测试清单`
- 遗留问题 / 下一步: 实景清单 `filename` 为相对路径（如 `cercis_test/leaf01_000000.jpg`），后续评测用 `data/leaf/<filename>` 拼接

---

## [2026-10-19 13:58] T1.5 生成 320px 训练缓存

- 执行代理: 子代理（编码，opus / medium）+ 主代理复核
- 目标: 把五类 294 张原图缩到长边 320，降低 CPU 每轮解码开销
- 改动:
  - 新增 `tools/prepare_cache.py`
  - 生成 `data/cache/leaf_320/*.jpg`（294 张，已忽略，不入库）
- 命令与结果:
  - `python tools/prepare_cache.py --src data/Leaves --dst data/cache/leaf_320 --long-edge 320 --manifest data/splits/5class/all.csv` → 处理 294，跳过 0，失败 0，耗时 16.00s
  - 主代理复核：缓存 294 张；随机抽 5 张均为 320×240 RGB；全部长边 = 320
  - 再运行一次 → 处理 0、跳过 294，幂等性通过
- 验证结论: 通过（数据预览图 `fig2_1` 尚未生成，留待报告阶段）
- 提交: `feat(data): 新增 320px 图像缓存工具`
- 遗留问题 / 下一步: T2.3 训练脚本完成后先审查，再跑 A 冒烟

---

## [2026-10-19 13:31] T2.3 + T2.4 + T2.5 训练脚本、审查修复与 A 冒烟

- 执行代理: 子代理（编码 T2.3，opus / medium）、子代理（独立审查 T2.4，opus / medium）、子代理（修复，opus / medium）、主代理验证
- 目标: 实现训练入口，审查并修复风险，完成 A 模型 1 epoch 冒烟
- 改动:
  - 新增 `src/train.py`：完整训练/验证/测试流程，checkpoint 与日志产物齐全
  - 修改 `src/models.py`：仅在 `_self_test()` 中补充预训练权重等价抽查
  - 训练脚本新增 CSV↔注册表标签交叉校验、val/test 统一使用无平滑 CE、32 类自动 top-5
- 命令与结果:
  - 独立审查：无阻塞；1 条重要（标签静默错位风险）+ 5 条建议，重要项与 3 条建议已修复
  - A 冒烟（1 epoch，完整数据，cache，num-workers=4）：train loss 1.3892 acc 0.5000；val loss 1.3808 acc **0.3966** f1 0.3616；test acc 0.3448、macro-F1 0.2741、top-3 0.8103
  - 产物核对：`best.pt`/`last.pt`/`history.csv`/`history.json`/`config.json`/`test_metrics.json`/`train.log` 全部生成；checkpoint 字段与注册表一致；test_metrics 含 `topk_k=3`/`topk_accuracy`/`top3_accuracy`
  - 性能诊断：DataLoader（0 worker）1.08s/178 张；模型 A 6 个 batch 0.80s。冒烟 42.1s/epoch 来自 Windows 4 worker 多进程开销，正式训练改用 `--num-workers 0`
- 验证结论: 通过（val acc 39.66% > 20%）
- 提交: `feat(train): 实现训练流程并通过 A 冒烟`
- 遗留问题 / 下一步: 启动 A/B/C/D/E 40 epoch 全量训练（`--num-workers 0`）

---

## [2026-10-19 13:55] T3.1 + T3.2 五模型 40 epoch 全量训练

- 执行代理: 主代理（启动与验证）
- 目标: 完成 A/B/C/D/E 五类训练，产出可供评测与报告的 checkpoint
- 首次训练（patience=10）发现的问题: D 在 warmup 后 val F1 回落，best 锁定在 epoch 3，13 epoch 即早停（test acc 89.66%）；其余模型正常。归档在 `runs/_archive_patience10/`
- 处理:
  - 将五模型统一改为 `--patience 20` 重跑，保证协议一致；`plans.md §3.3` 已同步说明
  - 性能配置 `--num-workers 0`（Windows 4 worker 使每 epoch 从约 2s 劣化到 42s）
- 最终结果（seed=42，patience=20，均训练 40 epoch）:

| 模型 | best epoch | val acc | val F1 | test acc | test macro-F1 | test CE | top-3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_mnv3s | 40 | 1.0000 | 1.0000 | 0.9828 | 0.9831 | 0.1713 | 0.9828 |
| B_mnv2 | 25 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0963 | 1.0000 |
| C_leafnet_msff | 25 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1380 | 1.0000 |
| D_leafnet_msff_cbam | 22 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1328 | 1.0000 |
| E_mnv3s_cbam | 40 | 1.0000 | 1.0000 | 0.9828 | 0.9831 | 0.1546 | 1.0000 |

- 命令与结果:
  - `python -m src.train --models A B C D E ... --epochs 40 --patience 20 --num-workers 0` → exit 0
  - 主代理逐 run 校验：7 个产物齐全、history 40 行无 NaN、checkpoint 的 class_keys/flavia_labels/num_classes 与注册表一致、test n=58、topk_k=3、train.log 无 ERROR/Traceback → **ALL PASS**
- 验证结论: 通过。五类任务过于容易，A/B/C/D 在干净 test 上已接近饱和；模型差异需靠后续压力测试与实景测试区分（plans §4.3 已预判）
- 提交: `feat(train): 完成五模型 40 epoch 全量训练`
- 遗留问题 / 下一步: 实现 `src/evaluate.py` 与报告图表；压力测试 + 791 张实景跨域评测

---

## [2026-10-19 15:20] 诊断 训练饱和原因与 D 早停

- 执行代理: 主代理（只读分析）
- 目标: 判断高分是否来自过拟合或标签/背景捷径，并解释 D 的早停
- 命令与结果:
  - 五个 run 的 history：val loss 持续下降到 0.10–0.13，val acc 100%，train acc 约 99% → 无经典过拟合
  - dHash 近邻：test→train 距离 ≤6 的 18/58、≤4 的 6/58；最近邻像素 MAD 8.8–9.6，无完全重复图
  - 四角背景 RGB + nearest-centroid：test acc **63.8%**；16×16 灰度形状：**93.1%**；第 17 类背景明显偏绿
  - D 首轮 patience=10：best F1 0.863（epoch 3，val loss 0.549）；epoch 13 时 val loss 已降至 0.395 但 F1 未超过 0.863，触发早停；patience=20 后 epoch 22 达 100%
- 验证结论: 高分主要来自任务本身简单 + 白底/采集批次相关性，不是训练过久过拟合；D 早停是 warmup 后优化波动 + 58 张验证集 F1 噪声 + patience 过短共同造成
- 提交: `docs(analysis): 记录训练饱和与 D 早停诊断`
- 遗留问题 / 下一步: 优先做实景 791 张跨域评测；32 类训练作为补充实验，不作为消除捷径的手段

---

## [2026-10-19 16:00] T4.1 生成 A/B/C/D 四模型对比图

- 执行代理: 子代理（编码，opus / medium）+ 主代理目视复核
- 目标: 训练完成后自动生成四模型对比图，并固化为强制流程
- 改动:
  - 新增 `tools/make_comparison_plots.py`（可复用；每次训练完成后重跑）
  - 新增 `pic/` 下 5 张图：训练 loss 折线、训练 acc 折线、测试 acc 柱状、测试 loss 柱状、3 样本 × 4 模型识别对比
  - `AGENTS.md` §7 新增“训练完成后的强制图表门禁”（5 张图存在且非空才算训练完成）
  - `AGENTS.md` §8 同步移除残留 cato 约束，改为 `data/class_map.json` / §13 映射
  - `plans.md` 目录结构加入 `pic/`，§4.2 补充说明，勾选 T4.1
- 命令与结果:
  - `python tools/make_comparison_plots.py --runs runs/5class --models A B C D --data data/splits/5class --registry data/class_registry.json --out pic --dpi 160` → exit 0
  - 5 张图尺寸/大小：1472×864（120 KB）、1472×864（114 KB）、1344×832（53 KB）、1344×832（65 KB）、2560×1872（726 KB）
  - 主代理逐张目视检查：中文正常、四模型同图、柱顶数值正确、样本网格绿色/红色边框正确
  - 样本：2346.jpg（真实桂花，A 误判紫荆 28.0%，B/C/D 正确）、2070.jpg（海桐，全对）、1156.jpg（紫荆，全对）
- 验证结论: 通过
- 提交: `feat(plots): 生成四模型对比图并固化训练后绘图流程`
- 遗留问题 / 下一步: 压力测试与 791 张实景跨域评测

---

## [2026-10-19 16:30] T5.0 制定实景与 32 类扩展实验计划

- 执行代理: 主代理（计划设计，未执行训练）
- 目标: 定义两个量级（5 类 / 32 类）× 四个模型（A/B/C/D）的实景测试、实景训练与全类训练方案
- 现状核对:
  - `data/leaf/*_test/` 432 张（83/96/100/86/67）；`*_test_hard/` 227 张（45/39/50/40/53）
  - 按视频：train leaf01-03 = 301；val leaf05 = 79；test leaf04 = 52；hard test leaf04 = 35
  - 旧版 791 张清单已被当前数据取代，`data/splits/real/all.csv` 保留但不再使用
- 计划改动:
  - `plans.md` §1.5 更新实景数据口径与视频级划分
  - 新增 §15：§15.1 两个量级与四模型、§15.2 数据/清单、§15.3 实验矩阵、§15.4 输出与图表命名、§15.5 执行命令、§15.6 工具改动、§15.7 风险、§15.8 验收
  - 新增 §14.5 任务 T5.1–T5.8
  - 同步更新 §3.6 命令、§4.3 诊断、§6 实景流程、§7 目录结构、§8 验证清单、§10 报告对应、§14 T1.4 说明
- 实验矩阵:
  - E1：W5（`runs/5class`）→ `_test` / `_test_hard`，输出 `spe-train-live-test/` 两张组合柱状图
  - E2：L5 视频级实景训练（301/79/52），输出 `spelive-train-test/` 标准 5 图 + 困难测试图
  - E3a：W32 白底训练（1798 → 1082/358/358），输出 `spe-train-test-all/`
  - E3b：W32 → 实景 5 类子集，输出 `spe-train-live-test-all/`
  - E3c：L32 实景训练，依赖 32 类实景数据；缺失时标记 BLOCKED
- 验证结论: 计划文档自检通过；未创建代码、未启动训练
- 提交: `docs(plan): 新增实景与 32 类扩展实验计划`
- 遗留问题 / 下一步: 按 §15.5 从 T5.1 开始执行；E3c 需先补齐 27 类实景数据

---

## [2026-10-19 16:40] T5.0a 对齐实验输出目录到 `pic/` 子目录

- 执行代理: 主代理
- 目标: 按用户已创建的 `pic/` 子目录统一图表输出路径
- 改动:
  - `plans.md` §7、§14.5、§15.3–15.5、§15.8 的输出路径统一为 `pic/spec-train-test*`、`pic/spe-train-live-test*`、`pic/spelive-train-test*`
  - `AGENTS.md` §7 绘图门禁：五类白底输出 `pic/spec-train-test`，32 类输出 `pic/spec-train-test-all`
  - 原 `pic/fig4_*.png` 归档到 `pic/spec-train-test/`
- 说明: `pic/spec-train-test-all/` 当前是同名占位副本，W32 训练后重绘覆盖；空目录 `pic/spe-*`、`pic/spelive-*` 由后续脚本写入
- 验证: 计划中的命令路径与磁盘目录一致；未启动训练
- 提交: `docs(plan): 对齐实景实验输出目录到 pic 子目录`

---

## [2026-10-19 16:50] T5.0b 取消 E3c（32 类实景训练）

- 执行代理: 主代理
- 用户决策: 其它种类的实景不做演示
- 改动: `plans.md` §7/§8/§14.5/§15.1/§15.3/§15.4/§15.5/§15.7/§15.8 将 E3c 标记为不执行；T5.7 勾选为“不执行”
- 说明: 32 类模型只做白底测试（E3a）和 5 类实景子集测试（E3b）；不生成 `runs/live32`，不写 `pic/spelive-train-test-all/`
- 验证: 计划中已无 E3c 训练命令与产物要求
- 提交: `docs(plan): 取消 32 类实景训练（E3c）`
- 下一步: T5.1 实景清单 → E1 → E2 → E3a → E3b

---

## [2026-10-19 17:00] T5.1 生成实景清单与视频级划分

- 执行代理: 子代理（编码，opus / medium）+ 主代理复核
- 目标: 生成 E1/E2/E3b 所需的实景清单，并防止同一视频跨 split
- 改动:
  - 新增 `tools/build_live_dataset.py`
  - 新增 `data/splits/live_all/{test,hard_test}.csv`（432 / 227）
  - 新增 `data/splits/live5/{train,val,test,hard_test}.csv`（301 / 79 / 52 / 35）
  - 新增 `data/splits/live_all32/{test,hard_test}.csv`（432 / 227，32 类 py_label）
- 命令与结果:
  - `build_live_dataset.py --mode all` → live_all 432/227
  - `build_live_dataset.py --mode video-split` → train 301（leaf01-03）/ val 79（leaf05）/ test 52（leaf04）/ hard_test 35（leaf04）
  - `build_live_dataset.py --mode all --registry class_registry_all32.json` → live_all32 432/227
  - 主代理复核：各类计数与 §15.2 完全一致；train/val/test 文件名两两无交集；视频前缀严格分离；live_all32 的 py_label = [3,11,13,15,18] 与 32 类注册表一致；hard_test 全部为 leaf04
  - 子代理 SHA-256：live5 467 行 unique=467、跨 split 重复 0；live_all/32 各 659 行 unique=659；dHash 跨 split 近重复 0
- 验证结论: 通过
- 提交: `feat(data): 生成实景清单与视频级划分`
- 遗留问题 / 下一步: E1 评测；E2 实景训练

---

## [2026-10-19 17:10] T5.4 生成 32 类数据、注册表与缓存

- 执行代理: 主代理（数据构建）+ 子代理（截断图修复，opus / medium）
- 目标: 为 E3a/E3b 准备 32 类白底数据与缓存
- 改动:
  - 新增 `data/class_registry_all32.json`（32 类，py_label 0–31）
  - 新增 `data/splits/all32/{train,val,test,all}.csv` 与 `split_stats.json`（1082/358/358/1798）
  - 生成 `data/cache/leaf_320_all32/`（1798 张，长边 320）
  - `tools/prepare_cache.py`、`src/data.py`、`tools/make_comparison_plots.py` 启用 `ImageFile.LOAD_TRUNCATED_IMAGES = True`
- 命令与结果:
  - `build_dataset.py --classes all --subset all32 ...` → 32 类，1798 张，train/val/test = 1082/358/358；三集合两两不相交；跨类 SHA-256 重复 0；VALIDATION PASS
  - `prepare_cache.py --manifest data/splits/all32/all.csv` → 首次失败 1 张（3023.jpg 截断 51 字节）；启用容错后补全，最终 1798 张、失败 0
  - `3023.jpg` 普通打开失败、容错打开成功（1600×1200 RGB）；缓存 320×240 RGB；训练/验证 tensor `(3,128,128)` float32
- 验证结论: 通过
- 提交: `feat(data): 生成 32 类数据与缓存并修复截断图读取`
- 遗留问题 / 下一步: E3a 训练进行中；E3b 待 W32 完成后评测
