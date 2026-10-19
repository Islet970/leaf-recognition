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
