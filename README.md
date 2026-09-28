# model-training-core

从一个实际分类项目中拆出的可复用训练工程：**定义任务 → 检查数据 → 训练 → 保存/加载 → 预测 → 评测与追溯**。

这里的“通用”指训练流程和接口可以复用。当前真正实现的训练能力是 **TF-IDF + 可选数值特征 + Logistic / LightGBM 分类**；上传研判保留为兼容实例，另用三分类文本示例验证跨任务复用。语言生成、视觉、神经网络微调、分布式训练尚未实现。

## 从一个小例子开始

需要 Python 3.11 和 `uv`。在仓库目录运行以下 PowerShell 命令；示例数据都是公开合成数据，不需要准备业务数据或下载正式模型。

```powershell
uv sync --locked --extra dev --python 3.11
uv run model-workflow validate-data --config configs/text_demo.json
$result = uv run model-workflow train --config configs/text_demo.json | ConvertFrom-Json
uv run model-workflow predict --artifact $result.artifact --input examples/text_dev.jsonl
uv run model-workflow evaluate --artifact $result.artifact --data examples/text_independent.jsonl --purpose independent_test --independent --provenance "Held-out synthetic engineering fixture"
uv run pytest -q
```

`$result.artifact` 是新建的运行目录。跨平台使用时，也可以直接把训练命令输出的 `artifact` 路径传给后续命令。

这个例子使用 24 条训练、6 条开发、6 条保留测试数据，标签为 `food`、`travel`、`technology`。它验证工程能运行，不代表真实场景的模型效果。`independent_test_eligible` 只表示声明与可检查的数据隔离条件通过。

## 代码如何拆分

| 目录 | 负责什么 |
| --- | --- |
| `src/training_core/` | 样本与任务协议、数据校验、用途隔离、运行记录、模型产物清单、通用命令 |
| `src/training_backends/` | 实际可复用的稀疏分类训练、预测与持久化 |
| `src/training_tasks/` | 任务自己的输入、目标、特征、输出与评测；显式注册任务 |
| `src/upload_judge/` | 原上传接口、规则和模型文件兼容；训练已委托共享后端 |
| `examples/`、`configs/` | 可直接运行的合成数据与配置 |
| `pipelines/` | 三个历史兼容模块；不是新工程主入口 |

核心不导入上传模块，也不要求结果包含“成功概率”或固定四类标签。换任务时，先定义输入和目标，再实现适配器；算法不适用时另接后端，不能只改任务名字。

## 每次运行留下什么

产物进入 `runs/<run_id>/`：配置、环境、数据统计、分区报告、运行状态；训练还会保存模型、SHA-256 清单与哈希成员记录。解析与数据 hash 使用同一份字节，防止读到的批次与记录的批次错位。失败和可处理的中断会记录状态；不加载未成功完成的训练制品。

默认预测/评测输出跟随当前工作目录，也可用 `MODEL_TRAINING_HOME` 指定工作区。正式安装包不会把运行结果写进 Python 安装环境。配置中的 `run_root` 和 CLI 的 `--run-root` 仍可显式指定位置。

训练只读取 `train` 和 `development`。评测显式区分 `development`、`historical_regression`、`independent_test`，检查 ID、内容、模板及分组交叉和指纹覆盖。新模型目录的 `evaluation_history/` 在模型执行前记录用途，并通过本地跨进程锁管理并发；失败也保留使用记录。移动产物时复制整个目录。框架无法证明人工声明真实，也无法追踪框架之外的选型过程。

默认禁止覆盖已有输出；模型、数据、历史报告等受保护路径不可作为输出。原始业务数据、正式模型、本地配置和运行结果不进入 Git。旧 joblib 可以通过 `configs/upload_legacy.example.json` 的本地副本引用；只加载自己生成或可信来源的模型文件。

## 验证与下一步

0.2 验收结果：**361 项测试通过，3 项符号链接测试因 Windows 权限限制跳过；真实 wheel 安装的 13 个步骤全部通过**。对应源码、测试、配置和依赖指纹见下方报告。原项目兼容验证包含 168 项测试基线、3,009 条历史输入完整输出一致、两种后端的合成训练对照，以及 173 个原资产文件的哈希核对。历史回归一致性不等于独立泛化准确率。

- [工程验收结果](docs/engineering-verification.json) · [原项目兼容验证](docs/verification.json)
- [0.2 可靠性改进与验证边界](docs/reliability.md) · [正式安装包验收](docs/wheel-verification.json)
- [运行与新任务接入手册](docs/generic_training.md)
- [底层训练逻辑](docs/logic-design.md)
- [实施计划与完成状态](docs/implementation-plan.md)
- [合成示例说明](examples/README.md)

先运行三分类示例，再用自己的任务替换输入、标签与评测。确定实际需求后，再选择是否接入语言或视觉训练后端。
