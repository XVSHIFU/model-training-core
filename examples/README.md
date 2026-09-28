# 公开合成示例

本目录全部是人为编写的合成记录，不含业务流量、客户数据或生产模型。用途是验证数据检查、真实拟合、保存/加载与评测工程，不代表真实业务精度。

| 文件 | 内容与用途 |
|---|---|
| `text_train.jsonl` | 24 条英文主题分类训练记录；food、travel、technology 各 8 条 |
| `text_dev.jsonl` | 6 条开发记录；每类 2 条 |
| `text_independent.jsonl` | 6 条单独保留的测试夹具；每类 2 条，仅验证独立性检查流程 |
| `upload_train.jsonl` | 49 条纯合成上传记录；成功/失败各 24 条，unknown 1 条 |

文本输入仅包含 `inputs.text` 和空的 `inputs.features`，类别、来源、编号及分组不作为模型特征。三个文本分区之间没有 ID、精确内容、模板或分组交叉。测试夹具在训练前保存；查看其结果并据此调试后，应把它视为已使用的工程回归资产。

上传示例的请求、响应和标签与 `scripts/capture_baseline.py` 中的 `upload_fixture()` 一致，另补充合成来源、分组及标注状态。unknown 是合法语义标签，在当前上传二分类训练时明确排除。该小型集合只用于训练流程烟测，不是独立测试。

## 配置

- `configs/text_demo.json`：非上传三分类，训练集与开发集，Logistic Regression，seed 42。
- `configs/text_independent.json`：登记同一训练/开发数据以及单独保留的合成测试夹具。
- `configs/upload_train.json`：纯合成上传训练，沿用上传阈值；新候选写入 `runs/`。
- `configs/upload_legacy.example.json`：仅用于读取本地已有上传模型及历史回归数据的模板。

配置中的相对路径以配置文件所在目录为基准，运行结果写入仓库的 `runs/`。从仓库根目录可以开始：

```text
uv sync --extra dev
uv run model-workflow validate-data --config configs/text_demo.json
uv run model-workflow train --config configs/text_demo.json
uv run model-workflow validate-data --config configs/text_independent.json
uv run model-workflow validate-data --config configs/upload_train.json
uv run model-workflow train --config configs/upload_train.json
```

把下面的 `<TEXT_TRAIN_RUN>` 替换为文本训练命令返回的 `artifact` 路径：

```text
uv run model-workflow predict --artifact "<TEXT_TRAIN_RUN>" --input examples/text_dev.jsonl
uv run model-workflow evaluate --artifact "<TEXT_TRAIN_RUN>" --data examples/text_independent.jsonl --purpose independent_test --independent --provenance "synthetic held-out engineering fixture"
```

这里的独立资格仅说明合成夹具通过当前用途和成员隔离检查，不是对真实业务泛化的认证。完整入口参数见仓库主 README 和 `uv run model-workflow --help`。

## 本地旧模型模板

`upload_legacy.example.json` 不附带生产模型或真实数据，也不能直接用于训练。需要使用时：

1. 在被 Git 忽略的 `local/` 中准备可信旧模型 `champion_v4.joblib` 和 `upload_regression.jsonl`；或把配置复制为被忽略的 `configs/upload_legacy.local.json`，填写自己的本地路径。
2. 回归JSONL使用上传任务的 `event_id`、`request`、`response` 结构；`label` 或 `gold_verdict` 提供四类语义标签。模板默认 `evaluation_mode=semantic`；若数据只有平台二类标签，改用 `tdp_v2` 评测口径。
3. 以只读评测方式使用该配置。它把数据声明为 `historical_regression`，保护 `local/` 不被新输出覆盖，结果写入新的 `runs/` 目录。

旧模型缺少可携带的训练指纹清单时，可以做历史回归，但不能据此声明完成独立泛化验收。不要将实际模型、流量或私有路径填入这个公开模板后提交。

本地文件准备好后，使用旧模型评测的命令是：

```text
uv run model-workflow evaluate --config configs/upload_legacy.example.json --data local/upload_regression.jsonl --purpose historical_regression --provenance "local historical regression" --mode semantic
```

`--config` 和 `--artifact` 互斥；前者在评测命令中只读加载已有模型，后者加载新框架生成的完整制品。
