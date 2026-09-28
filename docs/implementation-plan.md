# 通用训练框架实施计划与状态

日期：2026-09-28。状态：P0–P5 已完成，传统 ML 训练框架完成工程迁移验收；不代表新场景泛化效果已获验证。

本页保留首版迁移计划与验收结果。后续 0.2 针对安装、数据身份和中断/并发补强，见[可靠性改进](reliability.md)；当前验证以[工程报告](engineering-verification.json)和[安装报告](wheel-verification.json)为准。

依据：[底层逻辑](logic-design.md)。实现位于独立新仓库，原交付目录只读。原模型、真实流量和历史报告未进入公开仓库；examples 仅含合成工程夹具。

## 1. 交付范围

已经落地：任务配置 → 数据检查与目标转换 → 实际训练 → 保存/加载 → 预测 → 评测和制品记录。

- 公共核心管理任务、数据用途与制品，不导入上传业务模块。
- 稀疏后端支持 Logistic Regression/LightGBM，文本与外部数值特征，多类别目标。
- 上传适配器保留原输入、24 维特征、标签映射、规则短路与四类语义。
- 新 model-workflow 和旧 upload-judge 同时存在；禁止覆盖受保护资产是明确的兼容行为收紧。
- 语言、视觉仅有可扩展协议，没有对应训练后端。

## 2. 实际结构

```text
src/
  training_core/             # contracts/data/splits/artifacts/runner/cli
  training_backends/
    sparse_classifier.py    # 独立的稀疏分类拟合与持久化
  training_tasks/
    registry.py             # 任务选择与组合入口
    text_classification.py  # 非上传分类示例
    upload/                 # adapter/representation/fingerprints/policy/evaluation
  upload_judge/             # 旧公共接口、规则、特征与模型格式兼容
configs/                    # 合成示例配置与本地旧模型模板
examples/                   # 公开合成数据
docs/                       # 设计、手册与验证证据
runs/                       # 每次新建的运行目录，Git 忽略
local/                      # 外部资产与保护配置，Git 忽略
```

代码入口见[核心](../src/training_core/)、[后端](../src/training_backends/)、[任务](../src/training_tasks/)。保留的[旧数据构建](../pipelines/build_datasets.py)和[旧合并脚本](../pipelines/build_train_v4.py)部分默认路径依赖外部历史资产，不是公开示例的一键入口。

## 3. 阶段状态与验收

| 阶段 | 状态 | 交付与证据 |
|---|---|---|
| P0 基线 | 已完成 | 本地旧代码/环境基线、3,009 条输出、173 个资产 hash；原项目测试基线 168 项通过 |
| P1 公共约定 | 已实现并完成阶段测试 | [contracts](../src/training_core/contracts.py)、[artifacts](../src/training_core/artifacts.py)；非分类输出契约、路径保护 |
| P2 数据与真实训练 | 已实现并完成阶段测试 | [data](../src/training_core/data.py)、[splits](../src/training_core/splits.py)、[稀疏后端](../src/training_backends/sparse_classifier.py)；多分类真实拟合、保存加载、目标与维度校验 |
| P3 上传接回 | 专项验证完成 | [上传适配器](../src/training_tasks/upload/adapter.py)；3,009 条完整输出差异 0；两后端小型训练对照一致 |
| P4 运行闭环 | 已实现并运行示例 | [runner](../src/training_core/runner.py)、[CLI](../src/training_core/cli.py)、[配置](../configs/)；24 条文本训练、6 条开发完成实际流程 |
| P5 工程交付 | 首版已完成 | 首版 276 项测试通过，失败/错误/跳过均为 0；已安装命令入口退出码 0；文档链接有效，173 个原资产最终 hash 复核无变化；后续结果见当前报告 |

专项数字见[verification.json](verification.json)。原项目测试基线在依赖版本一致的新解释器中只读运行，不修改原环境。详细真实数据和模型保留在本机受保护区域，公开文件仅记录计数与一致性结果。

### P2：不能只包装旧类

共享后端只接收文本、外部数值特征和离散类别，不认识 UploadEvent 或成功/失败含义。上传类和非上传文本任务都调用同一个后端。预测只转换已拟合的表示；外部特征宽度与有限数值均有检查。

上传 24 维含义和顺序保留在上传代码。通用数据层保留原目标，转换后的目标写入样本元数据，并统计排除原因。标签覆盖优先级与训练权重分开。

### P3：兼容范围

同一旧模型的事件顺序、结论、证据、原因和对外分数保持一致；规则命中仍不运行模型。旧 joblib 七个顶层字段、公共导入路径和 CLI 参数继续兼容。

保存行为有意收紧：受保护路径及已有输出报错。旧训练 CLI 在拟合前检查模型和 manifest；不能借旧命令覆盖 champion。新算法、特征和阈值优化不属于本轮。

### P4：可执行入口

在仓库根目录执行，完整说明见[手册](generic_training.md)：

```powershell
uv sync --locked --extra dev
uv run --locked model-workflow validate-data --config configs/text_demo.json
$trainResult = uv run --locked model-workflow train --config configs/text_demo.json | ConvertFrom-Json
$artifactPath = $trainResult.artifact
uv run --locked model-workflow predict --artifact $artifactPath --input examples/text_dev.jsonl
uv run --locked model-workflow evaluate --artifact $artifactPath --data examples/text_dev.jsonl --purpose development --provenance "synthetic development fixture" --mode classification
```

命令已经实现。合成示例用于工程验证，不代表业务准确率。

## 4. 数据用途与独立性

| 用途 | 规则 |
|---|---|
| train | 拟合表示与模型；当前每次配置恰好一个训练分区，多来源先显式合并 |
| development | 选型和调整方案；所有参与选择的数据都应登记在训练配置 |
| historical_regression | 可与训练/开发交叉，如实报告，用于兼容与回归 |
| independent_test | 来源和独立声明，加上训练/开发 lineage 及已记录评测使用历史检查 |

指纹无交叉和 independent 声明不能代替真实采集与用途管理。看过测试结果并用于调整方案后，该集合对后续候选应改为开发或回归用途。

新制品在 evaluation_history 追加评测成员记录，不修改冻结模型与 manifest。重复评测同一冻结模型允许用于复验，不算新数据。旧模型无训练 lineage 时不能获独立资格。软件不能自动发现仓库外的人为选型；完整目录和真实用途声明仍然必要。

## 5. 工程完成条件

- [x] 原正式模型 3,009 条完整业务输出一致，规则短路保留。
- [x] 专项验证中登记的 173 个原资产 hash 未变化。
- [x] 非上传三分类通过共享后端实际拟合、保存、加载、预测和评测。
- [x] 公共核心不导入上传代码，不强制概率、四类标签或证据。
- [x] 目标状态、排除、非法标签及冲突有处理与测试。
- [x] 分区指纹与 lineage 检查已实现，历史回归不冒充独立测试。
- [x] 新旧保存入口有路径保护，旧模型格式继续可加载。
- [x] 文档区分实际支持、协议扩展和未实现能力。
- [x] 动态评测历史检查纳入最后一次整合测试。
- [x] 全套测试、安装入口和文档链接最终复验完成并记录。
- [x] 最终代码状态下再次确认 173 个原资产 hash 无变化，完成 P5 收口。

首版最终整合检查为 276 项通过。后续改动使用[复验脚本](../scripts/verify_engineering.py)重新检查，以当次实际结果为准；操作见[使用手册](generic_training.md)。

## 6. 回滚与未纳入事项

新运行全部写入独立 runs 目录，原交付仍可独立使用。本轮没有自动发布、部署、champion 晋级或替换步骤。出现上传行为差异应停止整合，定位语义、特征或编排差异，不顺带调整算法。

未实现语言生成/视觉训练、分布式训练、通用自动调参、自动校准与大规模流式管线。新后端须实现任务协议并实际验证后，才能宣称支持。
