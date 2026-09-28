# 换电脑与新会话接手说明

更新：2026-09-28。本文件保存项目方向和状态，不需要原聊天记录才能继续。

## 用户当前决定

- 保留现有架构，继续做可复用的底层训练工程，不再以 OWASP / SOC 专项为目标。
- 预算低，主机内存最多 32GB；优先 CPU 和很小的模型，不以购买显卡为前提。
- 希望逐个训练任务小模型，将经过验证的反馈用于改进共享底座，再验证后续任务是否受益。
- 先做小规模可解释实验；没有指定语言或视觉业务场景。
- 工程框架与可训练的底座权重是两个概念，必须分别记录版本和证据。

## 已经完成

实际支持 TF-IDF + 可选数值特征 + Logistic / LightGBM；包括通用工作流、文本分类适配器、上传兼容适配器、数据用途检查、制品完整性和运行追溯。源码、测试、示例、配置、依赖锁和验证报告均在 Git 中。

历史验收：361 项测试通过、3 项 Windows 符号链接测试跳过，wheel 安装验证 13 步通过。证据见 [工程验收](engineering-verification.json)与 [wheel 验收](wheel-verification.json)。这是对应代码和环境的历史结果，不代表新电脑已经验证通过。

语言/视觉神经网络后端、共享编码器、反馈训练循环尚未实现。小模型计划中的性能、内存和改进效果尚未实测。

## 新电脑开始

先安装 Git 和 uv，然后在 PowerShell 中执行：

```powershell
git clone https://github.com/XVSHIFU/model-training-core.git
cd model-training-core
uv sync --locked --extra dev --python 3.11
uv run --locked model-workflow validate-data --config configs/text_demo.json
$training = uv run --locked model-workflow train --config configs/text_demo.json | ConvertFrom-Json
uv run --locked model-workflow predict --artifact $training.artifact --input examples/text_dev.jsonl
uv run --locked pytest -q
```

这些命令使用公开合成示例，不依赖旧电脑。训练产物写入 runs/，它不进入 Git。其他 shell 需调整变量和 JSON 解析语法。

## 下一步执行顺序

1. 阅读 [CPU 小模型反馈循环计划](small-model-feedback-plan.md)，这是最新优先方向；前一版调研中的 GPU 预算只是参考，不是采购要求。
2. 先运行现有示例和测试，确认新机器环境，再做可选实验目录，不替换现有 SparseClassifier。
3. 实现小型共享编码器与任务头，按计划保存 B0、候选 B1 和父子血缘；使用教学数据先测资源。
4. 做等预算对照、多个种子、旧任务退化检查和保留任务评测；失败或无提升也如实报告。
5. 验证有效的接口再接入现有 TaskAdapter。新增深度学习依赖保持可选，明确锁定实验环境；基础 CPU 分类安装不应被迫依赖它们。

只把有真实标签或其他可核验依据的反馈用于训练；不把模型自己的输出自动当真值。测试集不能参与反复调参。同一图像的不同任务标签或变体必须继承同一分区。

## GitHub 能恢复什么

仓库足以恢复代码、公开教学示例、测试、依赖版本和计划，并从头生成新的示例模型。

以下内容**未上传，克隆仓库无法恢复**：原文件上传交付项目、真实业务数据和正式模型、local/、runs/、本地配置、虚拟环境及构建缓存。旧制品若需保留，应通过合适的私有备份渠道复制完整制品目录（含 manifest、lineage、evaluation_history），不能只复制单个模型文件。不要为了迁移而强制加入公开仓库。

原始交付并非继续小模型实验的前置条件；但重做历史上传模型一致性验收需要另行提供原资产。依赖原机器绝对路径的本地配置也需要重新配置。

## 给新会话的接手提示

> 请先阅读 README.md、docs/handoff.md 和 docs/small-model-feedback-plan.md。保留当前架构，按 CPU、最多 32GB 主机内存的预算实现小模型反馈循环实验。先验证环境和现有测试，再新增可选实验；不要把计划当成已实现能力。区分工程框架与共享底座权重，记录数据分区、父子制品、等预算对照、旧任务退化和资源开销。无提升也如实报告，不自动晋级候选底座。
