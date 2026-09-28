# 通用训练流程使用手册

当前支持文本分类和上传结果研判，共享 TF-IDF + 外部结构特征的 Logistic Regression / LightGBM 后端。语言生成、视觉、分布式训练尚未实现；协议可扩展不等于对应能力已经可用。

## 1. 安装与最小运行

Python 要求和命令入口见 [pyproject.toml](../pyproject.toml)。在仓库根目录执行：

```powershell
uv sync --locked --extra dev
uv run --locked model-workflow --help
```

公开文本实例有 24 条训练、6 条开发记录，类别为 food、travel、technology，全部为合成数据。四个入口的完整示例：

```powershell
uv run --locked model-workflow validate-data --config configs/text_demo.json
$trainResult = uv run --locked model-workflow train --config configs/text_demo.json | ConvertFrom-Json
$artifactPath = $trainResult.artifact
uv run --locked model-workflow predict --artifact $artifactPath --input examples/text_dev.jsonl
uv run --locked model-workflow evaluate --artifact $artifactPath --data examples/text_dev.jsonl --purpose development --provenance "synthetic development fixture" --mode classification
```

这是开发评测。合成示例高分只能说明工程流程可运行，不能代替业务泛化验证。数据说明见 [examples](../examples/README.md)。

每个 model-workflow 命令新建一个运行目录，并返回含 `run_dir` 的 JSON。训练还返回 `artifact`，预测返回 `predictions` 路径。进入运行目录后失败会记录在 `run.json`；配置解析或建目录前的保护错误可能直接退出，不产生运行目录。

## 2. 配置、路径和命令参数

[text_demo.json](../configs/text_demo.json) 是完整可运行配置：

| 字段 | 含义 |
|---|---|
| task_id | 当前为 text_classification 或 upload |
| backend | logistic 或 lightgbm |
| seed | 随机种子；完整环境记录仍然必要 |
| datasets | 分区名称对应的数据路径、purpose 和 provenance |
| parameters | 由任务解释的参数，未知任务参数报错 |
| run_root | 新运行目录的父目录 |
| protected_paths | 本次配置额外禁止写入的路径 |
| artifact_path | 旧模型配置直接加载外部模型的路径 |
| evaluation_mode | 文本用 classification；上传使用 semantic 或 TDP 口径 |

配置中的相对路径以**配置文件所在目录**为基准。CLI 的 `--input`、`--data`、`--run-root` 等相对路径以执行命令的当前目录为基准。

文本任务允许 `parameters.vectorizer`、`parameters.classifier` 设置后端参数。上传任务目前只开放 `parameters.thresholds`，保留原训练设置。切换其他任务不应靠修改上传标签或 24 维特征来实现。

| 命令 | 必填参数 | 其他参数 |
|---|---|---|
| validate-data | --config | 无 |
| train | --config | 无；恰好一个训练分区 |
| predict | --artifact 或 --config 二选一，--input | --run-root |
| evaluate | --artifact 或 --config 二选一，--data、--purpose | --run-root、--provenance、--independent、--mode |

评测 purpose 可选 development、historical_regression、independent_test，不能填 train。评测/预测使用 `--config` 时必须有 artifact_path；新框架制品优先用 `--artifact`。

## 3. 样本、目标和标注状态

读取器支持 JSON 对象、数组、JSONL/NDJSON。任务负责输入 schema；公共层要求样本 ID 非空且在同一文件中唯一。

文本样本：

```json
{
  "sample_id": "example-001",
  "inputs": {"text": "Book a train ticket for tomorrow", "features": []},
  "target": "travel",
  "source": "synthetic",
  "group_id": "example-group-001",
  "label_status": "labeled"
}
```

文本类别为非空字符串，结构特征为等宽、有限的数值列表。训练使用了结构特征，预测必须提供相同宽度；未使用时保持空列表。ID、source、group_id 和目标不会自动进入模型特征。

上传使用原 event_id、request、response，标签来自 label 或 gold_verdict。它仍判断文件接收/保存结果，不能把上传成功解释为代码执行成功。

| label_status | 当前监督训练处理 |
|---|---|
| labeled | 合法目标可以映射后参与拟合 |
| unlabeled | 排除，记录未标注数量 |
| unresolved | 排除，保留待确认状态 |
| conflict | 排除，保留冲突状态 |

目标缺失也被排除。上传 unknown 是合法目标值，不是标注状态；该二分类任务明确排除它。固定映射任务中的非法标签报错，不会因为标为 unresolved 而被悄悄略过。原目标保留，转换后的目标存放在准备后样本的 metadata.training_target。

当前内置训练是监督分类。协议允许目标暂缺，不等于已经实现无监督或自监督训练。

## 4. 数据的四种用途

| purpose | 用途与限制 |
|---|---|
| train | 拟合词表、表示和模型；一次配置恰好一个训练分区 |
| development | 选模型、调参数或策略，必须与训练隔离 |
| historical_regression | 检查兼容和退步，可与历史训练数据交叉，报告必须如实说明 |
| independent_test | 检验冻结方案，需要来源记录、独立声明及数据用途/成员隔离依据 |

validate-data 读取配置中的全部分区；train 只读取训练和开发。多来源训练数据应先显式合并；公共层提供 merge_samples，没有单独的合并 CLI。

检查维度包括 ID、精确内容、任务模板和 group。空指纹不被当成公共分组。不能只换 ID 避开内容交叉；也不能把“没有指纹交叉”当作真实独立性的证明。

**所有参与选型的数据应在训练配置中登记为 development。** 训练后另外查看数据并据此调参，应保留该使用记录，并纳入下一版 development。仓库外的人为选型不能被软件自动发现。

只有实际独立且有来源依据的数据，才使用下面形式；占位内容需替换为真实路径与记录：

```text
uv run --locked model-workflow evaluate --artifact <完整训练运行目录> --data <独立评测数据> --purpose independent_test --independent --provenance "采集批次、冻结时间与独立性说明"
```

--independent 是调用者声明，不是软件认证。公开 text_independent.jsonl 只用于演练检查；查看并用于开发后，也应视作已消费的工程资产。测试失败后据此改模型，再把相同集合报成新独立测试是不正确的。

## 5. manifest、lineage 和评测历史

训练运行通常包含：

```text
runs/<run_id>/
  config.json               # 已解析配置
  run.json                  # command、状态和摘要
  environment.json          # Python 与依赖版本
  data_summary.json         # 使用/排除数量与原因
  split_report.json         # 分区交叉与用途检查
  input_hashes.json          # 实际训练/开发文件 hash
  model                     # 后端保存的模型文件或目录
  lineage.json              # 训练/开发成员哈希键
  development_metrics.json  # 开发指标
  manifest.json             # 制品路径、文件 hash、源码指纹等
  evaluation_history/       # 后续评测追加的使用记录
```

预测和评测仍分别新建输出运行目录。预测有 predictions.jsonl；评测还有 evaluation_spec.json、lineage_report.json、metrics.json。不要覆盖旧运行来刷新结果。

manifest 关联模型、配置、lineage 和相关记录，并校验登记文件的完整性。lineage 保存经过哈希处理的 ID、内容、模板和分组成员键，使制品迁移后仍可查交叉；它不是完整数据备份，也不能证明来源真实性。

后续评测在制品的 evaluation_history 追加使用记录，**不修改冻结的模型和 manifest**。再次申请独立资格时，除训练 lineage 外，还检查记录中的 development/historical 使用数据。同一冻结模型重复运行 independent 可以复验，但不是新增的独立数据。外部旧 joblib 没有训练 lineage，仍不能取得独立资格。

**迁移模型要复制整个训练 run 目录，包括 evaluation_history。** 使用同版本代码和依赖，将新目录或其中的 manifest.json 传给 --artifact。只复制 model 会丢失配置和 lineage；删除或漏带 history 会丢失后续使用审计。应使用可信、完整的制品，不能把缺失记录解释为未使用。

制品内旧训练路径用于追溯。加载完整制品后，预测不要求原机器训练文件存在，新输出默认写当前仓库 runs，也可用 --run-root 指定。配置可能含私有路径，分享前应审查；不要修改冻结文件后继续沿用旧 hash。

joblib 只加载可信来源。hash 能检测变化，不能让不可信序列化文件变安全。

## 6. 上传兼容和原资产保护

[upload_train.json](../configs/upload_train.json) 使用公开合成上传数据，会创建新候选，不覆盖原 champion：

```powershell
uv run --locked model-workflow validate-data --config configs/upload_train.json
uv run --locked model-workflow train --config configs/upload_train.json
```

需要原模型时，参考 [upload_legacy.example.json](../configs/upload_legacy.example.json)，在本地忽略目录准备可信旧模型与回归文件，或复制为 .local.json 后填写自己的外部只读路径。公开仓库不附带这些资产。

```powershell
uv run --locked model-workflow evaluate --config configs/upload_legacy.example.json --data local/upload_regression.jsonl --purpose historical_regression --provenance "local historical regression" --mode semantic
```

该命令要求模板指定的本地文件已准备好。平台二类标签和四类语义真值含义不同，按数据选择 tdp_v2 或 semantic。没有训练 lineage 的旧模型可以回归，不能靠重新命名数据获得独立资格。

上传规则顺序、旧 joblib 格式和 upload-judge 参数继续兼容。新旧保存入口都拒绝覆盖已有文件。默认保护仓库 models、data、reports、.git、.venv；原交付目录可登记到被忽略的 local/protected.json 的 paths 数组，或通过 MODEL_TRAINING_PROTECTED_PATHS 指定。配置 protected_paths 添加本次保护项。

旧训练 CLI 在拟合前检查模型和 manifest，其他输出也做预检。不要直接运行保留的历史脚本默认流程：它们可能依赖旧目录或自行写报告，并不都经过新运行器。

[专项验证](verification.json)记录 3,009 条完整旧模型输出一致、173 个原资产 hash 不变及两后端的小型训练对照一致。这是工程迁移证据，不是新的独立泛化成绩。

## 7. 新任务最小实现

参考 [TextClassificationTask](../src/training_tasks/text_classification.py)，实现 [TaskAdapter 协议](../src/training_core/contracts.py)：

1. parse(record)：校验输入并返回 Sample，保留目标、状态、来源和分组。
2. fingerprints(sample)：提供内容、模板和必要的分组键，规则应对应真实同源关系。
3. prepare(samples)：明确目标转换与排除，返回可拟合样本和统计，不覆盖原目标。
4. fit(samples, config)：真正执行训练，只在传入训练样本上拟合表示，不自行读取最终测试。
5. predict(model, samples)：返回同数量、同 ID 顺序的 Prediction；适配器校验自己的输出结构。
6. save/load：保存完整模型及预处理，调用路径保护。若使用模型目录，推理必需文件都置于其中。
7. evaluate：定义任务指标和标签有效性；不默认所有任务都有概率、四类或 accuracy。

在[显式注册处](../src/training_tasks/registry.py)增加任务 ID，添加配置与小型工程夹具，验证拟合、保存加载、预测、评测及错误处理。不要让 training_core 导入业务模块。

神经网络任务还需要自己的真实后端、预处理、loss、优化和制品加载。当前 runner 拒绝没有可用监督样本的训练；自监督、无监督或大规模流式任务需要明确调整实现，不能仅注册名称就宣称支持。

## 8. 排查与工程验证

| 现象 | 先检查 |
|---|---|
| 输出路径被拒绝 | 已有文件、输入、制品目录和受保护路径；使用新运行目录 |
| train/development 交叉 | ID、内容、模板与 group；不要只改编号 |
| 独立资格失败 | 来源声明、训练 lineage、评测使用历史及数据交叉 |
| 结构维度不匹配 | 每条宽度和拟合时是否相同 |
| 没有有效训练样本 | 标注状态、目标映射与排除项 |
| 制品完整性错误 | 是否漏复制文件，或修改了 manifest 登记内容 |

完整工程复验使用[verify_engineering.py](../scripts/verify_engineering.py)，仅依赖公开合成夹具，不要求原项目的真实数据和正式模型：

```powershell
uv sync --locked --extra dev
uv run --locked python scripts/verify_engineering.py
```

脚本运行全套测试和已安装的 model-workflow 命令入口，检查执行期间源码指纹未变；为日志、JUnit 结果和摘要新建 runs/engineering-* 目录，并刷新仓库中的[工程验收报告](engineering-verification.json)。刷新该汇总报告是脚本的明确行为，不会覆盖模型运行制品。

本轮验收为 276 项测试通过，失败、错误、跳过均为 0；已安装命令入口退出码为 0。只需快速运行测试时，可执行 `uv run --locked pytest -q`。后续复验的数量和状态以当次执行为准。

工程测试、合成样例和[历史回归](verification.json)都不能替代真实新场景的独立验收。
