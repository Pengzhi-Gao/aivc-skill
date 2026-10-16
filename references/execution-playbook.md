# AIVC 执行手册

## 四阶段

1. 第一次回复按 `SKILL.md` 的启动模板核对六项：分析数据、分析目录、运行环境、KO/OE/药筛任务、模型、目标基因或药物库；把缺项合并提问，已明确的内容沿用。只说“测试一下”也需要补齐，不能自动采用历史示例。
2. 六项齐全后回显有效设置并检查 `input-contracts.md`；模型特有的必要输入仍缺少时继续补问。不匹配时请用户补数据或重选，未补齐前仅检查文件和环境，不启动推理或训练，不自动切换模型、数据或目标。输入完整后直接执行，无需额外的通用启动确认。
3. 按 [模型资源与执行入口](model-setup.md) 获取或复用所选模型资源，转换输入，运行每个目标/药物条件并落盘原始结果。
4. 按 `downstream-results.md` 选择适用的默认图组，生成表格、图、质量记录和 report.md。

## 运行环境

从 [模型资源与执行入口](model-setup.md) 读取当前后端的官方资源清单及调用步骤。执行者完成下载、版本匹配和路径配置；用户已有数据/资源直接复用。药筛历史脚本在分析目录中复制并适配，所需修改点见该文档的“复用药筛示例的配置步骤”。

各后端在用户指定的环境范围内选择兼容解释器，按 `backend-environments.md` 核对依赖，不要在主环境里跑 CellOracle。该文档的历史本机路径只作为候选。用户只给环境根目录时先检查其解释器及子环境，唯一匹配可直接记录，多个匹配则补问；需要新建时按用户指定安装位置及已有授权执行。启动前核实实际解释器路径和后端版本，并写进 `manifest`。分析目录、输入目录和环境目录分别处理。

KO/OE 分别使用 `examples/ko_oe/celloracle_run.py` 或 `examples/ko_oe/geneformer_route.py`（参数见同目录 `README.md`），两条路线独立执行。CellOracle 需要用户的 scRNA、匹配的 GRN 先验及目标基因；Geneformer 需要匹配版本的原始计数、tokenizer 和权重。脚本提供可重复调用的参数化入口，只有实际在用户数据和模型环境中运行并检查产物后才登记为完成。GEARS 仅在匹配扰动训练数据存在且用户选择时运行。

## 运行记录

每次运行在用户指定的分析目录内保存，使用唯一运行 ID 保存旧结果：

```text
<analysis_dir>/results/<run_id>/
  manifest.json
  provenance.json
  raw/
  tables/
  figures/               # 包含 figure_manifest.csv
  validation.json
  errors.jsonl
  report.md
```

`manifest.json` 记录六项启动设置：`data_selection`、`analysis_dir`、`environment`、`task`、`backend`、`intervention`，包括用户已选内容及本次补充的选择依据。另记实际解释器及版本、数据路径/用途/哈希、模型版本、权重/先验路径、目标基因与逐个/联合方式，或药物库来源/版本/哈希及筛选范围，以及运行参数。固定背景模型明确记录实际自带基线；用户疾病参照单列。适配脚本未自动写出的启动字段由执行者补入运行记录，保留原有模型元数据。

`validation.json` 分开记录输入检查、预测状态与独立验证状态。Geneformer 的 `delete`/`overexpress` 保存 embedding 影响及统计；CellOracle 导入原始计数后先做 log1p 变换，再经 KNN 平滑得到内部 `imputed_count`。模拟表达及 `delta_X` 按这一内部尺度标注。

批量药筛以 `(drug, dose, time, cell_type)` 为最小运行单元。已完成条件写入 checkpoint，失败条件写入 errors，已完成结果保留。

## 结果展示执行

1. 检查实际保存的输出是表达、网络、状态还是排名，并核对基线、单位、条件和覆盖范围。
2. 根据 `downstream-results.md` 的任务图组生成适用的主图和配套视图；有疾病参照、剂量、目标状态或实测数据时再启用相应扩展。
3. 每张图先保存源数值表和绘图参数，再绘图；使用保存的源数值表绘图。图形失败只重画相关图，不重跑成功的模型条件。
4. 检查实际导出图的文字、轴、色标、图例、对照和源数据，登记 `figures/figure_manifest.csv`。
5. 缺少图形所需输入时登记 `skipped` 和原因；绘制失败登记 `failed`。二者均与模型预测的运行状态分开。

只读取或整理已有运行时，使用现有产物补图，不自动启动新的模型、疾病差异分析或实验。富集等下游计算按已授权范围和实际数据执行。

## 报告内容

`report.md` 包含：

- 任务和后端
- 输入文件、预处理和参数
- 完成/跳过/失败条件
- 结果文件路径
- 按问题组织的结果图及各自源表，说明变量、单位和直接数值现象
- 图形清单，以及未生成的默认图及其原因
- 输出类型和单位
- 覆盖率、OOD、重复性或其他技术检查
- 未独立验证项目
- 未评估项目和原因

结果解读遵循 `SKILL.md` 的报告范围，解释指标定义与直接数值现象。

历史示例报告保留原有完成状态；新增图组执行并检查导出文件后登记完成。结果描述、质量检查和实测验证分别呈现。

## 失败状态

- `no_prediction`: 后端不覆盖目标
- `failed`: 运行报错
- `not_applicable`: 数据或任务不适用
- `partial`: 部分条件完成
- `validation_status: unvalidated`: 独立于预测状态记录

保留可运行条件的原始结果。输入无法解释、后端无输出、非有限结果或关键字段缺失时阻断对应条件；部分失败记 `partial`。响应变化、评分区分度和独立验证分开记录，判据见 `downstream-results.md`。

复用示例代码时，药筛先调用 `RunDir.save_predictions` 保存全量数据，再调用 `condition_quality` 检查；用 `top_response_table` 按候选 ID 提取热图源表。`RunDir.finish` 从原始响应与逐条件分数生成恢复/未恢复药物清单、`drug_recovery_summary.csv`、`drug_recovery_by_condition.csv`、`drug_top30_genes.csv` 和 `drug_result_rules.json`，覆盖全部药物与条件，而不是只展示排名前 20。结果页先显示两类药物清单及其他状态，再按药物及剂量查看 Top 30 基因；无效或未评分条件保留“无法判定”的原因。运行目录使用唯一 ID，禁止覆盖历史运行。绘图扩展仍按实际执行登记，以导出文件及检查记录为完成依据。
