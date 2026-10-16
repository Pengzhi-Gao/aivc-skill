# KO/OE：两条独立示例路线

开始分析时，按 [`SKILL.md`](../../SKILL.md) 的启动模板核对数据、分析目录、运行环境、KO/OE 任务、模型和目标基因，合并询问缺项。已有选择直接沿用，补齐后运行。

CellOracle 与 Geneformer 分别读取用户指定的输入、独立执行。模型与数据不匹配时说明缺口，请用户补数据或重选模型。

代码、checkpoint、词典和 base GRN 按 [模型资源与执行入口](../../references/model-setup.md) 获取及匹配，由执行者配置路径。用户选定 GEARS 时读取 [GEARS 数据、训练/加载与调用示例](../../references/gears-workflow.md)。

| 路线 | 用户数据与模型资源 | KO/OE 语义 | 原生结果 |
| --- | --- | --- | --- |
| CellOracle | 原始单细胞计数 `.h5ad`、匹配物种与基因符号的 base GRN `.parquet` | 目标 TF 设为 0 / 设置在 imputed 表达范围内的固定高值 | `simulated_count`、`delta_X`、拟合 GRN、UMAP 状态位移及随机化投影对照 |
| Geneformer | 人类完整转录组原始计数 `.h5ad`，`.var['ensembl_id']`、`.obs['n_counts']`，匹配 V1/V2 的本地 checkpoint | 删除目标 token / 把目标 token 移到排序编码前端 | 官方 ISP 余弦统计与表征影响 |

## CellOracle

在单独的 CellOracle 环境中运行 [`celloracle_run.py`](celloracle_run.py)。以下命令中的路径、目标基因和参数均须按用户选定的数据填写：

```powershell
python .\examples\ko_oe\celloracle_run.py --h5ad 'D:\data\cells.h5ad' --base-grn 'D:\data\base_grn.parquet' --target-gene STAT3 --mode ko --output-dir 'D:\results\celloracle_STAT3_ko'
python .\examples\ko_oe\celloracle_run.py --h5ad 'D:\data\cells.h5ad' --base-grn 'D:\data\base_grn.parquet' --target-gene STAT3 --mode oe --oe-percentile 95 --output-dir 'D:\results\celloracle_STAT3_oe'
```

`--counts-layer` 可指定原始计数层；`--group-col` 只用于分组描述，`--grn-unit-col` 指定按哪个细胞分组拟合网络。后者未给出时由 base GRN 先验拟合全体细胞共用的 GRN。OE 可用 `--oe-value` 明确设置内部 `imputed_count` 尺度的固定值（原始计数经 log1p 与 KNN 平滑），或用 `--oe-percentile` 从该尺度的数据中选择；基线高于设定值的细胞会被下调，需单独报告。数据使用用户选定的文件，base GRN 复用已有先验或由执行者下载后传入。

输出目录必须是新的或空目录。`raw/` 保存模型的全量基线、模拟表达、`delta_X` 和实际用于模拟的 GRN 系数矩阵；`tables/` 保存基因效应、状态位移及系数矩阵索引。使用 `--grn-unit-col` 时还保存筛选后的分群 Links 边。`figures/` 保存图和生成/跳过清单；根目录有 `manifest.json`、`validation.json` 和 `report.md`。随机化 `delta_X` 的投影作为技术对照一并保存。

## Geneformer

在已安装[官方 Geneformer](https://huggingface.co/ctheodoris/Geneformer)和相应权重、CUDA 可用的独立环境中运行 [`geneformer_route.py`](geneformer_route.py)：

```powershell
python .\examples\ko_oe\geneformer_route.py --input-h5ad 'D:\data\human_raw_counts.h5ad' --model-dir 'D:\models\Geneformer-V2' --model-version V2 --target-ensembl ENSG00000168610 --perturbation both --confirm-raw-full-transcriptome --output-dir 'D:\results\geneformer_STAT3'
```

`--confirm-raw-full-transcriptome` 表示用户确认输入未经 HVG 选择、未归一化或 log 转换。脚本抽样检查整数原始计数及 `n_counts`，转录组完整性由用户依据数据处理记录确认。V1/V2 的 tokenizer、词典和 checkpoint 必须匹配；目标使用无版本号的人类 Ensembl ID。当前官方 `InSilicoPerturber` 只执行 `delete`、`overexpress`；`inhibit`、`activate` 虽见于参数说明，实际会被拒绝。

脚本在输出目录保存 `tokenized/` 下的 `.dataset`、`raw/KO|OE` 下的官方原始 pickle、`tables/KO|OE` 下的统计 CSV 和余弦影响表、`figures/KO|OE` 下的分布图，以及根目录的 `manifest.json`、`provenance.json`、`validation.json`、`errors.jsonl`、`figures/figure_manifest.csv` 和 `report.md`。KO 与 OE 独立记录；一个条件失败时，成功条件保留，整体标为 `partial`。V1 的 `cell` 模式汇总对齐基因 embedding 的余弦值，V2 的 `cls` 模式比较 CLS token 表征，按版本分别展示。表中 `observation_index` 标识结果行，原始细胞 ID 对应关系未提供。该路线交付 embedding 统计，表达图组需要另有表达预测输出。

运行验收包括：用户选定的数据、匹配先验/权重、实际执行记录及产物检查。完成情况与科学验证分别登记，缺少独立实验验证时标记 `unvalidated`。
