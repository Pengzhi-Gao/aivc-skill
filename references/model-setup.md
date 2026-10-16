# 模型资源与执行入口

按用户选定的模型读取对应段落。Skill 提供规则、流程和示例；执行者在运行时获取代码、权重、词典、先验和训练配套文件，并配置到用户指定的环境与分析目录。

## 共同配置规则

1. 沿用六项启动选择。用户已有兼容资源时直接核验并复用；资源缺失时按下表从官方来源获取。版本、物种、细胞背景能由数据和任务确定时直接配置并记录依据；多个资源代表不同研究问题时向用户说明差异后选择。
2. 在 `<analysis_dir>/resources/<backend>/` 保存本次资源清单或下载文件，环境安装在用户指定的位置。可以复用外部缓存，记录其实际路径。示例中的历史目录按本次设置替换。
3. 代码固定到实际使用的 release/commit；权重与配置、基因顺序、tokenizer、编码器和标准化统计量成套匹配。记录来源 URL、版本、许可证、路径与 SHA256。下载得到的 Git LFS 指针或 HTML 错误页不能作为权重使用。
4. 先用本次数据中的少量有效条件检查输入轴、数值尺度、模型加载和输出类型，再按用户选定范围运行。没有匹配 checkpoint 的 GEARS 路线使用扰动数据训练；训练范围和参数写入分析设置。

| 路线 | 获取哪些资源 | 输入、调用及结果指导 |
| --- | --- | --- |
| CellOracle | 官方包、物种对应 base GRN；GRN 在用户表达数据上拟合 | [本页 CellOracle](#celloracle)；[参数化代码](../examples/ko_oe/celloracle_run.py) |
| Geneformer | 官方代码、选定 checkpoint、同版本词典/基因统计 | [本页 Geneformer](#geneformer)；[参数化代码](../examples/ko_oe/geneformer_route.py) |
| GEARS | 官方包、GO 图资源、扰动数据、匹配 checkpoint 或本次训练产物 | [GEARS 完整流程与示例](gears-workflow.md) |
| PRnet | 官方代码、LINCS/SciPlex 权重及配套基因顺序/数据 | [本页 PRnet](#prnet)；[外部基线示例](../examples/lung_cancer/drug_screen/run_prnet.py) |
| chemCPA | 官方代码、checkpoint/config、训练基因与协变量、RDKit 表征 | [本页 chemCPA](#chemcpa)；[推理示例](../examples/lung_cancer/drug_screen/run_chempa.py) |
| UniPert-G2CP | UniPert 编码器、G2CP 代码、指定细胞系数据与权重 | [本页 UniPert-G2CP](#unipert-g2cp)；[推理示例](../examples/lung_cancer/drug_screen/run_unipert_g2cp.py) |
| UniCure | UniCure 权重、细胞与药物表征；新样本另需 UCE 配套资源 | [UniCure 完整流程与示例](unicure-workflow.md) |
| DrugReflector | 官方代码、三个 fold checkpoint | [DrugReflector 完整流程与示例](drugreflector-workflow.md) |

## CellOracle

### 资源获取

来源：[官方仓库](https://github.com/morris-lab/CellOracle)、[base GRN 教程](https://morris-lab.github.io/CellOracle.documentation/tutorials/base_grn.html)、[先验加载源码](https://github.com/morris-lab/CellOracle/blob/master/celloracle/data/load_promoter_base_GRN.py)。安装使用所选版本的依赖；现有 Windows 0.20.0 环境的搭建方法见 [环境记录](../docs/environment-celloracle-windows.md)。

CellOracle 从表达数据拟合 GRN，通常需要下载的是 base GRN 先验。人类 hg19 示例：

```python
from pathlib import Path
import celloracle as co

def acquire_human_grn(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    grn = co.data.load_human_promoter_base_GRN(version="hg19_gimmemotifsv5_fpr2")
    grn.to_parquet(destination, index=False)
    return destination
```

其他物种选择对应 loader，例如 `load_mouse_promoter_base_GRN` 或 `load_mouse_scATAC_atlas_base_GRN`；记录基因组版本。已有匹配 ATAC 数据时按官方教程用 peaks、TSS 注释和 motif 构建样本先验。宽表包含 `peak_id`、`gene_short_name` 和各 TF 列，不能把 TF 列表当成边表读取。

### 输入与调用

1. 把用户数据整理成 cells × genes 的 AnnData；从 `.X` 或指定 layer 读取原始非负整数计数。使用与先验一致的 gene symbol，先解决重复 ID，再筛选目标细胞群。
2. 以 [celloracle_run.py](../examples/ko_oe/celloracle_run.py) 处理 HVG、PCA/UMAP、KNN 平滑和 GRN 拟合。目标 TF 强制保留；检查其出现在表达矩阵、先验 TF 列及拟合 GRN。
3. 按用户选择传 `--mode ko`，或 `--mode oe` 加 `--oe-value` / `--oe-percentile`。完整命令及分群参数见 [KO/OE 示例说明](../examples/ko_oe/README.md#celloracle)。批量单基因逐个调用，分别设置输出目录。

### 结果处理

保存 `simulated_count`、`delta_X`、基线、拟合系数与细胞/基因顺序；以内部 imputed log1p 尺度描述响应。基因影响按平均有符号变化与平均绝对变化分别统计。状态投影同时保留原始与随机化 `delta_X` 的投影；OE 另外报告设置值及基线高于该值的细胞比例。图组按 [下游规范](downstream-results.md) 生成。

## Geneformer

### 资源获取

来源：[官方模型及代码](https://huggingface.co/ctheodoris/Geneformer)、[tokenizer](https://geneformer.readthedocs.io/en/latest/geneformer.tokenizer.html)、[扰动 API](https://geneformer.readthedocs.io/en/latest/geneformer.in_silico_perturber.html)。

先选 V1 或 V2、模型规模及适用域，再固定 Hugging Face revision。V1 与 V2 的序列长度、词典及统计文件不同；仓库根目录的默认模型也可能与用户指定版本不同。检查所选模型卡、`config.json` 和文件树后下载对应子目录，而非把整个仓库根目录直接当成任意版本 checkpoint。

```python
from huggingface_hub import HfApi, snapshot_download

def acquire_geneformer(resource_dir, model_subdir):
    repo = "ctheodoris/Geneformer"
    revision = HfApi().model_info(repo).sha
    files = HfApi().list_repo_files(repo, revision=revision)
    prefix = model_subdir.rstrip("/") + "/" if model_subdir else ""
    weights = [f for f in files if f.startswith(prefix)
               and f.endswith((".safetensors", "pytorch_model.bin"))]
    if not weights:
        raise ValueError("所选模型目录没有权重，请核对官方文件树")
    patterns = ["geneformer/**", "setup.py", "pyproject.toml", "requirements*", "README.md"]
    patterns += [prefix + "*"] if prefix else ["config.json", "*.safetensors", "*.bin", "*.json"]
    path = snapshot_download(repo, revision=revision, local_dir=resource_dir,
                             allow_patterns=patterns)
    return path, revision
```

`model_subdir` 取官方文件树中已选版本的目录名。安装下载的官方 Python 包，并检查该版本 tokenizer 默认的 token dictionary、gene median 与 gene mapping 文件均已下载；文件树中的配套资源不在 `geneformer/` 时加入下载清单。也可用官方 `git lfs` 克隆方式，按本次所选模型限制权重下载范围。用所选解释器核对 `geneformer.__file__`、torch/CUDA 和模型配置。

### 输入与调用

1. 保留全转录组原始计数；用物种及注释版本对应的映射生成无版本后缀的 ENSG ID。多个原始列映射到同一 ENSG 时在计数尺度合并，记录未映射基因。
2. 写入 `.var['ensembl_id']`；从完整原始计数计算 `.obs['n_counts']`，保留细胞 ID、分组及筛选依据。仅持有 HVG 或已 log 转换矩阵时需取回完整原始计数。
3. 调用 [geneformer_route.py](../examples/ko_oe/geneformer_route.py)，使用 `--input-h5ad`、`--model-dir`、`--model-version`、`--target-ensembl`、`--perturbation` 与新的 `--output-dir`。原始全转录组来源核实后设置 `--confirm-raw-full-transcriptome`；完整命令见 [示例说明](../examples/ko_oe/README.md#geneformer)。

### 结果处理

保存 tokenizer 输出、官方 ISP 原始 pickle 和统计 CSV。`delete` / `overexpress` 分别登记；V1 的 cell 汇总指标与 V2 的 CLS 指标分开展示。默认示例输出 embedding 影响；若研究目标需要 gene embedding 或目标状态比较，按官方 API 配置 `emb_mode`、状态字典和 `InSilicoPerturberStats`，同时保存基因 token 映射、状态定义和实际比较指标。

## 复用药筛示例的配置步骤

复制所选脚本和 [common.py](../examples/lung_cancer/drug_screen/common.py) 到 `<analysis_dir>/code/`，在副本中按本次选择修改：

- `common.py` 的 `SCRNA`、`DATA`、`LIBRARY`、`CONDITIONS`、`DISEASE_VEC`；同步调整 `load_baseline` 的 ID 映射、原始计数层、归一化参数，以及 `load_disease_direction` 的列名和来源。
- 各脚本顶部的源码目录、数据目录、checkpoint/config；检查导入实际来自所选资源目录。
- `RunDir(..., base_dir=analysis_dir)` 指向本次分析目录；修改后保存代码与输入清单，再运行副本。
- 示例的疾病文件名和肺癌标签一并替换。`score_response` 中用于来源哈希的文件路径也要绑定本次疾病参照。无疾病参照时，移除强制读该文件的步骤，使用全 NaN 疾病向量并保存 `scoring_status: skipped`；原始预测及 Top 30 正常交付。

表达型药筛统一保留条件表、基因顺序、预测、基线与响应；先 `RunDir.save_predictions`，再做质量检查、可选评分和 `RunDir.finish`。恢复分类、逐条件 Top 30 及其他状态按 [下游规范](downstream-results.md) 输出。

## PRnet

### 资源获取

[官方代码](https://github.com/Perturbation-Response-Prediction/PRnet)包含模型定义、`test_demo.py`、`test_lincs.py`、`test_sciplex.py` 和自有数据预处理 notebook。权重从官方 README 的模型下载入口或仓库 `checkpoint/` 获取；配套数据可从 [Zenodo 14230870](https://zenodo.org/records/14230870) 获取。选择 LINCS 或 SciPlex 时同时选择其数据、权重与基因顺序。按仓库 requirements 配置所选环境。

### 输入与调用

- LINCS 路线使用 [screen_a549_prnet.py](../examples/lung_cancer/screen_a549_prnet.py) 的配对方式：取用户选定细胞系的条件，并保留 `paired_control_index` 指向的全部对照。检查 `SMILES`、`dose`、`cov_drug_name` 和 split 列。推理 split 可全部标 test。
- 外部基线路线使用 [run_prnet.py](../examples/lung_cancer/drug_screen/run_prnet.py)。按 checkpoint 的完整基因轴排序，LINCS 示例为 978 genes；归一化必须与对应训练数据一致，缺测必需基因先补数据。
- 该 LINCS 示例的药物条件是 FCFP4 指纹 × `log10(dose_uM + 1)`，复制到每个基线细胞。调用 `modelPGM(control, condition, noise)`，按示例的均值输出处理；保存随机种子、抽样基线及噪声设置。

### 结果处理

以预测处理后表达减对应对照得到响应；先存全量结果再做恢复方向与 Top 30。配对实测处理数据单独用于验证。外部细胞背景记录训练域适配状态，检查不同药物的响应差异；药名含逗号时按示例的逐行格式读取原生输出索引。

## chemCPA

### 资源获取

按[官方仓库 Getting started](https://github.com/theislab/chemCPA)取得环境配置、checkpoint 压缩包、`finetuning_num_genes.json`、配套处理后数据及分子 embeddings。官方提供 [checkpoint](https://f003.backblazeb2.com/file/chemCPA-models/chemCPA_models.zip) 和 [config](https://f003.backblazeb2.com/file/chemCPA-models/finetuning_num_genes.json)。模型 hash、配置记录及数据集必须对应；安装仓库后检查实际导入目录。

### 输入与调用

1. [run_chempa.py](../examples/lung_cancer/drug_screen/run_chempa.py) 对应 SciPlex3 finetuning 示例，977 genes、A549/K562/MCF7 协变量。其他 checkpoint 从自己的 config/data 重新读取基因和协变量。
2. 对齐用户基线的基因顺序和训练尺度；按训练类别构造协变量。剂量转成该示例的 nM，保留原单位。药物按 canonical SMILES 对齐 RDKit 描述符表，缺表征时补同版本编码或登记未覆盖。
3. 从 `finetuning_num_genes.json` 选对应 hash 并写回 `config['config_hash']`；设置 `notebooks.utils.CHECKPOINT_DIR`，调用该文件的 `load_model`。批量输入为 `predict(genes, drugs_idx, dosages, covariates)`，药物、剂量和协变量扩展到基线细胞数；官方 `compute_prediction` 可作对照。

### 结果处理

响应为预测表达减基线。保存剂量、协变量、SMILES、表征来源和缺失项；检查剂量变化与跨药物区分度，再生成恢复方向和逐条件 Top 30。旧环境中某个 checkpoint 的 doser 异常记录见 [环境说明](backend-environments.md#chemcpa)，新配置以本次检查结果登记。新分子描述符沿用训练 mean/std 与列顺序；本机历史近似重建列需明确记录，不能当作精确官方编码。

## UniPert-G2CP

### 资源获取

分别获取 [UniPert 编码器](https://github.com/lynn-1998/UniPert) 和 [G2CP 复现代码](https://github.com/lynn-1998/UniPert-G2CP_reproduce)。[官方 Zenodo 20355906](https://doi.org/10.5281/zenodo.20355906) 分发 `UniPert:current_model.zip`、`G2CP:data.zip`、`G2CP:saved_models.zip`；冒号是远程文件键，Windows 下载到合法的本地文件名。按所选细胞系解压数据和 checkpoint，并保留 G2CP 的目录关系。

### 输入与调用

1. 用户选择模型自带细胞背景和药物库。保存药物唯一 ID、名称、SMILES；统一结构后，用 `UniPert.encode_compounds(compound_dict={名称: SMILES})` 生成 256 维表征，保留未编码清单。
2. 采用 [run_unipert_g2cp.py](../examples/lung_cancer/drug_screen/run_unipert_g2cp.py)：设置 `UNIPERT_DIR`、`GEARS_ROOT`、`DATA_PATH`、`MODEL_PATH`；`PertData.load` → `prepare_split` → G2CP 的 `GEARS` → `load_pretrained(..., 'finetune_model')` → `new_predict`。
3. 这里导入的是 G2CP 自带的修改版 GEARS，与遗传扰动的 `cell-gears` 在独立进程/环境运行。记录 `gears.__file__`。细胞背景、数据名、split 和 checkpoint 目录使用同一套资源。

### 结果处理

输出按模型数据的 gene symbol 顺序整理；响应为预测减自带 ctrl 均值。剂量和时间标为模型未建模字段，不从外部药库生成虚假的剂量曲线。记录 NumPy/torch 种子与每条件抽取 300 个 ctrl 的方式，保存实际背景均值。随后执行统一恢复分类与 Top 30 流程。

## ORBIT：可选 pathway/program 扩展

仅在用户选择程序依赖或通路重排分析时读取本段。当前已知源码是用户本机的 `D:/WorkBuddy/模型论文复现/ORBIT/ORBIT-main`；其 README 的仓库地址仍为占位文本，不能据此构造公开下载 URL。可复用这份代码或用户指定的作者代码归档，并记录来源及哈希。

1. 检查 `scripts/main.py`、`src/orbit/model.py`、`training.py`、`data_pipeline.py` 与词表配置。按该版本 `requirements.txt` 配环境；从用户表达矩阵、细胞类型和疾病/对照标签构建输入，统一 gene symbol 与 GMT 成员矩阵。
2. 将 `main.py` 复制到分析目录，配置 `BASE`、`CONFIG` 的输入/输出、标签、词表、训练与置换参数。检查实际源码的参数；本机 README 与 main.py 的部分训练设定不同，以执行代码和记录为准。移除不属于本任务的脑组织数据自动下载，并将数据加载段接到用户 AnnData。
3. 本机入口使用 `from model import ORBIT` 等平级导入，运行前把其 `src/orbit` 加入模块搜索路径。按代码的数据加载 → 词表/掩码 → 训练/权重加载 → 注意力提取 → 分组差异及置换计算执行。训练所需程序顺序、基因掩码和标签编码与 checkpoint 一起保存。
4. 输出程序分数、注意力矩阵、细胞类型/条件对应关系和有向边差异；启用置换时保存次数、FDR 和随机种子。用程序热图、网络和分组差异图展示，结果类型为 `network` / `state_embedding`。本段作为程序分析扩展，不参与逐药恢复分类或 Top 30 基因导出。

本页官方入口与 API 于 2026-10-16 核对；每次执行记录实际资源版本。
