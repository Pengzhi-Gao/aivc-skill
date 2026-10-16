# AIVC 后端能力登记

各后端的解释器、权重、先验文件和已知安装障碍见 `backend-environments.md`。

执行时读取 [模型资源与执行入口](model-setup.md) 的对应路线，完成资源获取、输入转换、模型调用和结果整理。已有示例中的路径由执行者替换为用户本次指定的路径。

开始新分析时，把可访问的用户数据文件和可选模型一并列给用户选择。KO/OE 的 CellOracle 与 Geneformer 是两条独立路线：前者模拟 GRN 传播和模型表达差值，后者评估排序编码扰动后的 embedding 影响。用户选择后检查输入契约，按所选后端的原生定义保存结果。GEARS 仅在有匹配扰动训练数据且用户选择时可用。

## CellOracle

执行指导：[资源、输入与结果步骤](model-setup.md#celloracle)。

- 任务：单细胞 TF/调控网络扰动模拟
- KO：将目标表达设为 0，模拟网络传播和状态位移
- OE：在 CellOracle 内部表达尺度设置数据范围内的固定值；同时报告基线高于设定值的细胞及其模拟下降量
- 输入：scRNA 原始计数、与物种及 gene symbol 匹配的 TF-靶基因先验、可拟合 GRN、目标调控因子；示例自行生成 PCA/UMAP，分组标签可选
- 输出：模拟表达/`delta_X`、拟合网络及其传播关系、embedding/状态位移、轨迹/向量场；只有实际保存了前后两套网络权重时才能称“网络变化”
- 对照：空扰动差值为零；`calculate_randomized=True` 打乱 `delta_X` 并随机翻转符号后计算 `delta_embedding_random`，得到固定 GRN 下的随机化表达变化投影对照；原始与随机投影的向量差作为描述量
- 数据粒度：单细胞
- 示例：`examples/ko_oe/celloracle_run.py`；参数和输出见同目录 `README.md`
- 官方文档：https://morris-lab.github.io/CellOracle.documentation/
- 官方模拟 API：https://morris-lab.github.io/CellOracle.documentation/modules/celloracle.html
- 官方源码：https://github.com/morris-lab/CellOracle

## GEARS

执行指导：[资源、自有数据、训练/加载与预测代码](gears-workflow.md)。

- 任务：单细胞遗传扰动转录响应预测
- KO：仅在训练数据确实包含匹配 KO/CRISPRi 语义时使用
- OE：仅在训练数据确实包含匹配 CRISPRa/OE 语义时使用
- 输入：AnnData、基因名、condition、cell_type、匹配扰动训练数据
- 输出：逐基因预测表达响应及模型指标
- 适用范围：单细胞扰动数据；跨细胞类型迁移需单独验证，组合扰动需要组合扰动训练数据
- 官方仓库：https://github.com/snap-stanford/GEARS

## Geneformer

执行指导：[版本匹配、资源下载与输入处理](model-setup.md#geneformer)。

- 任务：单细胞 foundation model 的排序编码虚拟扰动
- KO 路线：`delete` 从排序编码中移除目标基因 token
- OE 路线：`overexpress` 调整目标基因 token 的排序位置
- 当前官方可执行操作：`delete`、`overexpress`。`inhibit`、`activate` 虽出现在参数说明中，当前实现会拒绝执行，不作为可运行选项
- 当前示例输入：人类原始全转录组 scRNA 计数、`.var['ensembl_id']`、`.obs['n_counts']`，以及匹配的 Geneformer 权重、token 字典与 tokenizer 版本；先转换成 `.dataset`
- 当前示例输出：扰动前后表征的余弦影响及聚合统计。V1 的 `cell` 模式汇总对齐基因 embedding 的余弦值，V2 的 `cls` 模式比较 CLS token 表征；两个版本的统计定义不同，分别展示。官方工具支持其他 embedding 分析模式，但须单独配置并验收。目标状态比较还需要另有明确定义的状态标签和统计配置
- 结果类型：cell/gene embedding 影响；逐基因表达与剂量响应需使用相应表达预测后端
- 版本要求：目标基因在 token 字典中，模型、token 字典和预处理统计文件使用匹配版本
- 示例：`examples/ko_oe/geneformer_route.py`；参数和输出见同目录 `README.md`
- 官方模型与示例：https://huggingface.co/ctheodoris/Geneformer
- 官方文档：https://geneformer.readthedocs.io/en/latest/geneformer.in_silico_perturber.html

## PRnet

执行指导：[资源、配对对照/外部基线与结果步骤](model-setup.md#prnet)。

- 任务：化合物结构+剂量到转录响应
- 模型接口：对照表达 + 化合物编码/剂量 + 噪声，可接外部表达；具体权重的适用样本范围需核对训练域。
- LINCS 适配器：使用自带 AnnData，978 landmark 基因，含 `SMILES`/`dose`/`cov_drug_name`/`paired_control_index`；按细胞背景取子集并保留配对对照。
- 外部基线适配器：示例 `drug_screen/run_prnet.py` 直接前向，需核对完整基因顺序、尺度和训练域，标为 OOD 探索
- 输出：逐药逐基因预测响应、对照和效应量
- 限制：训练域、编码可用性、输入适配是否成功分别记录；无法编码标 `no_prediction`；训练未见单独标注。
- 覆盖检查：LINCS 子集按已有细胞系和药物取记录；外部化合物检查合法结构、编码器和权重兼容性，再单列训练是否见过
- 工作区入口：`D:/WorkBuddy/模型论文复现/PRnet/PRnet-repro/test_demo.py`、`test_lincs.py`、`test_sciplex.py`

## UniCure

执行指导：[资源、表征生成、前向预测代码与结果整理](unicure-workflow.md)。

- 任务：细胞基线与药物条件的逐基因响应；本 Skill 示例采用 LINCS 模型
- 输入：匹配 cell embedding、Uni-Mol drug embedding、剂量和所选 checkpoint 的基因空间；LINCS 示例为 978 基因，新样本按官方流程生成细胞表征
- 输出：逐药逐基因预测响应和药物推荐计算
- 适用范围：LINCS 细胞系转录响应
- 工作区入口：`D:/WorkBuddy/模型论文复现/UniCure/UniCure-main/UniCure-main/quick_pred.py`、`repro/03_reproduce_inference.py`

## chemCPA

执行指导：[资源、输入对齐、调用与结果步骤](model-setup.md#chemcpa)。

- 任务：化合物（RDKit 2D 描述符）+ 剂量 + 细胞系协变量 → 处理后表达
- 输入：外部基线表达矩阵（可以是用户自己的单细胞数据）、药物 SMILES、剂量、细胞类型 one-hot
- 基因空间：977 个 landmark 基因，顺序取自训练数据集
- 输出：预测处理后表达；响应 = 预测 − 基线
- 限制：细胞类型协变量只能在训练用的三个细胞系（A549/K562/MCF7）里选；上游药物表征查表缺失可能静默填零，适配器必须拦截并标 `no_prediction`，或用经过一致性验证的同版编码流程生成表征
- 工作区入口：`D:/WorkBuddy/Work_推送/2610/261002 ChemCPA药物扰动/chemCPA_repro`，推理用 `notebooks/utils.py` 的 `load_model` + `chemCPA.train.compute_prediction`

## UniPert-G2CP

执行指导：[编码器与响应模型资源、调用和结果步骤](model-setup.md#unipert-g2cp)。

- 任务：化合物（UniPert 统一表征）→ 978 landmark 基因的表达效应
- 输入：化合物名称 + 256 维 UniPert 表征；**不接受外部基线矩阵，也没有剂量维度**
- 细胞背景：固定在训练 checkpoint 对应的细胞系（当前本地只有 A549）
- 输出：978 基因的预测表达（gene symbol）；响应 = 预测 − 模型自带对照细胞均值
- 限制：`new_predict` 每次随机抽 300 个对照细胞且无随机种子接口；本机只有 A549 权重
- 训练未见判定：编码覆盖与训练集覆盖分别记录。训练未见判定使用训练词表对 canonical SMILES 做差集，并先归一盐形式和立体异构体
- 疾病方向比较：输出为 gene symbol 的 landmark 空间，与 Ensembl 体系的疾病参照直接取交集前必须先映射；映射率或交集过小时不计算逆转分数
- 工作区入口：`D:/WorkBuddy/Work_推送/2610/261003 UniPert-G2CP`，UniPert 编码用 `official_unipert`，G2CP 推理用 `official_reproduce/G2CP/run_scripts/inference_unseen_a549_local.py` 的写法

## DrugReflector

执行指导：[签名方向、权重下载、调用代码与原生排名](drugreflector-workflow.md)。

- 任务：转录特征到化合物类别/排名
- 输出：模型原生化合物得分和排名
- 图表：原生分数、排名和覆盖情况
- 官方代码归档：https://zenodo.org/records/16921928
- 官方代码与 checkpoint 入口：https://github.com/Cellarity/drugreflector 、https://zenodo.org/records/16912444

## 其他后端的定位

- scTenifoldKnk：不作为 AIVC 路线
- ORBIT：用于 pathway/program 分析；本机源码、配置及结果步骤见 [可选扩展](model-setup.md#orbit可选-pathwayprogram-扩展)
- BulkFormer：当前版本不作为化合物扰动后端
