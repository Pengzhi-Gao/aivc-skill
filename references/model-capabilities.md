# AIVC 后端能力登记

各后端的解释器、权重、先验文件和已知安装障碍见 `backend-environments.md`。

## CellOracle

- 任务：单细胞 TF/调控网络扰动模拟
- KO：将目标表达设为 0，模拟网络传播和状态位移
- OE：在同一表达尺度设置数据范围内的高表达值
- 输入：scRNA 原始计数、TF-靶基因先验、可拟合 GRN、目标调控因子
- 输出：模拟表达/`delta_X`、调控网络变化、embedding/状态位移、轨迹/向量场
- 不输出：实测表达、实验倍数、临床疗效或因果结论
- 对照：没有"未扰动向量场"，用随机化 GRN 的 `delta_embedding_random` 作对照
- bulk：不直接适用
- 官方文档：https://morris-lab.github.io/CellOracle.documentation/
- 官方源码：https://github.com/morris-lab/CellOracle

## GEARS

- 任务：单细胞遗传扰动转录响应预测
- KO：仅在训练数据确实包含匹配 KO/CRISPRi 语义时使用
- OE：仅在训练数据确实包含匹配 CRISPRa/OE 语义时使用
- 输入：AnnData、基因名、condition、cell_type、匹配扰动训练数据
- 输出：逐基因预测表达响应及模型指标
- 限制：官方说明不面向 bulk，不保证跨细胞类型迁移；组合扰动需要组合扰动训练数据
- 官方仓库：https://github.com/snap-stanford/GEARS

## Geneformer

- 任务：单细胞 foundation model 的排序编码虚拟扰动
- 支持：`delete`、`overexpress`、`inhibit`、`activate` 参数（按版本检查）
- 输出：cell/gene embedding 影响和状态距离
- 不输出：真实 KO/OE 后逐基因表达量、实验倍数、剂量响应
- 官方文档：https://geneformer.readthedocs.io/en/latest/geneformer.in_silico_perturber.html

## PRnet

- 任务：化合物结构+剂量到转录响应
- 输入：模型自带的 LINCS AnnData（978 landmark 基因，obs 需含 `SMILES`/`dose`/`cov_drug_name`/`paired_control_index`）；筛某个细胞背景 = 从该矩阵按 `cell_type` 取子集，不是喂入新矩阵
- 输出：逐药逐基因预测响应、对照和效应量
- 限制：需要匹配训练域；未覆盖化合物不得补预测
- 覆盖检查：运行前先确认目标细胞系在 `cell_type` 里、目标药物在 `pert_iname` 里
- 工作区入口：`D:/WorkBuddy/模型论文复现/PRnet/PRnet-repro/test_demo.py`、`test_lincs.py`、`test_sciplex.py`

## UniCure

- 任务：LINCS 细胞系和药物条件的逐基因响应
- 输入：匹配 cell embedding、Uni-Mol drug embedding、剂量和 978 基因空间
- 输出：逐药逐基因预测响应和药物推荐计算
- 限制：主要限于 LINCS 训练域，不能直接当患者/体内疗效预测
- 工作区入口：`D:/WorkBuddy/模型论文复现/UniCure/UniCure-main/UniCure-main/quick_pred.py`、`repro/03_reproduce_inference.py`

## chemCPA

- 任务：化合物（RDKit 2D 描述符）+ 剂量 + 细胞系协变量 → 处理后表达
- 输入：外部基线表达矩阵（可以是用户自己的单细胞数据）、药物 SMILES、剂量、细胞类型 one-hot
- 基因空间：977 个 landmark 基因，顺序取自训练数据集
- 输出：预测处理后表达；响应 = 预测 − 基线
- 限制：细胞类型协变量只能在训练用的三个细胞系（A549/K562/MCF7）里选；药物表征按 SMILES 从 parquet 查，查不到的**填零向量**
- 工作区入口：`D:/WorkBuddy/Work_推送/2610/261002 ChemCPA药物扰动/chemCPA_repro`，推理用 `notebooks/utils.py` 的 `load_model` + `chemCPA.train.compute_prediction`

## UniPert-G2CP

- 任务：化合物（UniPert 统一表征）→ 978 landmark 基因的表达效应
- 输入：化合物名称 + 256 维 UniPert 表征；**不接受外部基线矩阵，也没有剂量维度**
- 细胞背景：固定在训练 checkpoint 对应的细胞系（当前本地只有 A549）
- 输出：978 基因的预测表达；响应 = 预测 − 模型自带对照细胞均值
- 限制：`new_predict` 每次随机抽 300 个对照细胞且无随机种子接口；本机只有 A549 权重
- 工作区入口：`D:/WorkBuddy/Work_推送/2610/261003 UniPert-G2CP`，UniPert 编码用 `official_unipert`，G2CP 推理用 `official_reproduce/G2CP/run_scripts/inference_unseen_a549_local.py` 的写法

## DrugReflector

- 任务：转录特征到化合物类别/排名
- 输出：模型原生化合物得分和排名
- 不默认输出：每个药物逐基因响应
- 官方代码归档：https://zenodo.org/records/16921928

## 不使用

- scTenifoldKnk：不作为 AIVC 路线
- ORBIT：仅作 pathway/program 分析，不冒充单基因 KO/OE
- BulkFormer：当前版本不作为化合物扰动后端