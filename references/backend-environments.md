# AIVC 后端运行环境

各后端的依赖、权重位置和已知安装障碍。这些是实测结论，不是推测。

## 环境分配

| 后端 | 解释器 | 说明 |
| --- | --- | --- |
| PRnet | `D:/WorkBuddy/env`（主环境，Python 3.13） | 已装 scanpy、torch；权重在 PRnet 工作区 |
| CellOracle | `D:/WorkBuddy/env/celloracle310`（Python 3.10） | 与主环境隔离，因为 celloracle 锁定 numpy 1.26 / pandas ≤1.5.3 |
| UniCure | 见其工作区 `environment.yml` | 未在本次测试中运行 |

不要用主环境跑 CellOracle：它的 `numpy==1.26.4` / `pandas<=1.5.3` 约束会和主环境冲突。

## CellOracle

### 权重与先验

- base GRN（人类 promoter，hg19，gimmemotifs v5，fpr2）：
  `https://raw.githubusercontent.com/morris-lab/CellOracle/master/celloracle/data/promoter_base_GRN/hg19_TFinfo_dataframe_gimmemotifsv5_fpr2_threshold_10_20210630.parquet`
  约 5.5 MB，37,003 行，1094 个 TF 列，27,150 个靶基因。首次运行自动下载并缓存。
- 该先验是**基因×TF 的宽表**：前两列 `peak_id`、`gene_short_name`，其余列名即 TF。
  取 TF 数量用 `df.shape[1] - 2`，没有 `TF` 列。

### 已知安装障碍（Windows，无 C 编译器）

- `velocyto` 只有源码包，且无 conda-forge / bioconda 的 win-64 构建。CellOracle 的计算路径
  依赖它的 `estimation.colDeltaCor*` 与 `diffusion.Diffusion`，必须真的编译出来。
  可行做法：用 pip 包 `ziglang` 自带的 zig 当 C 编译器，并且**必须用 Cython 3 从官方
  `speedboosted.pyx` 重新生成 C**（sdist 自带的 `speedboosted.c` 太旧，Python 3.10 下报
  `expression is not assignable` / `no member named 'tp_print'`）。编译后的 `_colDeltaCor`
  与 numpy 手写的 Pearson 参照实现比对，非对角元素最大绝对误差 1.7e-16。
- `velocyto/__init__.py` 会连带导入需要 `pysam` 的计数模块，而 `pysam` 没有可用的
  Windows 轮子。只导入 `diffusion` / `estimation` / `serialization` 即可，计算代码不受影响。
- `gimmemotifs` 与 `pybedtools` 只在 `celloracle.motif_analysis` 的 import 阶段被需要，
  那条路径是"从基因组序列 + motif 库扫描构建 base GRN"。直接加载预构建 base GRN 时用不到，
  可用占位模块满足导入。
- 正常导入时会打印 `which: no R` 和 jupyter 版本告警，都不影响计算路径。

完整搭建步骤见工作区 `AIVC-Skill/test/ENV_CELLORACLE.md`。

### 输入要求

- `adata.X` 必须是原始计数（不能有负值），embedding 放 `obsm`，cluster 列放 `obs`。
- 基因数应在 1000–3000 之间，否则 CellOracle 会告警。实际做法是先取高变基因
  （如 `n_top_genes=1500`），并把目标基因强制保留在基因集里。
- 目标基因要同时满足三个条件才会通过模拟前的检查：在表达矩阵里、在 base GRN 的 TF 列里、
  在推断出的 GRN 里有足够的调控连接。
- **不存在"未扰动向量场"**。`simulate_shift` 传播的是 `输入 - 实测` 这个差值，
  空扰动条件下差值恒为 0，得到的位移也是 0。扰动分析应改用
  `estimate_transition_prob(calculate_randomized=True)` 产出的 `delta_embedding_random`
  作为随机化 GRN 对照，报告"扰动位移 − 随机对照位移"。

### 单次运行参考耗时

703 细胞 × 1502 基因，bagging 20，单个扰动条件约 30 秒（4 线程，CPU）。

## PRnet

### 权重与数据

- 权重：`PRnet/PRnet-repro/checkpoint/lincs_best_epoch_all.pt`（LINCS 预训练）。
- 输入不是"任意表达矩阵"，而是**模型自带的 LINCS AnnData**：
  `PRnet/PRnet-repro/dataset/lincs_subset.h5ad`，978 个 landmark 基因，obs 里必须有
  `SMILES`、`dose`、`cov_drug_name`、`paired_control_index`。
- 筛选某个细胞背景的做法是**从这份矩阵里按 `cell_type` 取子集**，不是把新的 h5ad 喂进去。
  取子集时必须连同 `paired_control_index` 指向的对照一起保留，否则数据集构建会报错。
- 该子集覆盖 56 个细胞系，其中 **A549（人肺腺癌）有 4060 个 profile、458 种药物**，
  可直接用于肺癌药筛。

### 调用要点

- 全部条件放进同一个 split（如自建 `aivc_split` 列全填 `test`），train/valid 为空即可只跑推理。
- `predict_mean=True` 取分布均值，去掉单次采样噪声。
- 输出 `*_cov_drug_array.csv` 的每行是 `cov_drug_name`，**药物名可能含逗号**，
  读的时候按行读，不要用默认 csv 解析。
- 预测响应 = 预测表达 − 配对对照；用它与疾病方向向量的夹角做排序。

### 用外部基线时的坑

PRnet 的模型是 `forward(control, condition_label, noise)`，条件标签是 SMILES 的 FCFP4 指纹乘 `log10(剂量+1)`，剂量单位用 µM（训练数据是 0.04–10 µM）。自己拼 batch 时注意条件标签要复制到每个基线细胞，否则维度对不上。

把用户自己的单细胞矩阵当基线喂进去，模型会给出响应，但**药物之间往往没有区分度**：响应被一个与药物无关的共同成分主导。交付前必须做退化检查（见 `downstream-results.md`）。

## chemCPA

### 环境与权重

- 解释器：主环境 `D:/WorkBuddy/env`（torch 2.6.0+cu124 可用 CUDA）。
- 工作区：`D:/WorkBuddy/Work_推送/2610/261002 ChemCPA药物扰动/chemCPA_repro`，已 `pip install -e`。
- 权重：`project_folder/checkpoints/27b401db1845eea26c102fb614df9c33.pt`（SciPlex3，finetuning）。
- 配置：`notebooks/finetuning_num_genes.json`。注意 `load_config` 读出来的 config 里没有 `config_hash`，要手动补上，否则 `load_model` 会 KeyError。
- `load_model` 里是 `from chemCPA.paths import CHECKPOINT_DIR`，所以要改 `utils.CHECKPOINT_DIR`，改 `chemCPA.paths` 无效。

### 药物表征（重要）

- 默认从 `project_folder/embeddings/rdkit/rdkit2D_embedding_lincs_trapnell.parquet` 按 SMILES 查，**查不到的填零向量**，等于静默判定该药无效果。
- 生成流程（`embeddings/rdkit/embedding_rdkit.py`）：descriptastorus 的 `MakeGenerator(("RDKit2D",))` → 丢掉第一个值（SMILES 本身）→ NaN/Inf 置 0 → 丢掉标准差 ≤ 0.01 的列 → 用训练集统计量做 z 标准化（pandas 默认 ddof=1）。
- 训练集是 17869 个 SMILES × 194 维。要在库里加新药，用训练集重算原始描述符反解 mean/std 和保留列，并用 parquet 自身校验重建精度。实测 194 列里 180 列逐值一致；少数列因 RDKit/descriptastorus 版本差异无法重建（其中 33 号列方差 5e52、57 号列 99.85% 的行对不上），这两列对新增药物填训练均值，训练集已有药物用官方原值。
- 注意描述符里存在完全相同的重复列（raw 38 与 raw 40 逐值相等），做列对应时同分要优先取同位置。

### 调用要点

- 基因空间 977 个 landmark 基因，顺序取自训练数据集。
- `predict(genes, drugs_idx, dosages, covariates)`；`drugs_idx`/`dosages` 要复制到每个基线细胞。
- 细胞类型协变量 one-hot，类别顺序取 `datasets['ood'].covariate_names['cell_type']`。
- 剂量单位是 nM（训练数据 10–10000），对应 0.01–10 µM。

### 已知问题

该权重在本环境下**药物效应接近 0**：doser（带 BatchNorm 的 MLP）的 sigmoid 输出量级只有 5e-4，导致 `combo_embedding = transformed × sigmoid ≈ 0`。用模型自带的 A549 对照细胞走官方 `chemCPA.train.compute_prediction`，5 个不同药物的响应完全相同（两两相关 1.000000）。这不是调用姿势的问题，先不要把它当成可用后端，除非换权重或排查出 doser 权重加载异常。

## UniPert-G2CP

### 环境与权重

- 解释器：主环境 `D:/WorkBuddy/env`（需要 torch + torch_geometric）。
- UniPert 编码器：`official_unipert/current_model/unipert_model.pt`（3.7 MB），用 `from unipert import UniPert; up.encode_compounds(compound_dict={名称: SMILES})` 得到 256 维表征。化合物路径**不需要 mmseqs2**（只有基因路径需要）。
- G2CP 权重：`official_reproduce/G2CP/saved_models/LINCS_ts/LINCS-A549/new_pretrain-finetune-ts_0_0.0/finetune_model.pt`（4.4 MB）。**本机只有 A549 一个细胞系**，Zenodo 的 12.94 GB 全量归档没有下载。
- 数据：`official_reproduce/G2CP/data`，`PertData.load(data_name='LINCS-A549', ...)`。

### 调用要点

- 推理入口照 `official_reproduce/G2CP/run_scripts/inference_unseen_a549_local.py` 的写法：`PertData` → `prepare_split('pretrain-finetune-ts')` → `GEARS` → `load_pretrained` → `new_predict(化合物名列表, 表征字典)`。
- **不接受外部基线矩阵**：`new_predict` 用 `self.adata[condition=='ctrl']` 作对照，每次随机抽 300 个细胞（无种子接口）。A549 的 ctrl 只有 9 个细胞，实际是重复抽样。
- 没有剂量维度，一个化合物就是一个条件。
- 基因空间 978 个 landmark 基因，输出为 gene symbol。与 Ensembl 体系的疾病参照比较前先做 ID 映射。
- 单次参考耗时：2333 个化合物 7~18 分钟（GPU，视机器负载；瓶颈在逐化合物循环，GPU 利用率只有 8%~44%，不是显存问题）。
- 对照细胞抽样不可复现：A549 的 ctrl 只有 9 个细胞，`new_predict` 每次随机抽 300 个且没有种子接口。同一批化合物两次运行的预测会有差异，跨运行比较分数前先确认这一点，不要把运行间差异当成药物间差异。
