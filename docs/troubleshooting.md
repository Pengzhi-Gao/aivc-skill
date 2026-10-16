# 常见问题与踩坑记录

本页汇总本地实测中的现象、已定位原因和处理方式。

## 安装

### CellOracle 在 Windows 上装不下来

`pip install celloracle` 会拉进 `velocyto` 和 `gimmemotifs` 两个只有源码包的依赖，conda-forge 与 bioconda 都没有 win-64 构建。`velocyto` 唯一的 Cython 模块 `speedboosted` 提供 CellOracle 计算路径要用的 `colDeltaCor` 系列函数，必须真的编译。

处理方式见 [`environment-celloracle-windows.md`](environment-celloracle-windows.md)。要点：用 pip 包 `ziglang` 自带的 zig 当 C 编译器，并且必须用 Cython 3 从官方 `.pyx` 重新生成 C——sdist 里自带的 `speedboosted.c` 太旧，Python 3.10 下会报 `expression is not assignable` 和 `no member named 'tp_print'`。

### CellOracle 装完导入时报 `which: no R` 和 jupyter 版本告警

不阻断。前者是 `test_R_libraries_installation()` 找不到 R，本流程不用 R；后者只是版本提示。

### CellOracle 使用隔离环境

`celloracle 0.20.0` 锁定 `numpy==1.26.4` 和 `pandas<=1.5.3`，和带新版 scanpy 的主环境冲突。两个解释器分开装，靠 `manifest` 记录各自路径。

## 输入

### PRnet 的输入契约

PRnet 的 LINCS 适配器使用自带 AnnData（978 landmark 基因），需要 `SMILES`、`dose`、`cov_drug_name`、`paired_control_index`，按 `cell_type` 取子集。模型前向接口也能接外部基线，适用性由预处理、权重与训练域共同决定。

取子集时必须连同 `paired_control_index` 指向的对照一起保留，否则数据集构建直接报错。A549（人肺腺癌）有 4060 个 profile、458 种药物。

### chemCPA 查不到药物表征会静默填零向量

chemCPA 按 SMILES 从 parquet 查药物表征，查不到时会静默填零向量，适配器需先检查覆盖。2333 个 FDA 药物中，训练表征覆盖 768 个，剩余 1565 个需按兼容流程补充表征或标为无预测。

补表征的做法是照它的生成流程重算：descriptastorus 的 RDKit2D 描述符 → NaN/Inf 置 0 → 丢低方差列 → 用训练集统计量做 z 标准化。标准化用的 mean/std 没存在 parquet 里，要用训练集 SMILES 重算原始描述符反解，再拿 parquet 自身校验。实测 194 列里 180 列逐值一致。

### chemCPA 的 config 缺字段

`load_config` 读出来的 config 里没有 `config_hash`，要手动补，否则 `load_model` 报 KeyError。另外 `load_model` 里是 `from chemCPA.paths import CHECKPOINT_DIR`，要改 `utils` 命名空间里的那份，改 `chemCPA.paths` 无效。

## 结果

### CellOracle 的空扰动与随机化对照

`simulate_shift` 传播的是"输入 − 实测"这个差值。扰动条件为空时差值恒为 0，位移也是 0。扰动位移使用下述随机化投影作技术对照。

`estimate_transition_prob(calculate_randomized=True)` 会打乱 `delta_X` 并随机翻转符号，计算随机化表达变化的投影对照 `delta_embedding_random`，GRN 保持固定。原始与对照并排展示，两者向量差长度按描述量解读。

### 药筛排名看起来有数，实际不随药物变化

历史外部基线测试中，模型输出由共同成分主导，药物间响应差异较小。

按 `../references/downstream-results.md` 分开检查预测有限性、逐基因响应范围和评分区分度。NaN/Inf 按条件失败；评分相同或余弦接近 1 不足以判预测失败。响应同方向但幅度不同可以是有效输出。评分无法区分时保留预测，将排名标 `unresolved`；数值容差用于分辨数值差异，生物学显著性另需适用检验。

实测：同一批 FDA 药物库上，chemCPA 极差 3.4e-06、相关中位数 1.00000；PRnet 0.0084、0.99994；UniPert-G2CP 0.307、0.958。这些历史数值使用旧版判据，当前状态需分别核对预测、响应和排名。

### chemCPA 药物效应接近 0

除了"基线在训练域之外"，chemCPA 这个权重本身还有问题：剂量响应模块是带 BatchNorm 的 MLP，输出经 sigmoid 后只有 5e-4 量级，而 `combo_embedding = transformed × sigmoid`，药物对隐变量的贡献被压到 0.01 左右。

用模型自带的 A549 对照细胞、走官方 `chemCPA.train.compute_prediction` 原样复算，5 个不同药物的响应两两相关 1.000000。该结果复现了当前环境与权重下的低响应区分度。

### 单细胞矩阵投到 landmark 基因空间会掉基因

chemCPA 是 977 个 landmark 基因，PRnet 和 UniPert-G2CP 是 978 个。单细胞矩阵投过去覆盖率约 96%，缺失基因与实测零值分开。当前示例没有经过验证的插补流程，缺模型必需基因时停止该路推理；评分使用共同有效基因并保存掩码。

SciPlex 与 LINCS 各自按对应权重的训练预处理核对归一化口径。记录归一化所用基因范围、目标总量及 log 变换。

## 交付

### 缺失候选的保存方式

无覆盖候选标 `no_prediction`，预测值留缺失；有效预测中的零效应正常保留。

### 运行与验证状态分别记录

缺少独立实验或留出数据时标 `unvalidated`，并保存原始模型结果。预测与实测相关系数需注明来源属于训练内比较、留出测试还是外部验证。
