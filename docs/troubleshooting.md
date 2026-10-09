# 常见问题与踩坑记录

这些是实测踩出来的，不是推测。每条都注明了现象、原因和处理方式。

## 安装

### CellOracle 在 Windows 上装不下来

`pip install celloracle` 会拉进 `velocyto` 和 `gimmemotifs` 两个只有源码包的依赖，conda-forge 与 bioconda 都没有 win-64 构建。`velocyto` 唯一的 Cython 模块 `speedboosted` 提供 CellOracle 计算路径要用的 `colDeltaCor` 系列函数，必须真的编译。

处理方式见 [`environment-celloracle-windows.md`](environment-celloracle-windows.md)。要点：用 pip 包 `ziglang` 自带的 zig 当 C 编译器，并且必须用 Cython 3 从官方 `.pyx` 重新生成 C——sdist 里自带的 `speedboosted.c` 太旧，Python 3.10 下会报 `expression is not assignable` 和 `no member named 'tp_print'`。

### CellOracle 装完导入时报 `which: no R` 和 jupyter 版本告警

不阻断。前者是 `test_R_libraries_installation()` 找不到 R，本流程不用 R；后者只是版本提示。

### 不要用主环境跑 CellOracle

`celloracle 0.20.0` 锁定 `numpy==1.26.4` 和 `pandas<=1.5.3`，和带新版 scanpy 的主环境冲突。两个解释器分开装，靠 `manifest` 记录各自路径。

## 输入

### PRnet 的输入不是任意表达矩阵

PRnet 的输入是模型自带的 LINCS AnnData（978 个 landmark 基因），`obs` 里必须有 `SMILES`、`dose`、`cov_drug_name`、`paired_control_index`。想筛某个细胞背景，做法是**从这份矩阵里按 `cell_type` 取子集**，不是把新数据喂进去。

取子集时必须连同 `paired_control_index` 指向的对照一起保留，否则数据集构建直接报错。A549（人肺腺癌）有 4060 个 profile、458 种药物。

### chemCPA 查不到药物表征会静默填零向量

chemCPA 按 SMILES 从 parquet 查药物表征，查不到的不报错，直接填零向量，等于判定该药无效果。2333 个 FDA 药物里训练表征只覆盖 768 个，剩下 1565 个必须显式补，不能默认"跑完就有结果"。

补表征的做法是照它的生成流程重算：descriptastorus 的 RDKit2D 描述符 → NaN/Inf 置 0 → 丢低方差列 → 用训练集统计量做 z 标准化。标准化用的 mean/std 没存在 parquet 里，要用训练集 SMILES 重算原始描述符反解，再拿 parquet 自身校验。实测 194 列里 180 列逐值一致。

### chemCPA 的 config 缺字段

`load_config` 读出来的 config 里没有 `config_hash`，要手动补，否则 `load_model` 报 KeyError。另外 `load_model` 里是 `from chemCPA.paths import CHECKPOINT_DIR`，要改 `utils` 命名空间里的那份，改 `chemCPA.paths` 无效。

## 结果

### CellOracle 没有"未扰动向量场"

`simulate_shift` 传播的是"输入 − 实测"这个差值。扰动条件为空时差值恒为 0，位移自然也是 0，所以**不能靠跑一次空扰动来拿基线**。

正确做法是用 `estimate_transition_prob(calculate_randomized=True)` 产出的 `delta_embedding_random` 作随机化 GRN 对照，报告"扰动位移 − 随机对照位移"。实测扰动特异位移 0.49–0.70，随机对照 0.28–0.30，两者差得开才说明信号不是网络结构本身带来的。

### 药筛排名看起来有数，实际不随药物变化

把用户自己的样本当基线喂给在细胞系上训练过的模型，很容易得到这种结果。模型输出被一个与药物无关的共同成分主导，药物那一侧被淹没。

交付前必须过退化检查，两个指标：

- `score_spread_across_conditions`：分数在条件间的极差
- `median_pairwise_response_correlation`：不同条件响应向量两两相关的中位数

参考量级：正常结果约极差 0.31–0.36、相关中位数 0.04；退化结果极差 < 0.01、相关中位数 > 0.999。判退化时状态标 `failed`，排名表只留作排查。

实测：同一批 FDA 药物库上，chemCPA 极差 3.4e-06、相关中位数 1.00000；PRnet 0.0084、0.99994；UniPert-G2CP 0.307、0.958。前两个都退化。

### chemCPA 药物效应接近 0

除了"基线在训练域之外"，chemCPA 这个权重本身还有问题：剂量响应模块是带 BatchNorm 的 MLP，输出经 sigmoid 后只有 5e-4 量级，而 `combo_embedding = transformed × sigmoid`，药物对隐变量的贡献被压到 0.01 左右。

用模型自带的 A549 对照细胞、走官方 `chemCPA.train.compute_prediction` 原样复算，5 个不同药物的响应两两相关 1.000000。所以不是调用姿势的问题。

### 单细胞矩阵投到 landmark 基因空间会掉基因

chemCPA 是 977 个 landmark 基因，PRnet 和 UniPert-G2CP 是 978 个。单细胞矩阵投过去覆盖率约 96%，未覆盖的基因置 0，数量要写进运行记录。

另外归一化口径要和训练数据一致：SciPlex 和 LINCS 都是先在全转录组做 `normalize_total` 再投到 landmark 基因，不要在子集上重新归一化。

## 交付

### 不要用零向量或库均值补缺失候选

无覆盖的候选标 `no_prediction`，不补预测值。用零效应补齐会让排名看起来完整，实际上把"没跑"和"没效果"混在一起了。

### 结果有值不等于已验证

缺少独立实验或留出数据时标 `unvalidated`，保留原始结果但不写成已验证。药筛那类"预测响应 vs 实测响应"的相关系数是训练域内的内部一致性，不能当实验验证。
