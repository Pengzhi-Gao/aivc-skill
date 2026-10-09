# 示例：非小细胞肺癌（NSCLC）

这是一轮完整的端到端测试，覆盖 AIVC-skill 的三条路线。数据全部来自 GEO，癌种为肺癌。测试机器为 32G 内存 / 8G 显存，三个分支都没有用满显存。

测试结果与踩坑记录见 [`reports/reproduction-report.md`](reports/reproduction-report.md)。

## 数据来源

| 用途 | 编号 | 内容 | 规模 | 下载地址 |
| --- | --- | --- | --- | --- |
| 疾病参照（bulk） | GSE81089 | NSCLC bulk RNA-seq，cufflinks gene FPKM | 218 样本（199 肿瘤 / 19 正常），63130 基因 | `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE81nnn/GSE81089/suppl/` |
| 未干预基线（scRNA） | GSE198291 | LUAD 单细胞计数 | 720 细胞 × 22335 基因，QC 后 703 细胞 | `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE198nnn/GSE198291/suppl/GSE198291_allcounts2.csv.gz` |
| 药物库 | Drug Repurposing Hub | Broad Institute 药物重定位库 | 6798 药 → 筛 `clinical_phase == "Launched"` → 去重后 2333 药 | `https://clue.io/repurposing` |

另有 CellOracle 官方人类 promoter base GRN（hg19，gimmemotifs v5，fpr2），首次运行时自动下载并缓存。

## 运行环境

| 分支 | 解释器 | 关键依赖 |
| --- | --- | --- |
| bulk / scRNA 数据准备、PRnet、chemCPA、UniPert-G2CP | 主环境，Python 3.13 | scanpy、torch（CUDA 可用）、rdkit、statsmodels |
| CellOracle KO/OE | 独立环境，Python 3.10 | celloracle 0.20.0、numpy 1.26.4、pandas ≤1.5.3 |

CellOracle 环境的搭建（Windows 无 C 编译器时用 `ziglang` 编译 velocyto 的 Cython 模块）见 [`../../docs/environment-celloracle-windows.md`](../../docs/environment-celloracle-windows.md)。

## 运行顺序

```bash
# 0. 把 GEO 原始文件放进 data/
#    GSE81089_FPKM_cufflinks.tsv.gz / GSE81089_series_matrix.txt.gz
#    GSE198291_allcounts2.csv.gz / GSE198291_series_matrix.txt.gz

# 1. 疾病方向向量（主环境）
python prepare_luad_bulk.py

# 2. 单细胞 QC（主环境）
python prepare_geo_scrna.py

# 3. 虚拟敲除 / 过表达（CellOracle 环境）
<python310> run_celloracle_ko_oe.py

# 4. 药筛：PRnet 在自带 LINCS A549 子集上（主环境）
python screen_a549_prnet.py

# 5. 药筛：用户矩阵 + FDA 药物库，三个后端（主环境）
cd drug_screen
python build_fda_library.py
python run_chempa.py
python run_prnet.py
python run_unipert_g2cp.py
python compare_backends.py
```

脚本里的后端工作区路径（PRnet、chemCPA、UniPert-G2CP）是绝对路径，换机器时按需修改。

## 参考耗时

| 步骤 | 耗时 |
| --- | --- |
| bulk 差异表达（63129 基因 × 218 样本） | 约 35 秒 |
| 单细胞 QC | 约 15 秒 |
| CellOracle：GRN 推断 + 四个扰动条件 | 约 4 分钟（4 线程，CPU） |
| PRnet：2367 个条件 | 约 1 分钟（GPU） |
| 三后端药筛：每个 6999 个条件 | chemCPA 约 1.5 分钟、PRnet 约 35 秒、UniPert-G2CP 约 7 分钟（含化合物编码） |

## 输出

每个脚本产出独立的 `results/<run_id>/` 目录，结构见 skill 的 `references/execution-playbook.md`。关键结果：

- `results/geo_gse81089/`：全基因差异表、978 landmark 基因方向向量、火山图
- `results/geo_gse198291/`：QC 后的 AnnData、样本元数据
- `results/gse198291_celloracle_ko_oe/`：每个条件的 `delta_X` 全量矩阵、embedding 位移表（含随机化 GRN 对照）、逐基因效应表、向量场图和 Top 基因条形图
- `results/a549_prnet_screen/`：2367 条条件表、458 行药物排名表、预测与实测一致性散点图
- `drug_screen/results/`：三个后端各自的排名与图，以及 `comparison/` 下的横向对比
