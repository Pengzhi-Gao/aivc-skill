"""AIVC 测试 - 药筛分支输入准备

从 GSE81089（NSCLC bulk RNA-seq，gene-level FPKM）构建：
  1. 基因层面表达矩阵（log2(FPKM+1)）
  2. 肿瘤 vs 正常肺的差异表达表（Welch t 检验 + BH 校正）
  3. 限制在 PRnet/LINCS 978 landmark 基因空间上的疾病方向向量

这一步对应 SKILL.md 第 2 步「检查输入」中「疾病/对照参照」的构造。
"""
from pathlib import Path
import gzip
import json
import time

import numpy as np
import pandas as pd
import requests
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'results' / 'geo_gse81089'
OUT.mkdir(parents=True, exist_ok=True)

FPKM = DATA / 'GSE81089_FPKM_cufflinks.tsv.gz'
SERIES = DATA / 'GSE81089_series_matrix.txt.gz'
GENE_SPACE = DATA / 'lincs978_genes.csv'
ACCESSION = 'GSE81089'
SOURCE = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE81nnn/GSE81089/suppl/'
CACHE = DATA / 'ensembl_to_symbol_cache.csv'


# ---------------------------------------------------------------- 1. 读表达
x = pd.read_csv(FPKM, sep='\t', index_col=0)
x.index.name = 'ensembl_gene_id'
assert x.index.is_unique, 'duplicated Ensembl gene ids'

# 两列带测序批次后缀（L608T_2122 / L771T_1），对应系列矩阵里的 L608T / L771T
renamed = {c: c.rsplit('_', 1)[0] if c.rsplit('_', 1)[-1].isdigit() else c
           for c in x.columns}
x = x.rename(columns=renamed)
assert x.columns.is_unique, 'column collision after stripping run suffix'

# 单个缺失值直接丢基因，不做插补
n_genes_raw = x.shape[0]
x = x.dropna(axis=0)
dropped_na_genes = n_genes_raw - x.shape[0]

# cufflinks 用 -1 表示该基因在该样本无法计算 FPKM，按未表达处理（置 0）
neg_mask = x.to_numpy() < 0
n_neg = int(neg_mask.sum())
assert x.to_numpy()[neg_mask].min() == -1.0 if n_neg else True
x = x.clip(lower=0.0)
assert (x.to_numpy() >= 0).all(), 'negative FPKM value after clipping'
print(f'[fpkm] genes {n_genes_raw} -> {x.shape[0]} (dropped {dropped_na_genes} with NaN), '
      f'samples {x.shape[1]}, negative sentinels clipped to 0: {n_neg}')

# ------------------------------------------------------- 2. 读样本分组标签
with gzip.open(SERIES, 'rt') as f:
    label_line = None
    for line in f:
        if line.startswith('!Sample_characteristics_ch1') and 'tumor (t) or normal (n)' in line:
            label_line = line.rstrip('\n')
            break
assert label_line is not None, 'group characteristics line not found'
vals = [v.strip('"') for v in label_line.split('\t')[1:]]
groups = pd.Series(
    [('tumor' if v.rsplit(':', 1)[1].strip().endswith('T') else 'normal') for v in vals],
    index=[v.rsplit(':', 1)[1].strip() for v in vals],
    name='group',
)
assert groups.index.is_unique, 'duplicated sample names in series matrix'
groups = groups.reindex(x.columns)
assert groups.notna().all(), 'matrix / series-matrix sample mismatch'
assert set(groups.unique()) == {'tumor', 'normal'}

log2x = np.log2(x + 1.0)

# ------------------------------------------------- 3. 差异表达（全基因空间）
normal = log2x.loc[:, groups.eq('normal').to_numpy()].to_numpy(dtype=np.float64)
tumor = log2x.loc[:, groups.eq('tumor').to_numpy()].to_numpy(dtype=np.float64)
stat = ttest_ind(tumor, normal, axis=1, equal_var=False, nan_policy='omit')
pvals = np.nan_to_num(stat.pvalue, nan=1.0)
fdr = multipletests(pvals, method='fdr_bh')[1]

full = pd.DataFrame({
    'ensembl_gene_id': x.index,
    'tumor_mean_log2': tumor.mean(axis=1),
    'normal_mean_log2': normal.mean(axis=1),
    'tumor_minus_normal_log2': tumor.mean(axis=1) - normal.mean(axis=1),
    'p_value': pvals,
    'fdr_bh': fdr,
}).sort_values('p_value').reset_index(drop=True)
full.to_csv(OUT / 'GSE81089_gene_differential.csv', index=False)


# ---------------------- 4. 把 978 landmark 基因（symbol）映射到 Ensembl id
# 只映射真正需要的 978 个基因，请求量小、可重试、结果可缓存。
space = pd.read_csv(GENE_SPACE, index_col=0)
landmark = list(space.index)
assert len(landmark) == 978


def query_mygene(symbols, chunk=300, retries=4):
    """symbol -> ensembl gene id。返回 {symbol: [ensembl, ...]}。"""
    out = {}
    for k in range(0, len(symbols), chunk):
        part = symbols[k:k + chunk]
        for attempt in range(retries):
            try:
                r = requests.post(
                    'https://mygene.info/v3/query',
                    data={'q': ','.join(part), 'scopes': 'symbol',
                          'fields': 'ensembl.gene', 'species': 'human'},
                    timeout=60,
                )
                r.raise_for_status()
                for hit in r.json():
                    q = hit.get('query')
                    ens = hit.get('ensembl')
                    if not q or not ens:
                        continue
                    ens = ens if isinstance(ens, list) else [ens]
                    ids = []
                    for e in ens:
                        g = e.get('gene') if isinstance(e, dict) else e
                        if isinstance(g, list):
                            ids.extend(g)
                        elif g:
                            ids.append(g)
                    if ids:
                        out.setdefault(q, []).extend(ids)
                break
            except Exception as e:
                print(f'[mygene] chunk {k // chunk} attempt {attempt + 1} failed: {e!r}')
                time.sleep(2 + 3 * attempt)
        else:
            print(f'[mygene] chunk {k // chunk} gave up after {retries} attempts')
    return out


if CACHE.exists():
    _c = pd.read_csv(CACHE, dtype=str)
    sym2ens = {}
    for s, g in zip(_c['symbol'], _c['ensembl_gene_id']):
        sym2ens.setdefault(s, []).append(g)
    print(f'[mygene] cache hit: {len(sym2ens)} symbols')
else:
    sym2ens = query_mygene(landmark)
    rows = [(s, g) for s, gs in sym2ens.items() for g in gs]
    pd.DataFrame(rows, columns=['symbol', 'ensembl_gene_id']).to_csv(CACHE, index=False)
    print(f'[mygene] resolved {len(sym2ens)} / {len(landmark)} landmark symbols')

full_idx = full.set_index('ensembl_gene_id')

# ------------------------------------- 5. 投到 978 landmark 基因空间上的签名
recs = []
for g in landmark:
    ens = sym2ens.get(g)
    if not ens:
        continue
    hit = full_idx.reindex(ens).dropna(subset=['tumor_minus_normal_log2'])
    if hit.empty:
        continue
    hit = hit.assign(_abs=hit['tumor_minus_normal_log2'].abs()) \
             .sort_values('_abs', ascending=False)
    row = hit.iloc[0]
    recs.append({'gene_symbol': g, 'ensembl_gene_id': hit.index[0],
                 'tumor_minus_normal_log2': row['tumor_minus_normal_log2'],
                 'p_value': row['p_value'], 'fdr_bh': row['fdr_bh']})

sig_out = pd.DataFrame(recs)
assert len(sig_out) > 0, 'no landmark gene mapped'
sig_out['significant_fdr05'] = sig_out['fdr_bh'] < 0.05
sig_out.to_csv(OUT / 'GSE81089_landmark978_signature.csv', index=False)

# 与 PRnet 基因顺序严格对齐的方向向量（未覆盖的 landmark 记 0，并在 summary 中报告）
vec = sig_out.set_index('gene_symbol')['tumor_minus_normal_log2'].reindex(landmark)
n_uncovered = int(vec.isna().sum())
vec = vec.fillna(0.0)
vec.rename('tumor_minus_normal_log2').to_frame().to_csv(
    OUT / 'disease_direction_vector_978.csv')

# ------------------------------------------------------------------ 6. 图
fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
p = np.maximum(full['fdr_bh'].to_numpy(), np.finfo(float).tiny)
sig_mask = (full['fdr_bh'] < 0.05) & (full['tumor_minus_normal_log2'].abs() > 1)
ax.scatter(full.loc[~sig_mask, 'tumor_minus_normal_log2'],
           -np.log10(p[~sig_mask.to_numpy()]), s=3, c='#9aa2a8', alpha=.4)
ax.scatter(full.loc[sig_mask, 'tumor_minus_normal_log2'],
           -np.log10(p[sig_mask.to_numpy()]), s=5, c='#b94d48', alpha=.6)
ax.set(xlabel='Tumor - normal (log2 FPKM+1)', ylabel='-log10(BH adjusted p)',
       title=f'{ACCESSION}: NSCLC vs normal lung (gene level)')
fig.tight_layout()
fig.savefig(OUT / 'GSE81089_volcano.png')
plt.close(fig)

summary = {
    'accession': ACCESSION,
    'source': SOURCE,
    'technology': 'bulk RNA-seq, cufflinks gene FPKM (Ensembl gene id)',
    'samples': {'tumor': int(groups.eq('tumor').sum()),
                'normal': int(groups.eq('normal').sum())},
    'genes_total': int(len(full)),
    'genes_dropped_all_nan': int(dropped_na_genes),
    'negative_fpkm_sentinels_clipped_to_zero': int(n_neg),
    'landmark978_covered': int(len(sig_out)),
    'landmark978_not_covered': int(n_uncovered),
    'landmark978_coverage_fraction': round(len(sig_out) / 978, 4),
    'significant_genes_fdr_lt_0.05_abs_log2fc_gt_1': int(sig_mask.sum()),
    'comparison': 'unpaired Welch t-test on log2(FPKM+1), BH adjusted',
    'limitation': ('tumor and normal are not all from the same patients; stage/histology/'
                   'smoking covariates not modeled. This is a disease-direction reference, '
                   'not a perturbation training set.'),
}
(OUT / 'run_summary.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
