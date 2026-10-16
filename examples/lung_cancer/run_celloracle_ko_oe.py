"""AIVC 测试 - 虚拟敲除(KO) / 虚拟过表达(OE) 分支端到端运行

后端：CellOracle 0.20.0（独立环境 D:/WorkBuddy/env/celloracle310，Python 3.10）
数据：GSE198291 人肺癌单细胞计数（QC 后 703 细胞），由 prepare_geo_scrna.py 产出
先验：CellOracle 官方人类 promoter base GRN（hg19, gimmemotifs v5, fpr2）

流程（对应 SKILL.md 第 3 步「路由后端并运行」）：
  1. 取高变基因（强制保留目标基因），算 UMAP embedding
  2. import_anndata_as_raw_count -> import_TF_data -> perform_PCA -> knn_imputation
  3. get_links / filter_links / fit_GRN_for_simulation 建调控网络
  4. 基线状态：estimate_transition_prob -> calculate_embedding_shift -> p_mass/mass_filter
  5. 逐个目标基因跑 KO（表达置 0）与 OE（置数据范围内高值），各自独立落盘
  6. 导出 delta_X、embedding 位移、受影响基因表与图

只报告实际算出的数值与技术状态；不做机制、疗效或靶点结论。
"""
from pathlib import Path
import json
import sys
import traceback
import hashlib
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import celloracle as co

ROOT = Path(__file__).resolve().parent
RUN_ID = f'gse198291_celloracle_ko_oe_{datetime.now():%Y%m%dT%H%M%S%f}'
OUT = ROOT / 'results' / RUN_ID
RAW = OUT / 'raw'
TAB = OUT / 'tables'
FIG = OUT / 'figures'
for d in (RAW, TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

H5AD = ROOT / 'results' / 'geo_gse198291' / 'GSE198291_counts_qc200.h5ad'
GRN_CACHE = ROOT / 'data' / 'celloracle_base_grn_hg19.parquet'
GRN_URL = ('https://raw.githubusercontent.com/morris-lab/CellOracle/master/'
           'celloracle/data/promoter_base_GRN/'
           'hg19_TFinfo_dataframe_gimmemotifsv5_fpr2_threshold_10_20210630.parquet')

TARGETS = ['STAT3', 'JUN']
CLUSTER_COL = 'tissue'
# 本数据集没有细胞类型注释，且每类细胞数偏少（最少 16 个）。
# 这里按「全体细胞一个 GRN 单元」建网络，组织来源标签只用于分组汇总。
GRN_UNIT_COL = 'grn_unit'
N_HVG = 1500
N_PCS = 30
N_NEIGHBORS = 100
SEED = 123
errors = []


def log_err(stage, exc):
    errors.append({'stage': stage, 'error': repr(exc),
                   'traceback': traceback.format_exc()[-2000:]})
    print(f'[ERROR] {stage}: {exc!r}')


def sha16(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()[:16]


# ------------------------------------------------------------ 1. 准备输入
adata = ad.read_h5ad(H5AD)
adata.obs[CLUSTER_COL] = adata.obs[CLUSTER_COL].astype(str)
print(f'[input] cells={adata.n_obs} genes={adata.n_vars} '
      f'clusters={adata.obs[CLUSTER_COL].nunique()}')

# 目标基因必须存在
missing_targets = [g for g in TARGETS if g not in adata.var_names]
assert not missing_targets, f'target genes absent from matrix: {missing_targets}'

# HVG 选择在 log 标准化后的副本上做，CellOracle 仍吃原始计数
tmp = adata.copy()
sc.pp.normalize_total(tmp, target_sum=1e4)
sc.pp.log1p(tmp)
sc.pp.highly_variable_genes(tmp, n_top_genes=N_HVG, flavor='seurat')
hvg = set(tmp.var_names[tmp.var['highly_variable']]) | set(TARGETS)
keep = [g for g in adata.var_names if g in hvg]
adata = adata[:, keep].copy()
print(f'[hvg] genes kept={adata.n_vars} (targets forced in: {TARGETS})')

# embedding
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.pca(adata, n_comps=N_PCS, random_state=SEED)
sc.pp.neighbors(adata, n_neighbors=15, random_state=SEED)
sc.tl.umap(adata, random_state=SEED)

# CellOracle 需要原始计数：把 log 化的 X 换成原始计数副本
raw = ad.read_h5ad(H5AD)[:, keep].copy()
adata.X = raw.X.copy()
adata.layers.clear()
adata.obs[GRN_UNIT_COL] = 'all'
cell_tissue = adata.obs[CLUSTER_COL].copy()   # 单独留存，供按组织分组汇总

# ---------------------------------------------------------- 2. base GRN
if not GRN_CACHE.exists():
    import urllib.request
    print('[grn] downloading human promoter base GRN ...')
    urllib.request.urlretrieve(GRN_URL, GRN_CACHE)
base_grn = pd.read_parquet(GRN_CACHE)
# 官方 base GRN 是宽表：前两列是 peak_id / gene_short_name，其余列名即 TF
n_tf = base_grn.shape[1] - 2
n_tg = base_grn['gene_short_name'].nunique()
print(f'[grn] base GRN rows={len(base_grn)} TFs={n_tf} target genes={n_tg}')
assert 'STAT3' in base_grn.columns, 'STAT3 missing from base GRN TF columns'

# ------------------------------------------------------------ 3. 建网络
oracle = co.Oracle()
oracle.import_anndata_as_raw_count(
    adata=adata, cluster_column_name=GRN_UNIT_COL, embedding_name='X_umap')
oracle.import_TF_data(TF_info_matrix=base_grn)
oracle.perform_PCA()

k_imp = min(15, adata.n_obs - 1)
oracle.knn_imputation(n_pca_dims=min(N_PCS, adata.n_obs - 1), k=k_imp,
                      balanced=True, b_sight=min(150, adata.n_obs - 1),
                      b_maxl=min(150, adata.n_obs - 1), n_jobs=4)
print('[grn] knn imputation done')

links = oracle.get_links(cluster_name_for_GRN_unit=GRN_UNIT_COL, alpha=10,
                         bagging_number=20, verbose_level=0, n_jobs=4)
links.filter_links(p=0.001, weight='coef_abs', threshold_number=1000)
n_links = int(sum(len(v) for v in links.filtered_links.values()))
print(f'[grn] links after filtering = {n_links}')
assert n_links > 0, 'no GRN links survived filtering'

oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
oracle.fit_GRN_for_simulation(alpha=10, GRN_unit='whole',
                              use_cluster_specific_TFdict=False, verbose_level=0)

# --------------------------------------------------------- 4. 基线与对照
# CellOracle 的 simulate_shift 传播的是"扰动信号"：delta_input = 输入 - 实测，
# 空扰动条件下 delta_input 恒为 0，因此不存在"未扰动向量场"。
# 这里用 CellOracle 自带的随机化表达变化投影对照（不是重建随机 GRN）
# （estimate_transition_prob 的 calculate_randomized=True -> delta_embedding_random）。

# 目标基因的 imputed 表达范围（OE 取值依据）
# Save the full baseline and fitted network for downstream views.
for unit, edges in links.filtered_links.items():
    edges.assign(grn_unit=unit).to_csv(TAB / f'baseline_network_{unit}.csv', index=False)
imputed = pd.DataFrame(
    np.asarray(oracle.adata.layers['imputed_count'].todense()
               if hasattr(oracle.adata.layers['imputed_count'], 'todense')
               else oracle.adata.layers['imputed_count']),
    index=oracle.adata.obs_names, columns=oracle.adata.var_names)
imputed.to_csv(RAW / 'baseline_imputed_expression.csv.gz')
range_rows = []
for g in TARGETS:
    v = imputed[g].to_numpy()
    range_rows.append({'gene': g, 'min': float(v.min()), 'mean': float(v.mean()),
                       'max': float(v.max()), 'p95': float(np.percentile(v, 95))})
pd.DataFrame(range_rows).to_csv(TAB / 'target_baseline_expression_range.csv', index=False)
rng = {r['gene']: r for r in range_rows}
print('[targets] baseline imputed expression:', json.dumps(rng, indent=None))

# ------------------------------------------------- 5. KO / OE 逐条件运行
results = {}
for gene in TARGETS:
    for mode in ('ko', 'oe'):
        tag = f'{gene}_{mode}'
        try:
            if mode == 'ko':
                value = 0.0
            else:
                # 数据范围内的高表达值：取 imputed 的 p95，仍在 CellOracle 安全范围内
                value = float(rng[gene]['p95'])
                if value <= 0:
                    raise ValueError('non-positive OE value from imputed range')

            oracle.simulate_shift(perturb_condition={gene: value}, n_propagation=3)
            delta_x = pd.DataFrame(
                np.asarray(oracle.adata.layers['simulated_count']),
                index=oracle.adata.obs_names,
                columns=oracle.adata.var_names) - imputed
            delta_x.to_csv(RAW / f'{tag}_deltaX.csv')

            oracle.estimate_transition_prob(
                n_neighbors=min(N_NEIGHBORS, adata.n_obs - 1), knn_random=True,
                sampled_fraction=1, n_jobs=4, random_seed=SEED,
                calculate_randomized=True)
            oracle.calculate_embedding_shift(sigma_corr=0.05)
            shift = oracle.delta_embedding.copy()
            shift_random = oracle.delta_embedding_random.copy()
            shift_mag = np.linalg.norm(shift, axis=1)
            random_mag = np.linalg.norm(shift_random, axis=1)
            # 两种投影的向量差仅作描述；不能解释为去噪或扰动特异信号
            spec = shift - shift_random
            spec_mag = np.linalg.norm(spec, axis=1)

            emb = pd.DataFrame({
                'cell': oracle.adata.obs_names,
                'cluster': cell_tissue.reindex(oracle.adata.obs_names).to_numpy(),
                'umap_x': np.asarray(oracle.embedding)[:, 0],
                'umap_y': np.asarray(oracle.embedding)[:, 1],
                'embedding_shift_magnitude': shift_mag,
                'random_shift_magnitude': random_mag,
                'vector_difference_magnitude': spec_mag,
                'dx': shift[:, 0], 'dy': shift[:, 1],
                'dx_random': shift_random[:, 0], 'dy_random': shift_random[:, 1],
            })
            emb.to_csv(TAB / f'{tag}_embedding_shift.csv', index=False)

            gene_stat = pd.DataFrame({
                'gene': delta_x.columns,
                'mean_delta': delta_x.mean(axis=0).to_numpy(),
                'mean_abs_delta': delta_x.abs().mean(axis=0).to_numpy(),
                'frac_cells_up': (delta_x > 0).mean(axis=0).to_numpy(),
                'frac_cells_down': (delta_x < 0).mean(axis=0).to_numpy(),
            }).sort_values('mean_abs_delta', ascending=False)
            gene_stat.to_csv(TAB / f'{tag}_gene_effects.csv', index=False)

            by_cluster = emb.groupby('cluster')['embedding_shift_magnitude'] \
                            .agg(['count', 'mean', 'median']).reset_index()
            by_cluster.to_csv(TAB / f'{tag}_shift_by_cluster.csv', index=False)

            # 图：向量场。按组织来源给细胞和箭头分别上色，否则所有箭头同色、
            # 看不出哪一群细胞位移更大，图就没有信息量。
            umap = np.asarray(oracle.embedding)
            clus = cell_tissue.reindex(oracle.adata.obs_names).to_numpy()
            uniq = sorted(pd.unique(clus))
            palette = plt.get_cmap('tab20')
            cmap = {c: palette(i % 20) for i, c in enumerate(uniq)}
            arrow_colors = np.array([cmap[c] for c in clus])

            fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=140)
            ax.scatter(umap[:, 0], umap[:, 1], c='#dcdfe3', s=6, linewidths=0)
            for c in uniq:
                m = clus == c
                ax.scatter(umap[m, 0], umap[m, 1], s=9, color=cmap[c],
                           linewidths=0, label=f'{c} (n={int(m.sum())})')
            ax.quiver(umap[:, 0], umap[:, 1], shift[:, 0], shift[:, 1],
                      scale=30, headaxislength=7, headlength=11, headwidth=8,
                      linewidths=0.25, width=0.0045, edgecolors='k',
                      color=arrow_colors, alpha=1)
            ax.axis('off')
            ax.set_title(f'CellOracle：{gene} {"敲除" if mode == "ko" else "过表达"}后的细胞状态位移')
            ax.legend(fontsize=7.5, loc='center left', bbox_to_anchor=(1.01, 0.5),
                      frameon=False, markerscale=2.0, handletextpad=0.3,
                      labelspacing=0.4, title='组织来源', title_fontsize=8)
            fig.tight_layout(rect=(0, 0, 0.79, 1))
            fig.savefig(FIG / f'{tag}_quiver.png')
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7, 4.4), dpi=140)
            top = gene_stat.head(15).iloc[::-1]
            ax.barh(top['gene'], top['mean_delta'],
                    color=np.where(top['mean_delta'] >= 0, '#b94d48', '#3f6f9f'))
            ax.axvline(0, color='#444', lw=.8)
            ax.set(xlabel='Mean delta X (imputed expression units); selected by mean |delta X|',
                   title=f'{gene} {"KO" if mode == "ko" else "OE"}: top 15 affected genes')
            fig.tight_layout()
            fig.savefig(FIG / f'{tag}_top_genes.png')
            plt.close(fig)

            results[tag] = {
                'gene': gene, 'mode': mode, 'perturb_value': value,
                'delta_x_finite': bool(np.isfinite(delta_x.to_numpy()).all()),
                'delta_x_row_std_mean': float(np.mean(delta_x.std(axis=0))),
                'embedding_shift_finite': bool(np.isfinite(shift).all()),
                'embedding_shift_mean_magnitude': float(shift_mag.mean()),
                'random_shift_mean_magnitude': float(random_mag.mean()),
                'vector_difference_mean_magnitude': float(spec_mag.mean()),
                'target_gene_mean_delta': float(delta_x[gene].mean()),
                'target_cells_above_setting': int((imputed[gene] > value).sum()) if mode == 'oe' else 0,
                'oe_semantics': 'fixed target value; cells above this value decrease' if mode == 'oe' else None,
                'top_affected_genes': gene_stat.head(10)['gene'].tolist(),
                'status': ('result_available' if np.isfinite(delta_x.to_numpy()).all()
                           and np.isfinite(shift).all() and np.isfinite(shift_random).all()
                           else 'failed'),
            }
            print(f'[{tag}] value={value:g} shift_mag_mean={shift_mag.mean():.4f} '
                  f'random_mag_mean={random_mag.mean():.4f} '
                  f'vector_difference_mag_mean={spec_mag.mean():.4f} '
                  f'target_delta={delta_x[gene].mean():.4f}')
        except Exception as e:
            log_err(tag, e)
            results[tag] = {'gene': gene, 'mode': mode, 'status': 'failed',
                            'error': repr(e)}

# ----------------------------------------------------------- 6. 交付元数据
summary = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'condition'})
summary.to_csv(TAB / 'ko_oe_summary.csv', index=False)

validation = {
    'backend': 'CellOracle',
    'backend_version': co.__version__,
    'environment': 'D:/WorkBuddy/env/celloracle310 (Python 3.10)',
    'task': ['ko', 'oe'],
    'dataset': 'GSE198291 LUAD single cell, QC200',
    'cells': int(adata.n_obs),
    'genes_used': int(adata.n_vars),
    'grn_unit_column': GRN_UNIT_COL,
    'tissue_labels': sorted(cell_tissue.unique().tolist()),
    'base_grn': {'source': GRN_URL, 'tf': int(n_tf), 'target': int(n_tg),
                 'sha256_16': sha16(GRN_CACHE)},
    'grn_links_after_filter': n_links,
    'conditions': {k: v.get('status') for k, v in results.items()},
    'independent_validation': 'none',
    'notes': ('CellOracle 模拟的是调控网络上的表达状态位移，不是实测表达倍数；'
              '细胞类型未注释，用组织来源标签当 cluster；'
              'OE 表示在 imputed 表达尺度上把目标基因设为数据范围内的高值。'),
}
(OUT / 'validation.json').write_text(
    json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')

manifest = {
    'run_id': RUN_ID,
    'task': ['ko', 'oe'],
    'created': datetime.now().isoformat(timespec='seconds'),
    'backend': {'name': 'CellOracle', 'version': co.__version__},
    'inputs': {'h5ad': str(H5AD), 'h5ad_sha256_16': sha16(H5AD),
               'base_grn': str(GRN_CACHE)},
    'parameters': {'n_hvg': N_HVG, 'targets': TARGETS, 'n_pcs': N_PCS,
                   'n_neighbors': N_NEIGHBORS, 'n_propagation': 3,
                   'seed': SEED, 'alpha': 10, 'bagging_number': 20},
    'outputs': {'raw': sorted(p.name for p in RAW.iterdir()),
                'tables': sorted(p.name for p in TAB.iterdir()),
                'figures': sorted(p.name for p in FIG.iterdir())},
    'errors': len(errors),
}
(OUT / 'manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

with open(OUT / 'errors.jsonl', 'w', encoding='utf-8') as f:
    for e in errors:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')

print(json.dumps(validation, ensure_ascii=False, indent=2))
print(summary.to_string(index=False))
