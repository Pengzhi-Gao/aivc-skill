"""AIVC 测试 - 药筛分支端到端运行

后端：PRnet（LINCS L1000 预训练权重）
细胞背景：A549（人肺腺癌）
疾病参照：GSE81089 NSCLC vs 正常肺（由 prepare_luad_bulk.py 产出）

流程：
  1. 从 LINCS 子集里取出 A549 全部条件，构造只含 test split 的 AnnData
  2. 用 PRnet 预训练权重推理，得到每个条件的预测表达
  3. 预测响应 = 预测表达 - 配对对照基线
  4. 用疾病方向向量的负余弦/负相关给药物排序（药把基因往病相反方向推 = 高分）
  5. 落盘 raw / tables / figures / validation / errors / manifest

注意：这里只报告模型实际算出的数值和运行状态，不给出疗效或靶点结论。
"""
from pathlib import Path
import json
import sys
import traceback
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'drug_screen'))
from common import RunDir, load_disease_direction, top_response_table
PRNET = Path(r'D:/WorkBuddy/模型论文复现/PRnet/PRnet-repro')
run = RunDir('a549_prnet_screen', base_dir=ROOT)
RUN_ID = run.run_id
OUT = run.root
RAW = OUT / 'raw'
TAB = OUT / 'tables'
FIG = OUT / 'figures'
for d in (RAW, TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

SPLIT_KEY = 'aivc_split'
CELL = 'A549'
CHECKPOINT = PRNET / 'checkpoint' / 'lincs_best_epoch_all.pt'
DATA_H5AD = PRNET / 'dataset' / 'lincs_subset.h5ad'
SIG_CSV = ROOT / 'results' / 'geo_gse81089' / 'disease_direction_vector_978.csv'

errors = []


def log_err(stage, exc):
    errors.append({'stage': stage, 'error': repr(exc),
                   'traceback': traceback.format_exc()[-1500:]})
    print(f'[ERROR] {stage}: {exc!r}')


# ------------------------------------------------------------------ 1. 取子集
a = sc.read_h5ad(DATA_H5AD)
a.obs['cell_type'] = a.obs['cell_type'].astype(str)
sub = a[a.obs['cell_type'] == CELL].copy()
print(f'[{CELL}] profiles={sub.n_obs} genes={sub.n_vars}')
assert sub.n_vars == 978

# 配对对照必须留在同一份 AnnData 里
missing = set(sub.obs['paired_control_index'].astype(str)) - set(sub.obs.index.astype(str))
assert not missing, f'{len(missing)} paired controls missing from the {CELL} subset'

sub.obs[SPLIT_KEY] = 'test'   # 全部走 test，train/valid 为空 -> 不构建训练集
sub.write_h5ad(RAW / f'{CELL}_screening_input.h5ad')

# ------------------------------------------------------ 2. 载入模型并推理
sys.path.insert(0, str(PRNET))
import torch
from trainer.PRnetTrainer import PRnetTrainer  # noqa: E402

sc.pp.normalize_total(sub)
sc.pp.log1p(sub)

trainer = PRnetTrainer(
    sub, batch_size=512, comb_num=1, split_key=SPLIT_KEY,
    model_save_dir=str(PRNET / 'checkpoint') + '/',
    results_save_dir=str(RAW) + '/',
    x_dimension=978, hidden_layer_sizes=[128], z_dimension=64,
    adaptor_layer_sizes=[128], comb_dimension=64, drug_dimension=1024,
    dr_rate=0.05, n_genes=20, loss=['GUSS'], obs_key='cov_drug_name',
)
trainer.predict_mean = True   # 取分布均值，去掉单次采样噪声
trainer.test(str(CHECKPOINT))

# cov_drug_name 里可能含逗号，按行读，不要用 csv 解析
with open(RAW / f'{SPLIT_KEY}_cov_drug_array.csv', encoding='utf-8') as f:
    cov = np.array([ln.rstrip('\n') for ln in f if ln.strip() != ''])
y_pre = np.loadtxt(RAW / f'{SPLIT_KEY}_y_pre_array.csv', delimiter=',')
x_true = np.loadtxt(RAW / f'{SPLIT_KEY}_x_true_array.csv', delimiter=',')
y_true = np.loadtxt(RAW / f'{SPLIT_KEY}_y_true_array.csv', delimiter=',')
assert cov.shape[0] == y_pre.shape[0] == x_true.shape[0] == y_true.shape[0]
print(f'[inference] conditions={cov.shape[0]} genes={y_pre.shape[1]}')

genes = list(sub.var_names)

# ------------------------------------------------------------- 3. 预测响应
pred_resp = y_pre - x_true          # 模型预测的扰动响应
obs_resp = y_true - x_true          # 配对对照差（用于内部一致性核对）

cond = pd.DataFrame({'cov_drug_name': cov})
cond['cell'] = cond['cov_drug_name'].str.split('_').str[0]
cond['drug'] = cond['cov_drug_name'].str.split('_').str[1]

# 从 obs 取回剂量 / 时间 / 观察响应
meta = sub.obs.copy()
meta['cov_drug_name'] = meta['cov_drug_name'].astype(str)
meta = meta[meta['dose'].astype(float) != 0.0]
meta = meta.reset_index(drop=True)
assert len(meta) == len(cov), f'obs/cov length mismatch {len(meta)} vs {len(cov)}'
assert np.array_equal(meta['cov_drug_name'].to_numpy(), cov), 'condition order mismatch; join by identity before scoring'
cond['dose'] = meta['dose'].to_numpy(dtype=float)
cond['pert_time'] = meta['pert_time'].to_numpy(dtype=float)
cond['smiles'] = meta['SMILES'].to_numpy()
cond['pert_id'] = meta['pert_id'].astype(str).to_numpy()
cond['n_genes'] = len(genes)

# 每行的响应强度（用于覆盖性/退化检查）
cond['pred_resp_l2'] = np.linalg.norm(pred_resp, axis=1)
cond['pred_resp_absmean'] = np.abs(pred_resp).mean(axis=1)
cond['obs_resp_absmean'] = np.abs(obs_resp).mean(axis=1)

# --------------------------------------------------- 4. 疾病方向逆转评分
disease, dis_missing = load_disease_direction(genes)
condition_meta = run.save_predictions(cond, genes, y_pre, x_true, pred_resp)
cond.insert(0, 'condition_id', condition_meta['condition_id'].to_numpy())
cos_all, corr_all, reversal = run.score_response(pred_resp, disease, 'PRnet')
deg = run.condition_quality(condition_meta, pred_resp, reversal)
pd.DataFrame({'gene': genes, 'disease_effect': disease,
              'used_for_scoring': np.isfinite(disease)}).to_csv(TAB / 'disease_gene_mask.csv', index=False)


cond['cos_pred_vs_disease_common_genes'] = cos_all
cond['corr_pred_vs_disease_common_genes'] = corr_all
cond['reversal_score'] = reversal
cond['ranking_status'] = deg['ranking_status']

# 内部一致性：预测响应与实测响应在同一基因空间上的相关
agree = []
for i in range(pred_resp.shape[0]):
    p, o = pred_resp[i], obs_resp[i]
    if p.std() > 0 and o.std() > 0:
        agree.append(float(np.corrcoef(p, o)[0, 1]))
    else:
        agree.append(np.nan)
cond['pred_obs_response_corr'] = agree

cond.to_csv(TAB / 'a549_conditions.csv', index=False)

# --------------------------------------------------------- 5. 药物层面汇总
g = cond.loc[np.isfinite(cond['reversal_score'])].groupby('drug')
drug = pd.DataFrame({
    'n_conditions': g.size(),
    'n_doses': g['dose'].nunique(),
    'dose_min': g['dose'].min(),
    'dose_max': g['dose'].max(),
    'reversal_score_mean': g['reversal_score'].mean(),
    'reversal_score_max': g['reversal_score'].max(),
    'cos_pred_vs_disease_mean': g['cos_pred_vs_disease_common_genes'].mean(),
    'pred_resp_absmean_mean': g['pred_resp_absmean'].mean(),
    'pred_obs_response_corr_mean': g['pred_obs_response_corr'].mean(),
}).reset_index()
drug = drug.sort_values('reversal_score_mean', ascending=False).reset_index(drop=True)
drug.insert(0, 'rank', np.arange(1, len(drug) + 1) if deg['ranking_status'] == 'available'
            else pd.array([pd.NA] * len(drug), dtype='Int64'))
drug['ranking_status'] = deg['ranking_status']
drug.to_csv(TAB / 'a549_drug_ranking.csv', index=False)

# 排名靠前药物的逐基因预测响应
top_drugs = drug['drug'].head(10).tolist()
top_rows = cond[cond['drug'].isin(top_drugs)]
idx = top_rows.index.to_numpy()
per_gene = pd.DataFrame(pred_resp[idx], columns=genes)
per_gene.insert(0, 'cov_drug_name', cond.loc[idx, 'cov_drug_name'].to_numpy())
per_gene.insert(1, 'drug', cond.loc[idx, 'drug'].to_numpy())
per_gene.insert(2, 'dose', cond.loc[idx, 'dose'].to_numpy())
per_gene.to_csv(TAB / 'a549_top10_drug_per_gene_predicted_response.csv', index=False)

# ------------------------------------------------------------------- 6. 图
fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=140)
top = drug.head(20).iloc[::-1]
ax.barh(top['drug'], top['reversal_score_mean'], color='#3f6f9f')
ax.set(xlabel='Reversal score  (-cos(predicted response, LUAD direction))',
       title=f'PRnet / {CELL}: top 20 drugs by LUAD-signature reversal')
ax.axvline(0, color='#444', lw=.8)
if deg['ranking_status'] != 'available':
    ax.set_title('Diagnostic only: ranking unresolved / unavailable')
fig.tight_layout()
fig.savefig(FIG / 'a549_drug_reversal_ranking.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=140)
ax.scatter(cond['pred_obs_response_corr'], cond['reversal_score'], s=6,
           c='#b94d48', alpha=.45)
ax.set(xlabel='corr(predicted response, observed response) per condition',
       ylabel='Reversal score',
       title=f'PRnet / {CELL}: internal agreement vs reversal score')
ax.axhline(0, color='#444', lw=.8)
fig.tight_layout()
fig.savefig(FIG / 'a549_agreement_vs_reversal.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.6, 5.6), dpi=140)
ax.hist(cond['reversal_score'].dropna(), bins=60, color='#6b8f71')
ax.set(xlabel='Reversal score', ylabel='conditions',
       title=f'PRnet / {CELL}: reversal score distribution ({len(cond)} conditions)')
fig.tight_layout()
fig.savefig(FIG / 'a549_reversal_distribution.png')
plt.close(fig)

# -------------------------------------------------------------- 7. 交付元数据
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


finite = bool(np.isfinite(y_pre).all())
row_std = float(np.mean(pred_resp.std(axis=0)))
validation = {
    'backend': 'PRnet',
    'checkpoint': str(CHECKPOINT),
    'checkpoint_sha256_16': sha(CHECKPOINT),
    'task': 'drug',
    'cell_background': CELL,
    'conditions_scored': int(len(cond)),
    'drugs_scored': int(drug.shape[0]),
    'genes': int(len(genes)),
    'prediction_finite': finite,
    'prediction_row_std_mean': row_std,
    'response_status': deg['response_status'],
    'internal_agreement': {
        'metric': 'per-condition Pearson corr between predicted and observed response',
        'mean': float(np.nanmean(cond['pred_obs_response_corr'])),
        'median': float(np.nanmedian(cond['pred_obs_response_corr'])),
        'note': 'LINCS profiles; training/test overlap not established by relabeling the inference split',
    },
    'status': deg['prediction_status'],
    'ranking_status': deg['ranking_status'],
    'degeneracy_check': deg,
    'validation_status': 'unvalidated',
    'independent_validation': 'none',
}
# Validation and provenance are written together by RunDir.finish below.

manifest = {
    'run_id': RUN_ID,
    'task': 'drug',
    'created': datetime.now().isoformat(timespec='seconds'),
    'backend': {'name': 'PRnet', 'version': 'lincs_best_epoch_all.pt',
                'workspace': str(PRNET)},
    'inputs': {
        'lincs_subset': {'path': str(DATA_H5AD), 'sha256_16': sha(DATA_H5AD)},
        'disease_reference': str(SIG_CSV),
        'cell_line': CELL,
    },
    'parameters': {'batch_size': 512, 'predict_mean': True,
                   'obs_key': 'cov_drug_name', 'split_key': SPLIT_KEY,
                   'seed': 2024, 'device': str(trainer.device)},
    'outputs': {
        'raw': sorted(p.name for p in RAW.iterdir()),
        'tables': sorted(p.name for p in TAB.iterdir()),
        'figures': sorted(p.name for p in FIG.iterdir()),
    },
    'errors': len(errors),
}
run.errors.extend(errors)

run.finish(manifest, validation)
print(json.dumps(validation, ensure_ascii=False, indent=2))
print('\ntop 10 by reversal:')
print(drug.head(10)[['rank', 'drug', 'n_conditions', 'reversal_score_mean',
                     'pred_obs_response_corr_mean']].to_string(index=False))
