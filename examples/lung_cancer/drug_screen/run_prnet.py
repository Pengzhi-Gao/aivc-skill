"""AIVC 药筛测试 - 后端 2/3：PRnet

输入：用户单细胞矩阵（GSE198291 LUAD）+ FDA 已上市药物库（2333 药 × 3 剂量）
输出：每个 (药物, 剂量) 的预测处理后表达与 LUAD 疾病方向逆转分数

PRnet 的预测是「对照表达 + 药物 FCFP4 指纹 × log10(剂量+1) → 处理后表达」。
接口可接外部基线，但当前 LINCS 权重对患者单细胞输入属于 OOD 探索，需核对预处理与完整基因空间。

不走 PRnetTrainer.test()：它的 Dataset.__getitem__ 用 list.index() 找配对对照，
条件数一多就是 O(n²)。这里直接构建模型输入做批量前向。
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RunDir, load_baseline, load_disease_direction,  # noqa: E402
                    reversal_scores, degeneracy_check, top_response_table, sha16, CONDITIONS, LIBRARY)

PRNET = Path(r'D:/WorkBuddy/模型论文复现/PRnet/PRnet-repro')
CHECKPOINT = PRNET / 'checkpoint' / 'lincs_best_epoch_all.pt'
LINCS_H5AD = PRNET / 'dataset' / 'lincs_subset.h5ad'
N_BASE = 25          # 每个条件抽多少个基线细胞
COND_CHUNK = 200     # 每次前向处理多少个条件
SEED = 2026

run = RunDir('fda_prnet')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
sys.path.insert(0, str(PRNET))

lib = pd.read_csv(LIBRARY)
cond = pd.read_csv(CONDITIONS)
print(f'[input] drugs={len(lib)} conditions={len(cond)}')

# 基因空间取自训练数据
import scanpy as sc  # noqa: E402
lincs = sc.read_h5ad(LINCS_H5AD)
gene_space = list(lincs.var_names)
assert len(gene_space) == 978
del lincs

baseline_df, cov_info = load_baseline(gene_space)
print(f'[baseline] {baseline_df.shape} coverage={cov_info["coverage_fraction"]}')
disease, dis_missing = load_disease_direction(gene_space)
print(f'[disease] non-zero={int((disease != 0).sum())} missing={dis_missing}')

# 训练数据进入模型时的尺度（用于核对基线量纲）
tr = sc.read_h5ad(LINCS_H5AD)
sc.pp.normalize_total(tr)
sc.pp.log1p(tr)
Xtr = tr.X.toarray() if hasattr(tr.X, 'toarray') else np.asarray(tr.X)
train_mean, train_max = float(Xtr.mean()), float(Xtr.max())
del tr
base_X = baseline_df.to_numpy(dtype=np.float32)
print(f'[scale] LINCS(model input) mean={train_mean:.4f} max={train_max:.4f} | '
      f'ours mean={base_X.mean():.4f} max={base_X.max():.4f}')

# ------------------------------------------------------------------ 模型
from data._utils import Drug_dose_encoder  # noqa: E402
from trainer.PRnetTrainer import PRnetTrainer  # noqa: E402

# 用一个极小的 AnnData 建出模型结构，再手动载入权重（避开 O(n^2) 的 Dataset）
dummy = ad.AnnData(
    X=np.zeros((4, 978), dtype=np.float32),
    obs=pd.DataFrame({
        'dose': [0.0, 0.0, 0.0, 1.0],
        'SMILES': ['C', 'C', 'C', 'C'],
        'cov_drug_name': ['ctrl_control'] * 3 + ['ctrl_drug'],
        'paired_control_index': ['c0', 'c1', 'c2', 'c0'],
        'aivc_split': ['test'] * 4,
    }, index=['c0', 'c1', 'c2', 't0']),
    var=pd.DataFrame(index=gene_space),
)
trainer = PRnetTrainer(
    dummy, batch_size=512, comb_num=1, split_key='aivc_split',
    model_save_dir=str(PRNET / 'checkpoint') + '/',
    results_save_dir=str(run.raw) + '/',
    x_dimension=978, hidden_layer_sizes=[128], z_dimension=64,
    adaptor_layer_sizes=[128], comb_dimension=64, drug_dimension=1024,
    dr_rate=0.05, n_genes=20, loss=['GUSS'], obs_key='cov_drug_name',
)
trainer.modelPGM.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
trainer.modelPGM.eval()
model = trainer.modelPGM.to(device)
print(f'[model] loaded {CHECKPOINT.name} on {device}')

# ------------------------------------------------------------------ 推理
n_base = min(N_BASE, base_X.shape[0])
base_sel = rng.choice(base_X.shape[0], size=n_base, replace=False)
base_sub = base_X[base_sel]
base_mean = base_sub.mean(axis=0)

smiles_list = cond['smiles'].tolist()
dose_list = cond['dose_uM'].tolist()
labels = Drug_dose_encoder(smiles_list, dose_list, comb_num=1)
print(f'[label] {labels.shape}')

preds = np.zeros((len(cond), len(gene_space)), dtype=np.float32)
with torch.no_grad():
    for start in range(0, len(cond), COND_CHUNK):
        end = min(start + COND_CHUNK, len(cond))
        m = end - start
        ctrl = torch.tensor(np.tile(base_sub, (m, 1)), dtype=torch.float32, device=device)
        # 条件标签要复制到每个基线细胞，batch 维才能对上
        lab = torch.tensor(np.repeat(labels[start:end], n_base, axis=0),
                           dtype=torch.float32, device=device)
        noise = torch.randn(m * n_base, 10, device=device)
        out = model(ctrl, lab, noise)
        mean = out[:, :out.shape[1] // 2]
        preds[start:end] = mean.reshape(m, n_base, -1).mean(dim=1).cpu().numpy()
        if start % (COND_CHUNK * 10) == 0:
            print(f'  [predict] {end}/{len(cond)}')

resp = preds - base_mean[None, :]
condition_meta = cond.copy()
condition_meta = run.save_predictions(condition_meta, gene_space, preds, base_mean, resp)
pd.DataFrame({'gene': gene_space, 'disease_effect': disease,
              'used_for_scoring': np.isfinite(disease)}).to_csv(
    run.tables / 'disease_gene_mask.csv', index=False)
cos, corr, reversal = run.score_response(resp, disease, 'PRnet')
deg = run.condition_quality(condition_meta, resp, reversal)
print(f'[degeneracy] {deg}')

res = cond.copy()
res['pred_response_l2'] = np.linalg.norm(resp, axis=1)
res['cos_pred_vs_disease'] = cos
res['corr_pred_vs_disease'] = corr
res['reversal_score'] = reversal
res.insert(0, 'condition_id', condition_meta['condition_id'].to_numpy())
res['prediction_status'] = np.where(np.isfinite(resp).all(axis=1), 'result_available', 'failed')
res['ranking_status'] = deg['ranking_status']
res.to_csv(run.tables / 'conditions.csv', index=False)

drug = res.loc[np.isfinite(res['reversal_score']) & (res['prediction_status'] == 'result_available')].groupby('drug').agg(
    smiles=('smiles', 'first'),
    n_conditions=('dose_uM', 'size'),
    reversal_score_mean=('reversal_score', 'mean'),
    reversal_score_max=('reversal_score', 'max'),
    cos_mean=('cos_pred_vs_disease', 'mean'),
    pred_response_absmean=('pred_response_l2', 'mean'),
).reset_index().sort_values('reversal_score_mean', ascending=False).reset_index(drop=True)
drug.insert(0, 'rank', np.arange(1, len(drug) + 1) if deg['ranking_status'] == 'available'
            else pd.array([pd.NA] * len(drug), dtype='Int64'))
drug['ranking_status'] = deg['ranking_status']
drug = drug.merge(lib[['drug', 'mechanism', 'target']], on='drug', how='left')
drug.to_csv(run.tables / 'drug_ranking.csv', index=False)

pg = top_response_table(res, resp, gene_space, drug['drug'].tolist())
pg.to_csv(run.tables / 'top20_drug_per_gene_predicted_response.csv', index=False)

if np.isfinite(reversal).any():
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=140)
    t = drug.head(20).iloc[::-1]
    ax.barh(t['drug'], t['reversal_score_mean'], color='#3f6f9f')
    ax.axvline(0, color='#444', lw=.8)
    ax.set(xlabel='Reversal score  (-cos(predicted response, LUAD direction))',
           title='PRnet: top 20 FDA drugs by LUAD-signature reversal')
    if deg['ranking_status'] != 'available':
        ax.set_title('Diagnostic only: ranking unresolved / unavailable')
    fig.tight_layout()
    fig.savefig(run.figures / 'drug_reversal_ranking.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=140)
    ax.hist(res['reversal_score'].dropna(), bins=60, color='#6b8f71')
    ax.axvline(0, color='#444', lw=.8)
    ax.set(xlabel='Reversal score', ylabel='conditions',
           title=f'PRnet: reversal score distribution ({len(res)} conditions)')
    fig.tight_layout()
    fig.savefig(run.figures / 'reversal_distribution.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=140)
    for d, grp in res.groupby('dose_uM'):
        ax.hist(grp['reversal_score'].dropna(), bins=50, alpha=.5, label=f'{d} µM')
    ax.axvline(0, color='#444', lw=.8)
    ax.legend()
    ax.set(xlabel='Reversal score', ylabel='conditions', title='PRnet: reversal score by dose')
    fig.tight_layout()
    fig.savefig(run.figures / 'reversal_by_dose.png')
    plt.close(fig)

validation = {
    'backend': 'PRnet',
    'checkpoint': str(CHECKPOINT),
    'checkpoint_sha256_16': sha16(CHECKPOINT),
    'task': 'drug',
    'cell_background': 'user scRNA baseline (LUAD), not a LINCS cell line',
    'drugs_scored': int(drug.shape[0]),
    'conditions_scored': int(len(res)),
    'genes': len(gene_space),
    'baseline_coverage': cov_info,
    'baseline_scale_check': {
        'lincs_model_input_mean': train_mean,
        'lincs_model_input_max': train_max,
        'our_baseline_mean': float(base_mean.mean()),
        'our_baseline_max': float(base_mean.max()),
    },
    'dose_series_uM': sorted(res['dose_uM'].unique().tolist()),
    'prediction_finite': bool(np.isfinite(preds).all()),
    'degenerate_output': bool(np.abs(resp).max() < 1e-6),
    'reversal_score_range': [float(res['reversal_score'].min()),
                             float(res['reversal_score'].max())],
    'degeneracy_check': deg,
    'independent_validation': 'none',
    'status': deg['prediction_status'],
    'ranking_status': deg['ranking_status'],
    'validation_status': 'unvalidated',
}
manifest = {
    'task': 'drug',
    'backend': {'name': 'PRnet', 'checkpoint': CHECKPOINT.name},
    'inputs': {'library': str(LIBRARY), 'library_sha256_16': sha16(LIBRARY)},
    'parameters': {'n_base_cells': int(n_base), 'seed': SEED, 'device': str(device)},
}
run.finish(manifest, validation)
print(json.dumps(validation, ensure_ascii=False, indent=2))
print(drug.head(10)[['rank', 'drug', 'reversal_score_mean', 'mechanism']].to_string(index=False))
