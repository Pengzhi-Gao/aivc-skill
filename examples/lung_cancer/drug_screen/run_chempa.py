"""AIVC 药筛测试 - 后端 1/3：chemCPA

输入：用户单细胞矩阵（GSE198291 LUAD，703 细胞）+ FDA 已上市药物库（2333 药 × 3 剂量）
输出：每个 (药物, 剂量) 的预测处理后表达、预测响应、与 LUAD 疾病方向的逆转分数

chemCPA 的 predict 接口本身就是「基线表达 + 药物 + 剂量 → 预测处理后表达」，
接口可接外部基线，但患者单细胞输入与 A549 协变量的组合属于 OOD 探索；
不能因癌种相同就视为训练域内样本，且须核对尺度、协变量与完整基因空间。

药物表征：训练 parquet 只覆盖 768 个 FDA 药物，其余 1565 个由
solve_rdkit_norm.py 反解出的标准化参数补齐（见该脚本注释）。
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RunDir, load_baseline, load_disease_direction,  # noqa: E402
                    reversal_scores, degeneracy_check, top_response_table, sha16, DATA, CONDITIONS, LIBRARY)

CHEMPA = Path(r'D:/WorkBuddy/Work_推送/2610/261002 ChemCPA药物扰动/chemCPA_repro')
PROJECT = CHEMPA / 'project_folder'
EMB_PARQUET = DATA / 'rdkit_embedding_fda_extended.parquet'
CONFIG_JSON = CHEMPA / 'notebooks/finetuning_num_genes.json'
MODEL_HASH = '27b401db1845eea26c102fb614df9c33'
CELL_TYPE = 'A549'
N_BASE = 200
DRUG_CHUNK = 20
SEED = 2026

run = RunDir('fda_chempa')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------- 输入
lib = pd.read_csv(LIBRARY)
cond = pd.read_csv(CONDITIONS)
print(f'[input] drugs={len(lib)} conditions={len(cond)}')

sys.path.insert(0, str(CHEMPA))
sys.path.insert(0, str(CHEMPA / 'notebooks'))
import types
_fake = types.ModuleType('seml')
_fake.database = types.SimpleNamespace(get_collection=lambda *a, **k: None)
_fake.get_results = lambda *a, **k: None
sys.modules.setdefault('seml', _fake)

import chemCPA.paths as chem_paths  # noqa: E402
import utils as official_utils  # noqa: E402
# load_model 里是 from chemCPA.paths import CHECKPOINT_DIR，绑在 utils 自己的命名空间
official_utils.CHECKPOINT_DIR = PROJECT / 'checkpoints'
chem_paths.CHECKPOINT_DIR = PROJECT / 'checkpoints'

cfg_all = json.loads(CONFIG_JSON.read_text())
config = next(c['config'] for c in cfg_all if c['config_hash'] == MODEL_HASH)
config = json.loads(json.dumps(config))
config['config_hash'] = MODEL_HASH
dp = dict(config['dataset']['data_params'])
dp['dataset_path'] = str(PROJECT / 'datasets' / Path(dp['dataset_path']).name)
config['dataset']['data_params'] = dp
config['model']['embedding']['directory'] = str(EMB_PARQUET)

# 基因空间：用训练数据集的基因顺序
import anndata as ad  # noqa: E402
train_adata = ad.read_h5ad(dp['dataset_path'])
gene_space = list(train_adata.var_names)
config['dataset']['n_vars'] = len(gene_space)
print(f'[gene space] {len(gene_space)} genes')

baseline_df, cov_info = load_baseline(gene_space)
print(f'[baseline] {baseline_df.shape} coverage={cov_info["coverage_fraction"]}')
disease, dis_missing = load_disease_direction(gene_space)
print(f'[disease] non-zero entries={int((disease != 0).sum())} '
      f'missing_in_vector={dis_missing}')

# 训练集对照的尺度参考，用于核对基线是否在同一量纲
ctrl_train = train_adata.obs['condition'].astype(str).eq('control').to_numpy()
Xtr = train_adata.X
Xtr = Xtr.toarray() if hasattr(Xtr, 'toarray') else np.asarray(Xtr)
train_ctrl_mean = float(Xtr[ctrl_train].mean())
train_ctrl_max = float(Xtr[ctrl_train].max())
print(f'[scale] train control mean={train_ctrl_mean:.4f} max={train_ctrl_max:.4f} | '
      f'our baseline mean={float(baseline_df.to_numpy().mean()):.4f} '
      f'max={cov_info["value_max"]:.4f}')

# 细胞类型协变量的类别顺序
from chemCPA.data import load_dataset_splits  # noqa: E402
datasets = load_dataset_splits(**dp, return_dataset=False)
cov_names = datasets['ood'].covariate_names['cell_type']
print(f'[covariate] cell_type order = {cov_names}')
assert CELL_TYPE in list(cov_names)
cl_idx = list(cov_names).index(CELL_TYPE)

# 药物表征（覆盖全部 FDA 药物）
from chemCPA.embedding import get_chemical_representation  # noqa: E402
drug_smiles = lib['smiles'].tolist()
emb_table = pd.read_parquet(EMB_PARQUET)
missing_embeddings = [s for s in drug_smiles if s not in emb_table.index]
if missing_embeddings:
    raise ValueError(f'{len(missing_embeddings)} missing drug embeddings; zero fallback prohibited')
embedding = get_chemical_representation(
    smiles=drug_smiles, embedding_model='rdkit',
    data_path=EMB_PARQUET, device=device)
print(f'[drug embedding] {tuple(embedding.weight.shape)}')
assert embedding.weight.shape[0] == len(drug_smiles)

# 模型
model, _ = official_utils.load_model(config, drug_smiles)
model = model.to(device).eval()

# ---------------------------------------------------------------- 推理
base_X = baseline_df.to_numpy(dtype=np.float32)
n_base = min(N_BASE, base_X.shape[0])
base_sel = rng.choice(base_X.shape[0], size=n_base, replace=False)
base_sub = torch.tensor(base_X[base_sel], dtype=torch.float32, device=device)
base_mean = base_X[base_sel].mean(axis=0)

dose_nM = (cond['dose_uM'] * 1000.0).to_numpy(dtype=np.float32)
smiles_to_idx = {s: i for i, s in enumerate(drug_smiles)}
drug_idx_all = np.array([smiles_to_idx[s] for s in cond['smiles']], dtype=np.int64)

preds = np.zeros((len(cond), len(gene_space)), dtype=np.float32)
with torch.no_grad():
    for start in range(0, len(cond), DRUG_CHUNK * 3):
        end = min(start + DRUG_CHUNK * 3, len(cond))
        m = end - start
        genes = base_sub.repeat(m, 1)
        # 每个条件对 n_base 个基线细胞各预测一次，药物/剂量要复制到逐行
        didx = torch.tensor(np.repeat(drug_idx_all[start:end], n_base),
                            dtype=torch.long, device=device).unsqueeze(1)
        doses = torch.tensor(np.repeat(dose_nM[start:end], n_base),
                             dtype=torch.float32, device=device).unsqueeze(1)
        cov = torch.zeros((m * n_base, len(cov_names)), dtype=torch.float32, device=device)
        cov[:, cl_idx] = 1.0
        out, _ = model.predict(genes=genes, drugs_idx=didx, dosages=doses,
                               covariates=[cov])
        mean = out[:, :out.shape[1] // 2]
        preds[start:end] = mean.reshape(m, n_base, -1).mean(dim=1).cpu().numpy()
        if start % (DRUG_CHUNK * 3 * 40) == 0:
            print(f'  [predict] {end}/{len(cond)}')

resp = preds - base_mean[None, :]
condition_meta = cond.copy()
condition_meta = run.save_predictions(condition_meta, gene_space, preds, base_mean, resp)
pd.DataFrame({'gene': gene_space, 'disease_effect': disease,
              'used_for_scoring': np.isfinite(disease)}).to_csv(
    run.tables / 'disease_gene_mask.csv', index=False)
cos, corr, reversal = run.score_response(resp, disease, 'chemCPA')
deg = run.condition_quality(condition_meta, resp, reversal)
print(f'[degeneracy] {deg}')

res = cond.copy()
res['pred_response_l2'] = np.linalg.norm(resp, axis=1)
res['cos_pred_vs_disease'] = cos
res['corr_pred_vs_disease'] = corr
res['reversal_score'] = reversal
res['baseline_mean_expr'] = base_mean.mean()
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

# 逐药全基因预测响应（Top 20）
pg = top_response_table(res, resp, gene_space, drug['drug'].tolist())
pg.to_csv(run.tables / 'top20_drug_per_gene_predicted_response.csv', index=False)

# ---------------------------------------------------------------- 图
if np.isfinite(reversal).any():
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=140)
    t = drug.head(20).iloc[::-1]
    ax.barh(t['drug'], t['reversal_score_mean'], color='#3f6f9f')
    ax.axvline(0, color='#444', lw=.8)
    ax.set(xlabel='Reversal score  (-cos(predicted response, LUAD direction))',
           title=f'chemCPA / {CELL_TYPE}: top 20 FDA drugs by LUAD-signature reversal')
    if deg['ranking_status'] != 'available':
        ax.set_title('Diagnostic only: ranking unresolved / unavailable')
    fig.tight_layout()
    fig.savefig(run.figures / 'drug_reversal_ranking.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=140)
    ax.hist(res['reversal_score'].dropna(), bins=60, color='#6b8f71')
    ax.axvline(0, color='#444', lw=.8)
    ax.set(xlabel='Reversal score', ylabel='conditions',
           title=f'chemCPA / {CELL_TYPE}: reversal score distribution ({len(res)} conditions)')
    fig.tight_layout()
    fig.savefig(run.figures / 'reversal_distribution.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=140)
    for d, grp in res.groupby('dose_uM'):
        ax.hist(grp['reversal_score'].dropna(), bins=50, alpha=.5, label=f'{d} µM')
    ax.axvline(0, color='#444', lw=.8)
    ax.legend()
    ax.set(xlabel='Reversal score', ylabel='conditions',
           title='chemCPA: reversal score by dose')
    fig.tight_layout()
    fig.savefig(run.figures / 'reversal_by_dose.png')
    plt.close(fig)

    # ---------------------------------------------------------------- 元数据
validation = {
    'backend': 'chemCPA',
    'model_hash': MODEL_HASH,
    'checkpoint': str(PROJECT / 'checkpoints' / (MODEL_HASH + '.pt')),
    'task': 'drug',
    'cell_background': CELL_TYPE,
    'input_scrna': str(Path(__file__).resolve().parent.parent /
                       'results/geo_gse198291/GSE198291_counts_qc200.h5ad'),
    'drug_library': str(LIBRARY),
    'drugs_scored': int(drug.shape[0]),
    'conditions_scored': int(len(res)),
    'genes': len(gene_space),
    'baseline_coverage': cov_info,
    'baseline_scale_check': {
        'train_control_mean': train_ctrl_mean,
        'train_control_max': train_ctrl_max,
        'our_baseline_mean': float(base_mean.mean()),
        'our_baseline_max': float(base_mean.max()),
        'same_order_of_magnitude': bool(abs(np.log10(
            (base_mean.mean() + 1e-9) / (train_ctrl_mean + 1e-9))) < 1),
    },
    'drug_embedding': {
        'parquet': str(EMB_PARQUET),
        'note': 'training parquet covers 768 of 2333 FDA drugs; the rest computed '
                'with normalization parameters solved from the training parquet',
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
if deg['ranking_status'] != 'available':
    validation['ranking_note'] = '评分区分度不足或无有效评分；保留预测，评分图仅供诊断。'
manifest = {
    'task': 'drug',
    'backend': {'name': 'chemCPA', 'model_hash': MODEL_HASH},
    'inputs': {'scrna': str(SCRNA_ := str(Path(__file__).resolve().parent.parent /
                                         'results/geo_gse198291/GSE198291_counts_qc200.h5ad')),
               'library': str(LIBRARY), 'library_sha256_16': sha16(LIBRARY),
               'embedding_parquet': str(EMB_PARQUET)},
    'parameters': {'n_base_cells': int(n_base), 'seed': SEED,
                   'dose_nM': sorted(set(dose_nM.tolist())),
                   'cell_type_covariate': CELL_TYPE},
}
run.finish(manifest, validation)
print(json.dumps(validation, ensure_ascii=False, indent=2))
print(drug.head(10)[['rank', 'drug', 'reversal_score_mean', 'mechanism']].to_string(index=False))
