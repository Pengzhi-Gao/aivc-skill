"""AIVC 药筛测试 - 后端 3/3：UniPert-G2CP

输入：FDA 已上市药物库（2333 药）
输出：每个药物的预测处理后表达与 LUAD 疾病方向逆转分数

重要限制（必须如实记录）：G2CP 的 new_predict 用的是模型自带 LINCS-A549
对照细胞（每次随机抽 300 个），**不接受外部基线矩阵**，也没有剂量维度。
所以这个后端吃不到用户的单细胞矩阵，只能改变药物那一侧。
"""
from pathlib import Path
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RunDir, load_disease_direction, reversal_scores,  # noqa: E402
                    degeneracy_check, sha16, LIBRARY)

G2CP_ROOT = Path(r'D:/WorkBuddy/Work_推送/2610/261003 UniPert-G2CP')
UNIPERT_DIR = G2CP_ROOT / 'official_unipert'
GEARS_ROOT = G2CP_ROOT / 'official_reproduce/G2CP/backbone_model/GEARS'
DATA_PATH = G2CP_ROOT / 'official_reproduce/G2CP/data'
MODEL_PATH = (G2CP_ROOT / 'official_reproduce/G2CP/saved_models/LINCS_ts/LINCS-A549'
              / 'new_pretrain-finetune-ts_0_0.0')
REPS_CACHE = Path(__file__).resolve().parent / 'data' / 'unipert_fda_reps.pkl'
SEED = 2026

run = RunDir('fda_unipert_g2cp')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

lib = pd.read_csv(LIBRARY)
print(f'[input] drugs={len(lib)}')

# ---------------------------------------------------------- 1. UniPert 表征
if REPS_CACHE.exists():
    with open(REPS_CACHE, 'rb') as f:
        reps = pickle.load(f)
    print(f'[unipert] loaded cached reps: {len(reps)}')
else:
    sys.path.insert(0, str(UNIPERT_DIR))
    from unipert import UniPert
    up = UniPert()
    t0 = time.time()
    compound_dict = dict(zip(lib['drug'], lib['smiles']))
    out_embs, invalid = up.encode_compounds(compound_dict=compound_dict)
    print(f'[unipert] encoded {len(out_embs)} compounds in {time.time()-t0:.1f}s, '
          f'invalid={len(invalid) if invalid is not None else 0}')
    reps = {k: np.asarray(v).reshape(-1) for k, v in out_embs.items()}
    with open(REPS_CACHE, 'wb') as f:
        pickle.dump(reps, f)
print(f'[unipert] embedding dim = {len(next(iter(reps.values())))}')

# ---------------------------------------------------------- 2. G2CP 推理
sys.path.insert(0, str(GEARS_ROOT))
from gears import PertData, GEARS  # noqa: E402

pert_data = PertData(data_path=str(DATA_PATH))
pert_data.load(data_name='LINCS-A549', pert_type_col_name='pert_type',
               perturbagen_col_name='condition', control_key='ctrl')
pert_data.prepare_split(split='pretrain-finetune-ts', seed=0,
                        train_ptbg_set_frac=0.0, val_ptbg_set_frac=0.1)
pert_data.get_dataloader(batch_size=32, test_batch_size=128)
gene_space = list(pert_data.adata.var_names)
print(f'[g2cp] gene space {len(gene_space)}')

model = GEARS(pert_data, device=device, weight_bias_track=False,
              proj_name='LINCS-A549', exp_name='aivc_fda',
              save_path=str(MODEL_PATH))
model.load_pretrained(str(MODEL_PATH), 'finetune_model')

np.random.seed(SEED)
torch.manual_seed(SEED)
ctrl = pert_data.adata[pert_data.adata.obs['condition'] == 'ctrl']
ctrl_mean = np.asarray(ctrl.X.mean(axis=0)).reshape(-1)
print(f'[g2cp] control cells={ctrl.n_obs} mean={ctrl_mean.mean():.4f}')

query = [d for d in lib['drug'] if d in reps]
print(f'[g2cp] predicting {len(query)} compounds ...')
t0 = time.time()
preds_dict = model.new_predict(query, reps)
print(f'[g2cp] done in {time.time()-t0:.1f}s, predicted={len(preds_dict)}')

names = list(preds_dict.keys())
preds = np.vstack([np.asarray(preds_dict[n]).reshape(-1) for n in names])
resp = preds - ctrl_mean[None, :]

disease, dis_missing = load_disease_direction(gene_space)
cos, corr, reversal = reversal_scores(resp, disease)
deg = degeneracy_check(resp, reversal)
print(f'[degeneracy] {deg}')

res = pd.DataFrame({
    'drug': names,
    'pred_response_l2': np.linalg.norm(resp, axis=1),
    'cos_pred_vs_disease': cos,
    'corr_pred_vs_disease': corr,
    'reversal_score': reversal,
})
res = res.merge(lib[['drug', 'smiles', 'mechanism', 'target']], on='drug', how='left')
res.to_csv(run.tables / 'conditions.csv', index=False)

drug = res.sort_values('reversal_score', ascending=False).reset_index(drop=True)
drug.insert(0, 'rank', np.arange(1, len(drug) + 1))
drug.to_csv(run.tables / 'drug_ranking.csv', index=False)

sel = np.arange(min(20, len(names)))
pg = pd.DataFrame(resp[sel], columns=gene_space)
pg.insert(0, 'drug', [names[i] for i in sel])
pg.to_csv(run.tables / 'top20_drug_per_gene_predicted_response.csv', index=False)

fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=140)
t = drug.head(20).iloc[::-1]
ax.barh(t['drug'], t['reversal_score'], color='#3f6f9f')
ax.axvline(0, color='#444', lw=.8)
ax.set(xlabel='Reversal score  (-cos(predicted response, LUAD direction))',
       title='UniPert-G2CP / LINCS-A549: top 20 FDA drugs by LUAD-signature reversal')
fig.tight_layout()
fig.savefig(run.figures / 'drug_reversal_ranking.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=140)
ax.hist(res['reversal_score'].dropna(), bins=60, color='#6b8f71')
ax.axvline(0, color='#444', lw=.8)
ax.set(xlabel='Reversal score', ylabel='drugs',
       title=f'UniPert-G2CP: reversal score distribution ({len(res)} drugs)')
fig.tight_layout()
fig.savefig(run.figures / 'reversal_distribution.png')
plt.close(fig)

validation = {
    'backend': 'UniPert-G2CP',
    'checkpoint': str(MODEL_PATH / 'finetune_model.pt'),
    'unipart_model': str(UNIPERT_DIR / 'current_model/unipert_model.pt'),
    'task': 'drug',
    'cell_background': 'LINCS-A549 (模型自带对照细胞，非用户矩阵)',
    'accepts_user_baseline': False,
    'accepts_dose': False,
    'drugs_requested': int(len(lib)),
    'drugs_scored': int(len(res)),
    'unipert_reps': int(len(reps)),
    'genes': len(gene_space),
    'control_cells': int(ctrl.n_obs),
    'prediction_finite': bool(np.isfinite(preds).all()),
    'degenerate_output': bool(np.abs(resp).max() < 1e-6),
    'reversal_score_range': [float(res['reversal_score'].min()),
                             float(res['reversal_score'].max())],
    'degeneracy_check': deg,
    'limitation': ('new_predict 每次随机抽 300 个 LINCS-A549 对照细胞，没有随机种子接口；'
                   '本次运行用 np.random.seed 固定了抽样，但多次运行结果会有差异。'),
    'independent_validation': 'none',
    'status': 'failed' if deg['degenerate'] else 'result_available',
}
manifest = {
    'task': 'drug',
    'backend': {'name': 'UniPert-G2CP', 'cell_line': 'A549'},
    'inputs': {'library': str(LIBRARY), 'library_sha256_16': sha16(LIBRARY)},
    'parameters': {'seed': SEED, 'n_control_cells_per_pert': 300, 'device': device},
}
run.finish(manifest, validation)
print(json.dumps(validation, ensure_ascii=False, indent=2))
print(drug.head(10)[['rank', 'drug', 'reversal_score', 'mechanism']].to_string(index=False))
