"""AIVC 药筛测试 - 构建 FDA 已上市药物库（带剂量）

数据源：Broad Institute Drug Repurposing Hub（CLUE 镜像）
  - repurposing_drugs_20200324.txt   ：药物名 / 临床阶段 / 作用机制 / 靶点 / 适应症
  - repurposing_samples_20200324.txt ：药物名 / SMILES / InChIKey / PubChem CID

筛选规则：clinical_phase == "Launched"（即已获 FDA 批准上市）。
剂量：三个模型都是剂量条件模型（UniPert-G2CP 除外），训练分布都落在微摩尔量级
（SciPlex 约 10 nM–10 µM，LINCS 约 0.04–10 µM）。这里取 0.1 / 1 / 10 µM 三个
体外筛选浓度，不对应任何临床给药剂量。
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'data'
DOSE_UM = [0.1, 1.0, 10.0]

drugs = pd.read_csv(DATA / 'repurposing_drugs_20200324.txt', sep='\t',
                    skiprows=9, dtype=str)
samples = pd.read_csv(DATA / 'repurposing_samples_20200324.txt', sep='\t',
                      skiprows=9, dtype=str)
print(f'[source] drugs={len(drugs)} samples={len(samples)}')

launched = drugs[drugs['clinical_phase'].eq('Launched')].copy()
print(f'[filter] clinical_phase == Launched: {len(launched)}')

# 每个药物可能有多条样品记录，取 SMILES 解析成功、纯度最高的一条
samples = samples[samples['qc_incompatible'].astype(str).isin(['0', 'False', 'nan'])]
samples = samples.dropna(subset=['smiles'])
samples['purity_num'] = pd.to_numeric(samples['purity'], errors='coerce').fillna(0)
samples = samples.sort_values('purity_num', ascending=False)


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


best = {}
for name, grp in samples.groupby('pert_iname'):
    for smi in grp['smiles']:
        c = canon(smi)
        if c:
            best[name] = c
            break

lib = launched.copy()
lib['smiles'] = lib['pert_iname'].map(best)
n_before = len(lib)
lib = lib.dropna(subset=['smiles'])
print(f'[join] with valid canonical SMILES: {len(lib)} / {n_before}')

# 同一 SMILES 只保留一个药物名（避免同一分子重复计分）
lib = lib.drop_duplicates('smiles').reset_index(drop=True)
lib = lib.rename(columns={'pert_iname': 'drug', 'moa': 'mechanism',
                          'target': 'target', 'disease_area': 'disease_area',
                          'indication': 'indication'})
lib['n_heavy_atoms'] = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in lib['smiles']]
lib['mw'] = [round(Descriptors.MolWt(Chem.MolFromSmiles(s)), 2)
             for s in lib['smiles']]
lib.to_csv(OUT / 'fda_approved_library.csv', index=False)

# 展开成 (drug, dose) 条件表，供三个后端共用
rows = []
for r in lib.itertuples(index=False):
    for d in DOSE_UM:
        rows.append({'drug': r.drug, 'smiles': r.smiles, 'dose_uM': d,
                     'mechanism': r.mechanism, 'target': r.target})
cond = pd.DataFrame(rows)
cond.to_csv(OUT / 'fda_dose_conditions.csv', index=False)

summary = {
    'source': 'Broad Institute Drug Repurposing Hub (CLUE mirror)',
    'source_files': ['repurposing_drugs_20200324.txt',
                     'repurposing_samples_20200324.txt'],
    'filter': 'clinical_phase == "Launched"',
    'drugs_in_library': int(len(lib)),
    'dose_series_uM': DOSE_UM,
    'conditions': int(len(cond)),
    'dose_semantics': ('体外筛选浓度（µM），落在 SciPlex / LINCS 训练剂量范围内；'
                       '不是临床给药剂量，也没有换算成 Cmax。'),
    'mechanism_coverage': int(lib['mechanism'].notna().sum()),
    'smiles_unique': bool(lib['smiles'].is_unique),
}
(OUT / 'library_summary.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
