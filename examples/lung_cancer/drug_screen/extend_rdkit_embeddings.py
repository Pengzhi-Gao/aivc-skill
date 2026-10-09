"""AIVC 药筛测试 - 复现 chemCPA 的 RDKit 药物表征，并扩展到 FDA 药物库

chemCPA 推理时按 SMILES 从 parquet 查药物表征，查不到的**填零向量**，
等于静默地判定"这个药没效果"。FDA 库里 1565 个药不在训练表征里，必须补出来。

生成流程（embeddings/rdkit/embedding_rdkit.py）：
  1. 规范化 SMILES
  2. descriptastorus MakeGenerator(("RDKit2D",)).process(smiles)
  3. 丢掉第 0 个值（SMILES 字符串本身）
  4. NaN/Inf 置 0（embed_smiles_list 里做的）
  5. 丢掉标准差 <= 0.01 的列
  6. 用训练集统计量做 z 标准化，pandas 默认 ddof=1

这里不照抄第 5/6 步的实现，而是用训练集重算原始描述符后**反解**：
逐个原始列去比对 parquet 的 194 列，确定保留列与 mean/std。
再用 parquet 自身校验重建精度，并把逐列误差写进 summary。
"""
from pathlib import Path
import json
import multiprocessing as mp

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
CHEMPA = Path(r'D:/WorkBuddy/Work_推送/2610/261002 ChemCPA药物扰动/chemCPA_repro')
TRAIN_PARQUET = CHEMPA / 'project_folder/embeddings/rdkit/rdkit2D_embedding_lincs_trapnell.parquet'
RAW_CACHE = DATA / '_train_raw_descriptors.npz'
OUT_PARQUET = DATA / 'rdkit_embedding_fda_extended.parquet'
STD_THRESHOLD = 0.01


def raw_descriptors(smi):
    from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
    gen = MakeGenerator(("RDKit2D",))
    res = gen.process(smi)
    if res is None:
        return None
    return np.asarray(res[1:], dtype=np.float64)


def compute_many(smiles, n_jobs=8):
    with mp.Pool(processes=n_jobs) as pool:
        out = pool.map(raw_descriptors, smiles, chunksize=64)
    ok = [(s, d) for s, d in zip(smiles, out) if d is not None]
    bad = [s for s, d in zip(smiles, out) if d is None]
    return [s for s, _ in ok], np.vstack([d for _, d in ok]), bad


def main():
    train = pd.read_parquet(TRAIN_PARQUET)
    train_smiles = list(train.index)

    if RAW_CACHE.exists():
        z = np.load(RAW_CACHE, allow_pickle=True)
        ok_smiles, X = list(z['smiles']), z['X']
        print(f'[cache] raw descriptors {X.shape}')
    else:
        ok_smiles, X, bad = compute_many(train_smiles, n_jobs=8)
        np.savez_compressed(RAW_CACHE, smiles=np.array(ok_smiles), X=X)
        print(f'[raw] {X.shape}, failed {len(bad)}')

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    cand = np.where(sd > STD_THRESHOLD)[0]
    print(f'[solve] raw columns {X.shape[1]} -> sd > {STD_THRESHOLD}: {len(cand)}')

    # 低方差筛完之后，再按"必须能在 parquet 里找到对应列"过滤。
    # RDKit / descriptastorus 版本差异会让个别描述符定义变化，这类列在本机
    # 有方差、但训练时的版本里没有，不能凭阈值保留。
    P_all = train.loc[ok_smiles].to_numpy()
    Zc = (X[:, cand] - mu[cand]) / sd[cand]
    Zc = (Zc - Zc.mean(0)) / np.where(Zc.std(0) == 0, 1, Zc.std(0))
    Pz = (P_all - P_all.mean(0)) / P_all.std(0)
    corr = (Zc.T @ Pz) / Zc.shape[0]
    maxcorr = corr.max(axis=1)
    matched_raw = corr.argmax(axis=1)
    # 描述符里存在完全相同的重复列（例如 raw 38 与 raw 40 逐值相等），
    # 会让 argmax 选到同分的另一列。同分时优先取同位置。
    for i in range(len(matched_raw)):
        j = matched_raw[i]
        if i < corr.shape[1] and corr[i, i] >= corr[i, j] - 1e-9:
            matched_raw[i] = i
    keep_local = maxcorr > 0.9
    dropped = cand[~keep_local]
    if len(dropped):
        print(f'[solve] dropped columns without a parquet counterpart: '
              f'{[int(i) for i in dropped]}')
    kept = cand[keep_local]
    matched_raw = matched_raw[keep_local]
    print(f'[solve] kept {len(kept)} columns')
    assert len(kept) == train.shape[1], (
        f'kept {len(kept)} columns but parquet has {train.shape[1]}')

    identity = bool(np.array_equal(matched_raw, np.arange(len(kept))))
    print(f'[solve] parquet column j corresponds to the j-th kept raw column: {identity}')
    assert identity, (
        f'column correspondence broken at {[(i, int(a), int(b)) for i, (a, b) in enumerate(zip(kept, matched_raw)) if a != b][:10]}')

    Z = (X[:, kept] - mu[kept]) / sd[kept]
    P = P_all
    per_col = np.abs(Z - P).max(axis=0)
    # 逐行看误差：多数列只是个别离群分子，真正系统性对不上的是 57 号列
    # （99.85% 的行误差 > 0.01），33 号列本身方差是 5e52、数值病态。
    # 这两列对新增药物不可靠，填训练均值（z=0）；已有药物仍用官方原值。
    row_frac = (np.abs(Z - P) > 0.01).mean(axis=0)
    unrep = np.where((per_col >= 1e-6) & (row_frac > 0.5))[0]
    unstable = np.where(sd[kept] > 1e12)[0]
    impute_cols = sorted(set(unrep.tolist()) | set(unstable.tolist()))
    print(f'[solve] columns imputed to training mean for new drugs: {impute_cols}')
    n_exact = int((per_col < 1e-6).sum())
    print(f'[verify] columns reproduced within 1e-6: {n_exact} / {len(kept)}')
    print(f'[verify] max abs error overall: {float(per_col.max()):.3e}')
    imperfect = np.where(per_col >= 1e-6)[0]
    if len(imperfect):
        print(f'[verify] imperfect columns (parquet idx, max err): '
              f'{[(int(i), round(float(per_col[i]), 5)) for i in imperfect]}')

    # ---- 新 SMILES ----
    lib = pd.read_csv(DATA / 'fda_approved_library.csv')
    new_smiles = [s for s in lib['smiles'] if s not in train.index]
    print(f'[fda] total {len(lib)}  in train {len(lib) - len(new_smiles)}  '
          f'to compute {len(new_smiles)}')

    raw_new, Xn, bad_new = compute_many(new_smiles, n_jobs=8)
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    Zn = (Xn[:, kept] - mu[kept]) / sd[kept]
    if impute_cols:
        Zn[:, impute_cols] = 0.0
    new_df = pd.DataFrame(Zn, index=raw_new, columns=train.columns)
    print(f'[new] computed {new_df.shape}, failed {len(bad_new)}')

    full = pd.concat([train, new_df])
    full = full[~full.index.duplicated(keep='first')]
    full.to_parquet(OUT_PARQUET)

    covered = sum(1 for s in lib['smiles'] if s in full.index)
    summary = {
        'train_embedding': str(TRAIN_PARQUET),
        'train_smiles': int(len(train_smiles)),
        'raw_descriptor_dim': int(X.shape[1]),
        'columns_kept': int(len(kept)),
        'std_threshold': STD_THRESHOLD,
        'std_ddof': 1,
        'columns_reproduced_within_1e-6': n_exact,
        'columns_imperfect': [int(i) for i in imperfect],
        'columns_imputed_for_new_drugs': impute_cols,
        'max_abs_reproduction_error': float(per_col.max()),
        'fda_drugs': int(len(lib)),
        'fda_from_training_parquet': int(len(lib) - len(new_smiles)),
        'fda_newly_computed': int(new_df.shape[0]),
        'fda_failed_descriptors': int(len(bad_new)),
        'fda_final_coverage': int(covered),
        'output': str(OUT_PARQUET),
        'note': ('重建用训练集 SMILES 反解出的 mean/std；194 列中多数与官方 parquet 逐值一致。'
                 'columns_imputed_for_new_drugs 里的少数列因描述符定义随 RDKit/descriptastorus '
                 '版本变化而无法重建，对新增药物填训练均值（z=0），训练集已有药物仍用官方原值。'),
    }
    (DATA / 'rdkit_embedding_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
