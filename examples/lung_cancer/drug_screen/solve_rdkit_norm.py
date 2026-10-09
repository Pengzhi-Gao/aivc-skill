"""反解 chemCPA RDKit 表征的列筛选与标准化参数

不猜 `df.std()` 的自由度，改为直接反解：
对每个原始描述符列 j，用训练集 SMILES 算 z_j = (x_j - mean_j) / std_j，
再和 parquet 的 194 列逐一比对。能对上 1e-6 的，说明该列被保留，
同时确定了它在 parquet 中的位置；对不上的说明被低方差筛掉了。

这样得到的 (保留列, mean, std) 是**从 parquet 反推出来的**，不依赖对原码的复述。
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
PARAMS = DATA / 'rdkit_norm_params.npz'


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
    bad = [s for s, d in zip(smiles, out) if d is None]
    ok = [(s, d) for s, d in zip(smiles, out) if d is not None]
    return [s for s, _ in ok], np.vstack([d for _, d in ok]), bad


def main():
    train = pd.read_parquet(TRAIN_PARQUET)
    train_smiles = list(train.index)

    if RAW_CACHE.exists():
        z = np.load(RAW_CACHE, allow_pickle=True)
        ok_smiles = list(z['smiles'])
        X = z['X']
        print(f'[cache] loaded raw descriptors {X.shape}')
    else:
        ok_smiles, X, bad = compute_many(train_smiles, n_jobs=8)
        np.savez_compressed(RAW_CACHE, smiles=np.array(ok_smiles), X=X)
        print(f'[raw] computed {X.shape}, failed {len(bad)}')

    # 原始代码在标准化之前把 NaN/Inf 置 0（embedding_rdkit.py embed_smiles_list）
    n_bad = int((~np.isfinite(X)).sum())
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f'[clean] non-finite entries replaced with 0: {n_bad}')

    P = train.loc[ok_smiles].to_numpy()
    n, d = X.shape
    print(f'[train] raw {X.shape}, parquet {P.shape}')

    best = {}
    for ddof in (0, 1):
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=ddof)
        Z = (X - mu) / sd
        matched = {}
        for j in range(d):
            diff = np.abs(Z[:, [j]] - P).max(axis=0)   # (n_parquet,)
            k = int(np.argmin(diff))
            if diff[k] < 1e-6:
                matched[j] = k
        print(f'[ddof={ddof}] columns matched to parquet: {len(matched)} / {d}')
        best[ddof] = (matched, mu, sd)

    ddof = max(best, key=lambda k: len(best[k][0]))
    matched, mu, sd = best[ddof]
    print(f'[choose] ddof={ddof} with {len(matched)} matched columns')

    assert len(matched) == train.shape[1], (
        f'only {len(matched)} columns matched, expected {train.shape[1]}')

    # 校验顺序完全一致
    order_ok = all(matched[j] == j for j in range(train.shape[1]))
    print(f'[order] kept columns are the first {train.shape[1]} raw columns: {order_ok}')
    kept = sorted(matched, key=lambda j: matched[j])
    assert [matched[j] for j in kept] == list(range(train.shape[1]))

    # 用全量训练 SMILES 重算的统计量做最终校验
    Zk = (X[:, kept] - mu[kept]) / sd[kept]
    err = np.abs(Zk - P).max()
    print(f'[verify] max abs error vs parquet: {err:.3e}')
    assert err < 1e-6

    np.savez_compressed(PARAMS, kept=np.array(kept), mu=mu, sd=sd,
                        ddof=np.array([ddof]))
    print(json.dumps({'ddof': ddof, 'kept_columns': len(kept),
                      'raw_columns': int(d), 'max_abs_error': float(err),
                      'kept_are_leading_columns': bool(order_ok)},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
