"""AIVC 药筛测试 - 三个后端共用的输入与打分

输入形态（本次测试的口径）：
  - 用户自己的单细胞表达矩阵（GSE198291，LUAD，QC 后 703 细胞）
  - 一个药物库（FDA 已上市药物 + 剂量）

共用步骤：
  1. 把单细胞矩阵压成后端基因空间上的基线表达（每个细胞一个基线样本）
  2. 把 GSE81089 得到的 LUAD 疾病方向向量投到同一基因空间
  3. 预测响应 = 预测处理后表达 − 基线；用与疾病方向的夹角给药物排序
"""
from pathlib import Path
import json
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TEST_ROOT = ROOT.parent
DATA = ROOT / 'data'

SCRNA = TEST_ROOT / 'results/geo_gse198291/GSE198291_counts_qc200.h5ad'
DISEASE_VEC = TEST_ROOT / 'results/geo_gse81089/disease_direction_vector_978.csv'
LIBRARY = DATA / 'fda_approved_library.csv'
CONDITIONS = DATA / 'fda_dose_conditions.csv'


def sha16(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()[:16]


def degeneracy_check(resp, scores):
    """判断结果是否退化：药物之间没有区分度时不能当成排名交付。

    两个指标：
      score_spread  —— 逆转分数在药物间的极差；
      median_pair_corr —— 不同药物响应向量两两相关的中位数。
    只要响应几乎不随药物变化，就判退化。
    """
    r = np.asarray(resp, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    spread = float(s.max() - s.min()) if s.size else 0.0

    n = r.shape[0]
    idx = np.linspace(0, n - 1, min(200, n)).astype(int)
    sub = r[idx]
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    u = sub / norms
    cm = u @ u.T
    iu = np.triu_indices(cm.shape[0], k=1)
    med_corr = float(np.median(cm[iu])) if iu[0].size else 0.0

    degenerate = bool(spread < 1e-4 or med_corr > 0.999)
    return {
        'score_spread_across_conditions': spread,
        'median_pairwise_response_correlation': med_corr,
        'degenerate': degenerate,
        'criterion': 'score_spread < 1e-4 或 median_pairwise_response_correlation > 0.999',
    }


def load_baseline(gene_space, target_sum=1e4):
    """把单细胞矩阵转成 gene_space 上的 log1p 归一化基线。

    返回 (cells x genes 的 DataFrame, 覆盖信息 dict)。
    在完整转录组上做 normalize_total + log1p，再取子集——与训练数据
    （SciPlex/LINCS 都是先在全转录组归一化再投到 landmark 基因）一致。
    """
    import scanpy as sc
    a = sc.read_h5ad(SCRNA)
    sc.pp.normalize_total(a, target_sum=target_sum)
    sc.pp.log1p(a)
    X = a.X.toarray() if hasattr(a.X, 'toarray') else np.asarray(a.X)

    gene_space = list(gene_space)
    idx = {g: i for i, g in enumerate(a.var_names)}
    present = [g for g in gene_space if g in idx]
    M = np.zeros((a.n_obs, len(gene_space)), dtype=np.float32)
    col = [gene_space.index(g) for g in present]
    M[:, col] = X[:, [idx[g] for g in present]]

    df = pd.DataFrame(M, index=a.obs_names, columns=gene_space)
    info = {
        'scrna_cells': int(a.n_obs),
        'scrna_genes_total': int(a.n_vars),
        'target_gene_space': len(gene_space),
        'genes_covered': len(present),
        'genes_missing_filled_zero': len(gene_space) - len(present),
        'coverage_fraction': round(len(present) / len(gene_space), 4),
        'normalization': f'normalize_total(target_sum={target_sum}) on full transcriptome, then log1p',
        'rowsum_mean': float(df.sum(axis=1).mean()),
        'value_max': float(df.to_numpy().max()),
    }
    return df, info


def load_disease_direction(gene_space):
    v = pd.read_csv(DISEASE_VEC, index_col=0)['tumor_minus_normal_log2']
    v = v.reindex(list(gene_space))
    n_missing = int(v.isna().sum())
    v = v.fillna(0.0)
    return v.to_numpy(dtype=np.float64), n_missing


def reversal_scores(pred_response, disease):
    """pred_response: (n_cond, n_genes)；disease: (n_genes,)。"""
    p = np.asarray(pred_response, dtype=np.float64)
    d = np.asarray(disease, dtype=np.float64)
    pn = np.linalg.norm(p, axis=1)
    dn = np.linalg.norm(d)
    cos = np.full(p.shape[0], np.nan)
    ok = (pn > 0) & (dn > 0)
    cos[ok] = (p[ok] @ d) / (pn[ok] * dn)
    corr = np.full(p.shape[0], np.nan)
    ds = d.std()
    if ds > 0:
        pc = p - p.mean(axis=1, keepdims=True)
        dc = d - d.mean()
        denom = np.linalg.norm(pc, axis=1) * np.linalg.norm(dc)
        good = denom > 0
        corr[good] = (pc[good] @ dc) / denom[good]
    return cos, corr, -cos


class RunDir:
    """统一的结果目录：raw / tables / figures + manifest / validation / errors。"""

    def __init__(self, run_id):
        self.run_id = run_id
        self.root = ROOT / 'results' / run_id
        self.raw = self.root / 'raw'
        self.tables = self.root / 'tables'
        self.figures = self.root / 'figures'
        for d in (self.raw, self.tables, self.figures):
            d.mkdir(parents=True, exist_ok=True)
        self.errors = []

    def log_error(self, stage, exc):
        import traceback
        self.errors.append({'stage': stage, 'error': repr(exc),
                            'traceback': traceback.format_exc()[-1500:]})
        print(f'[ERROR] {stage}: {exc!r}')

    def finish(self, manifest, validation):
        manifest = dict(manifest)
        manifest.setdefault('run_id', self.run_id)
        manifest['created'] = datetime.now().isoformat(timespec='seconds')
        manifest['outputs'] = {
            'raw': sorted(p.name for p in self.raw.iterdir()),
            'tables': sorted(p.name for p in self.tables.iterdir()),
            'figures': sorted(p.name for p in self.figures.iterdir()),
        }
        manifest['errors'] = len(self.errors)
        (self.root / 'manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        (self.root / 'validation.json').write_text(
            json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
        with open(self.root / 'errors.jsonl', 'w', encoding='utf-8') as f:
            for e in self.errors:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')
