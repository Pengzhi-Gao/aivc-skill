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


def degeneracy_check(resp, scores, *, response_atol=1e-8, response_rtol=1e-6,
                     score_atol=1e-4):
    """Separate finite predictions, response variation and ranking resolution.

    Tolerances describe numerical resolution, not biological significance.
    Cosine similarity is diagnostic only: proportional responses can differ in size.
    """
    r = np.asarray(resp, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] == 0 or s.shape != (r.shape[0],):
        raise ValueError('expected conditions x genes and one score per condition')
    finite = np.isfinite(r).all(axis=1)
    valid = r[finite]
    score_ok = finite & np.isfinite(s)
    usable = s[score_ok]
    spread = float(np.ptp(usable)) if usable.size else None
    identical = False
    max_spread = None
    if len(valid) >= 2:
        ranges = np.ptp(valid, axis=0)
        tolerance = response_atol + response_rtol * np.max(np.abs(valid), axis=0)
        identical = bool(np.all(ranges <= tolerance))
        max_spread = float(ranges.max())
    idx = np.linspace(0, len(valid) - 1, min(200, len(valid))).astype(int)
    sub = valid[idx]
    norms = np.linalg.norm(sub, axis=1)
    nonzero = norms > 0
    u = sub[nonzero] / norms[nonzero, None]
    cm = np.clip(u @ u.T, -1, 1)
    pairs = cm[np.triu_indices(len(u), k=1)]
    median_cosine = float(np.median(pairs)) if pairs.size else None
    status = ('failed' if not finite.any() else
              'partial' if not finite.all() else 'result_available')
    ranking = ('not_available' if usable.size == 0 else
               'insufficient_conditions' if usable.size < 2 else
               'unresolved' if identical or spread <= score_atol else 'available')
    return {
        'prediction_status': status,
        'invalid_condition_indices': np.flatnonzero(~finite).tolist(),
        'unscored_condition_indices': np.flatnonzero(~score_ok).tolist(),
        'response_status': ('insufficient_conditions' if len(valid) < 2 else
                            'indistinguishable' if identical else 'varying'),
        'ranking_status': ranking,
        'score_spread_across_conditions': spread,
        'max_per_gene_response_range': max_spread,
        'median_pairwise_response_cosine': median_cosine,
        'zero_response_conditions': int(np.sum(np.linalg.norm(valid, axis=1) == 0)),
        'degenerate': identical,
        'tolerances': {'response_atol': response_atol, 'response_rtol': response_rtol,
                       'score_atol': score_atol},
        'criterion': 'finite rows; per-gene range <= atol + rtol * max(abs(response)); '
                     'score resolution assessed separately; cosine does not decide failure',
    }


def load_baseline(gene_space, target_sum=1e4):
    """把单细胞矩阵转成 gene_space 上的 log1p 归一化基线。

    返回 (cells x genes 的 DataFrame, 覆盖信息 dict)。
    这里使用 full-transcriptome normalize_total + log1p；是否匹配具体权重
    仍需核对其预处理，不能假定不同平台使用同一变换。
    """
    import scanpy as sc
    a = sc.read_h5ad(SCRNA)
    sc.pp.normalize_total(a, target_sum=target_sum)
    sc.pp.log1p(a)
    X = a.X.toarray() if hasattr(a.X, 'toarray') else np.asarray(a.X)

    gene_space = list(gene_space)
    idx = {g: i for i, g in enumerate(a.var_names)}
    present = [g for g in gene_space if g in idx]
    missing = [g for g in gene_space if g not in idx]
    if missing:
        raise ValueError(f'{len(missing)} model input genes are unmeasured/unmapped; '
                         f'no validated imputation configured: {missing[:10]}')
    M = X[:, [idx[g] for g in gene_space]].astype(np.float32)

    df = pd.DataFrame(M, index=a.obs_names, columns=gene_space)
    info = {
        'scrna_cells': int(a.n_obs),
        'scrna_genes_total': int(a.n_vars),
        'target_gene_space': len(gene_space),
        'genes_covered': len(present),
        'genes_missing': len(missing),
        'missing_policy': 'reject; no zero imputation',
        'coverage_fraction': round(len(present) / len(gene_space), 4),
        'normalization': f'normalize_total(target_sum={target_sum}) on full transcriptome, then log1p',
        'rowsum_mean': float(df.sum(axis=1).mean()),
        'value_max': float(df.to_numpy().max()),
    }
    return df, info


def load_disease_direction(gene_space):
    # Read the mapped table: the historical direction vector had missing genes filled with 0.
    source = DISEASE_VEC.with_name('GSE81089_landmark978_signature.csv')
    if not source.exists():
        return np.full(len(gene_space), np.nan), len(gene_space)
    v = pd.read_csv(source).set_index('gene_symbol')['tumor_minus_normal_log2']
    if not v.index.is_unique:
        raise ValueError('duplicate disease gene IDs; resolve mapping before scoring')
    v = v.reindex(list(gene_space))
    n_missing = int(v.isna().sum())
    return v.to_numpy(dtype=np.float64), n_missing


def reversal_scores(pred_response, disease):
    """pred_response: (n_cond, n_genes)；disease: (n_genes,)。"""
    p = np.asarray(pred_response, dtype=np.float64)
    d = np.asarray(disease, dtype=np.float64)
    if p.ndim != 2 or d.shape != (p.shape[1],):
        raise ValueError('disease and response gene order/shape must match')
    # One shared measured-gene mask for every candidate. Do not drop genes per drug.
    keep = np.isfinite(d)
    valid_rows = np.isfinite(p).all(axis=1)
    if keep.sum() < 2:
        empty = np.full(p.shape[0], np.nan)
        return empty.copy(), empty.copy(), empty.copy()
    p, d = p[:, keep], d[keep]
    pn = np.linalg.norm(p, axis=1)
    dn = np.linalg.norm(d)
    cos = np.full(p.shape[0], np.nan)
    ok = valid_rows & (pn > 0) & (dn > 0)
    cos[ok] = (p[ok] @ d) / (pn[ok] * dn)
    corr = np.full(p.shape[0], np.nan)
    ds = d.std()
    if ds > 0:
        pc = p - p.mean(axis=1, keepdims=True)
        dc = d - d.mean()
        denom = np.linalg.norm(pc, axis=1) * np.linalg.norm(dc)
        good = valid_rows & (denom > 0)
        corr[good] = (pc[good] @ dc) / denom[good]
    return cos, corr, -cos


def top_response_table(conditions, response, genes, ranked_drugs, n=20):
    """Select by drug ID in rank order, preserving all corresponding conditions."""
    r = np.asarray(response)
    if r.shape != (len(conditions), len(genes)):
        raise ValueError('condition/gene axes do not match response')
    wanted = list(dict.fromkeys(ranked_drugs))[:n]
    ids = conditions['drug'].astype(str).to_numpy()
    blocks = [np.flatnonzero(ids == str(drug)) for drug in wanted]
    if any(len(ix) == 0 for ix in blocks):
        raise ValueError('ranked drug absent from response condition index')
    selected = np.concatenate(blocks) if blocks else np.array([], dtype=int)
    meta = conditions.iloc[selected].reset_index(drop=True).copy()
    cols = [c for c in ('condition_id', 'drug', 'dose_uM', 'time', 'cell_type') if c in meta]
    return pd.concat([meta[cols], pd.DataFrame(r[selected], columns=genes)], axis=1)


def drug_result_tables(conditions, response, genes, scores, *, scoring_status,
                       ranking_status, top_n=30):
    """Describe disease-direction reversal and top gene effects for every condition.

    A positive negative-cosine score is a descriptive direction label, not
    efficacy or statistical significance. Conditions remain separate by dose.
    """
    r = np.asarray(response, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if r.shape != (len(conditions), len(genes)) or s.shape != (len(conditions),):
        raise ValueError('condition, gene and score axes do not match')
    if 'condition_id' not in conditions or 'drug' not in conditions:
        raise ValueError('condition_id and drug are required')
    if conditions['condition_id'].duplicated().any() or len(set(genes)) != len(genes):
        raise ValueError('condition and gene IDs must be unique')
    if top_n < 1:
        raise ValueError('top_n must be positive')
    out = conditions.reset_index(drop=True).copy()
    out['reversal_score'] = s
    finite = np.isfinite(r).all(axis=1)
    out['gene_response_status'] = np.where(finite, 'available', 'not_available')
    out['recovery_status'] = 'indeterminate'
    out['recovery_reason'] = 'score_not_available'
    out.loc[~finite, 'recovery_reason'] = 'invalid_response'
    zero_response = finite & (np.linalg.norm(r, axis=1) == 0)
    out.loc[zero_response, 'recovery_reason'] = 'zero_response'
    if scoring_status == 'descriptive_only' and ranking_status == 'available':
        usable = finite & ~zero_response & np.isfinite(s)
        out.loc[usable & (s > 0), 'recovery_status'] = 'predicted_reversal'
        out.loc[usable & (s <= 0), 'recovery_status'] = 'no_predicted_reversal'
        out.loc[usable, 'recovery_reason'] = 'negative_cosine_sign_only'
    elif ranking_status != 'available' and scoring_status == 'descriptive_only':
        out.loc[finite & ~zero_response, 'recovery_reason'] = f'ranking_{ranking_status}'
    labels = {'predicted_reversal': '预测有疾病方向恢复',
              'no_predicted_reversal': '未见预测恢复方向',
              'indeterminate': '无法判定',
              'condition_dependent': '随剂量或条件而异'}
    out['recovery_label_zh'] = out['recovery_status'].map(labels)

    rows = []
    for i in np.flatnonzero(finite):
        order = np.argsort(-np.abs(r[i]), kind='stable')
        order = order[np.abs(r[i, order]) > 0][:top_n]
        meta = out.iloc[i]
        for rank, j in enumerate(order, start=1):
            row = {k: meta[k] for k in ('condition_id', 'drug', 'dose_uM', 'time', 'cell_type')
                   if k in out.columns}
            row.update(rank=rank, gene=str(genes[j]), predicted_delta=float(r[i, j]),
                       abs_predicted_delta=float(abs(r[i, j])),
                       effect_direction='up' if r[i, j] > 0 else 'down')
            rows.append(row)
    top_columns = [c for c in ('condition_id', 'drug', 'dose_uM', 'time', 'cell_type')
                   if c in out.columns] + ['rank', 'gene', 'predicted_delta',
                                           'abs_predicted_delta', 'effect_direction']
    top = pd.DataFrame(rows, columns=top_columns)
    counts = out.groupby('drug', sort=False)['recovery_status'].value_counts().unstack(fill_value=0)
    for status in ('predicted_reversal', 'no_predicted_reversal', 'indeterminate'):
        if status not in counts:
            counts[status] = 0
    summary = counts.reset_index().rename(columns={
        'predicted_reversal': 'n_predicted_reversal',
        'no_predicted_reversal': 'n_no_predicted_reversal',
        'indeterminate': 'n_indeterminate'})
    summary['n_conditions'] = summary[['n_predicted_reversal', 'n_no_predicted_reversal',
                                       'n_indeterminate']].sum(axis=1)
    summary['recovery_status'] = np.select(
        [summary.n_indeterminate > 0,
         (summary.n_predicted_reversal > 0) & (summary.n_no_predicted_reversal > 0),
         summary.n_predicted_reversal > 0],
        ['indeterminate', 'condition_dependent', 'predicted_reversal'],
        default='no_predicted_reversal')
    summary['recovery_label_zh'] = summary['recovery_status'].map(labels)
    summary['top_gene_rows'] = summary['drug'].map(top.groupby('drug').size()).fillna(0).astype(int)
    return out, summary, top


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


class RunDir:
    """统一的结果目录：raw / tables / figures + manifest / validation / errors。"""

    def __init__(self, run_id, base_dir=None):
        # Each execution gets a new directory; historical results are not overwritten.
        self.run_id = f'{run_id}_{datetime.now():%Y%m%dT%H%M%S%f}'
        self.root = Path(base_dir or ROOT) / 'results' / self.run_id
        self.raw = self.root / 'raw'
        self.tables = self.root / 'tables'
        self.figures = self.root / 'figures'
        for d in (self.raw, self.tables, self.figures):
            d.mkdir(parents=True, exist_ok=True)
        self.errors = []

    def save_predictions(self, conditions, genes, prediction, baseline, response):
        """Persist all predictions before filtering/ranking, including invalid rows for diagnosis."""
        p, b, r = map(np.asarray, (prediction, baseline, response))
        if len(set(genes)) != len(genes) or set(genes) & set(conditions.columns):
            raise ValueError('gene IDs must be unique and distinct from condition metadata columns')
        if p.shape != r.shape or p.shape != (len(conditions), len(genes)):
            raise ValueError('prediction axes do not match condition/gene metadata')
        if b.shape not in ((len(genes),), p.shape):
            raise ValueError('baseline must be one gene vector or one vector per condition')
        if not np.allclose(p - b, r, equal_nan=True):
            raise ValueError('response is inconsistent with prediction minus baseline')
        meta = conditions.reset_index(drop=True).copy()
        meta.insert(0, 'condition_id', [f'condition_{i:06d}' for i in range(len(meta))])
        meta.to_csv(self.raw / 'condition_index.csv', index=False)
        np.savez_compressed(self.raw / 'predictions.npz', prediction=p, baseline=b,
                            response=r, genes=np.asarray(genes, dtype=str),
                            condition_id=meta['condition_id'].to_numpy(dtype=str))
        pd.concat([meta, pd.DataFrame(r, columns=genes)], axis=1).to_csv(
            self.tables / 'all_conditions_gene_response.csv.gz', index=False)
        return meta

    def score_response(self, response, disease, backend):
        """Skip optional reversal scoring until its scale contract has been documented."""
        path = DATA / f'{backend}_scoring_contract.json'
        contract = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        required = ('response_scale', 'disease_scale', 'scale_justification',
                    'comparison_scope', 'minimum_common_genes')
        reason = None
        source = DISEASE_VEC.with_name('GSE81089_landmark978_signature.csv')
        if not all(contract.get(k) for k in required):
            reason = 'missing documented scoring scale/context contract'
        elif contract.get('metric') != 'negative_cosine':
            reason = 'this adapter implements negative_cosine only'
        elif not source.exists() or contract.get('disease_source_sha256_16') != sha16(source):
            reason = 'scoring contract does not match disease source'
        elif not isinstance(contract['minimum_common_genes'], int) or contract['minimum_common_genes'] < 2:
            reason = 'minimum_common_genes must be an integer >= 2 with task-specific justification'
        elif np.isfinite(disease).sum() < contract['minimum_common_genes']:
            reason = 'insufficient common measured genes'
        meta = {'status': 'skipped' if reason else 'descriptive_only',
                'reason': reason, 'contract_path': str(path), 'contract': contract,
                'common_genes': int(np.isfinite(disease).sum()),
                'disease_source': str(source), 'null_model': 'not_tested'}
        (self.tables / 'scoring_metadata.json').write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        if reason:
            empty = np.full(len(response), np.nan)
            return empty.copy(), empty.copy(), empty.copy()
        return reversal_scores(response, disease)

    def condition_quality(self, conditions, response, scores):
        check = degeneracy_check(response, scores)
        q = conditions.reset_index(drop=True).copy()
        q['prediction_status'] = 'result_available'
        q.loc[check['invalid_condition_indices'], 'prediction_status'] = 'failed'
        q['score_status'] = 'available'
        q.loc[check['unscored_condition_indices'], 'score_status'] = 'not_available'
        q.to_csv(self.tables / 'condition_quality.csv', index=False)
        for i in check['invalid_condition_indices']:
            self.errors.append({'condition_id': str(q.iloc[i]['condition_id']),
                                'stage': 'prediction_quality', 'error': 'non-finite response'})
        return check

    def log_error(self, stage, exc):
        import traceback
        self.errors.append({'stage': stage, 'error': repr(exc),
                            'traceback': traceback.format_exc()[-1500:]})
        print(f'[ERROR] {stage}: {exc!r}')

    def export_drug_results(self, validation):
        """Export all-drug status and per-condition top genes from saved predictions."""
        pack = self.raw / 'predictions.npz'
        conditions_path = self.tables / 'conditions.csv'
        if not pack.exists() or not conditions_path.exists():
            return None
        with np.load(pack, allow_pickle=False) as saved:
            response, genes, ids = saved['response'], saved['genes'], saved['condition_id']
        conditions = pd.read_csv(conditions_path)
        if 'condition_id' not in conditions or conditions['condition_id'].astype(str).tolist() != ids.tolist():
            raise ValueError('conditions.csv is not aligned with saved predictions')
        if 'reversal_score' not in conditions:
            raise ValueError('conditions.csv has no reversal_score column')
        scoring_path = self.tables / 'scoring_metadata.json'
        scoring = json.loads(scoring_path.read_text(encoding='utf-8')) if scoring_path.exists() else {}
        by_condition, by_drug, top_genes = drug_result_tables(
            conditions, response, genes,
            pd.to_numeric(conditions['reversal_score'], errors='coerce').to_numpy(),
            scoring_status=scoring.get('status'), ranking_status=validation.get('ranking_status'))
        by_condition.to_csv(self.tables / 'drug_recovery_by_condition.csv', index=False)
        by_drug.to_csv(self.tables / 'drug_recovery_summary.csv', index=False)
        top_genes.to_csv(self.tables / 'drug_top30_genes.csv', index=False)
        for status, name in (
            ('predicted_reversal', 'drugs_predicted_reversal.csv'),
            ('no_predicted_reversal', 'drugs_no_predicted_reversal.csv'),
            ('condition_dependent', 'drugs_condition_dependent.csv'),
            ('indeterminate', 'drugs_indeterminate.csv')):
            by_drug.loc[by_drug.recovery_status == status].to_csv(self.tables / name, index=False)
        (self.tables / 'drug_result_rules.json').write_text(json.dumps({
            'recovery_metric': 'negative_cosine',
            'classification_scope': 'per condition; drug summary requires all conditions classified',
            'predicted_reversal_rule': 'finite response and score > 0, scoring contract documented, ranking available',
            'no_predicted_reversal_rule': 'finite response and score <= 0 under the same requirements',
            'condition_dependent_rule': 'classified conditions include both directions',
            'indeterminate_rule': 'missing or invalid score/response, skipped scoring, or unresolved ranking',
            'top_gene_rule': 'up to 30 nonzero finite genes per drug-condition, descending absolute predicted delta; stable ties',
            'response_scale': scoring.get('contract', {}).get('response_scale'),
            'interpretation': 'descriptive predicted transcriptomic direction, not clinical efficacy or statistical significance',
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        return by_drug

    def response_views(self, validation):
        """Basic reproducible response views; figure errors do not erase model output."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        records = []
        base = {'task': 'drug', 'condition': 'multiple', 'backend': validation.get('backend'),
                'output_type': 'expression', 'baseline_source': validation.get('cell_background'),
                'visual_review': 'pending'}
        def record(name, status, source='', reason='', metric=''):
            records.append(dict(base, figure_id=name, question=name, status=status,
                                plot_path=f'{name}.png' if status == 'generated' else '',
                                source_table=source, metric=metric, unit='model scale',
                                reason=reason))
        def draw(name, table, title, metric, cmap, limit=None):
            source = self.tables / f'{name}_source.csv'
            table.to_csv(source)
            fig = None
            try:
                fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
                im = ax.imshow(table.to_numpy(), aspect='auto', cmap=cmap,
                               vmin=-limit if limit else None, vmax=limit)
                ax.set_xticks(range(len(table.columns)), table.columns, rotation=90, fontsize=6)
                ax.set_yticks(range(len(table.index)), table.index, fontsize=6)
                ax.set_title(title)
                fig.colorbar(im, ax=ax, label=metric)
                fig.tight_layout()
                fig.savefig(self.figures / f'{name}.png')
                record(name, 'generated', '../tables/' + source.name, metric=metric)
            except Exception as exc:
                record(name, 'failed', '../tables/' + source.name, repr(exc), metric)
                self.log_error('figure:' + name, exc)
            finally:
                if fig is not None:
                    plt.close(fig)
        pack = self.raw / 'predictions.npz'
        if pack.exists():
            with np.load(pack, allow_pickle=False) as saved:
                response, genes = saved['response'], saved['genes']
            meta = pd.read_csv(self.raw / 'condition_index.csv')
            finite = np.isfinite(response).all(axis=1)
            valid_idx = np.flatnonzero(finite)
            rank_path = self.tables / 'drug_ranking.csv'
            if not rank_path.exists():
                rank_path = self.tables / 'a549_drug_ranking.csv'
            if len(valid_idx):
                if validation.get('ranking_status') == 'available' and rank_path.exists():
                    ranked = pd.read_csv(rank_path)['drug'].tolist()
                    table = top_response_table(meta.loc[finite].reset_index(drop=True), response[finite], genes, ranked)
                    rows = table.head(24)
                    matrix = rows[list(genes)].to_numpy()
                    labels = rows['condition_id'].astype(str).to_numpy()
                    title = 'Predicted response: ranked candidates'
                else:
                    ix = valid_idx[np.argsort(-np.linalg.norm(response[valid_idx], axis=1), kind='stable')[:24]]
                    matrix, labels = response[ix], meta.iloc[ix]['condition_id'].astype(str).to_numpy()
                    title = 'Response examples by magnitude (not candidate ranking)'
                order = np.argsort(-np.mean(np.abs(response[valid_idx]), axis=0), kind='stable')[:30]
                table = pd.DataFrame(matrix[:, order], index=labels, columns=genes[order])
                if len(table):
                    draw('drug_response_heatmap', table, title, 'predicted delta', 'RdBu_r',
                         max(float(np.abs(table.to_numpy()).max()), 1e-12))
                else:
                    record('drug_response_heatmap', 'skipped', reason='no ranked conditions')
                nz = valid_idx[np.linalg.norm(response[valid_idx], axis=1) > 0]
                ix = nz[np.linspace(0, len(nz)-1, min(40, len(nz))).astype(int)]
                if len(ix) >= 2:
                    u = response[ix] / np.linalg.norm(response[ix], axis=1, keepdims=True)
                    ids = meta.iloc[ix]['condition_id'].astype(str).tolist()
                    table = pd.DataFrame(np.clip(u @ u.T, -1, 1), index=ids, columns=ids)
                    draw('response_similarity', table, 'Diagnostic: response cosine similarity',
                         'cosine similarity', 'RdBu_r', 1)
                else:
                    record('response_similarity', 'skipped', reason='fewer than two nonzero finite responses')
            else:
                for name in ('drug_response_heatmap', 'response_similarity'):
                    record(name, 'skipped', reason='no finite response conditions')
        else:
            for name in ('drug_response_heatmap', 'response_similarity'):
                record(name, 'skipped', reason='missing_source_data: full prediction archive absent')
        for p in sorted(self.figures.glob('*.png')):
            if any(row['figure_id'] == p.stem for row in records):
                continue
            rank = 'ranking' in p.stem
            source = ('a549_drug_ranking.csv' if p.stem.startswith('a549') else 'drug_ranking.csv') if rank else (
                'a549_conditions.csv' if p.stem.startswith('a549') else 'conditions.csv')
            record(p.stem, 'generated', '../tables/' + source,
                   'diagnostic_only' if validation.get('ranking_status') != 'available' else '',
                   'see scoring_metadata.json; plotting code retained in adapter')
        if not any('ranking' in row['figure_id'] for row in records):
            record('drug_ranking', 'skipped', reason='no valid scoring contract / score')
        if not any('distribution' in row['figure_id'] for row in records):
            record('score_distribution', 'skipped', reason='no valid scores')
        record('reversal_null', 'skipped', reason='no justified null model; descriptive scoring only')
        pd.DataFrame(records).to_csv(self.figures / 'figure_manifest.csv', index=False)
        return records

    def finish(self, manifest, validation):
        drug_summary = None
        try:
            drug_summary = self.export_drug_results(validation)
        except Exception as exc:
            self.log_error('export_drug_results', exc)
        try:
            self.response_views(validation)
        except Exception as exc:
            self.log_error('response_views', exc)
            pd.DataFrame([{'figure_id': 'response_views', 'status': 'failed',
                           'reason': repr(exc)}]).to_csv(self.figures / 'figure_manifest.csv', index=False)
        manifest = dict(manifest)
        manifest.setdefault('run_id', self.run_id)
        manifest['created'] = datetime.now().isoformat(timespec='seconds')
        manifest['outputs'] = {
            'raw': sorted(p.name for p in self.raw.iterdir()),
            'tables': sorted(p.name for p in self.tables.iterdir()),
            'figures': sorted(p.name for p in self.figures.iterdir()),
        }
        manifest['errors'] = len(self.errors)
        manifest['source_files'] = {str(p.relative_to(self.root)): sha16(p)
                                    for folder in (self.raw, self.tables) for p in folder.iterdir()
                                    if p.is_file()}
        (self.root / 'provenance.json').write_text(
            json.dumps({'source_files': manifest['source_files'], 'backend': manifest.get('backend'),
                        'parameters': manifest.get('parameters'), 'inputs': manifest.get('inputs')},
                       ensure_ascii=False, indent=2), encoding='utf-8')
        (self.root / 'manifest.json').write_text(
            json.dumps(json_safe(manifest), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        (self.root / 'validation.json').write_text(
            json.dumps(json_safe(validation), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        recovery_note = ''
        if drug_summary is not None:
            counts = drug_summary['recovery_status'].value_counts()
            recovery_note = (
                "## 药物筛选结果\n\n"
                f"预测有疾病方向恢复：{counts.get('predicted_reversal', 0)} 个药物；"
                f"未见预测恢复方向：{counts.get('no_predicted_reversal', 0)} 个；"
                f"随剂量或条件而异：{counts.get('condition_dependent', 0)} 个；"
                f"无法判定：{counts.get('indeterminate', 0)} 个。\n\n"
                "[预测有恢复方向的药物](tables/drugs_predicted_reversal.csv) · "
                "[未见预测恢复方向的药物](tables/drugs_no_predicted_reversal.csv) · "
                "[其余药物及全部状态](tables/drug_recovery_summary.csv) · "
                "[逐条件分数与状态](tables/drug_recovery_by_condition.csv) · "
                "[每个药物条件的 Top 30 影响基因](tables/drug_top30_genes.csv) · "
                "[判定规则](tables/drug_result_rules.json)。\n\n"
                "恢复方向按负余弦分数符号作描述性分类：大于 0 表示预测表达变化与疾病方向相反。"
                "独立验证状态见上方记录；临床疗效需另有实验或临床证据。"
                "Top 基因按模型尺度下预测变化的绝对值排序，同时保留上调或下调方向。\n\n")
        (self.root / 'report.md').write_text(
            f"# {self.run_id}\n\nBackend: {validation.get('backend')}\n\n"
            f"Prediction: {validation.get('status')}; ranking: {validation.get('ranking_status')}; "
            f"validation: {validation.get('validation_status', 'unvalidated')}\n\n"
            "Full predictions: [archive](raw/predictions.npz), [conditions](raw/condition_index.csv).\n\n" +
            recovery_note +
            "Figure generation and omissions: [manifest](figures/figure_manifest.csv). "
            "Visual review remains pending until the exported files are inspected.\n",
            encoding='utf-8')
        with open(self.root / 'errors.jsonl', 'w', encoding='utf-8') as f:
            for e in self.errors:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')
