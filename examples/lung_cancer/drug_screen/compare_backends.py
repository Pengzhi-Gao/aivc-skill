"""AIVC 药筛测试 - 三后端横向对比

把 chemCPA / PRnet / UniPert-G2CP 在同一个 FDA 药物库上的结果汇总到一张表，
并画出各自的逆转分数分布，便于看出哪个后端在这个输入形态下真的能区分药物。
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
OUT = RES / 'comparison'
OUT.mkdir(parents=True, exist_ok=True)

rows, dists = [], {}
for d in sorted(RES.iterdir()):
    if not d.is_dir() or not (d / 'validation.json').exists() or not (d / 'tables/conditions.csv').exists():
        continue
    v = json.loads((d / 'validation.json').read_text())
    dg = v.get('degeneracy_check', {})
    cond = pd.read_csv(d / 'tables' / 'conditions.csv')
    score_col = 'reversal_score'
    dists[f'{v.get("backend", "unknown")} / {d.name}'] = cond[score_col].dropna().to_numpy()
    rows.append({
        'run': d.name,
        'backend': v.get('backend'),
        'drugs_scored': v.get('drugs_scored'),
        'conditions_scored': v.get('conditions_scored', v.get('drugs_scored')),
        'status': v.get('status'),
        'accepts_user_baseline': v.get('accepts_user_baseline', True),
        'accepts_dose': v.get('accepts_dose', True),
        'score_spread': dg.get('score_spread_across_conditions'),
        'median_pairwise_response_cosine': dg.get('median_pairwise_response_cosine'),
        'legacy_metric_unverified': dg.get('median_pairwise_response_correlation'),
        'ranking_status': v.get('ranking_status', 'legacy_not_reassessed'),
        'degenerate': dg.get('degenerate'),
    })

cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(OUT / 'backend_comparison.csv', index=False)
print(cmp_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=140)
for name, vals in dists.items():
    ax.hist(vals, bins=60, alpha=.55, label=f'{name} (n={len(vals)})')
ax.axvline(0, color='#444', lw=.8)
ax.set(xlabel='Reversal score  (-cos(predicted response, LUAD direction))',
       ylabel='conditions',
       title='FDA drug library: reversal score by backend')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'reversal_by_backend.png')
plt.close(fig)

summary = {
    'library': 'FDA approved (Drug Repurposing Hub, clinical_phase == Launched)',
    'drugs': int(cmp_df['drugs_scored'].max()),
    'doses_uM': [0.1, 1.0, 10.0],
    'rankable_runs': cmp_df.loc[cmp_df['ranking_status'] == 'available', 'run'].tolist(),
    'degenerate_backends': cmp_df.loc[cmp_df['degenerate'] == True, 'backend'].tolist(),
    'note': '按各运行实际状态汇总；旧判据未自动复核。不同评分尺度/背景不能据分布高低比较模型优劣。',
}
(OUT / 'comparison_summary.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
