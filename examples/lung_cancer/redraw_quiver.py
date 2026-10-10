"""从已保存的 embedding 位移表重绘向量场图（不重跑模型）。

位移表由 run_celloracle_ko_oe.py 产出，含每个细胞的 UMAP 坐标、组织来源标签
和扰动位移 (dx, dy)，所以重绘不需要再加载 CellOracle。
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent
TAB = ROOT / 'results/gse198291_celloracle_ko_oe/tables'
FIG = ROOT / 'results/gse198291_celloracle_ko_oe/figures'
FIG.mkdir(parents=True, exist_ok=True)


def draw(csv_path, out_path, title):
    d = pd.read_csv(csv_path)
    umap = d[['umap_x', 'umap_y']].to_numpy()
    shift = d[['dx', 'dy']].to_numpy()
    clus = d['cluster'].astype(str).to_numpy()

    uniq = sorted(pd.unique(clus))
    palette = plt.get_cmap('tab20')
    cmap = {c: palette(i % 20) for i, c in enumerate(uniq)}
    arrow_colors = np.array([cmap[c] for c in clus])

    fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=140)
    ax.scatter(umap[:, 0], umap[:, 1], c='#dcdfe3', s=6, linewidths=0)
    for c in uniq:
        m = clus == c
        ax.scatter(umap[m, 0], umap[m, 1], s=9, color=cmap[c],
                   linewidths=0, label=f'{c} (n={int(m.sum())})')
    ax.quiver(umap[:, 0], umap[:, 1], shift[:, 0], shift[:, 1],
              scale=30, headaxislength=7, headlength=11, headwidth=8,
              linewidths=0.25, width=0.0045, edgecolors='k',
              color=arrow_colors, alpha=1)
    ax.axis('off')
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc='center left', bbox_to_anchor=(1.01, 0.5),
              frameon=False, markerscale=2.0, handletextpad=0.3,
              labelspacing=0.4, title='组织来源', title_fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.79, 1))
    fig.savefig(out_path)
    plt.close(fig)
    print('saved', out_path)


for gene in ('STAT3', 'JUN'):
    for mode, cn in (('ko', '敲除'), ('oe', '过表达')):
        tag = f'{gene}_{mode}'
        draw(TAB / f'{tag}_embedding_shift.csv', FIG / f'{tag}_quiver.png',
             f'CellOracle：{gene} {cn}后的细胞状态位移')
