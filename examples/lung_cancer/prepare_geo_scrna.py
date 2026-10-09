from pathlib import Path
import csv
import gzip
import json
import re
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

root = Path(__file__).resolve().parent
source = root / 'data' / 'GSE198291_allcounts2.csv.gz'
metadata = root / 'data' / 'GSE198291_series_matrix.txt.gz'
out = root / 'results' / 'geo_gse198291'
out.mkdir(parents=True, exist_ok=True)

with gzip.open(metadata, 'rt', encoding='utf-8', errors='replace') as fh:
    fields = {row[0]: row[1:] for row in csv.reader(fh, delimiter='\t')
              if row and row[0] in {'!Sample_geo_accession', '!Sample_source_name_ch1', '!Sample_characteristics_ch1'}}
meta = pd.DataFrame({'geo_accession': fields['!Sample_geo_accession'],
                     'geo_source_name': fields['!Sample_source_name_ch1'],
                     'tissue': fields['!Sample_characteristics_ch1']})
meta['tissue'] = meta['tissue'].str.removeprefix('tissue: ')

frame = pd.read_csv(source, index_col=0)
assert frame.index.is_unique and frame.columns.is_unique
assert len(frame.columns) == len(meta)
# GEO source names contain repeated barcodes; pandas mangles duplicate matrix headers.
# Retain per-column GEO accession by position and flag the one unexpected source-name mismatch.
meta['matrix_cell_id'] = frame.columns.to_numpy()
meta['source_name_match'] = [str(m).split('.')[0] == str(s)
                             for m, s in zip(meta['matrix_cell_id'], meta['geo_source_name'])]
meta['donor'] = meta['matrix_cell_id'].str.extract(r'^(PT\d+)_')[0]
assert np.isfinite(frame.to_numpy(dtype=np.float32)).all()
meta = meta.set_index('matrix_cell_id')
meta['cell_type'] = 'unannotated'

counts = frame.to_numpy(dtype=np.float32, copy=True).T
adata = sc.AnnData(X=sparse.csr_matrix(counts))
adata.obs_names = meta.index.copy()
adata.var_names = frame.index.copy()
adata.obs = meta.copy()
adata.var_names_make_unique()
adata.obs_names_make_unique()
adata.obs['n_genes'] = np.asarray((adata.X > 0).sum(axis=1)).ravel()
adata = adata[adata.obs['n_genes'] >= 200].copy()
adata.write_h5ad(out / 'GSE198291_counts_qc200.h5ad', compression='gzip')
meta.to_csv(out / 'sample_metadata.csv')
summary = {
    'accession': 'GSE198291',
    'source': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE198nnn/GSE198291/suppl/GSE198291_allcounts2.csv.gz',
    'technology': 'single-cell RNA-seq counts',
    'raw_shape_cells_genes': [int(counts.shape[0]), int(counts.shape[1])],
    'qc_cells': int(adata.n_obs),
    'tissue_counts': adata.obs['tissue'].value_counts().to_dict(),
    'donors': adata.obs['donor'].nunique(),
    'source_name_mismatches': int((~meta['source_name_match']).sum()),
    'duplicate_geo_source_names': int(meta['geo_source_name'].duplicated().sum()),
    'cell_type': 'unannotated; no validated cell-type label in GEO matrix',
    'target_coverage': {g: int((adata[:, g].X.toarray() > 0).sum()) for g in ['STAT3', 'JUN']},
    'caution': 'No KO/OE condition labels. Does not constitute perturbation training data.'
}
(out / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
