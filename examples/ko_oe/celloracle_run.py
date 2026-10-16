"""Run one CellOracle TF KO/OE simulation on a user-selected scRNA dataset.

The input matrix must contain raw, nonnegative integer counts with gene symbols
matching the supplied CellOracle base GRN. No data or GRN is downloaded. This
script simulates a network response; it does not measure a perturbation effect.

Heavy dependencies are imported only after CLI parsing so --help works in a
general AIVC environment. Run this file with the CellOracle interpreter.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback


FIGURE_FIELDS = [
    "figure_id", "task", "condition", "backend", "output_type", "question",
    "plot_path", "source_table", "metric", "unit", "baseline_source",
    "status", "reason",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", type=Path, required=True, help="Selected scRNA AnnData with raw counts in X or --counts-layer")
    p.add_argument("--counts-layer", help="Use this AnnData layer as raw counts instead of X")
    p.add_argument("--base-grn", type=Path, required=True, help="Local CellOracle TF-target base GRN parquet; must match species/gene symbols")
    p.add_argument("--target-gene", required=True, help="Gene symbol to simulate; must be an active TF in the fitted GRN")
    p.add_argument("--mode", required=True, choices=("ko", "oe"))
    p.add_argument("--output-dir", type=Path, required=True, help="New or empty output directory")
    p.add_argument("--group-col", help="AnnData obs column for descriptive summaries and plots (e.g. cell_type or tissue)")
    p.add_argument("--grn-unit-col", help="AnnData obs column for cluster-specific GRNs; omitted means one GRN for all cells")
    oe = p.add_mutually_exclusive_group()
    oe.add_argument("--oe-value", type=float, help="Fixed target value on CellOracle's imputed log1p(raw-count) scale")
    oe.add_argument("--oe-percentile", type=float, help="Choose fixed target value from imputed target-gene percentile, 0 < p <= 100")
    p.add_argument("--n-hvg", type=int, default=1500)
    p.add_argument("--n-pcs", type=int, default=30)
    p.add_argument("--n-neighbors", type=int, default=100, help="CellOracle embedding projection neighbors")
    p.add_argument("--n-propagation", type=int, default=3)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--bagging-number", type=int, default=20)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--min-grn-cells", type=int, default=30, help="Minimum cells per GRN unit (preflight heuristic, not model validation)")
    p.add_argument("--seed", type=int, default=123)
    return p


def validate_arguments(a: argparse.Namespace) -> None:
    if not a.target_gene.strip():
        raise ValueError("--target-gene cannot be empty")
    if a.mode == "oe" and a.oe_value is None and a.oe_percentile is None:
        raise ValueError("OE requires an explicit --oe-value or --oe-percentile")
    if a.mode == "ko" and (a.oe_value is not None or a.oe_percentile is not None):
        raise ValueError("OE settings are not applicable to KO")
    if a.oe_value is not None and not (0 < a.oe_value < float("inf")):
        raise ValueError("--oe-value must be finite and positive")
    if a.oe_percentile is not None and not (0 < a.oe_percentile <= 100):
        raise ValueError("--oe-percentile must be in (0, 100]")
    for name in ("n_hvg", "n_pcs", "n_neighbors", "n_propagation", "bagging_number", "n_jobs", "min_grn_cells"):
        if getattr(a, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not (0 < a.alpha < float("inf")):
        raise ValueError("--alpha must be finite and positive")
    for name in ("h5ad", "base_grn"):
        value = getattr(a, name)
        if not value.is_file():
            raise FileNotFoundError(f"{name}: {value}")
    if a.h5ad.resolve() == a.base_grn.resolve():
        raise ValueError("The AnnData and base GRN paths must differ")
    if a.output_dir.exists():
        if not a.output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {a.output_dir}")
        if any(a.output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {a.output_dir}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def dense(matrix):
    import numpy as np
    return np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix)


def check_raw_counts(adata, layer: str | None) -> None:
    import numpy as np
    if layer and layer not in adata.layers:
        raise ValueError(f"Counts layer is absent: {layer}")
    matrix = adata.layers[layer] if layer else adata.X
    values = matrix.data if hasattr(matrix, "tocoo") else np.asarray(matrix)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Raw counts contain no values or NaN/Inf")
    if (values < 0).any() or not np.allclose(values, np.rint(values), atol=1e-6, rtol=0):
        raise ValueError("CellOracle raw-count input must be nonnegative integer counts, not normalized/log expression")
    if adata.n_obs < 4 or adata.n_vars < 4:
        raise ValueError("Need at least four cells and four genes for this workflow")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Cell and gene identifiers must be unique")
    if not adata.obs_names.notna().all() or not adata.var_names.notna().all():
        raise ValueError("Cell and gene identifiers cannot be missing")
    if np.asarray(matrix.sum(axis=1)).reshape(-1).min() <= 0:
        raise ValueError("Every retained cell must have a positive library size")


def choose_oe_value(values, explicit: float | None, percentile: float | None) -> tuple[float, dict]:
    import numpy as np
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all() or x.min() < 0:
        raise ValueError("Invalid imputed target-gene expression")
    value = float(explicit if explicit is not None else np.percentile(x, percentile))
    if not np.isfinite(value) or value <= 0 or value < x.min() or value > x.max():
        raise ValueError("OE target must be positive and inside the imputed target-gene range")
    return value, {
        "imputed_min": float(x.min()), "imputed_max": float(x.max()),
        "imputed_mean": float(x.mean()), "chosen_value": value,
        "choice": "explicit_value" if explicit is not None else f"percentile_{percentile:g}",
        "cells_increased_at_setting": int((x < value).sum()),
        "cells_unchanged_at_setting": int((x == value).sum()),
        "cells_decreased_at_setting": int((x > value).sum()),
        "note": "CellOracle sets a fixed imputed log1p(raw-count) scale value; cells above it are decreased. This is not an experimental fold change.",
    }


def check_labels(adata, column: str, min_cells: int = 1) -> list[str]:
    if column not in adata.obs:
        raise ValueError(f"AnnData obs column is absent: {column}")
    labels = adata.obs[column]
    if labels.isna().any() or (labels.astype(str).str.strip() == "").any():
        raise ValueError(f"AnnData obs column contains missing/blank labels: {column}")
    sizes = labels.astype(str).value_counts()
    if sizes.min() < min_cells:
        raise ValueError(f"GRN unit {sizes.idxmin()!r} has {sizes.min()} cells; minimum is {min_cells}")
    return sorted(sizes.index.tolist())


def prepare_oracle_grn_labels(adata, column: str) -> None:
    """CellOracle's palette setup requires categorical GRN-unit labels."""
    if column not in adata.obs:
        raise ValueError(f"AnnData obs column is absent: {column}")
    if adata.obs[column].isna().any():
        raise ValueError(f"AnnData obs column contains missing labels: {column}")
    adata.obs[column] = adata.obs[column].astype("category")


def add_figure(rows: list[dict], a: argparse.Namespace, figure_id: str, question: str,
               source_table: str, metric: str, unit: str, status: str,
               plot_path: str = "", reason: str = "") -> None:
    rows.append({
        "figure_id": figure_id, "task": a.mode, "condition": f"{a.target_gene}_{a.mode}",
        "backend": "CellOracle", "output_type": "simulation", "question": question,
        "plot_path": plot_path, "source_table": source_table, "metric": metric,
        "unit": unit, "baseline_source": "raw/baseline_imputed_expression.csv.gz",
        "status": status, "reason": reason,
    })


def make_figures(a, output: Path, gene_effects, cell_shift, imputed_target, oe_value, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    fig_dir = output / "figures"
    tag = f"{a.target_gene} {a.mode.upper()}"
    top = gene_effects.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.34 * len(top) + 1.5)), dpi=150)
    ax.barh(top["gene"], top["mean_delta"], color=np.where(top["mean_delta"] >= 0, "#bb504b", "#4379a1"))
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set(xlabel="Mean simulated delta_X (CellOracle imputed log1p(raw-count) scale)",
           title=f"{tag}: genes selected by mean absolute delta_X")
    fig.tight_layout()
    fig.savefig(fig_dir / "gene_effects.png")
    plt.close(fig)
    add_figure(rows, a, "gene_effects", "Which genes have the largest simulated changes?",
               "tables/gene_effects.csv", "mean_delta; selected by mean_abs_delta",
               "CellOracle imputed log1p(raw-count) scale", "generated", "figures/gene_effects.png")

    # Draw both projections on identical UMAP coordinates and arrow scale.
    x, y = cell_shift["umap_x"].to_numpy(), cell_shift["umap_y"].to_numpy()
    orig = cell_shift[["dx", "dy"]].to_numpy()
    random = cell_shift[["dx_random", "dy_random"]].to_numpy()
    extent = max(float(np.ptp(x)), float(np.ptp(y)), 1e-8)
    max_vector = max(float(np.linalg.norm(orig, axis=1).max()),
                     float(np.linalg.norm(random, axis=1).max()), 1e-8)
    common_scale = max_vector / (0.09 * extent)
    sampled = np.sort(np.random.default_rng(a.seed).choice(len(x), size=min(len(x), 400), replace=False))
    groups = cell_shift["group"].astype(str).to_numpy()
    uniques = sorted(pd.unique(groups))
    palette = plt.get_cmap("tab20")
    colors = {group: palette(i % 20) for i, group in enumerate(uniques)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, sharex=True, sharey=True)
    for ax, vectors, label in zip(axes, (orig, random), ("Simulated perturbation", "Randomized delta_X projection")):
        for group in uniques:
            mask = groups == group
            ax.scatter(x[mask], y[mask], s=4, color=colors[group], alpha=0.55,
                       label=f"{group} (n={int(mask.sum())})")
        ax.quiver(x[sampled], y[sampled], vectors[sampled, 0], vectors[sampled, 1],
                  angles="xy", scale_units="xy", scale=common_scale, width=0.003,
                  color="#202b35", alpha=0.78)
        ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title=label)
    if len(uniques) <= 12:
        axes[1].legend(fontsize=7, loc="best", title=a.group_col or "all cells")
    fig.suptitle(f"{tag}: CellOracle state shift (common arrow scale)")
    fig.tight_layout()
    fig.savefig(fig_dir / "state_shift.png")
    plt.close(fig)
    add_figure(rows, a, "state_shift", "How do simulated cell-state shifts compare with the randomized projection?",
               "tables/cell_state_shift.csv.gz", "delta_embedding; delta_embedding_random",
               "UMAP coordinate projection", "generated", "figures/state_shift.png")

    labels = uniques if len(uniques) <= 20 else ["all_cells"]
    plot_data = cell_shift if len(uniques) <= 20 else cell_shift.assign(group="all_cells")
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.7 + 2), 4.8), dpi=150)
    positions, values, face = [], [], []
    for i, group in enumerate(labels):
        subset = plot_data.loc[plot_data["group"] == group]
        for offset, column, color in ((-0.18, "shift_magnitude", "#bb504b"),
                                      (0.18, "random_shift_magnitude", "#4379a1")):
            positions.append(i + offset)
            values.append(subset[column].to_numpy())
            face.append(color)
    boxes = ax.boxplot(values, positions=positions, widths=0.28, patch_artist=True, showfliers=False)
    for box, color in zip(boxes["boxes"], face):
        box.set_facecolor(color)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set(ylabel="Embedding shift magnitude (UMAP projection units)",
           title=f"{tag}: descriptive cell-level shift distribution")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#bb504b", label="Simulated"),
                       Patch(color="#4379a1", label="Randomized projection")])
    fig.tight_layout()
    fig.savefig(fig_dir / "shift_distribution.png")
    plt.close(fig)
    add_figure(rows, a, "shift_distribution", "How do shift magnitudes differ by selected group?",
               "tables/cell_state_shift.csv.gz", "L2 magnitude of projected shift",
               "UMAP coordinate projection", "generated", "figures/shift_distribution.png",
               reason="Cell-level descriptive distribution; no biological replicate test")

    if a.mode == "oe":
        fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
        ax.hist(imputed_target, bins=30, color="#4379a1", edgecolor="white")
        ax.axvline(oe_value, color="#bb504b", linewidth=2, label=f"Fixed target value = {oe_value:.4g}")
        ax.set(xlabel="Baseline imputed target-gene expression (log1p raw-count scale)",
               ylabel="Cells", title=f"{tag}: chosen input setting")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "oe_setting.png")
        plt.close(fig)
        add_figure(rows, a, "oe_setting", "Where does the OE setting fall in the observed imputed range?",
                   "tables/target_expression_setting.csv", "imputed target-gene expression",
                   "CellOracle imputed log1p(raw-count) scale", "generated", "figures/oe_setting.png")


def run(a: argparse.Namespace) -> Path:
    output = a.output_dir.resolve()
    for section in ("raw", "tables", "figures"):
        (output / section).mkdir(parents=True, exist_ok=True)
    status_file = output / "manifest.json"
    manifest = {
        "schema_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "backend": {"name": "CellOracle", "version": "pending"},
        "task": a.mode, "target_gene": a.target_gene,
        "inputs": {"h5ad": str(a.h5ad.resolve()), "h5ad_sha256": sha256(a.h5ad),
                   "base_grn": str(a.base_grn.resolve()), "base_grn_sha256": sha256(a.base_grn),
                   "counts_layer": a.counts_layer},
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
                       if k not in {"h5ad", "base_grn", "output_dir"}},
        "software": {"python": sys.version.split()[0]},
        "output_type": "network_simulation", "independent_validation": "none",
        "celloracle_internal_scale": "log1p(raw counts), then KNN smoothing into imputed_count; native simulated_count and delta_X retain that internal scale",
    }
    provenance = {
        "schema_version": "1.0", "status": "running",
        "created_utc": manifest["created_utc"],
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "inputs": dict(manifest["inputs"]),
        "backend": dict(manifest["backend"]),
        "software": dict(manifest["software"]),
        "grn": {"base_grn_path": str(a.base_grn.resolve()),
                "base_grn_sha256": manifest["inputs"]["base_grn_sha256"],
                "scope": "pending", "grn_unit_column": a.grn_unit_col,
                "alpha": a.alpha, "bagging_number": a.bagging_number,
                "min_grn_cells": a.min_grn_cells},
        "independent_validation": "none",
    }
    dump_json(status_file, manifest)
    dump_json(output / "provenance.json", provenance)
    (output / "errors.jsonl").touch()
    (output / "figures" / "figure_manifest.csv").write_text(",".join(FIGURE_FIELDS) + "\n", encoding="utf-8")
    stage = "dependencies"
    figure_rows: list[dict] = []
    validation = {"input_status": "pending", "simulation_status": "pending",
                  "projection_status": "pending", "figure_status": "pending",
                  "independent_validation": "none"}
    try:
        import numpy as np
        import pandas as pd
        import anndata as ad
        import scanpy as sc
        import celloracle as co

        manifest["backend"]["version"] = co.__version__
        manifest["software"].update({"anndata": ad.__version__, "scanpy": sc.__version__,
                                      "numpy": np.__version__, "pandas": pd.__version__})
        provenance["backend"] = dict(manifest["backend"])
        provenance["software"] = dict(manifest["software"])
        dump_json(status_file, manifest)
        dump_json(output / "provenance.json", provenance)
        stage = "input_validation"
        adata = ad.read_h5ad(a.h5ad)
        check_raw_counts(adata, a.counts_layer)
        if a.counts_layer:
            adata.X = adata.layers[a.counts_layer].copy()
        if a.target_gene not in adata.var_names:
            raise ValueError(f"Target gene is absent from AnnData var_names: {a.target_gene}")
        if np.asarray(adata[:, a.target_gene].X.sum()).item() <= 0:
            raise ValueError("Target gene has zero raw counts across all cells")
        group_col = a.group_col or "__aivc_all_cells"
        if a.group_col:
            group_labels = check_labels(adata, a.group_col)
        else:
            if group_col in adata.obs:
                raise ValueError(f"Reserved obs column already exists: {group_col}")
            adata.obs[group_col] = "all_cells"
            group_labels = ["all_cells"]
        if a.grn_unit_col:
            grn_labels = check_labels(adata, a.grn_unit_col, a.min_grn_cells)
            grn_scope = "cluster"
            grn_col = a.grn_unit_col
        else:
            grn_col = "__aivc_grn_unit"
            if grn_col in adata.obs:
                raise ValueError(f"Reserved obs column already exists: {grn_col}")
            adata.obs[grn_col] = "all_cells"
            grn_labels = ["all_cells"]
            grn_scope = "whole"
        if adata.n_obs < a.min_grn_cells:
            raise ValueError(f"Dataset has {adata.n_obs} cells, below --min-grn-cells {a.min_grn_cells}")
        base_grn = pd.read_parquet(a.base_grn)
        if "gene_short_name" not in base_grn.columns or a.target_gene not in base_grn.columns:
            raise ValueError("Base GRN must contain gene_short_name and the target TF as a column; check species and gene IDs")
        overlap = len(set(adata.var_names) & set(base_grn["gene_short_name"].dropna().astype(str)))
        if overlap < 10:
            raise ValueError(f"Only {overlap} genes overlap the base GRN; check species and gene ID system")
        validation.update(input_status="passed", cells=int(adata.n_obs), genes_input=int(adata.n_vars),
                          group_column=a.group_col, group_labels=group_labels,
                          grn_unit_column=a.grn_unit_col, grn_units=grn_labels,
                          grn_scope=grn_scope, base_grn_gene_overlap=overlap)
        provenance["grn"].update(scope=grn_scope, grn_unit_column=a.grn_unit_col,
                                 base_grn_gene_overlap=overlap,
                                 fitting_source=("filtered cluster Links TF dictionaries"
                                                 if grn_scope == "cluster" else "base TF dictionary across all cells"))
        dump_json(output / "provenance.json", provenance)
        dump_json(output / "validation.json", validation)

        stage = "preprocessing"
        # Select HVGs on a normalized copy. The object imported into CellOracle
        # receives raw counts again after UMAP construction.
        if a.n_hvg < adata.n_vars:
            hvg_view = adata.copy()
            sc.pp.normalize_total(hvg_view, target_sum=1e4)
            sc.pp.log1p(hvg_view)
            sc.pp.highly_variable_genes(hvg_view, n_top_genes=a.n_hvg, flavor="seurat")
            selected = set(hvg_view.var_names[hvg_view.var["highly_variable"]]) | {a.target_gene}
            keep = [gene for gene in adata.var_names if gene in selected]
        else:
            keep = list(adata.var_names)
        selected_raw = adata[:, keep].copy()
        raw_matrix = selected_raw.X.copy()
        sc.pp.normalize_total(selected_raw, target_sum=1e4)
        sc.pp.log1p(selected_raw)
        n_pcs = min(a.n_pcs, selected_raw.n_obs - 1, selected_raw.n_vars - 1)
        if n_pcs < 2:
            raise ValueError("Too few selected genes or cells for PCA/UMAP")
        sc.pp.pca(selected_raw, n_comps=n_pcs, random_state=a.seed)
        sc.pp.neighbors(selected_raw, n_neighbors=min(15, selected_raw.n_obs - 1), random_state=a.seed)
        sc.tl.umap(selected_raw, random_state=a.seed)
        selected_raw.X = raw_matrix
        selected_raw.layers.clear()
        # X contains raw counts again; the PCA copy's log1p marker is now stale.
        selected_raw.uns.pop("log1p", None)
        prepare_oracle_grn_labels(selected_raw, grn_col)
        manifest["genes_used"] = int(selected_raw.n_vars)
        manifest["n_pcs_used"] = n_pcs
        dump_json(status_file, manifest)

        stage = "fit_grn"
        oracle = co.Oracle()
        oracle.import_anndata_as_raw_count(adata=selected_raw, cluster_column_name=grn_col,
                                            embedding_name="X_umap")
        oracle.import_TF_data(TF_info_matrix=base_grn)
        oracle.perform_PCA()
        oracle.knn_imputation(n_pca_dims=n_pcs, k=min(15, selected_raw.n_obs - 1),
                              balanced=True, b_sight=min(150, selected_raw.n_obs - 1),
                              b_maxl=min(150, selected_raw.n_obs - 1), n_jobs=a.n_jobs)
        n_links = None
        if grn_scope == "cluster":
            # Filtered Links define each cluster's TF dictionary. Whole-GRN
            # fitting instead uses the imported base TF dictionary directly.
            links = oracle.get_links(cluster_name_for_GRN_unit=grn_col, alpha=a.alpha,
                                     bagging_number=a.bagging_number, verbose_level=0, n_jobs=a.n_jobs)
            links.filter_links(p=0.001, weight="coef_abs", threshold_number=1000)
            n_links = sum(len(edges) for edges in links.filtered_links.values())
            if n_links <= 0:
                raise ValueError("No cluster GRN links survived filtering")
            for i, (unit, edges) in enumerate(links.filtered_links.items()):
                edges.assign(grn_unit=unit).to_csv(output / "tables" / f"filtered_cluster_links_{i:03d}.csv", index=False)
            oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
        oracle.fit_GRN_for_simulation(alpha=a.alpha, GRN_unit=grn_scope,
                                      use_cluster_specific_TFdict=(grn_scope == "cluster"), verbose_level=0)
        # These are the exact coefficients used by simulate_shift. In whole
        # mode CellOracle fits from the base TFdict, not the filtered Links.
        fitted = (oracle.coef_matrix_per_cluster if grn_scope == "cluster"
                  else {"all_cells": oracle.coef_matrix})
        fitted_index = []
        for i, (unit, coefficients) in enumerate(fitted.items()):
            relative = f"raw/simulation_grn_coefficients_{i:03d}.csv.gz"
            coefficients.to_csv(output / relative, compression="gzip")
            fitted_index.append({"grn_unit": str(unit), "coefficient_path": relative,
                                 "nonzero_coefficients": int((coefficients.to_numpy() != 0).sum())})
        pd.DataFrame(fitted_index).to_csv(output / "tables" / "simulation_grn_index.csv", index=False)
        manifest["filtered_grn_links"] = int(n_links) if n_links is not None else None
        manifest["filtered_links_used_for_simulation"] = grn_scope == "cluster"
        provenance["grn"].update(filtered_grn_links=int(n_links) if n_links is not None else None,
                                 filtered_links_used_for_simulation=(grn_scope == "cluster"),
                                 coefficient_index="tables/simulation_grn_index.csv")
        dump_json(output / "provenance.json", provenance)
        validation["filtered_grn_links"] = int(n_links) if n_links is not None else None
        validation["fitted_grn_source"] = ("cluster filtered Links TF dictionaries"
                                           if grn_scope == "cluster" else "base TF dictionary fitted across all cells")
        dump_json(status_file, manifest)

        stage = "simulation"
        cells = oracle.adata.obs_names
        genes = oracle.adata.var_names
        imputed = pd.DataFrame(dense(oracle.adata.layers["imputed_count"]), index=cells, columns=genes)
        imputed.to_csv(output / "raw" / "baseline_imputed_expression.csv.gz", compression="gzip")
        if a.mode == "ko":
            setting = 0.0
            target_info = {"chosen_value": 0.0, "choice": "ko_zero", "imputed_min": float(imputed[a.target_gene].min()),
                           "imputed_max": float(imputed[a.target_gene].max())}
        else:
            setting, target_info = choose_oe_value(imputed[a.target_gene].to_numpy(), a.oe_value, a.oe_percentile)
        pd.DataFrame({"cell": cells, "baseline_imputed": imputed[a.target_gene].to_numpy(),
                      "fixed_input_setting": setting}).to_csv(output / "tables" / "target_expression_setting.csv", index=False)
        oracle.simulate_shift(perturb_condition={a.target_gene: setting}, GRN_unit=grn_scope,
                              n_propagation=a.n_propagation)
        simulated = pd.DataFrame(dense(oracle.adata.layers["simulated_count"]), index=cells, columns=genes)
        delta = pd.DataFrame(dense(oracle.adata.layers["delta_X"]), index=cells, columns=genes)
        if not np.isfinite(simulated.to_numpy()).all() or not np.isfinite(delta.to_numpy()).all():
            raise ValueError("CellOracle produced NaN/Inf simulated expression or delta_X")
        if not np.allclose(delta.to_numpy(), simulated.to_numpy() - imputed.to_numpy(), atol=1e-8):
            raise ValueError("CellOracle delta_X does not match simulated_count minus imputed_count")
        simulated.to_csv(output / "raw" / "simulated_expression.csv.gz", compression="gzip")
        delta.to_csv(output / "raw" / "delta_X.csv.gz", compression="gzip")
        gene_effects = pd.DataFrame({
            "gene": genes, "mean_delta": delta.mean(axis=0).to_numpy(),
            "mean_abs_delta": delta.abs().mean(axis=0).to_numpy(),
            "fraction_cells_up": (delta > 0).mean(axis=0).to_numpy(),
            "fraction_cells_down": (delta < 0).mean(axis=0).to_numpy(),
        }).sort_values("mean_abs_delta", ascending=False)
        gene_effects.to_csv(output / "tables" / "gene_effects.csv", index=False)
        validation.update(simulation_status="result_available", target_setting=target_info,
                          delta_x_finite=True, target_gene_mean_delta=float(delta[a.target_gene].mean()),
                          note="delta_X is simulated_count minus imputed_count, both on CellOracle's internal imputed log1p(raw-count) scale; it is not log2FC or measured perturbation response")
        dump_json(output / "validation.json", validation)

        stage = "projection"
        oracle.estimate_transition_prob(n_neighbors=min(a.n_neighbors, selected_raw.n_obs - 1),
                                        knn_random=True, sampled_fraction=1, n_jobs=a.n_jobs,
                                        random_seed=a.seed, calculate_randomized=True)
        oracle.calculate_embedding_shift(sigma_corr=0.05)
        embedding = np.asarray(oracle.embedding)
        shift = np.asarray(oracle.delta_embedding)
        random_shift = np.asarray(oracle.delta_embedding_random)
        if embedding.shape != (len(cells), 2) or shift.shape != embedding.shape or random_shift.shape != embedding.shape:
            raise ValueError("Unexpected CellOracle embedding/projection shape")
        if not np.isfinite(embedding).all() or not np.isfinite(shift).all() or not np.isfinite(random_shift).all():
            raise ValueError("CellOracle embedding or state shift contains NaN/Inf")
        groups = oracle.adata.obs[group_col].astype(str).to_numpy()
        cell_shift = pd.DataFrame({
            "cell": cells, "group": groups, "umap_x": embedding[:, 0], "umap_y": embedding[:, 1],
            "dx": shift[:, 0], "dy": shift[:, 1], "dx_random": random_shift[:, 0],
            "dy_random": random_shift[:, 1], "shift_magnitude": np.linalg.norm(shift, axis=1),
            "random_shift_magnitude": np.linalg.norm(random_shift, axis=1),
            "vector_difference_magnitude": np.linalg.norm(shift - random_shift, axis=1),
        })
        cell_shift.to_csv(output / "tables" / "cell_state_shift.csv.gz", index=False, compression="gzip")
        cell_shift.groupby("group", observed=True)[["shift_magnitude", "random_shift_magnitude"]].agg(
            ["count", "mean", "median"]).to_csv(output / "tables" / "shift_by_group.csv")
        validation.update(projection_status="result_available", random_control="delta_X row permutation with sign randomization before embedding projection; not a randomized GRN",
                          mean_shift_magnitude=float(cell_shift.shift_magnitude.mean()),
                          mean_random_shift_magnitude=float(cell_shift.random_shift_magnitude.mean()))
        dump_json(output / "validation.json", validation)

        stage = "figures"
        try:
            make_figures(a, output, gene_effects, cell_shift, imputed[a.target_gene].to_numpy(), setting, figure_rows)
            validation["figure_status"] = "generated"
        except Exception as figure_error:
            validation["figure_status"] = "failed"
            validation["figure_error"] = repr(figure_error)
            (output / "errors.jsonl").write_text(json.dumps({"stage": stage, "error": repr(figure_error),
                "traceback": traceback.format_exc()}, ensure_ascii=False) + "\n", encoding="utf-8")
            add_figure(figure_rows, a, "requested_figures", "Default CellOracle figures",
                       "tables/gene_effects.csv;tables/cell_state_shift.csv.gz", "", "", "failed", reason=repr(figure_error))
        add_figure(figure_rows, a, "response_heatmap", "How do simulated effects vary by group?",
                   "raw/delta_X.csv.gz", "mean delta_X", "CellOracle imputed log1p(raw-count) scale", "skipped",
                   reason="This runner does not compute a group-level heatmap; use the complete saved delta_X with a reviewed grouping rule")
        add_figure(figure_rows, a, "pathway_enrichment", "Which pathways are affected?",
                   "tables/gene_effects.csv", "", "", "skipped",
                   reason="No user-approved gene-set database/version or mapping supplied")
        add_figure(figure_rows, a, "network_effects", "What is the target's fitted regulatory neighborhood?",
                   "tables/simulation_grn_index.csv", "simulation GRN coefficients", "", "skipped",
                   reason="Exact simulation coefficients are saved; no reviewed network layout/edge selection was supplied")
        add_figure(figure_rows, a, "perturbation_scores", "How does perturbation align with a reference trajectory?",
                   "", "", "", "skipped", reason="No trusted trajectory or target-state reference supplied")
        if a.mode == "oe":
            add_figure(figure_rows, a, "ko_oe_comparison", "How do matched KO and OE effects compare?",
                       "", "", "", "skipped", reason="This invocation runs OE only; no matched KO run was supplied")
        pd.DataFrame(figure_rows, columns=FIGURE_FIELDS).to_csv(output / "figures" / "figure_manifest.csv", index=False)
        dump_json(output / "validation.json", validation)
        report = [f"# CellOracle {a.target_gene} {a.mode.upper()} run", "",
                  f"Input: `{a.h5ad.resolve()}`; base GRN: `{a.base_grn.resolve()}`.",
                  f"{len(cells)} cells; {len(genes)} modeled genes; GRN scope: {grn_scope}.",
                  f"Fixed target setting: {setting:.6g} (CellOracle imputed log1p(raw-count) scale).",
                  f"Mean simulated target-gene delta_X: {validation['target_gene_mean_delta']:.6g}.",
                  f"Mean UMAP shift magnitude: {validation['mean_shift_magnitude']:.6g}; randomized projection: {validation['mean_random_shift_magnitude']:.6g}.",
                  "", "Full native simulated expression, baseline imputed expression and delta_X are in `raw/`.",
                  "Figure availability and source tables are in `figures/figure_manifest.csv`.",
                  "Input hashes, software versions and GRN fitting source are in `provenance.json`.",
                  "The random control permutes delta_X and signs before projection while keeping the fitted GRN fixed.",
                  "Result type: network simulation in the CellOracle expression scale. Independent experimental validation: unvalidated."]
        if n_links is not None:
            report.append(f"Cluster-specific TF dictionaries used {n_links} filtered Links edges.")
        if a.mode == "oe":
            report.append(f"The fixed OE setting decreases {target_info['cells_decreased_at_setting']} cells whose imputed baseline exceeds it.")
        (output / "report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
        provenance.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
        dump_json(output / "provenance.json", provenance)
        manifest.update({
            "status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "outputs": sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()),
            "scientific_status": "simulated_unvalidated",
        })
        dump_json(status_file, manifest)
        return output
    except Exception as exc:
        validation["failed_stage"] = stage
        validation["error"] = repr(exc)
        validation["simulation_status"] = "failed" if stage in {"input_validation", "preprocessing", "fit_grn", "simulation"} else validation["simulation_status"]
        if stage == "projection":
            validation["projection_status"] = "failed"
        dump_json(output / "validation.json", validation)
        manifest.update(status="failed", failed_stage=stage, error=repr(exc))
        dump_json(status_file, manifest)
        provenance.update(status="failed", failed_stage=stage, error=repr(exc),
                          failed_utc=datetime.now(timezone.utc).isoformat())
        dump_json(output / "provenance.json", provenance)
        with (output / "errors.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": stage, "error": repr(exc),
                                     "traceback": traceback.format_exc()}, ensure_ascii=False) + "\n")
        (output / "report.md").write_text(
            f"# CellOracle run failed\n\nStage: `{stage}`\n\nError: `{exc!r}`\n\nSee `errors.jsonl`, `validation.json`, and `provenance.json`.\n",
            encoding="utf-8")
        raise


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    try:
        validate_arguments(a)
        output = a.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        run(a)
        print(f"Completed CellOracle simulation: {output}")
        return 0
    except Exception as exc:
        print(f"CellOracle run failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
