"""Run Geneformer's independent representation-space KO/OE route.

This is an adapter for the official Geneformer package, not an expression
prediction model. KO means token deletion; OE means moving a token to the
front of a rank-value encoding. Neither operation supplies a molecular dose.

Example::

    python geneformer_route.py --input-h5ad cells.h5ad --model-dir path/to/Geneformer-V2 --model-version V2 --target-ensembl ENSG00000168610 --perturbation both --confirm-raw-full-transcriptome --output-dir results/geneformer
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import re
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


OFFICIAL_SOURCES = {
    "tokenizer": "https://geneformer.readthedocs.io/en/latest/geneformer.tokenizer.html",
    "perturber": "https://geneformer.readthedocs.io/en/latest/geneformer.in_silico_perturber.html",
    "stats": "https://geneformer.readthedocs.io/en/latest/geneformer.in_silico_perturber_stats.html",
}
PERTURB_TYPES = {"KO": "delete", "OE": "overexpress"}
ENSEMBL_RE = re.compile(r"^ENSG\d{11}$")
FIGURE_FIELDS = ["figure_id", "task", "condition", "backend", "output_type", "question", "plot_path", "source_table", "metric", "unit", "baseline_source", "status", "reason"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-h5ad", type=Path, required=True, help="User's human scRNA .h5ad; X must be raw counts, without feature selection")
    parser.add_argument("--model-dir", type=Path, required=True, help="Local Geneformer checkpoint directory, matched to --model-version")
    parser.add_argument("--model-version", choices=("V1", "V2"), required=True)
    parser.add_argument("--target-ensembl", required=True, help="Versionless human Ensembl gene ID, e.g. ENSG00000168610")
    parser.add_argument("--perturbation", choices=("KO", "OE", "both"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-raw-full-transcriptome", action="store_true", help="Confirm X is unselected raw scRNA counts, not log/normalized/HVG data")
    parser.add_argument("--model-type", choices=("Pretrained", "CellClassifier", "GeneClassifier", "MTLCellClassifier"), default="Pretrained")
    parser.add_argument("--num-classes", type=int, default=0, help="Required (>1) when using a classifier checkpoint")
    parser.add_argument("--cell-type-column", help="Optional obs column to restrict the analysis")
    parser.add_argument("--cell-type", help="Value in --cell-type-column to analyze")
    parser.add_argument("--max-ncells", type=int, help="Optional cell cap for a small pilot run")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--forward-batch-size", type=int, default=16)
    return parser.parse_args(argv)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_figure_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIGURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def figure_row(condition: str, figure_id: str, status: str, question: str, *,
               plot_path: str = "", source_table: str = "", metric: str = "",
               reason: str = "", embedding_kind: str = "") -> dict:
    return {
        "figure_id": f"{condition}_{figure_id}", "task": condition, "condition": condition,
        "backend": "Geneformer", "output_type": "embedding" if figure_id == "embedding_distance" else "expression",
        "question": question, "plot_path": plot_path, "source_table": source_table,
        "metric": metric, "unit": "dimensionless" if metric else "",
        "baseline_source": "original rank-token encoding" if embedding_kind else "",
        "status": status, "reason": reason,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_options(args: argparse.Namespace) -> tuple[Path, Path, Path, list[str]]:
    # Path.resolve() follows links and may be denied in restricted Windows
    # environments even for an otherwise readable, explicitly selected file.
    input_h5ad = args.input_h5ad.expanduser().absolute()
    model_dir = args.model_dir.expanduser().absolute()
    output_dir = args.output_dir.expanduser().absolute()
    if not input_h5ad.is_file() or input_h5ad.suffix.lower() != ".h5ad":
        raise ValueError("--input-h5ad must name one existing .h5ad file")
    if not model_dir.is_dir():
        raise ValueError("--model-dir must name an existing checkpoint directory")
    if not ENSEMBL_RE.fullmatch(args.target_ensembl):
        raise ValueError("--target-ensembl must be a versionless human ENSG ID")
    if not args.confirm_raw_full_transcriptome:
        raise ValueError("Confirm that X contains unselected raw human scRNA counts with --confirm-raw-full-transcriptome")
    if bool(args.cell_type_column) != bool(args.cell_type):
        raise ValueError("--cell-type-column and --cell-type must be supplied together")
    if args.model_type == "Pretrained" and args.num_classes != 0:
        raise ValueError("Pretrained checkpoint requires --num-classes 0")
    if args.model_type != "Pretrained" and args.num_classes < 2:
        raise ValueError("Classifier checkpoint requires --num-classes >= 2")
    if args.nproc < 1 or args.forward_batch_size < 1 or (args.max_ncells is not None and args.max_ncells < 1):
        raise ValueError("nproc, forward batch size, and any cell cap must be positive")
    if input_h5ad == output_dir or input_h5ad in output_dir.parents:
        raise ValueError("Output directory cannot be the input file or its parent")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if args.perturbation == "both":
        conditions = ["KO", "OE"]
    else:
        conditions = [args.perturbation]
    return input_h5ad, model_dir, output_dir, conditions


def validate_h5ad(input_h5ad: Path, target: str, cell_type_column: str | None, cell_type: str | None) -> dict:
    """Check schema and a deterministic count sample without claiming full-file QC."""
    try:
        import anndata
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Input validation needs anndata and numpy in the selected environment") from exc

    adata = anndata.read_h5ad(input_h5ad, backed="r")
    try:
        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("Input h5ad has no cells or genes")
        if "ensembl_id" not in adata.var:
            raise ValueError("Input .var needs an ensembl_id column")
        if "n_counts" not in adata.obs:
            raise ValueError("Input .obs needs an n_counts column")
        ids = adata.var["ensembl_id"].astype(str).tolist()
        if any(not ENSEMBL_RE.fullmatch(gene) for gene in ids):
            raise ValueError(".var['ensembl_id'] must contain versionless human ENSG IDs, with no null/symbol values")
        if target not in ids:
            raise ValueError(f"Target {target} is absent from .var['ensembl_id']")
        counts = np.asarray(adata.obs["n_counts"], dtype=float)
        selected_indices = np.arange(adata.n_obs)
        if "filter_pass" in adata.obs:
            passed = np.asarray(adata.obs["filter_pass"])
            if not np.isin(passed, [0, 1, False, True]).all():
                raise ValueError(".obs['filter_pass'] must contain 0/1 values")
            selected_indices = selected_indices[passed == 1]
        if cell_type_column:
            if cell_type_column not in adata.obs:
                raise ValueError(f"Cell type column {cell_type_column!r} is absent from .obs")
            selected_indices = selected_indices[np.asarray(adata.obs[cell_type_column].astype(str))[selected_indices] == cell_type]
        if len(selected_indices) == 0:
            raise ValueError("No cells remain after filter_pass and cell-type filters")
        if not np.isfinite(counts[selected_indices]).all() or (counts[selected_indices] <= 0).any():
            raise ValueError("Selected .obs['n_counts'] must contain positive finite totals")
        sample_indices = selected_indices[: min(32, len(selected_indices))]
        sample = adata[sample_indices, :].X
        if hasattr(sample, "toarray"):
            sample = sample.toarray()
        sample = np.asarray(sample, dtype=float)
        if not np.isfinite(sample).all() or (sample < 0).any() or not np.allclose(sample, np.rint(sample), atol=1e-6):
            raise ValueError("Sampled X values are not nonnegative integer raw counts")
        sampled_totals = sample.sum(axis=1)
        if not np.allclose(sampled_totals, counts[sample_indices], atol=1, rtol=1e-4):
            raise ValueError("Sampled X row sums differ from .obs['n_counts']; check counts and feature selection")
        target_indices = [i for i, gene in enumerate(ids) if gene == target]
        # Use only one advanced index when reading a backed AnnData matrix.
        target_slice = adata[:, target_indices].X
        if hasattr(target_slice, "toarray"):
            target_slice = target_slice.toarray()
        target_detected = int(np.count_nonzero(np.asarray(target_slice)[selected_indices].sum(axis=1)))
        return {
            "status": "passed_sampled_checks",
            "n_cells_input": int(adata.n_obs),
            "n_cells_selected": int(len(selected_indices)),
            "n_genes": int(adata.n_vars),
            "target_detected_cells": target_detected,
            "sampled_cells": int(len(sample_indices)),
            "sampled_count_check": "nonnegative integers; row sum matches n_counts",
            "full_transcriptome": "user_confirmed_not_machine_verifiable",
            "filter_pass_present": "filter_pass" in adata.obs,
        }
    finally:
        adata.file.close()


def load_geneformer():
    try:
        from geneformer import InSilicoPerturber, InSilicoPerturberStats, TranscriptomeTokenizer
    except ImportError as exc:
        raise RuntimeError("Geneformer is unavailable. Install its official package and dependencies in the selected Python environment") from exc
    return TranscriptomeTokenizer, InSilicoPerturber, InSilicoPerturberStats


def check_cuda() -> dict:
    """Current official ISP code moves cosine tensors to CUDA explicitly."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Geneformer inference needs PyTorch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Current official Geneformer ISP uses CUDA tensors; a CUDA-capable PyTorch environment is required")
    return {"cuda_available": True, "device_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__}


def check_model_config(model_dir: Path, version: str) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Checkpoint is missing config.json: {model_dir}")
    weights = sorted(path for path in model_dir.iterdir() if path.is_file() and path.suffix.lower() in {".bin", ".safetensors", ".pt", ".pth"})
    if not weights:
        raise ValueError(f"Checkpoint has no top-level model weights (.bin/.safetensors/.pt/.pth): {model_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    length = config.get("max_position_embeddings")
    expected = 2048 if version == "V1" else 4096
    if isinstance(length, int) and length != expected:
        raise ValueError(f"Checkpoint config max_position_embeddings={length} conflicts with {version} ({expected})")
    result = {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "max_position_embeddings": length,
        "weights": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in weights],
    }
    return result


def prepare_tokenizer_input(tokenizer, input_h5ad: Path) -> dict:
    """Restrict tokenizer glob to exactly the user-selected file."""
    tokenize_sig = inspect.signature(tokenizer.tokenize_data)
    siblings = list(input_h5ad.parent.glob("*.h5ad"))
    if "input_identifier" in tokenize_sig.parameters:
        matches = [path for path in siblings if input_h5ad.stem in path.stem]
        if matches != [input_h5ad]:
            raise ValueError("Geneformer input_identifier also matches other .h5ad files; place selected file in an isolated directory")
        return {"input_identifier": input_h5ad.stem}
    if siblings != [input_h5ad]:
        raise ValueError("This Geneformer tokenizer scans every .h5ad in the input directory; isolate the selected file first")
    return {}


def inspect_tokenized_target(dataset_path: Path, token_id: int, cell_type_column: str | None, cell_type: str | None) -> dict:
    """Detect token loss after V1/V2 vocabulary mapping and length cropping."""
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("Geneformer token coverage check needs the datasets package") from exc
    dataset = load_from_disk(str(dataset_path))
    selected = 0
    carrying_target = 0
    for row in dataset:
        if cell_type_column and str(row[cell_type_column]) != cell_type:
            continue
        selected += 1
        carrying_target += token_id in row["input_ids"]
    if selected == 0:
        raise ValueError("No tokenized cells remain after the requested cell-type filter")
    return {"n_tokenized_cells_selected_before_cap": selected, "target_token_cells_before_cap": carrying_target}


def parse_cosine_csv(path: Path, condition: str, target: str, output_path: Path, embedding_kind: str) -> dict:
    distances: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as source, output_path.open("w", newline="", encoding="utf-8") as target_file:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "Cosine_sim" not in reader.fieldnames:
            raise ValueError(f"Geneformer stats CSV lacks Cosine_sim: {path}")
        writer = csv.DictWriter(target_file, fieldnames=["condition", "target_ensembl", "observation_index", "cosine_similarity", "cosine_distance"])
        writer.writeheader()
        for index, row in enumerate(reader):
            cosine = float(row["Cosine_sim"])
            if not math.isfinite(cosine) or cosine < -1.00001 or cosine > 1.00001:
                raise ValueError(f"Nonfinite/out-of-range cosine similarity in {path}")
            distance = 1.0 - min(1.0, max(-1.0, cosine))
            distances.append(distance)
            writer.writerow({"condition": condition, "target_ensembl": target, "observation_index": index, "cosine_similarity": cosine, "cosine_distance": distance})
    if not distances:
        raise ValueError(f"Geneformer produced no cell embedding effects: {path}")
    return {
        "n_observations": len(distances),
        "mean_cosine_distance": statistics.mean(distances),
        "median_cosine_distance": statistics.median(distances),
        "min_cosine_distance": min(distances),
        "max_cosine_distance": max(distances),
        "std_cosine_distance": statistics.stdev(distances) if len(distances) > 1 else 0.0,
        "embedding_kind": embedding_kind,
        "interpretation": (
            "1 - mean of aligned per-gene embedding cosine similarities after rank-token perturbation; not a cosine between pooled cell vectors or gene expression change"
            if embedding_kind == "mean_gene_embedding_cosine" else
            "1 - cosine similarity of original and rank-token-perturbed CLS cell representations; not gene expression change"
        ),
    }


def write_histogram(csv_path: Path, svg_path: Path, condition: str, target: str, embedding_kind: str) -> None:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        values = [float(row["cosine_distance"]) for row in csv.DictReader(handle)]
    bins = [0] * 20
    upper = max(max(values), 1e-6)
    for value in values:
        bins[min(19, int(value / upper * 20))] += 1
    tallest = max(bins)
    bars = []
    for index, count in enumerate(bins):
        height = 210 * count / tallest
        bars.append(f'<rect x="{65 + index * 29}" y="{270 - height:.2f}" width="24" height="{height:.2f}" fill="#2e6685"/>')
    label = "mean gene embedding distance" if embedding_kind == "mean_gene_embedding_cosine" else "CLS cell representation distance"
    title = escape(f"Geneformer {condition} {target}: {label}")
    axis_label = "1 - mean aligned gene cosine" if embedding_kind == "mean_gene_embedding_cosine" else "1 - CLS cosine similarity"
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="340" viewBox="0 0 720 340">'
        '<rect width="720" height="340" fill="white"/>'
        f'<text x="65" y="30" font-size="17" font-family="sans-serif">{title}</text>'
        '<path d="M65 55V270H645" stroke="#333" fill="none"/>'
        + "".join(bars)
        + f'<text x="65" y="303" font-size="12" font-family="sans-serif">0</text><text x="590" y="303" font-size="12" font-family="sans-serif">{upper:.3g}</text>'
        + f'<text x="260" y="326" font-size="12" font-family="sans-serif">{escape(axis_label)}</text>'
        + '</svg>\n'
    )
    svg_path.write_text(svg, encoding="utf-8")


def append_error(path: Path, stage: str, exc: Exception, condition: str | None = None) -> None:
    entry = {
        "at_utc": now_utc(), "stage": stage, "condition": condition,
        "error_type": type(exc).__name__, "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def skipped_expression_figures(condition: str) -> list[dict]:
    reason = "Geneformer returns embedding effects, not per-gene predicted expression or expression statistics"
    figures = [
        ("gene_effects", "Which genes change expression after perturbation?"),
        ("response_heatmap", "How does predicted expression differ by gene and cell group?"),
        ("pathway_enrichment", "Which pathways change according to an expression response?"),
    ]
    if condition == "OE":
        figures.append(("oe_setting", "Where does a numeric expression setting fall in the observed range?"))
    return [figure_row(condition, key, "skipped", question, reason=reason) for key, question in figures]


def run(args: argparse.Namespace) -> int:
    try:
        input_h5ad, model_dir, output_dir, conditions = validate_options(args)
    except (OSError, ValueError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_root = output_dir / "raw"
    tables_root = output_dir / "tables"
    raw_root.mkdir()
    tables_root.mkdir()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir()
    figure_manifest_path = figures_dir / "figure_manifest.csv"
    figure_rows: list[dict] = []
    write_figure_manifest(figure_manifest_path, figure_rows)
    errors_path = output_dir / "errors.jsonl"
    errors_path.touch()
    validation = {"input_status": "pending", "model_status": "pending", "execution_status": "pending", "independent_validation": "not_performed"}
    write_json(output_dir / "validation.json", validation)
    write_json(output_dir / "provenance.json", {"input_h5ad": str(input_h5ad), "model_dir": str(model_dir), "status": "pending"})
    manifest = {
        "backend": "Geneformer",
        "route": "independent_rank_token_embedding_perturbation",
        "started_at_utc": now_utc(),
        "status": "running",
        "conditions": {condition: {"status": "pending", "perturb_type": PERTURB_TYPES[condition]} for condition in conditions},
        "result_kind": "rank_token_representation_cosine_effect",
        "embedding_kind": "mean_gene_embedding_cosine" if args.model_version == "V1" else "CLS_cell_embedding_cosine",
        "figure_manifest": "figures/figure_manifest.csv",
        "does_not_produce": ["post_perturbation_expression", "delta_X", "log2FC", "clinical_response"],
    }
    write_json(output_dir / "manifest.json", manifest)
    stage = "input_validation"
    try:
        validation.update(validate_h5ad(input_h5ad, args.target_ensembl, args.cell_type_column, args.cell_type))
        validation["input_status"] = "passed_sampled_checks"
        write_json(output_dir / "validation.json", validation)
        stage = "dependencies"
        tokenizer_class, perturber_class, stats_class = load_geneformer()
        hardware = check_cuda()
        stage = "model_validation"
        model_config = check_model_config(model_dir, args.model_version)
        custom_attrs = {args.cell_type_column: args.cell_type_column} if args.cell_type_column else None
        tokenizer = tokenizer_class(custom_attr_name_dict=custom_attrs, nproc=args.nproc, model_version=args.model_version)
        if args.target_ensembl not in tokenizer.gene_token_dict:
            raise ValueError(f"Target {args.target_ensembl} is absent from Geneformer {args.model_version} token dictionary")
        tokenizer_options = prepare_tokenizer_input(tokenizer, input_h5ad)
        validation["model_status"] = "passed_preflight"
        write_json(output_dir / "validation.json", validation)

        provenance = {
            "input_h5ad": str(input_h5ad),
            "input_sha256": sha256_file(input_h5ad),
            "model_dir": str(model_dir),
            "model_version": args.model_version,
            "emb_mode": "cell" if args.model_version == "V1" else "cls",
            "embedding_kind": manifest["embedding_kind"],
            "model_type": args.model_type,
            "num_classes": args.num_classes,
            "model_config": model_config,
            "target_ensembl": args.target_ensembl,
            "cell_type_filter": {"column": args.cell_type_column, "value": args.cell_type} if args.cell_type_column else None,
            "max_ncells": args.max_ncells,
            "nproc": args.nproc,
            "forward_batch_size": args.forward_batch_size,
            "geneformer_package_version": _package_version("geneformer"),
            "hardware": hardware,
            "geneformer_sources": {
                name: {"path": inspect.getfile(cls), "sha256": sha256_file(Path(inspect.getfile(cls)))}
                for name, cls in (("tokenizer", tokenizer_class), ("perturber", perturber_class), ("stats", stats_class))
            },
            "adapter_sha256": sha256_file(Path(__file__)),
            "official_sources": OFFICIAL_SOURCES,
            "input_check": "Sampled X and metadata validated; full-transcriptome status supplied by user",
        }
        write_json(output_dir / "provenance.json", provenance)

        stage = "tokenization"
        tokenized_dir = output_dir / "tokenized"
        tokenized_dir.mkdir()
        tokenizer.tokenize_data(input_h5ad.parent, tokenized_dir, "selected_cells", file_format="h5ad", **tokenizer_options)
        dataset_path = tokenized_dir / "selected_cells.dataset"
        if not dataset_path.exists():
            raise RuntimeError("Geneformer did not create the expected tokenized .dataset")
        validation.update(inspect_tokenized_target(dataset_path, tokenizer.gene_token_dict[args.target_ensembl], args.cell_type_column, args.cell_type))
        write_json(output_dir / "validation.json", validation)

        for condition in conditions:
            condition_stage = f"{condition}:eligibility"
            raw_dir = raw_root / condition
            tables_dir = tables_root / condition
            condition_figures_dir = figures_dir / condition
            raw_dir.mkdir()
            tables_dir.mkdir()
            condition_figures_dir.mkdir()
            manifest["conditions"][condition]["status"] = "running"
            write_json(output_dir / "manifest.json", manifest)
            try:
                if condition == "KO" and validation["target_token_cells_before_cap"] == 0:
                    raise ValueError("KO/delete requires target token in selected tokenized cells; OE can insert an absent target token")
                condition_stage = f"{condition}:inference"
                perturb = perturber_class(
                    perturb_type=PERTURB_TYPES[condition],
                    genes_to_perturb=[args.target_ensembl],
                    combos=0,
                    model_type=args.model_type,
                    num_classes=args.num_classes,
                    emb_mode="cell" if args.model_version == "V1" else "cls",
                    filter_data={args.cell_type_column: [args.cell_type]} if args.cell_type_column else None,
                    max_ncells=args.max_ncells,
                    forward_batch_size=args.forward_batch_size,
                    nproc=args.nproc,
                    model_version=args.model_version,
                )
                perturb.perturb_data(model_dir, dataset_path, raw_dir, f"{condition}_{args.target_ensembl}")
                raw_files = sorted(raw_dir.glob("*_raw.pickle"))
                if not raw_files:
                    raise RuntimeError(f"Geneformer produced no raw ISP pickle for {condition}")
                condition_stage = f"{condition}:aggregation"
                stats = stats_class(mode="aggregate_data", genes_perturbed=[args.target_ensembl], model_version=args.model_version)
                stats.get_stats(raw_dir, None, tables_dir, "geneformer_cell_cosine")
                official_csv = tables_dir / "geneformer_cell_cosine.csv"
                if not official_csv.exists():
                    raise RuntimeError(f"Geneformer did not produce aggregate_data CSV for {condition}")
                effects_csv = tables_dir / "representation_effects.csv"
                summary = parse_cosine_csv(official_csv, condition, args.target_ensembl, effects_csv, manifest["embedding_kind"])
                summary_path = tables_dir / "summary.json"
                write_json(summary_path, summary)
                condition_stage = f"{condition}:figure"
                figure_path = condition_figures_dir / "representation_distance_distribution.svg"
                write_histogram(effects_csv, figure_path, condition, args.target_ensembl, manifest["embedding_kind"])
                figure_rows.append(figure_row(
                    condition, "embedding_distance", "generated",
                    "How large is the cell representation change after rank-token perturbation?",
                    plot_path=str(figure_path.relative_to(output_dir)),
                    source_table=str(effects_csv.relative_to(output_dir)),
                    metric="1 - mean aligned gene cosine" if args.model_version == "V1" else "1 - CLS cosine similarity",
                    embedding_kind=manifest["embedding_kind"],
                ))
                manifest["conditions"][condition] = {
                    "status": "completed", "perturb_type": PERTURB_TYPES[condition],
                    "emb_mode": "cell" if args.model_version == "V1" else "cls",
                    "raw_files": [str(path.relative_to(output_dir)) for path in raw_files],
                    "official_stats": str(official_csv.relative_to(output_dir)),
                    "effects_table": str(effects_csv.relative_to(output_dir)),
                    "summary": str(summary_path.relative_to(output_dir)),
                    "figure": str(figure_path.relative_to(output_dir)),
                    "n_observations": summary["n_observations"],
                }
            except Exception as exc:
                append_error(errors_path, condition_stage, exc, condition)
                manifest["conditions"][condition] = {
                    "status": "failed", "perturb_type": PERTURB_TYPES[condition],
                    "failed_stage": condition_stage, "error": f"{type(exc).__name__}: {exc}",
                }
                figure_rows.append(figure_row(
                    condition, "embedding_distance", "failed",
                    "How large is the cell representation change after rank-token perturbation?",
                    reason=f"{type(exc).__name__}: {exc}", embedding_kind=manifest["embedding_kind"],
                ))
            finally:
                figure_rows.extend(skipped_expression_figures(condition))
                write_figure_manifest(figure_manifest_path, figure_rows)
                write_json(output_dir / "manifest.json", manifest)

        completed = sum(detail["status"] == "completed" for detail in manifest["conditions"].values())
        manifest["status"] = "completed" if completed == len(conditions) else "partial" if completed else "failed"
        manifest["completed_at_utc"] = now_utc()
        validation["execution_status"] = manifest["status"]
        write_json(output_dir / "validation.json", validation)
        write_json(output_dir / "manifest.json", manifest)
        write_report(output_dir, args, manifest)
        print(f"Geneformer embedding results: {output_dir}")
        return 0 if manifest["status"] == "completed" else 1
    except Exception as exc:
        append_error(errors_path, stage, exc)
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = now_utc()
        manifest["failed_stage"] = stage
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        for condition in conditions:
            if manifest["conditions"][condition]["status"] == "pending":
                manifest["conditions"][condition]["status"] = "skipped"
                manifest["conditions"][condition]["reason"] = f"Setup failed at {stage}"
                figure_rows.append(figure_row(condition, "embedding_distance", "skipped",
                                               "How large is the cell representation change after rank-token perturbation?",
                                               reason=f"Setup failed at {stage}"))
                figure_rows.extend(skipped_expression_figures(condition))
        write_figure_manifest(figure_manifest_path, figure_rows)
        validation["execution_status"] = "failed"
        validation["error"] = manifest["error"]
        write_json(output_dir / "validation.json", validation)
        write_json(output_dir / "manifest.json", manifest)
        write_report(output_dir, args, manifest)
        print(f"Geneformer route failed: {exc}. See {output_dir / 'manifest.json'}", file=sys.stderr)
        return 1


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_report(output_dir: Path, args: argparse.Namespace, manifest: dict) -> None:
    lines = [
        "# Geneformer representation perturbation report",
        "",
        f"Status: **{manifest['status']}**",
        f"Target: `{args.target_ensembl}`; requested route: `{args.perturbation}`; checkpoint series: `{args.model_version}`.",
        "",
        "KO uses Geneformer `delete`; OE uses `overexpress`. Both manipulate rank-value tokens. "
        "Output type: embedding effects of rank-token perturbation.",
        f"Embedding metric: `{manifest['embedding_kind']}`. V1 reports the mean of aligned per-gene cosine similarities; V2 reports cosine similarity of `<cls>` cell representations. Each checkpoint series is reported separately using its own metric definition.",
        "",
        "The observation index identifies a result row; mapping to original cell IDs is unavailable. "
        "Independent experimental validation: unvalidated. Target-state reference: unavailable in this route.",
        "The current official ISP implementation uses CUDA tensors; this adapter checks for a CUDA-capable PyTorch environment before inference.",
        "",
        "## Condition outputs",
        "",
    ]
    for condition, detail in manifest["conditions"].items():
        line = f"- **{condition}** (`{detail['perturb_type']}`): {detail['status']}"
        if "n_observations" in detail:
            line += f", {detail['n_observations']} representation observations"
        if "error" in detail:
            line += f"; {detail['error']}"
        if "reason" in detail:
            line += f"; {detail['reason']}"
        lines.append(line)
    if manifest.get("error"):
        lines += ["", "## Error", "", manifest["error"]]
    lines += ["", "Condition files are under `raw/KO|OE/`, `tables/KO|OE/`, and `figures/KO|OE/`. See `figures/figure_manifest.csv` for generated and skipped plots; `errors.jsonl` records failures. `manifest.json`, `validation.json`, and `provenance.json` record status, checks, input identity, and parameters.", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
