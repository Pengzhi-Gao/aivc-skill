"""Fast adapter tests; official Geneformer model inference is deliberately mocked."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "ko_oe" / "geneformer_route.py"
spec = importlib.util.spec_from_file_location("geneformer_route", MODULE_PATH)
assert spec and spec.loader
route = importlib.util.module_from_spec(spec)
spec.loader.exec_module(route)
TARGET = "ENSG00000168610"


class FakeTokenizer:
    gene_token_dict = {TARGET: 123}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def tokenize_data(self, data_directory, output_directory, output_prefix, file_format="loom", input_identifier=""):
        assert file_format == "h5ad"
        assert input_identifier == "cells"
        (Path(output_directory) / f"{output_prefix}.dataset").mkdir()


class FakePerturber:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)

    def perturb_data(self, model_directory, input_data_file, output_directory, output_prefix):
        assert Path(input_data_file).is_dir()
        (Path(output_directory) / f"{output_prefix}_cell_embs_raw.pickle").write_bytes(b"test placeholder")


class FakeStats:
    def __init__(self, **kwargs):
        assert kwargs["mode"] == "aggregate_data"
        assert kwargs["genes_perturbed"] == [TARGET]

    def get_stats(self, input_data_directory, null_dist_data_directory, output_directory, output_prefix):
        assert null_dist_data_directory is None
        assert list(Path(input_data_directory).glob("*_raw.pickle"))
        with (Path(output_directory) / f"{output_prefix}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", "Cosine_sim", "Gene"])
            writer.writerow(["0", "0.8", "TARGET"])
            writer.writerow(["1", "0.6", "TARGET"])


class FailKoPerturber(FakePerturber):
    calls = []

    def perturb_data(self, model_directory, input_data_file, output_directory, output_prefix):
        if self.kwargs["perturb_type"] == "delete":
            raise RuntimeError("mock inference failure")
        super().perturb_data(model_directory, input_data_file, output_directory, output_prefix)


class GeneformerRouteTests(unittest.TestCase):
    def test_both_routes_write_distinct_embedding_outputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            base = Path(directory)
            input_path = base / "cells.h5ad"
            input_path.write_bytes(b"synthetic h5ad placeholder")
            model = base / "model"
            model.mkdir()
            (model / "config.json").write_text('{"max_position_embeddings":4096}', encoding="utf-8")
            (model / "pytorch_model.bin").write_bytes(b"mock weights")
            output = base / "out"
            args = route.parse_args([
                "--input-h5ad", str(input_path), "--model-dir", str(model),
                "--model-version", "V2", "--target-ensembl", TARGET,
                "--perturbation", "both", "--output-dir", str(output),
                "--confirm-raw-full-transcriptome",
            ])
            FakePerturber.calls = []
            with mock.patch.object(route, "validate_h5ad", return_value={"status": "passed_sampled_checks", "target_detected_cells": 2}), \
                 mock.patch.object(route, "load_geneformer", return_value=(FakeTokenizer, FakePerturber, FakeStats)), \
                 mock.patch.object(route, "inspect_tokenized_target", return_value={"n_tokenized_cells_selected_before_cap": 2, "target_token_cells_before_cap": 2}), \
                 mock.patch.object(route, "check_cuda", return_value={"cuda_available": True}):
                self.assertEqual(route.run(args), 0)
            self.assertEqual([call["perturb_type"] for call in FakePerturber.calls], ["delete", "overexpress"])
            self.assertEqual([call["emb_mode"] for call in FakePerturber.calls], ["cls", "cls"])
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertIn("delta_X", manifest["does_not_produce"])
            self.assertEqual((output / "errors.jsonl").read_text(encoding="utf-8"), "")
            with (output / "figures" / "figure_manifest.csv").open(newline="", encoding="utf-8") as handle:
                figures = list(csv.DictReader(handle))
            self.assertEqual([row["status"] for row in figures if row["figure_id"].endswith("embedding_distance")], ["generated", "generated"])
            self.assertTrue(all(row["status"] == "skipped" for row in figures if row["output_type"] == "expression"))
            for condition in ("KO", "OE"):
                self.assertEqual(manifest["conditions"][condition]["n_observations"], 2)
                self.assertTrue((output / "tables" / condition / "representation_effects.csv").is_file())
                self.assertTrue((output / "figures" / condition / "representation_distance_distribution.svg").is_file())
                summary = json.loads((output / "tables" / condition / "summary.json").read_text(encoding="utf-8"))
                self.assertAlmostEqual(summary["mean_cosine_distance"], 0.3)
                self.assertEqual(summary["embedding_kind"], "CLS_cell_embedding_cosine")

    def test_ko_ineligible_does_not_block_oe(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            base = Path(directory)
            input_path = base / "cells.h5ad"
            input_path.write_bytes(b"synthetic h5ad placeholder")
            model = base / "model"
            model.mkdir()
            (model / "config.json").write_text('{"max_position_embeddings":4096}', encoding="utf-8")
            (model / "pytorch_model.bin").write_bytes(b"mock weights")
            output = base / "out"
            args = route.parse_args([
                "--input-h5ad", str(input_path), "--model-dir", str(model),
                "--model-version", "V2", "--target-ensembl", TARGET,
                "--perturbation", "both", "--output-dir", str(output),
                "--confirm-raw-full-transcriptome",
            ])
            FakePerturber.calls = []
            with mock.patch.object(route, "validate_h5ad", return_value={"status": "passed_sampled_checks", "target_detected_cells": 0}), \
                 mock.patch.object(route, "load_geneformer", return_value=(FakeTokenizer, FakePerturber, FakeStats)), \
                 mock.patch.object(route, "inspect_tokenized_target", return_value={"n_tokenized_cells_selected_before_cap": 2, "target_token_cells_before_cap": 0}), \
                 mock.patch.object(route, "check_cuda", return_value={"cuda_available": True}):
                self.assertEqual(route.run(args), 1)
            self.assertEqual([call["perturb_type"] for call in FakePerturber.calls], ["overexpress"])
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "partial")
            self.assertEqual(manifest["conditions"]["KO"]["status"], "failed")
            self.assertEqual(manifest["conditions"]["OE"]["status"], "completed")
            errors = [json.loads(line) for line in (output / "errors.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["condition"], "KO")
            with (output / "figures" / "figure_manifest.csv").open(newline="", encoding="utf-8") as handle:
                figures = list(csv.DictReader(handle))
            self.assertEqual({row["figure_id"]: row["status"] for row in figures if row["figure_id"].endswith("embedding_distance")},
                             {"KO_embedding_distance": "failed", "OE_embedding_distance": "generated"})

    def test_v1_inference_failure_does_not_block_oe(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            base = Path(directory)
            input_path = base / "cells.h5ad"
            input_path.write_bytes(b"synthetic h5ad placeholder")
            model = base / "model"
            model.mkdir()
            (model / "config.json").write_text('{"max_position_embeddings":2048}', encoding="utf-8")
            (model / "pytorch_model.bin").write_bytes(b"mock weights")
            output = base / "out"
            args = route.parse_args([
                "--input-h5ad", str(input_path), "--model-dir", str(model),
                "--model-version", "V1", "--target-ensembl", TARGET,
                "--perturbation", "both", "--output-dir", str(output),
                "--confirm-raw-full-transcriptome",
            ])
            FailKoPerturber.calls = []
            with mock.patch.object(route, "validate_h5ad", return_value={"status": "passed_sampled_checks", "target_detected_cells": 2}), \
                 mock.patch.object(route, "load_geneformer", return_value=(FakeTokenizer, FailKoPerturber, FakeStats)), \
                 mock.patch.object(route, "inspect_tokenized_target", return_value={"n_tokenized_cells_selected_before_cap": 2, "target_token_cells_before_cap": 2}), \
                 mock.patch.object(route, "check_cuda", return_value={"cuda_available": True}):
                self.assertEqual(route.run(args), 1)
            self.assertEqual([call["emb_mode"] for call in FailKoPerturber.calls], ["cell", "cell"])
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "partial")
            self.assertEqual(manifest["conditions"]["KO"]["failed_stage"], "KO:inference")
            self.assertEqual(manifest["conditions"]["OE"]["status"], "completed")
            summary = json.loads((output / "tables" / "OE" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["embedding_kind"], "mean_gene_embedding_cosine")
            errors = [json.loads(line) for line in (output / "errors.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["condition"], "KO")

    def test_rejects_missing_full_transcriptome_confirmation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            base = Path(directory)
            (base / "cells.h5ad").write_bytes(b"x")
            (base / "model").mkdir()
            args = route.parse_args([
                "--input-h5ad", str(base / "cells.h5ad"), "--model-dir", str(base / "model"),
                "--model-version", "V1", "--target-ensembl", TARGET,
                "--perturbation", "KO", "--output-dir", str(base / "out"),
            ])
            self.assertEqual(route.run(args), 2)
            self.assertFalse((base / "out").exists())

    def test_nonfinite_cosine_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            base = Path(directory)
            source = base / "official.csv"
            source.write_text("Cosine_sim,Gene\nnan,TARGET\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                route.parse_cosine_csv(source, "KO", TARGET, base / "effects.csv", "CLS_cell_embedding")

    def test_dependency_failure_writes_root_contract(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            base = Path(directory)
            input_path = base / "cells.h5ad"
            input_path.write_bytes(b"synthetic h5ad placeholder")
            model = base / "model"
            model.mkdir()
            output = base / "out"
            args = route.parse_args([
                "--input-h5ad", str(input_path), "--model-dir", str(model),
                "--model-version", "V2", "--target-ensembl", TARGET,
                "--perturbation", "both", "--output-dir", str(output),
                "--confirm-raw-full-transcriptome",
            ])
            with mock.patch.object(route, "validate_h5ad", return_value={"status": "passed_sampled_checks", "target_detected_cells": 1}), \
                 mock.patch.object(route, "load_geneformer", side_effect=RuntimeError("mock missing dependency")):
                self.assertEqual(route.run(args), 1)
            for name in ("raw", "tables", "figures", "errors.jsonl", "manifest.json", "validation.json", "provenance.json", "report.md"):
                self.assertTrue((output / name).exists(), name)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual({detail["status"] for detail in manifest["conditions"].values()}, {"skipped"})
            errors = [json.loads(line) for line in (output / "errors.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(errors[0]["stage"], "dependencies")
            with (output / "figures" / "figure_manifest.csv").open(newline="", encoding="utf-8") as handle:
                figures = list(csv.DictReader(handle))
            self.assertEqual({row["status"] for row in figures}, {"skipped"})


if __name__ == "__main__":
    unittest.main()
