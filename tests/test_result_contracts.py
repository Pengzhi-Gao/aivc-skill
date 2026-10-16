"""Adversarial regression tests; no model weights or network calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('screen_common', ROOT / 'examples/lung_cancer/drug_screen/common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


class ResultContracts(unittest.TestCase):
    def test_distinct_responses_equal_scores(self):
        r = np.array([[0, 1, 0], [0, 0, 1], [0, -1, 0.]])
        q = common.degeneracy_check(r, common.reversal_scores(r, [1, 0, 0])[2])
        self.assertEqual(q['prediction_status'], 'result_available')
        self.assertEqual(q['response_status'], 'varying')
        self.assertEqual(q['ranking_status'], 'unresolved')

    def test_proportional_responses_retain_amplitude(self):
        r = np.array([[1, 2, 3], [2, 4, 6], [10, 20, 30.]])
        q = common.degeneracy_check(r, common.reversal_scores(r, [1, 0, 0])[2])
        self.assertFalse(q['degenerate'])
        self.assertAlmostEqual(q['median_pairwise_response_cosine'], 1)

    def test_nan_is_condition_failure(self):
        q = common.degeneracy_check([[1, 0], [0, 1], [np.nan, np.nan]], [-1, 0, np.nan])
        self.assertEqual(q['prediction_status'], 'partial')
        self.assertEqual(q['invalid_condition_indices'], [2])
        self.assertEqual(q['unscored_condition_indices'], [2])

    def test_all_invalid_empty_and_single_condition(self):
        self.assertEqual(common.degeneracy_check([[np.inf]], [np.nan])['prediction_status'], 'failed')
        self.assertEqual(common.degeneracy_check([[1, 2]], [1])['response_status'], 'insufficient_conditions')
        self.assertEqual(common.degeneracy_check(np.empty((0, 2)), [])['prediction_status'], 'failed')

    def test_identical_responses_keep_predictions(self):
        q = common.degeneracy_check([[1, 2], [1, 2]], [0, 0])
        self.assertTrue(q['degenerate'])
        self.assertEqual(q['prediction_status'], 'result_available')
        self.assertEqual(q['ranking_status'], 'unresolved')

    def test_shared_disease_mask(self):
        r = np.array([[2., -1, 1000], [1, 1, 2000]])
        full = common.reversal_scores(r, [1, -1, np.nan])
        subset = common.reversal_scores(r[:, :2], [1, -1])
        np.testing.assert_allclose(full[0], subset[0])
        r[0, 2] = np.nan
        self.assertTrue(np.isnan(common.reversal_scores(r, [1, -1, np.nan])[0][0]))

    def test_top_selection_preserves_rank_ids_and_doses(self):
        meta = pd.DataFrame({'drug': ['A', 'B', 'C', 'B'], 'dose_uM': [1, 1, 1, 10]})
        r = np.arange(8).reshape(4, 2)
        top = common.top_response_table(meta, r, ['g1', 'g2'], ['C', 'B'], n=2)
        self.assertEqual(top.drug.tolist(), ['C', 'B', 'B'])
        np.testing.assert_equal(top[['g1', 'g2']].to_numpy(), r[[2, 1, 3]])
        with self.assertRaises(ValueError):
            common.top_response_table(meta, r, ['g1', 'g2'], ['unknown'])

    def test_drug_recovery_and_each_conditions_top_genes(self):
        genes = [f'g{i}' for i in range(35)]
        conditions = pd.DataFrame({
            'condition_id': ['c0', 'c1', 'c2', 'c3', 'c4'],
            'drug': ['A', 'A', 'B', 'C', 'D'],
            'dose_uM': [1, 10, 1, 1, 1]})
        response = np.array([
            np.arange(35), -np.arange(35), np.ones(35),
            np.full(35, np.nan), np.zeros(35)], dtype=float)
        by_condition, by_drug, top = common.drug_result_tables(
            conditions, response, genes, [0.5, -0.2, -0.4, np.nan, 0.1],
            scoring_status='descriptive_only', ranking_status='available')
        self.assertEqual(by_condition.recovery_status.tolist(), [
            'predicted_reversal', 'no_predicted_reversal',
            'no_predicted_reversal', 'indeterminate', 'indeterminate'])
        self.assertEqual(by_drug.set_index('drug').loc['A', 'recovery_status'], 'condition_dependent')
        self.assertEqual(by_drug.set_index('drug').loc['C', 'recovery_status'], 'indeterminate')
        self.assertEqual(top[top.condition_id == 'c0'].gene.tolist(),
                         [f'g{i}' for i in range(34, 4, -1)])
        self.assertEqual(top[top.condition_id == 'c1'].iloc[0].effect_direction, 'down')
        self.assertFalse((top.condition_id == 'c3').any())
        self.assertFalse((top.condition_id == 'c4').any())

        unavailable, summary, top_unavailable = common.drug_result_tables(
            conditions, response, genes, [0.5, -0.2, -0.4, np.nan, 0.1],
            scoring_status='skipped', ranking_status='not_available')
        self.assertTrue((unavailable.recovery_status == 'indeterminate').all())
        self.assertTrue((summary.recovery_status == 'indeterminate').all())
        self.assertEqual(len(top_unavailable), len(top))

    def test_drug_result_export_uses_saved_condition_order(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'ROOT', Path(tmp)):
            run = common.RunDir('drug_results')
            response = np.array([[2., -1.], [-3., 1.]])
            indexed = run.save_predictions(
                pd.DataFrame({'drug': ['A', 'B'], 'dose_uM': [1, 10]}),
                ['g1', 'g2'], response, np.zeros(2), response)
            indexed['reversal_score'] = [0.5, -0.5]
            indexed.to_csv(run.tables / 'conditions.csv', index=False)
            (run.tables / 'scoring_metadata.json').write_text(
                json.dumps({'status': 'descriptive_only'}), encoding='utf-8')
            run.finish({'backend': 'synthetic'},
                       {'backend': 'synthetic', 'status': 'result_available',
                        'ranking_status': 'available'})
            summary = pd.read_csv(run.tables / 'drug_recovery_summary.csv').set_index('drug')
            self.assertEqual(summary.loc['A', 'recovery_status'], 'predicted_reversal')
            self.assertEqual(summary.loc['B', 'recovery_status'], 'no_predicted_reversal')
            self.assertEqual(pd.read_csv(run.tables / 'drugs_predicted_reversal.csv').drug.tolist(), ['A'])
            self.assertEqual(pd.read_csv(run.tables / 'drugs_no_predicted_reversal.csv').drug.tolist(), ['B'])
            top = pd.read_csv(run.tables / 'drug_top30_genes.csv')
            self.assertEqual(top.groupby('drug').size().to_dict(), {'A': 2, 'B': 2})
            self.assertIn('药物筛选结果', (run.root / 'report.md').read_text(encoding='utf-8'))
            self.assertIn('tables/drug_top30_genes.csv', (run.root / 'report.md').read_text())

    def test_full_prediction_roundtrip_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'ROOT', Path(tmp)):
            run = common.RunDir('test')
            baseline = np.array([2., 3.])
            prediction = np.array([[3, 4], [4, 5], [np.nan, 4.]])
            response = prediction - baseline
            indexed = run.save_predictions(pd.DataFrame({'drug': ['A', 'B', 'C']}),
                                           ['g1', 'g2'], prediction, baseline, response)
            with np.load(run.raw / 'predictions.npz', allow_pickle=False) as saved:
                np.testing.assert_allclose(saved['response'], response, equal_nan=True)
                self.assertEqual(saved['condition_id'].tolist(), indexed.condition_id.tolist())
            self.assertEqual(len(pd.read_csv(run.tables / 'all_conditions_gene_response.csv.gz')), 3)
            q = run.condition_quality(indexed, response, [1, 2, np.nan])
            self.assertEqual(q['prediction_status'], 'partial')
            self.assertEqual(run.errors[0]['condition_id'], indexed.iloc[2].condition_id)
            with self.assertRaises(ValueError):
                run.save_predictions(pd.DataFrame({'drug': ['A', 'B', 'C']}),
                                     ['g1', 'g2'], prediction, baseline, response + 1)
            self.assertNotEqual(common.RunDir('test').root, run.root)

    def test_scoring_requires_scale_contract(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'ROOT', Path(tmp)), patch.object(common, 'DATA', Path(tmp)):
            run = common.RunDir('test')
            result = run.score_response(np.array([[1, 2.]]), np.array([1, -1.]), 'backend')
            self.assertTrue(np.isnan(result[2]).all())
            meta = json.loads((run.tables / 'scoring_metadata.json').read_text())
            self.assertEqual(meta['status'], 'skipped')

    def test_finish_exports_figures_and_strict_json(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'ROOT', Path(tmp)):
            run = common.RunDir('integration')
            response = np.array([[1., -1, 0], [0, 1, -1], [-1, 0, 1]])
            indexed = run.save_predictions(pd.DataFrame({'drug': ['A', 'B', 'C']}),
                                           ['g1', 'g2', 'g3'], response, np.zeros(3), response)
            q = run.condition_quality(indexed, response, [np.nan] * 3)
            run.finish({'backend': {'name': 'test'}},
                       {'backend': 'test', 'status': q['prediction_status'],
                        'ranking_status': q['ranking_status'], 'range': [np.nan, np.nan]})
            records = pd.read_csv(run.figures / 'figure_manifest.csv').set_index('figure_id')
            self.assertEqual(records.loc['drug_response_heatmap', 'status'], 'generated')
            self.assertEqual(records.loc['response_similarity', 'status'], 'generated')
            self.assertEqual(records.loc['reversal_null', 'status'], 'skipped')
            def reject_constant(s):
                raise ValueError(s)
            result = json.loads((run.root / 'validation.json').read_text(), parse_constant=reject_constant)
            self.assertEqual(result['range'], [None, None])
            self.assertTrue((run.root / 'provenance.json').exists())

    def test_scoring_contract_source_binding(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'ROOT', Path(tmp)), patch.object(common, 'DATA', Path(tmp)):
            base = Path(tmp)
            source = base / 'GSE81089_landmark978_signature.csv'
            source.write_text('gene_symbol,tumor_minus_normal_log2\nA,1\nB,-1\n', encoding='utf-8')
            contract = {'metric': 'negative_cosine', 'response_scale': 'test effect',
                        'disease_scale': 'test effect', 'scale_justification': 'same synthetic scale',
                        'comparison_scope': 'synthetic unit test', 'minimum_common_genes': 2,
                        'disease_source_sha256_16': common.sha16(source)}
            (base / 'test_scoring_contract.json').write_text(json.dumps(contract), encoding='utf-8')
            with patch.object(common, 'DISEASE_VEC', base / 'vector.csv'):
                run = common.RunDir('scoring')
                score = run.score_response(np.array([[-1., 1]]), np.array([1., -1]), 'test')[2]
                self.assertAlmostEqual(score[0], 1)
                source.write_text('changed', encoding='utf-8')
                score = run.score_response(np.array([[-1., 1]]), np.array([1., -1]), 'test')[2]
                self.assertTrue(np.isnan(score[0]))

    def test_disease_missing_distinct_from_real_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pd.DataFrame({'gene_symbol': ['A', 'B'], 'tumor_minus_normal_log2': [0., 2.]}).to_csv(
                base / 'GSE81089_landmark978_signature.csv', index=False)
            with patch.object(common, 'DISEASE_VEC', base / 'old_vector.csv'):
                d, count = common.load_disease_direction(['A', 'B', 'C'])
            self.assertEqual(count, 1)
            self.assertEqual(d[0], 0.)
            self.assertTrue(np.isnan(d[2]))


if __name__ == '__main__':
    unittest.main()
