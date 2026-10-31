#!/usr/bin/env python3
"""Adoption checks on in-memory quality records and temporary transaction files.

No numerical solver runs and no measured data are modified, including when
testing the apply transaction on synthetic files under a temporary directory.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import adopt_single_dual_gaps as adoption


class DualAdoption(unittest.TestCase):
    def test_readonly_eight_row_write_set_with_mocked_coverage_loader(self):
        # This isolates adoption construction while the last copied case may still
        # be pending. It does not claim or bypass production coverage admission.
        results=adoption.ROOT/'results_single/verified';inventory=adoption.ROOT/'notes/single_delivery_inventory.json';summary=adoption.ROOT/'reporting/recovery/single_delivery_v2/summary.json'
        if not summary.exists():self.skipTest('Actual independent sidecars are unavailable')
        inv=adoption.read(inventory);pairs={(r['case'],r['method']) for r in inv['invalid_diagnostics']}
        loaded=[dict(case=case,method=method,checkpoints=adoption.read(results/case/(method+'_diagnostics.json'))) for case,method in pairs]
        if any(row.get('gap_status')=='independent_dual_bracket_valid' for item in loaded for row in item['checkpoints']):self.skipTest('Root has already applied the reviewed adoption')
        plan=adoption.read(results/'plan.json');inputs={str(results/'plan.json'):adoption.sha(results/'plan.json')}
        with patch.object(adoption,'load_complete',return_value=(plan,loaded,inputs)):
            review,writes,originals=adoption.prepare(results,inventory,summary)
        self.assertEqual(review['rows'],8);self.assertFalse(review['quality_reviewed_by_explicit_cli']);self.assertFalse(review['apply_eligible'])
        self.assertTrue(all(path.read_bytes()==data for path,data in originals.items()))
        self.assertTrue(all(path.name=='manifest.json' or path.name.endswith('_diagnostics.json') for path in writes))
        self.assertEqual(sum(path.name.endswith('_diagnostics.json') for path in writes),7)
        self.assertEqual(sum(row['quality']['optimizer_success'] is False for row in review['changes']),3)

    def test_actual_sidecar_quality_and_rejection_mutations(self):
        summary=adoption.ROOT/'reporting/recovery/single_delivery_v2/summary.json'
        if not summary.is_file():self.skipTest('Actual independent solver sidecars are unavailable')
        records=[adoption.read(entry['diagnostic_file']) for entry in adoption.read(summary)['results']]
        metrics=[adoption.check_quality(record) for record in records]
        self.assertEqual(len(metrics),8);self.assertEqual(sum(m['optimizer_success'] is False for m in metrics),3)
        mutations={
            'gap_invalid':lambda r:r.update(gap_valid=False),
            'nonfinite_gap':lambda r:r.update(gap=float('nan')),
            'infeasible':lambda r:r['full_feasibility'].update(max_violation=1e-4),
            'wide_bracket':lambda r:r['numerical_quality'].update(primal_dual_width=1e-4),
            'stationarity':lambda r:r['numerical_quality'].update(stationarity_l2=1e-4),
            'negative_multiplier':lambda r:r['numerical_quality']['multipliers'].__setitem__(0,-1.),
            'failed_posterior_check':lambda r:r['numerical_quality']['checks'].update(primal_dual_width=False),
            'optimizer_objective_changed':lambda r:r.update(raw_objective=999.),
        }
        for name,mutate in mutations.items():
            with self.subTest(name=name):
                record=deepcopy(records[0]);mutate(record)
                with self.assertRaises(ValueError):adoption.check_quality(record)

    def test_original_target_refused(self):
        with self.assertRaisesRegex(ValueError,'copied results_single'):
            adoption.prepare(adoption.ROOT/'results/verified','unused','unused')

    def synthetic_transaction(self,directory,reviewed=True):
        root=Path(directory)/'verified';case=root/'synthetic_case';case.mkdir(parents=True)
        diag=case/'synthetic_diagnostics.json';manifest=case/'manifest.json';points=case/'synthetic_points.npz'
        diag.write_text('[{"fixture": "original"}]\n');manifest.write_text('{"fixture": "original counters"}\n');points.write_bytes(b'synthetic protected point archive')
        originals={p:p.read_bytes() for p in (diag,manifest)}
        writes={diag:b'[{"fixture": "new diagnostic"}]\n',manifest:b'{"fixture": "new counters"}\n'}
        review=dict(target=str(root),quality_reviewed_by_explicit_cli=reviewed,input_sha256={str(p):adoption.sha(p) for p in (diag,manifest,points)},helper_sha256=adoption.sha(adoption.__file__),rows=8,fixture_only=True)
        return root,points,review,writes,originals

    def test_explicit_review_required_without_mutation(self):
        with tempfile.TemporaryDirectory(prefix='cvi_adoption_fixture_') as directory:
            root,points,review,writes,originals=self.synthetic_transaction(directory,reviewed=False)
            with self.assertRaisesRegex(ValueError,'quality-reviewed'):adoption.apply(review,writes,originals)
            self.assertTrue(all(path.read_bytes()==data for path,data in originals.items()))
            self.assertFalse((root/'diagnostic_adoption_manifest.json').exists())

    def test_temporary_apply_preserves_backup_and_protected_pairs(self):
        with tempfile.TemporaryDirectory(prefix='cvi_adoption_fixture_') as directory:
            root,points,review,writes,originals=self.synthetic_transaction(directory)
            before=points.read_bytes();result=adoption.apply(review,writes,originals)
            self.assertEqual(result['state'],'applied');self.assertEqual(points.read_bytes(),before)
            self.assertTrue(all(path.read_bytes()==data for path,data in writes.items()))
            backup=Path(result['backup_directory'])
            self.assertTrue(all((backup/path.relative_to(root)).read_bytes()==data for path,data in originals.items()))
            self.assertEqual(result['helper_snapshot_sha256'],adoption.sha(adoption.__file__))
            self.assertFalse((root/'.diagnostic_adoption.lock').exists())

    def test_temporary_failure_rolls_back_every_written_primary_file(self):
        with tempfile.TemporaryDirectory(prefix='cvi_adoption_fixture_') as directory:
            root,points,review,writes,originals=self.synthetic_transaction(directory)
            original_atomic=adoption.atomic_bytes;calls=[0]
            def fail_second(path,data):
                calls[0]+=1
                if calls[0]==2:raise OSError('synthetic transaction failure')
                original_atomic(path,data)
            with patch.object(adoption,'atomic_bytes',side_effect=fail_second):
                with self.assertRaises(OSError):adoption.apply(review,writes,originals)
            self.assertTrue(all(path.read_bytes()==data for path,data in originals.items()))
            self.assertFalse((root/'diagnostic_adoption_manifest.json').exists())
            self.assertFalse((root/'.diagnostic_adoption.lock').exists())


if __name__=='__main__':unittest.main()
