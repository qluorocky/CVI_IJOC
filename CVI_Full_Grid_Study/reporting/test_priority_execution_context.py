#!/usr/bin/env python3
"""Synthetic scheduling-provenance checks; no actual experiment files change."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import priority_execution_context as context
import report_full_grid as reporting


def fixture(root):
    plan=root/'original_plan.json';plan.write_text(json.dumps(dict(synthetic_fixture=True,execution_policy=dict(workers=7))))
    runner=root/'runner_snapshot.py';runner.write_text('# SYNTHETIC priority runner source only\n')
    phase=root/'priority_phase_plan.json';phase.write_text(json.dumps(dict(synthetic_fixture=True,original_plan_file=str(plan),original_plan_sha256=context.sha(plan),priority_source_file=str(runner),priority_source_sha256=context.sha(runner),execution_policy=dict(original_worker_limit=7,priority_worker_limit=10,restored_worker_limit=7))))
    case=root/'multi_n100_m100_seed3';case.mkdir();pairs={}
    for method in sorted(context.METHODS):
        run=case/(method+'_run.json');run.write_text('SYNTHETIC run bytes '+method)
        points=case/(method+'_points.npz');points.write_bytes(('SYNTHETIC protected point bytes '+method).encode())
        pairs[method]=dict(run=context.sha(run),points=context.sha(points))
    record=dict(schema_version=1,phase_plan_file=str(phase),phase_plan_sha256=context.sha(phase),original_plan_sha256=context.sha(plan),priority_source_sha256=context.sha(runner),setting=dict(n=100,m=100),original_worker_limit=7,priority_worker_limit=10,logical_cpu_count=10,numerical_threads_per_worker=1,memory_budget_gib=8,modeled_case_resident_gib=.504,configured_pool_semantics='maximum admitted workers, not simultaneous-CPU guarantee',observed_peak_priority_workers=10,observed_active_at_launch=3,committed_before_launch_methods=['proposed'],methods_scheduled_in_priority=sorted(context.METHODS-{'proposed'}),prior_execution=dict(synthetic_fixture=True,worker_limit=7),timed_pair_sha256=pairs)
    (case/'priority_execution.json').write_text(json.dumps(record));return plan,case,record


class PriorityExecution(unittest.TestCase):
    def test_frozen_runner_sidecar_schema_with_study_and_pool_disclosure(self):
        with tempfile.TemporaryDirectory(prefix='cvi_SYNTHETIC_priority_') as directory:
            root=Path(directory);plan,case,record=fixture(root)
            record['setting']['study']='multi'
            record['configured_pool_semantics']='Maximum admitted workers; not a simultaneous-CPU or matched-panel workload guarantee'
            (case/'priority_execution.json').write_text(json.dumps(record))
            result=context.load_priority_execution(case,dict(n=100,m=100,seed=3),plan,{})
            self.assertEqual(result['recorded_sidecar'],record)
            record['setting']['study']='single'
            (case/'priority_execution.json').write_text(json.dumps(record))
            with self.assertRaises(ValueError):context.load_priority_execution(case,dict(n=100,m=100,seed=3),plan,{})

    def test_binds_sidecar_phase_source_base_plan_and_pairs(self):
        with tempfile.TemporaryDirectory(prefix='cvi_SYNTHETIC_priority_') as directory:
            root=Path(directory);plan,case,record=fixture(root);before=plan.read_bytes();hashes={}
            result=context.load_priority_execution(case,dict(n=100,m=100,seed=3),plan,hashes)
            self.assertEqual(result['priority_worker_limit'],10);self.assertEqual(result['original_worker_limit'],7)
            self.assertEqual(plan.read_bytes(),before);self.assertIn(str((case/'priority_execution.json').resolve()),hashes)
            self.assertIn(str((root/'priority_phase_plan.json').resolve()),hashes);self.assertIn(str((root/'runner_snapshot.py').resolve()),hashes)
            summary=context.summarize_priority_execution([result]);self.assertEqual(summary['priority_method_runs'],4);self.assertEqual(summary['previously_committed_method_runs'],1)
            text=context.priority_execution_text(summary);self.assertIn('ten workers',text);self.assertIn('original seven-worker',text);self.assertIn('do not guarantee simultaneous CPU',text)
            protocol=reporting.protocol(dict(settings=[dict(study='single',n=1000,m=1),dict(study='multi',n=100,m=100)],seeds=list(range(10)),methods=list(reporting.METHODS),seconds=1000),dict(priority_execution_overrides=summary))
            self.assertIn(r'\paragraph{Priority execution phase.}',protocol);self.assertIn('4 completed method trajectories',protocol)
            (case/'oe_true_points.npz').write_bytes(b'CHANGED SYNTHETIC point bytes')
            with self.assertRaises(ValueError):context.load_priority_execution(case,dict(n=100,m=100,seed=3),plan,{})

    def test_subset_plan_may_link_to_unchanged_original_without_rewriting_it(self):
        with tempfile.TemporaryDirectory(prefix='cvi_SYNTHETIC_priority_') as directory:
            root=Path(directory);plan,case,record=fixture(root);subset=root/'selected_plan.json';subset.write_text(json.dumps(dict(synthetic_fixture=True,source_full_plan_sha256=context.sha(plan))))
            context.load_priority_execution(case,dict(n=100,m=100,seed=3),subset,{},allow_subset_plan=True)
            with self.assertRaises(ValueError):context.load_priority_execution(case,dict(n=100,m=100,seed=3),subset,{})
            plan.write_text('CHANGED SYNTHETIC original plan')
            with self.assertRaises(ValueError):context.load_priority_execution(case,dict(n=100,m=100,seed=3),subset,{},allow_subset_plan=True)

    def test_invalid_observed_counts_or_overlapping_methods_are_refused(self):
        for mutation in ('peak','overlap','phase_hash','setting'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory(prefix='cvi_SYNTHETIC_priority_') as directory:
                root=Path(directory);plan,case,record=fixture(root)
                if mutation=='peak':record['observed_peak_priority_workers']=11
                if mutation=='overlap':record['methods_scheduled_in_priority'].append('proposed')
                if mutation=='phase_hash':record['phase_plan_sha256']='0'*64
                if mutation=='setting':record['setting']['n']=50
                (case/'priority_execution.json').write_text(json.dumps(record))
                with self.assertRaises(ValueError):context.load_priority_execution(case,dict(n=100,m=100,seed=3),plan,{})

    def test_absent_override_does_not_invent_ten_worker_execution(self):
        with tempfile.TemporaryDirectory(prefix='cvi_SYNTHETIC_priority_') as directory:
            root=Path(directory);hashes={}
            self.assertIsNone(context.load_priority_execution(root,dict(n=1000,m=1,seed=0),root/'unused.json',hashes))
            self.assertEqual(hashes,{})
            self.assertEqual(context.priority_execution_text(context.summarize_priority_execution([])),'')


if __name__=='__main__':unittest.main()
