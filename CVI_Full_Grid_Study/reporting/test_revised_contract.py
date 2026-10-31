#!/usr/bin/env python3
"""Lightweight synthetic reporting checks; never runs a numerical experiment.

All generated preview artifacts are explicitly labeled validation fixtures and
live in a temporary directory. No measured dataset or frozen source is changed.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import report_full_grid as reporter
import audit_table_statistics as statistics_audit
import audit_saved_records as structural_audit
import assemble_manuscript as assembler


def full_plan():
    return dict(settings=[dict(study='single',n=n,m=1) for n in (10,50,100,200,500,1000)]+
                [dict(study='multi',n=100,m=m) for m in (10,50,100,200,500)],
                seeds=list(range(10)),methods=list(reporter.METHODS),seconds=1000,
                source_hashes={'study.py':'a'*64},
                plan_amendment={'multi_n':100,'removed_multi_m':[1000]},
                inherited_single_provenance={'fixture_only':True},
                orchestrator_hashes={'batch_n100.py':'b'*64})


class RevisedContract(unittest.TestCase):
    def test_inherited_single_pair_lineage(self):
        source=Path('/private/tmp/cvi_full_grid_five_fixture/data')
        if not (source/'single_n10_m1_seed0').is_dir():self.skipTest('Earlier genuine worker fixture is unavailable')
        with tempfile.TemporaryDirectory(prefix='cvi_single_lineage_') as directory:
            root=Path(directory);case=root/'single_n10_m1_seed0'
            shutil.copytree(source/case.name,case)
            plan=json.loads((source/'plan.json').read_text());plan['settings']=[dict(study='single',n=10,m=1)]
            plan['inherited_single_provenance']=dict(source_results=str(source),source_plan_sha256=reporter.digest(source/'plan.json'),earlier_small_study_reused=False)
            reporter.write_json(root/'plan.json',plan)
            copied={p.name:reporter.digest(p) for p in case.iterdir() if p.is_file()}
            lineage=dict(source_case=str(source/case.name),source_results=str(source),source_plan_sha256=plan['inherited_single_provenance']['source_plan_sha256'],new_plan_sha256=reporter.digest(root/'plan.json'),copied_artifact_sha256=copied)
            reporter.write_json(case/'lineage.json',lineage)
            _,runs,hashes=reporter.load_complete(root,validation_subset=True)
            self.assertEqual(len(runs),5);self.assertIn(str(case/'lineage.json'),hashes)
            _,runs,_=statistics_audit.load_complete_runs(root,plan_scope=True);self.assertEqual(len(runs),5)
            command=[sys.executable,str(Path(structural_audit.__file__)),'--results',str(root),'--code',str(source.parent/'code'),'--plan-scope','--preflight']
            result=subprocess.run(command,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(json.loads((root/'report/structural/structural_audit.json').read_text())['status'],'preflight_only')
            # Diagnostics may acquire a separately recorded retry; timed pairs stay fixed.
            diag=case/'proposed_diagnostics.json';diag.write_text(diag.read_text()+'\n')
            reporter.load_complete(root,validation_subset=True);statistics_audit.load_complete_runs(root,plan_scope=True)
            run=case/'proposed_run.json';run.write_text(run.read_text()+'\n')
            with self.assertRaises(reporter.IncompleteDataset):reporter.load_complete(root,validation_subset=True)
            with self.assertRaises(statistics_audit.IncompleteDataset):statistics_audit.load_complete_runs(root,plan_scope=True)
            result=subprocess.run(command,capture_output=True,text=True);self.assertEqual(result.returncode,1,result.stdout+result.stderr)

    def test_explicit_single_release(self):
        plan=full_plan();plan['settings']=[s for s in plan['settings'] if s['study']=='single'];plan['release_scope']='single_only'
        reporter.check_plan(plan,single_only=True)
        statistics_audit.validate_plan(plan,single_only=True)
        issues,incomplete=[],[]
        expected=structural_audit.validate_plan(plan,False,issues,incomplete,single_only=True)
        self.assertFalse(issues+incomplete);self.assertEqual(len(expected),60)
        self.assertEqual(len(expected)*len(plan['methods']),300)
        self.assertIn('only the single-constraint experiments',reporter.protocol(plan))
        with self.assertRaises(reporter.IncompleteDataset):reporter.check_plan(plan)
        with self.assertRaises(statistics_audit.IncompleteDataset):statistics_audit.validate_plan(plan)
        without_scope=deepcopy(plan);without_scope.pop('release_scope')
        with self.assertRaises(reporter.IncompleteDataset):reporter.check_plan(without_scope,single_only=True)
        with self.assertRaises(statistics_audit.IncompleteDataset):statistics_audit.validate_plan(without_scope,single_only=True)
        mixed=deepcopy(plan);mixed['settings'].append(dict(study='multi',n=100,m=10))
        with self.assertRaises(reporter.IncompleteDataset):reporter.check_plan(mixed,single_only=True)
        with self.assertRaises(statistics_audit.IncompleteDataset):statistics_audit.validate_plan(mixed,single_only=True)

    def accepted_by_all(self,plan):
        reporter.check_plan(plan)
        statistics_audit.validate_plan(plan)
        issues,incomplete=[],[]
        expected=structural_audit.validate_plan(plan,False,issues,incomplete)
        self.assertFalse(issues+incomplete)
        self.assertEqual(len(expected),110)

    def rejected_by_all(self,plan):
        with self.assertRaises(reporter.IncompleteDataset):reporter.check_plan(plan)
        with self.assertRaises(statistics_audit.IncompleteDataset):statistics_audit.validate_plan(plan)
        issues,incomplete=[],[]
        structural_audit.validate_plan(plan,False,issues,incomplete)
        self.assertTrue(issues+incomplete)

    def test_exact_revised_contract(self):
        plan=full_plan();self.accepted_by_all(plan)
        self.assertEqual(len(plan['settings'])*len(plan['seeds'])*len(plan['methods']),550)
        singles,multi=assembler.settings_text(plan)
        self.assertEqual(singles,[10,50,100,200,500,1000])
        self.assertEqual({s['n'] for s in multi},{100})
        self.assertEqual([s['m'] for s in multi],[10,50,100,200,500])

    def test_reject_stale_and_malformed_release_contracts(self):
        mutations={
            'old_multi_n1000':lambda p:[s.update(n=1000) for s in p['settings'] if s['study']=='multi'],
            'mixed_multi_dimensions':lambda p:p['settings'][-1].update(n=1000),
            'removed_m1000_reintroduced':lambda p:p['settings'].append(dict(study='multi',n=100,m=1000)),
            'missing_row':lambda p:p['settings'].pop(),
            'duplicate_row':lambda p:p['settings'].append(deepcopy(p['settings'][0])),
            'float_dimension':lambda p:p['settings'][0].update(n=10.0),
            'study_identity':lambda p:p['settings'][0].update(study='multi'),
            'seed_order':lambda p:p['seeds'].reverse(),
            'boolean_seed':lambda p:p['seeds'].__setitem__(0,False),
            'method_order':lambda p:p['methods'].reverse(),
            'method_cap':lambda p:p.update(seconds=999),
            'source_hash':lambda p:p.update(source_hashes={'study.py':'invalid'}),
        }
        for name,mutate in mutations.items():
            with self.subTest(name=name):
                plan=full_plan();mutate(plan);self.rejected_by_all(plan)

    def test_revised_tables_protocol_and_figures(self):
        plan=full_plan();plan['validation_fixture']=True
        loaded=[];rows=[]
        for setting in plan['settings']:
            for seed in plan['seeds']:
                for method in plan['methods']:
                    key=(setting['n'],setting['m'],seed)
                    diag=[dict(n=key[0],m=key[1],seed=seed,method=method,checkpoint=i,
                               elapsed=t,gap=g,violation=0.,residual=g,gap_valid=True)
                          for i,(t,g) in enumerate(((0.,1.),(1.,1e-6)))]
                    loaded.append((key,method,diag));rows.extend(diag)
        records,issues=statistics_audit.derive_records(loaded);self.assertFalse(issues)
        runs=[dict(record,status='time_limit',invalid_gap_checkpoints=0) for record in records.values()]
        with tempfile.TemporaryDirectory(prefix='cvi_revised_reporting_') as directory:
            out=Path(directory)
            for study,filename,count in (('single','pfaos_table',18),('multi','mc_table',15)):
                for metric,suffix in (('residual',''),('gap','_gap_only')):
                    path=out/(filename+suffix+'.tex');reporter.table(path,plan,runs,study,metric)
                    check=statistics_audit.compare_table(path,records,plan['settings'],reporter.TABLE_METHODS,metric,10,study=='multi')
                    self.assertEqual(check['rows'],count);self.assertFalse(check['issues'])
                    self.assertIn('VALIDATION FIXTURE ONLY',path.read_text())
                    if study=='multi':
                        self.assertIn('Fix $n=100$.',path.read_text())
                        self.assertNotIn('Fix $n=1000$.',path.read_text())
            protocol=reporter.protocol(plan)
            self.assertIn('fixes $n=100$',protocol)
            self.assertIn(r'$m\in\{10,50,100,200,500\}$',protocol)
            self.assertIn('at $n=100$',reporter.discussion(plan,runs))
            # A one-seed, explicitly synthetic preview exercises five-panel layout.
            preview=deepcopy(plan);preview['seeds']=[0]
            subset=[row for row in rows if row['seed']==0]
            reporter.write_csv(out/'checkpoints.csv',subset)
            reporter.figures(out,preview,subset)
            provenance=json.loads((out/'full_grid_figure_provenance.json').read_text())
            self.assertEqual(sum(s['study']=='single' for s in provenance['settings']),30)
            self.assertEqual(sum(s['study']=='multi' for s in provenance['settings']),25)
            self.assertTrue(all(s['n']==100 and s['m']<=500 for s in provenance['settings'] if s['study']=='multi'))
            self.assertIn('$n=100$;',(out/'mc_full_grid_figure.tex').read_text())
            single=deepcopy(preview);single['settings']=[s for s in single['settings'] if s['study']=='single'];single['release_scope']='single_only'
            single_out=out/'single';single_out.mkdir();single_rows=[row for row in subset if row['m']==1]
            reporter.write_csv(single_out/'checkpoints.csv',single_rows);reporter.figures(single_out,single,single_rows)
            self.assertTrue((single_out/'pfaos_full_grid.pdf').exists());self.assertFalse(list(single_out.glob('mc_*')))
            single_discussion=reporter.discussion(single,[r for r in runs if r['m']==1])
            self.assertNotIn('multi_constraint_comparative',single_discussion)
            self.assertNotIn('MC-PFAOS',single_discussion)


if __name__=='__main__':unittest.main()
