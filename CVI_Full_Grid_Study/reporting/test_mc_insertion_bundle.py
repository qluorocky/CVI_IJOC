#!/usr/bin/env python3
"""Explicitly synthetic MC-bundle tests; no current data or delivery is changed."""
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest

import mc_insertion_bundle as bundle
import report_full_grid as reporting


def synthetic_inputs():
    # Pure in-memory contract fixture, never represented as a measured dataset.
    plan=dict(synthetic_fixture_only=True,settings=[dict(study='single',n=n,m=1) for n in (10,50,100,200,500,1000)]+[dict(study='multi',n=100,m=m) for m in bundle.MC_M],seeds=list(range(10)),methods=list(bundle.METHODS),seconds=1000,source_hashes={'SYNTHETIC_FIXTURE.py':'a'*64})
    loaded=[]
    for setting in plan['settings']:
        for seed in plan['seeds']:
            for method in plan['methods']:
                n,m=setting['n'],setting['m'];meta=dict(n=n,m=m,seed=seed,synthetic_fixture_only=True)
                points=[dict(n=n,m=m,seed=seed,method=method,checkpoint=i,k=i,elapsed=elapsed,gap=gap,violation=0.,residual=gap,gap_valid=True,gap_status='optimal',diagnostic_seconds=0.) for i,(elapsed,gap) in enumerate(((0.,.2),(1000.,.001)))]
                loaded.append(dict(case=f'SYNTHETIC_{setting["study"]}_n{n}_m{m}_seed{seed}',meta=meta,method=method,result=dict(status='time_limit',runtime=1000.,counts=None),checkpoints=points,samples=[]))
    return plan,loaded


class MCInsertionBundle(unittest.TestCase):
    def test_eight_single_invalid_rows_do_not_qualify_clean_mc_subset(self):
        plan,loaded=synthetic_inputs();singles=[item for item in loaded if item['meta']['m']==1]
        for item in singles[:8]:item['checkpoints'][0].update(gap=float('nan'),residual=float('nan'),gap_valid=False,gap_status='8')
        selected=bundle.select_mc(plan,loaded);inventory,rows,runs=bundle.mc_inventory(plan,selected)
        self.assertEqual(len(selected),250);self.assertEqual(len(runs),250);self.assertEqual(inventory['instances'],50)
        self.assertEqual(inventory['status'],'mc_complete');self.assertFalse(inventory['mc_has_exceptions']);self.assertEqual(inventory['unresolved_joint_diagnostics'],[])
        self.assertTrue(all(row['m'] in bundle.MC_M and row['n']==100 for row in rows))
        self.assertNotIn('single_constraint_comparative',reporting.discussion(dict(plan,settings=[s for s in plan['settings'] if s['study']=='multi']),runs))

    def test_actual_mc_exceptions_remain_qualified_and_seed_denominator_fixed(self):
        plan,loaded=synthetic_inputs();selected=bundle.select_mc(plan,loaded)
        for point in selected[0]['checkpoints']:point.update(gap=float('nan'),residual=float('nan'),gap_valid=False,gap_status='synthetic_failure')
        selected[0]['result']['status']='method_error'
        inventory,rows,runs=bundle.mc_inventory(plan,selected)
        self.assertTrue(inventory['mc_has_exceptions']);self.assertEqual(len(inventory['unresolved_joint_diagnostics']),2)
        self.assertEqual(len(inventory['algorithm_failures']),1);self.assertEqual(len(inventory['unavailable_aggregate_groups']),1)
        self.assertEqual(len(runs),250);self.assertEqual(inventory['seeds'],list(range(10)))

    def test_old_or_missing_mc_rows_are_refused(self):
        plan,loaded=synthetic_inputs()
        with self.assertRaises(ValueError):bundle.select_mc(plan,loaded[:-1])
        old=deepcopy(plan);old['settings'][-1]['m']=1000
        with self.assertRaises(reporting.IncompleteDataset):bundle.select_mc(old,loaded)
        old=deepcopy(plan);old['settings'][-1]['n']=1000
        with self.assertRaises(reporting.IncompleteDataset):bundle.select_mc(old,loaded)

    def test_full_audit_exceptions_are_permitted_but_failed_audits_are_not(self):
        plan,_=synthetic_inputs();validation=dict(status='complete_with_exceptions',instances=110,method_runs=550,plan_snapshot=plan,reporter_sha256='reporter')
        structural=dict(status='structural_pass',scope='requested_full_grid',expected_instances=110,audited_instances=110,expected_method_runs=550,audited_method_runs=550,coverage_complete=True,auditor_sha256='structural')
        stats=dict(status='passed',scope='full_requested_grid',expected_instances=110,expected_method_runs=550,auditor_sha256='stats')
        bundle.check_audits(plan,validation,structural,stats,'reporter','structural','stats')
        for evidence in (structural,stats):
            before=evidence['status'];evidence['status']='failed'
            with self.assertRaises(ValueError):bundle.check_audits(plan,validation,structural,stats,'reporter','structural','stats')
            evidence['status']=before

    def test_final_review_requires_mc_exception_acknowledgement(self):
        review=dict(decision='approved_for_mc_insertion_bundle',mc_review_pdf_sha256='a',mc_table_figure_pdf_sha256='b',references_resolved=True,checks={key:True for key in ('coverage','statistics','structural','visual_table','visual_figure','qualifications')})
        manifest=dict(mc_inventory=dict(mc_has_exceptions=False));hashes=dict(mc_review='a',mc_table_figure='b')
        bundle.check_final_review(review,manifest,hashes)
        manifest['mc_inventory']['mc_has_exceptions']=True
        with self.assertRaises(ValueError):bundle.check_final_review(review,manifest,hashes)
        review['acknowledges_mc_exceptions']=True;bundle.check_final_review(review,manifest,hashes)
        with self.assertRaises(ValueError):bundle.check_final_review(review,manifest,dict(mc_review='changed',mc_table_figure='b'))

    def test_synthetic_standalone_tables_and_figure_compile(self):
        if not shutil.which('pdflatex'):self.skipTest('pdfLaTeX is unavailable')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plan,loaded=synthetic_inputs();selected=bundle.select_mc(plan,loaded);inventory,rows,runs=bundle.mc_inventory(plan,selected)
        plan['validation_fixture']=True
        with tempfile.TemporaryDirectory(prefix='cvi_SYNTHETIC_mc_bundle_') as directory:
            root=Path(directory);assets=root/'MC_results';assets.mkdir()
            for metric,suffix in (('residual',''),('gap','_gap_only')):reporting.table(assets/('mc_table'+suffix+'.tex'),plan,runs,'multi',metric)
            self.assertEqual(sum(line[:1].isdigit() and '&' in line for line in (assets/'mc_table.tex').read_text().splitlines()),15)
            figure=plt.figure(figsize=(5,2));figure.text(.5,.5,'SYNTHETIC FIXTURE ONLY\nNo measured MC values',ha='center');figure.savefig(assets/'mc_full_grid.pdf');plt.close(figure)
            (assets/'mc_full_grid_figure.tex').write_text(bundle.figure_fragment(r'\begin{figure*}\centering\includegraphics[width=\textwidth]{mc_full_grid.pdf}\caption{SYNTHETIC FIXTURE ONLY.}\label{fig:mc_full_grid}\end{figure*}'))
            (assets/'mc_protocol.tex').write_text(r'\textbf{SYNTHETIC FIXTURE ONLY.}'+bundle.mc_protocol(inventory))
            (assets/'mc_discussion.tex').write_text(r'\textbf{SYNTHETIC FIXTURE ONLY.} Table~\ref{tab:multi_constraint_comparative} and Figure~\ref{fig:mc_full_grid} are formatting tests.')
            bundle.write_entries(root);bundle.compile_reviews(root)
            self.assertTrue((root/'mc_review.pdf').is_file());self.assertTrue((root/'mc_table_figure.pdf').is_file())
            self.assertFalse((root/'data').exists())


if __name__=='__main__':unittest.main()
