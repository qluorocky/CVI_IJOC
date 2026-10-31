#!/usr/bin/env python3
"""Pure lightweight packaging checks; never copies measured data or stages a ZIP."""
from copy import deepcopy
from pathlib import Path
import unittest
import package_single_latex as packaging


class PortablePackage(unittest.TestCase):
    def test_main_wrapper_adds_only_local_override(self):
        text='% original wrapper\n\\makeatletter\n\\def\\input@path{{JoC_submission/}}\n\\makeatother\n\\input{JoC_submission/INFORMS-IJOC.tex}\n'
        result,description=packaging.normalize_wrapper(text,'main.tex')
        self.assertEqual(result,'\\providecommand{\\ResultsDir}{measured_results}\n'+text)
        self.assertIn('only ResultsDir',description)

    def test_numerical_review_changes_only_path(self):
        text='\\documentclass{article}\n\\newcommand{\\ResultsDir}{../results_single/verified/report}\n\\begin{document}unchanged measured values\\end{document}\n'
        result,_=packaging.normalize_wrapper(text,'numerical_review.tex')
        self.assertEqual(result,text.replace('../results_single/verified/report','measured_results'))
        with self.assertRaises(ValueError):packaging.normalize_wrapper('no definition','numerical_review.tex')
        with self.assertRaises(ValueError):packaging.normalize_wrapper(text+text,'numerical_review.tex')

    def test_review_requires_actual_portable_pdf_identity_and_every_check(self):
        review=dict(decision='approved_for_portable_release',package_main_pdf_sha256='a'*64,references_resolved=True,citations_resolved=True,visual_inspection_passed=True,checks={k:True for k in ('references','citations','visual_tables','visual_figure','proofs_unchanged')})
        packaging.check_review(review,'a'*64)
        with self.assertRaises(ValueError):packaging.check_review(review,'b'*64)
        for key in review['checks']:
            bad=deepcopy(review);bad['checks'][key]=False
            with self.subTest(key=key):
                with self.assertRaises(ValueError):packaging.check_review(bad,'a'*64)
        bad=deepcopy(review);bad['decision']='pending'
        with self.assertRaises(ValueError):packaging.check_review(bad,'a'*64)

    def test_build_junk_excluded_and_bibliography_retained(self):
        for name in ('main.aux','main.log','main.blg','__pycache__/test.pyc','.DS_Store','draft.tmp'):
            self.assertFalse(packaging.included(Path(name)),name)
        for name in ('main.tex','main.bbl','main.pdf','plan.json','single_n1000_m1_seed9/proposed_points.npz'):
            self.assertTrue(packaging.included(Path(name)),name)


if __name__=='__main__':unittest.main()
