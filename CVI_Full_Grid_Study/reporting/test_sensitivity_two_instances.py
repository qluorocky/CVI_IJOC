#!/usr/bin/env python3
"""Synthetic format/preflight checks; never modifies measured datasets."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import plot_sensitivity_two_instances as plotter


class SensitivityFigure(unittest.TestCase):
    def test_exact_geometry_legend_strokes_and_axis_limits(self):
        panels=plotter.synthetic_panels();curves=[plotter.aggregate(panel,100) for panel in panels]
        figure,axes=plotter.make_figure(panels,curves,fixture=True)
        try:
            self.assertEqual(tuple(figure.get_size_inches()),(11.,4.8))
            self.assertEqual(tuple(axes[0].get_xlim()),(1e-4,1000));self.assertEqual(tuple(axes[0].get_ylim()),(1e-6,100))
            for value,expected in zip(axes[0].get_position().bounds,(.075,.19,.3217592592592593,.70)):self.assertAlmostEqual(value,expected,12)
            for value,expected in zip(axes[1].get_position().bounds,(.4482407407407407,.19,.3217592592592593,.70)):self.assertAlmostEqual(value,expected,12)
            self.assertEqual(len(figure.legends[0].get_texts()),7)
            self.assertEqual([text.get_fontsize() for text in figure.legends[0].get_texts()],[10.3]*7)
            self.assertEqual([line.get_color() for line in axes[0].lines[:5]],['#111111','#666666','#3c3c3c','#888888','#aaaaaa'])
            self.assertEqual([line.get_marker() for line in axes[0].lines[1:5]],['s','o','^','D'])
            self.assertTrue(all(line.get_linewidth()==1.65 for line in axes[0].lines[:5]))
            self.assertEqual(axes[0].get_ylabel(),'Combined error');self.assertEqual(axes[0].get_xlabel(),'Wall-clock time (s)')
            self.assertIn('n=100',axes[1].get_title());self.assertTrue(any('SYNTHETIC STYLE FIXTURE' in text.get_text() for text in figure.texts))
        finally:
            import matplotlib.pyplot as plt
            plt.close(figure)

    def test_fixed_ten_seed_common_horizon_no_extrapolation(self):
        panel=plotter.synthetic_panels()[0]
        panel['records']['proposed'][0]['rows']=[row for row in panel['records']['proposed'][0]['rows'] if row['elapsed']<=100]
        endpoint=panel['records']['proposed'][0]['rows'][-1]['elapsed'];curves=plotter.aggregate(panel,100)
        self.assertEqual(curves['proposed']['seed_count'],10);self.assertEqual(curves['proposed']['common_endpoint_seconds'],endpoint)
        self.assertTrue(np.all(curves['proposed']['time']<=endpoint));self.assertTrue(np.all(curves['proposed']['lower']<=curves['proposed']['upper']))
        panel['records']['proposed'].pop()
        with self.assertRaises(ValueError):plotter.aggregate(panel,100)

    def test_invalid_seed_is_unavailable_instead_of_dropped(self):
        panel=plotter.synthetic_panels()[1]
        panel['records']['oe_low'][0]['rows'][2]['gap_valid']=False
        curves=plotter.aggregate(panel,100)
        self.assertFalse(curves['oe_low']['available']);self.assertTrue(curves['oe_true']['available'])

    def test_audited_snapshot_requires_matching_setting_path_and_bytes(self):
        with tempfile.TemporaryDirectory(prefix='cvi_SYNTHETIC_audit_binding_') as directory:
            file=Path(directory)/'multi_n100_m100_seed0'/'oe_true_run.json';file.parent.mkdir();file.write_text('SYNTHETIC FIXTURE ONLY')
            identity=hashlib.sha256(file.read_bytes()).hexdigest()
            plotter.audit_binds(file,{'/synthetic/audited_copy/multi_n100_m100_seed0/oe_true_run.json':identity})
            with self.assertRaises(ValueError):plotter.audit_binds(file,{'/synthetic/audited_copy/multi_n50_m100_seed0/oe_true_run.json':identity})
            with self.assertRaises(ValueError):plotter.audit_binds(file,{'/synthetic/audited_copy/multi_n100_m100_seed0/oe_true_run.json':'0'*64})

    def test_missing_requested_setting_and_fixture_data_are_refused(self):
        with tempfile.TemporaryDirectory(prefix='cvi_SYNTHETIC_preflight_') as directory:
            root=Path(directory);dataset=root/'fixture_dataset';dataset.mkdir();code=root/'fixture_code';code.mkdir();(code/'fixture.py').write_text('# SYNTHETIC FIXTURE ONLY\n')
            plan=dict(synthetic_fixture_only=True,settings=[dict(study='multi',n=100,m=10)],seeds=list(range(10)),methods=list(plotter.PLAN_METHODS),seconds=1000,source_hashes={'fixture.py':plotter.sha(code/'fixture.py')})
            (dataset/'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'absent or duplicated'):plotter.load_setting(dataset,100,100,root/'missing_audit.json',code)
            plan['validation_fixture']=True;(dataset/'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'validation fixture'):plotter.load_setting(dataset,100,10,root/'missing_audit.json',code)
            self.assertFalse((root/'actual_figure').exists())


if __name__=='__main__':unittest.main()
