#!/usr/bin/env python3
"""Independently verify CSV and four LaTeX tables against complete raw runs.

No experiment/report module is imported. Threshold decisions use the exact
recorded gap and violation, with no tolerance added to the target. Main-table
hits require gap and violation at the same checkpoint. Conditional timing
statistics use successful runs only; final-V statistics use every planned run.
Incomplete plans are refused before any audit output is written.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

THRESHOLDS = (1e-2, 1e-3, 1e-4, 1e-5)
METHODS = ('proposed','oe_true','oe_low','oe_high','golden_ratio')
TABLE_METHODS = ('proposed','oe_true','golden_ratio')
FULL_SETTINGS = {('single',n,1) for n in (10,50,100,200,500,1000)} | {('multi',100,m) for m in (10,50,100,200,500)}
METHOD_NAMES = {'PFAOS': 'proposed', 'MC-PFAOS': 'proposed',
                r'OE ($\widehat M=M$)': 'oe_true', 'Golden Ratio': 'golden_ratio'}


class IncompleteDataset(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def validate_plan(plan, plan_scope=False, single_only=False):
    """Independently validate the revised release identity before reading cases."""
    if not isinstance(plan,dict) or any(k not in plan for k in ('settings','seeds','methods','seconds','source_hashes')):
        raise IncompleteDataset('Plan requires settings, seeds, methods, seconds, and source_hashes.')
    if not isinstance(plan['settings'],list) or not plan['settings']:
        raise IncompleteDataset('Empty or invalid setting list.')
    identities=[]
    for setting in plan['settings']:
        if not isinstance(setting,dict):raise IncompleteDataset('Setting must be an object.')
        n,m=setting.get('n'),setting.get('m')
        if type(n) is not int or type(m) is not int or n<2 or m<1:
            raise IncompleteDataset('Setting dimensions must be integer dimensions with n >= 2 and m >= 1.')
        study='single' if m==1 else 'multi'
        if setting.get('study',study)!=study:raise IncompleteDataset('Setting study label disagrees with constraint count.')
        identities.append((study,n,m))
    if len(identities)!=len(set(identities)):raise IncompleteDataset('Duplicate settings.')
    if tuple(plan['methods']) not in ((METHODS,TABLE_METHODS) if plan_scope else (METHODS,)):
        raise IncompleteDataset('Method set/order is not the declared comparison.')
    seeds=plan['seeds']
    if not isinstance(seeds,list) or not seeds or any(type(seed) is not int or seed<0 for seed in seeds) or len(seeds)!=len(set(seeds)):
        raise IncompleteDataset('Empty, invalid, or duplicate seeds.')
    if type(plan['seconds']) not in (int,float) or not finite(plan['seconds']) or plan['seconds']<=0:
        raise IncompleteDataset('Invalid method cap.')
    hashes=plan['source_hashes']
    if not isinstance(hashes,dict) or not hashes or any(not isinstance(name,str) or Path(name).name!=name or not name.endswith('.py') or not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value) for name,value in hashes.items()):
        raise IncompleteDataset('Invalid frozen source hash map.')
    if plan_scope and single_only and (not plan.get('validation_fixture') or any(study!='single' for study,n,m in identities)):raise IncompleteDataset('Combined plan/single scope is allowed only for explicitly marked single-constraint layout fixtures.')
    expected={('single',n,1) for n in (10,50,100,200,500,1000)} if single_only else FULL_SETTINGS
    if not plan_scope and (set(identities)!=expected or seeds!=list(range(10)) or plan['seconds']!=1000):
        raise IncompleteDataset('Setting/seed/cap identity differs from the explicitly selected release contract.')
    if single_only and not plan_scope and plan.get('release_scope')!='single_only':raise IncompleteDataset('Single-only release requires release_scope=single_only in plan.json.')
    if not single_only and not plan_scope and plan.get('release_scope')=='single_only':raise IncompleteDataset('Single-only plan requires explicit --single-only.')
    return identities


def load_complete_runs(root, plan_scope=False, single_only=False):
    """Preflight every planned case/method before deriving any outcomes."""
    plan_path = root / 'plan.json'
    if not plan_path.exists():
        raise IncompleteDataset('No plan.json exists.')
    plan = json.loads(plan_path.read_text())
    validate_plan(plan,plan_scope,single_only)
    expected = {(int(s['n']), int(s['m']), int(seed))
                for s in plan['settings'] for seed in plan['seeds']}
    if not expected or len(set(plan['methods'])) != len(plan['methods']):
        raise IncompleteDataset('The plan is empty or contains duplicate methods.')
    manifests = list(root.glob('*/manifest.json'))
    found = {}
    for path in manifests:
        meta = json.loads(path.read_text())
        key = (meta['n'], meta['m'], meta['seed'])
        if key in found:
            raise IncompleteDataset(f'Duplicate planned case: {key}.')
        found[key] = (path, meta)
    if set(found) != expected:
        raise IncompleteDataset(f'Case set differs from plan; missing={sorted(expected-set(found))}, extra={sorted(set(found)-expected)}.')
    loaded = []
    files = {str(plan_path): sha(plan_path)}
    for key in sorted(expected):
        manifest, meta = found[key]
        if meta['state'] != 'complete' or meta['source_hashes'] != plan['source_hashes']:
            raise IncompleteDataset(f'{manifest.parent.name}: incomplete state or source-hash mismatch.')
        if meta['cap_seconds'] != plan['seconds']:
            raise IncompleteDataset(f'{manifest.parent.name}: method budget differs from plan.')
        lineagepath=manifest.parent/'lineage.json'
        if lineagepath.exists():
            lineage=json.loads(lineagepath.read_text())
            origin=plan.get('inherited_single_provenance',{})
            if meta['m']!=1 or Path(lineage.get('source_case','')).name!=manifest.parent.name or not lineage.get('source_results') or not lineage.get('source_plan_sha256') or lineage.get('new_plan_sha256')!=sha(plan_path) or any(origin.get(key) is not None and lineage.get(key)!=origin[key] for key in ('source_results','source_plan_sha256')):
                raise IncompleteDataset(f'{manifest.parent.name}: inherited lineage identity or amended-plan hash mismatch.')
            copied=lineage.get('copied_artifact_sha256')
            if not isinstance(copied,dict) or not copied:raise IncompleteDataset(f'{manifest.parent.name}: missing inherited artifact hash map.')
            for name,value in copied.items():
                if not isinstance(name,str) or Path(name).name!=name or not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:raise IncompleteDataset(f'{manifest.parent.name}: invalid inherited artifact hash identity.')
                if name.endswith(('_run.json','_points.npz')) and (not (manifest.parent/name).is_file() or sha(manifest.parent/name)!=value):raise IncompleteDataset(f'{manifest.parent.name}: inherited timed artifact changed: {name}.')
            files[str(lineagepath)]=sha(lineagepath)
        files[str(manifest)] = sha(manifest)
        for method in plan['methods']:
            if meta.get('methods', {}).get(method, {}).get('stage') != 'complete':
                raise IncompleteDataset(f'{manifest.parent.name}/{method}: method stage is incomplete.')
            runpath = manifest.parent / f'{method}_run.json'
            diagpath = manifest.parent / f'{method}_diagnostics.json'
            if not runpath.exists() or not diagpath.exists():
                raise IncompleteDataset(f'{manifest.parent.name}/{method}: saved run/diagnostic file missing.')
            run = json.loads(runpath.read_text())
            pointpath = manifest.parent / f'{method}_points.npz'
            if not pointpath.exists():raise IncompleteDataset(f'{manifest.parent.name}/{method}: point archive missing.')
            metadata=run.get('metadata')
            if not metadata and not plan_scope:raise IncompleteDataset(f'{manifest.parent.name}/{method}: pair metadata missing.')
            if metadata:
                if any(metadata.get(field)!=meta[field] for field in ('n','m','seed','cap_seconds','instance_hash','source_hashes')) or metadata.get('method')!=method:
                    raise IncompleteDataset(f'{manifest.parent.name}/{method}: pair metadata mismatch.')
            if run.get('points_sha256'):
                if run['points_sha256']!=sha(pointpath):raise IncompleteDataset(f'{manifest.parent.name}/{method}: point archive checksum mismatch.')
            elif not plan_scope:raise IncompleteDataset(f'{manifest.parent.name}/{method}: point archive checksum missing.')
            files[str(pointpath)]=sha(pointpath)
            rows = json.loads(diagpath.read_text())
            if not rows or len(rows) != len(run['checkpoints']):
                raise IncompleteDataset(f'{manifest.parent.name}/{method}: diagnostic coverage is incomplete.')
            for index, (row, point) in enumerate(zip(rows, run['checkpoints'])):
                if (row['n'], row['m'], row['seed'], row['method'], row['checkpoint']) != (*key, method, index):
                    raise IncompleteDataset(f'{manifest.parent.name}/{method}: diagnostic identity mismatch.')
                if row['elapsed'] != point['elapsed'] or row['k'] != point['k']:
                    raise IncompleteDataset(f'{manifest.parent.name}/{method}: checkpoint identity mismatch.')
                if row['gap_valid'] and not all(finite(row[x]) and (x=='elapsed' or row[x]>=0) for x in ('gap','violation','residual','elapsed')):
                    raise IncompleteDataset(f'{manifest.parent.name}/{method}: diagnostic marked valid but nonfinite/negative.')
                if not row['gap_valid'] and not row.get('gap_status'):
                    raise IncompleteDataset(f'{manifest.parent.name}/{method}: unresolved diagnostic lacks reason.')
                if not 0 <= row['elapsed'] <= plan['seconds']:
                    raise IncompleteDataset(f'{manifest.parent.name}/{method}: checkpoint outside budget.')
                if index and row['elapsed'] < rows[index - 1]['elapsed']:
                    raise IncompleteDataset(f'{manifest.parent.name}/{method}: checkpoint times are not ordered.')
            files[str(runpath)] = sha(runpath)
            files[str(diagpath)] = sha(diagpath)
            loaded.append((key, method, rows))
    adoption_path=root/'diagnostic_adoption_manifest.json'
    adopted=[row for key,method,rows in loaded for row in rows if row.get('gap_status')=='independent_dual_bracket_valid']
    if adopted or adoption_path.exists():
        if not adoption_path.is_file():raise IncompleteDataset('Independent adopted gap rows lack the adoption manifest.')
        adoption=json.loads(adoption_path.read_text())
        if adoption.get('state')!='applied' or adoption.get('rows')!=len(adopted) or adoption.get('quality_reviewed_by_explicit_cli') is not True or adoption.get('timed_pairs_unchanged') is not True:raise IncompleteDataset('Invalid independent gap adoption manifest.')
        for path,value in adoption.get('input_sha256_after',{}).items():
            if not Path(path).is_file() or sha(path)!=value:raise IncompleteDataset(f'Adoption input changed: {path}.')
            files[path]=value
        snapshot=Path(adoption['helper_snapshot_file'])
        if sha(snapshot)!=adoption['helper_snapshot_sha256'] or adoption['helper_snapshot_sha256']!=adoption['helper_sha256']:raise IncompleteDataset('Adoption helper snapshot hash mismatch.')
        files[str(snapshot)]=sha(snapshot);files[str(adoption_path)]=sha(adoption_path)
    return plan, loaded, files


def derive_records(loaded):
    records, issues = {}, []
    for key, method, rows in loaded:
        def valid(row, metric):
            gap=bool(row['gap_valid']) and finite(row.get('gap')) and row['gap']>=0
            return gap if metric=='gap' else gap and finite(row.get('violation')) and row['violation']>=0
        last=rows[-1]
        record=dict(n=key[0],m=key[1],seed=key[2],method=method,
                    final_gap=last['gap'] if valid(last,'gap') else None,
                    final_violation=last['violation'] if finite(last.get('violation')) and last['violation']>=0 else None,
                    final_residual=max(last['gap'],last['violation']) if valid(last,'residual') else None)
        for row in rows:
            if valid(row,'residual'):
                joint=max(row['gap'],row['violation'])
                if abs(joint-row['residual'])>2e-15*max(1.,abs(joint)):
                    issues.append(f'{key}/{method}/{row["checkpoint"]}: redundant joint residual differs.')
        for threshold in THRESHOLDS:
            for metric in ('gap','residual'):
                hit=None
                for row in rows:
                    if valid(row,metric) and (row['gap'] if metric=='gap' else max(row['gap'],row['violation']))<=threshold:
                        hit=row['elapsed'];break
                invalid=[row for row in rows if not valid(row,metric)]
                record[f'{metric}_{threshold:g}']=hit
                record[f'{metric}_{threshold:g}_outcome']='observed_success' if hit is not None else 'unresolved_diagnostics' if invalid else 'no_observed_crossing'
                record[f'{metric}_{threshold:g}_invalid_before_hit']=bool(hit is not None and any(row['elapsed']<hit for row in invalid))
            tg,tj=record[f'gap_{threshold:g}'],record[f'residual_{threshold:g}']
            if tj is not None and (tg is None or tg>tj):issues.append(f'{key}/{method}/{threshold:g}: joint crossing lacks earlier/equal gap crossing.')
        records[(*key,method)]=record
    return records, issues


def parse_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def compare_csv(path, expected):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    issues, comparisons = [], 0
    indexed = {}
    for row in rows:
        try:
            key = (int(row['n']), int(row['m']), int(row['seed']), row['method'])
        except (KeyError, ValueError):
            issues.append('per_run_summary.csv: invalid identity fields.')
            continue
        if key in indexed:
            issues.append(f'per_run_summary.csv: duplicate row {key}.')
        indexed[key] = row
    if set(indexed) != set(expected):
        issues.append(f'per_run_summary.csv: record set mismatch; missing={sorted(set(expected)-set(indexed))}, extra={sorted(set(indexed)-set(expected))}.')
    fields = ('final_gap', 'final_violation', 'final_residual') + tuple(
        f'{metric}_{threshold:g}' for threshold in THRESHOLDS for metric in ('gap', 'residual'))
    for key in set(expected) & set(indexed):
        for field in fields:
            actual = parse_number(indexed[key].get(field))
            truth = expected[key][field]
            comparisons += 1
            # Raw CSV floats must round-trip the actual recorded values.
            equal = (str(indexed[key].get(field, '')).strip().lower() == 'nan'
                     and math.isnan(actual)) if truth is None else actual == truth
            if not equal:
                issues.append(f'per_run_summary.csv {key}/{field}: observed {indexed[key].get(field)!r}, expected {truth!r}.')
    categories=0
    for key in set(expected) & set(indexed):
        for threshold in THRESHOLDS:
            for metric in ('gap','residual'):
                for suffix in ('outcome','invalid_before_hit'):
                    field=f'{metric}_{threshold:g}_{suffix}';truth=expected[key][field];actual=indexed[key].get(field)
                    categories+=1
                    if str(actual).lower()!=str(truth).lower():issues.append(f'per_run_summary.csv {key}/{field}: outcome/uncertainty flag differs.')
    return dict(path=str(path), rows=len(rows), scalar_comparisons=comparisons, categorical_comparisons=categories, issues=issues)


def displayed_check(text, truth, kind):
    """Require the declared precision and agreement within half a display unit."""
    if kind == 'time':
        digits = 3 if truth < 1 else 2
        valid_format = re.fullmatch(r'\d+\.\d{' + str(digits) + r'}', text) is not None
        unit = Decimal(1).scaleb(-digits)
    else:
        valid_format = re.fullmatch(r'\d\.\d{2}e[+-]\d{2,}', text) is not None
        exponent = Decimal(str(abs(truth))).adjusted() if truth else 0
        unit = Decimal(1).scaleb(exponent - 2)
    value = parse_number(text)
    if not valid_format or not math.isfinite(value):
        return False, dict(displayed=text, expected=truth, reason='Incorrect numeric format or nonfinite value.')
    if truth == 0:
        return value == 0, dict(displayed=text, expected=truth, half_display_unit=float(unit / 2))
    error = abs(Decimal(text) - Decimal(str(truth)))
    # Four binary ULPs permit only representation/mean-summation roundoff;
    # the substantive acceptance interval is half the requested display unit.
    allowance = unit / 2 + Decimal(str(4 * math.ulp(truth)))
    return error <= allowance, dict(displayed=text, expected=truth,
                                    absolute_error=float(error), half_display_unit=float(unit / 2))


def triple(values):
    return (min(values), statistics.fmean(values), max(values))


def compare_table(path, records, settings, methods, metric, seed_count, multi):
    text = path.read_text()
    parsed, issues, details = {}, [], []
    fixed=sorted({s['n'] if multi else s['m'] for s in settings if (s['m']>1)==multi})
    if len(fixed)==1 and ('Fix $'+('n' if multi else 'm')+'='+str(fixed[0])+'$.') not in text:
        issues.append(f'{path.name}: fixed-dimension note disagrees with the plan.')
    final_triples=0
    for line in text.splitlines():
        if re.match(r'^\s*\d+\s*&', line):
            cells = [s.strip() for s in line.strip().removesuffix('\\\\').strip().split('&')]
            if len(cells) != 7 or cells[1] not in METHOD_NAMES:
                issues.append(f'{path.name}: unrecognized data row {line!r}.')
                continue
            key = (int(cells[0]), METHOD_NAMES[cells[1]])
            if key in parsed:
                issues.append(f'{path.name}: duplicate table row {key}.')
            parsed[key] = cells
    expected_rows = {}
    for setting in settings:
        n, m = setting['n'], setting['m']
        if (m > 1) != multi:
            continue
        for method in methods:
            selected = [r for (rn, rm, seed, meth), r in records.items()
                        if rn == n and rm == m and meth == method]
            key = (m if multi else n, method)
            if key in expected_rows:
                issues.append(f'{path.name}: dimension labels do not uniquely identify configurations.')
            expected_rows[key] = selected
    if set(parsed) != set(expected_rows):
        issues.append(f'{path.name}: table row set differs from planned rows.')
    for key in set(parsed) & set(expected_rows):
        cells, selected = parsed[key], expected_rows[key]
        if len(selected) != seed_count:
            issues.append(f'{path.name}/{key}: expected {seed_count} records, found {len(selected)}.')
            continue
        expected_name = ('MC-PFAOS' if multi else 'PFAOS') if key[1] == 'proposed' else (
            r'OE ($\widehat M=M$)' if key[1] == 'oe_true' else 'Golden Ratio')
        if cells[1] != expected_name:
            issues.append(f'{path.name}/{key}: method name is inconsistent with configuration.')
        for column, threshold in enumerate(THRESHOLDS, start=2):
            match = re.fullmatch(r'\\shortstack\{\$(\d+)/(\d+)(?:,\\ u=(\d+))?\$\\\\(.+)\}', cells[column])
            successes = [r[f'{metric}_{threshold:g}'] for r in selected if r[f'{metric}_{threshold:g}'] is not None]
            unknown=sum(r[f'{metric}_{threshold:g}_outcome']=='unresolved_diagnostics' for r in selected)
            info = dict(dimension=key[0], method=key[1], metric=metric, threshold=threshold,
                        expected_successes=len(successes), expected_total=seed_count, expected_unresolved=unknown,
                        expected_best_mean_worst=triple(successes) if successes else None,
                        displayed_cell=cells[column])
            details.append(info)
            if not match:
                issues.append(f'{path.name}/{key}/{threshold:g}: malformed count/timing cell.')
                continue
            numerator, denominator, unresolved, stats = int(match[1]), int(match[2]), int(match[3] or 0), match[4]
            if unresolved!=unknown:issues.append(f'{path.name}/{key}/{threshold:g}: unresolved count differs from raw diagnostics.')
            if (numerator, denominator) != (len(successes), seed_count):
                issues.append(f'{path.name}/{key}/{threshold:g}: success count differs from raw diagnostics.')
            if not successes:
                if stats != '--':
                    issues.append(f'{path.name}/{key}/{threshold:g}: no-success cell must have -- instead of timing statistics.')
            else:
                values = stats.split('/')
                if len(values) != 3:
                    issues.append(f'{path.name}/{key}/{threshold:g}: conditional timing triple missing.')
                    continue
                rounded = [displayed_check(v, truth, 'time') for v, truth in zip(values, triple(successes))]
                info['rounding_checks'] = [item[1] for item in rounded]
                if not all(item[0] for item in rounded):
                    issues.append(f'{path.name}/{key}/{threshold:g}: conditional timing triple differs beyond displayed rounding.')
        finals = [r['final_violation'] for r in selected]
        displayed = cells[6].split('/')
        final_info = dict(dimension=key[0], method=key[1], metric='final_violation',
                          expected_all_run_best_mean_worst=triple(finals) if all(v is not None for v in finals) else None, displayed_cell=cells[6])
        details.append(final_info)
        if any(v is None for v in finals):
            if cells[6] != r'\textit{unresolved}':issues.append(f'{path.name}/{key}: unavailable final-V must be unresolved.')
            continue
        final_triples+=1
        if len(displayed) != 3:
            issues.append(f'{path.name}/{key}: all-run final-V triple missing.')
        else:
            rounded = [displayed_check(v, truth, 'violation') for v, truth in zip(displayed, triple(finals))]
            final_info['rounding_checks'] = [item[1] for item in rounded]
            if not all(item[0] for item in rounded):
                issues.append(f'{path.name}/{key}: final-V triple differs beyond three-significant-figure rounding.')
    return dict(path=str(path), metric=metric, rows=len(parsed), threshold_cells=4 * len(expected_rows),
                final_violation_triples=final_triples, final_violation_rows=len(expected_rows), issues=issues, cell_checks=details)


def audit(root, report, output, plan_scope=False,single_only=False):
    plan, loaded, inputs = load_complete_runs(root, plan_scope,single_only)
    expected, issues = derive_records(loaded)
    csvpath = report / 'per_run_summary.csv'
    names = [('pfaos_table.tex', 'residual', False), ('mc_table.tex', 'residual', True),
             ('pfaos_table_gap_only.tex', 'gap', False), ('mc_table_gap_only.tex', 'gap', True)]
    if single_only:names=[entry for entry in names if not entry[2]]
    required = [csvpath] + [report / name for name, metric, multi in names]
    if any(not p.exists() for p in required):
        raise IncompleteDataset('The complete generated CSV and four LaTeX tables are not all available.')
    csvcheck = compare_csv(csvpath, expected)
    tablechecks = [compare_table(report / name, expected, plan['settings'], TABLE_METHODS, metric,
                                len(plan['seeds']), multi) for name, metric, multi in names]
    issues += csvcheck['issues']
    for check in tablechecks:
        issues += check['issues']
    inputs.update({str(p): sha(p) for p in required})
    summary = dict(status='passed' if not issues else 'failed',
                   completed_at_utc=datetime.now(timezone.utc).isoformat(),
                   scope='declared_plan_validation' if plan_scope else 'single_constraint_release' if single_only else 'full_requested_grid',
                   plan_snapshot=plan,
                   csv_categorical_comparisons=csvcheck['categorical_comparisons'],
                   results=str(root), report=str(report), auditor_sha256=sha(__file__),
                   expected_instances=len(plan['settings']) * len(plan['seeds']),
                   expected_method_runs=len(expected),
                   csv_scalar_comparisons=csvcheck['scalar_comparisons'],
                   table_threshold_cells=sum(t['threshold_cells'] for t in tablechecks),
                   table_final_V_triples=sum(t['final_violation_triples'] for t in tablechecks),
                   joint_implies_gap_checks=len(expected) * len(THRESHOLDS),
                   issues=issues, csv_check=csvcheck, table_checks=tablechecks,
                   raw_derived_records=list(expected.values()), file_sha256=inputs,
                   policy=dict(thresholds=list(THRESHOLDS), target_decision_tolerance=0.,
                               hits='First valid saved checkpoint meeting exact recorded conditions; no-hit trajectories with invalid diagnostics remain unresolved; gap and V tested at the same checkpoint for the joint criterion.',
                               timing_statistics='Best/mean/worst over successful runs only; no failed crossing is assigned the cap.',
                               final_V_statistics='Best/mean/worst over every planned run, or unresolved when any terminal V is unavailable.',
                               CSV='Raw float values must round-trip exactly; absent hits must be nonfinite.',
                               table_rounding='Times below one second: three decimals; other times: two decimals. Final V: three significant figures. Acceptance is half a display unit plus at most four binary ULPs for representation/summation roundoff.',
                               completeness='Every planned instance and method must be complete before any audit output is written.'))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'table_statistics_audit.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    lines = ['# Independent table-statistics audit', '', f'**Status: {summary["status"]}.**', '',
             f'Checked the complete plan: {summary["expected_instances"]} instances and {summary["expected_method_runs"]} method runs.',
             f'Compared {summary["csv_scalar_comparisons"]} raw CSV values, {summary["table_threshold_cells"]} LaTeX count/timing cells, and {summary["table_final_V_triples"]} all-run final-violation triples.', '',
             'The checker derives first observed gap and joint crossings directly from per-case diagnostic JSONs without importing the report implementation. Joint crossings require gap and violation at the same saved checkpoint. Every joint hit has an earlier or equal gap hit.', '',
             'Counts are exact; timing triples condition on observed successes; final-violation triples include all planned seeds. CSV floats must round-trip exactly. Displayed times are checked within half the relevant decimal unit (three decimals below one second, two otherwise); final violations use three significant figures. Four binary ULPs allow only representation/summation roundoff. No tolerance is added to any threshold decision.', '', '## Findings', '']
    lines += [f'- {s}' for s in issues] if issues else ['No discrepancies found.']
    lines += ['', 'This verifies numerical reporting and aggregation of the recorded diagnostics. It does not independently establish rigorous gap bounds or fill missing observations.', '',
              'Detailed per-cell expected statistics, rounding checks, raw derived crossings, and input SHA256 values are in `table_statistics_audit.json`.']
    (output / 'table_statistics_audit.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: summary[k] for k in ('status', 'expected_instances', 'expected_method_runs', 'csv_scalar_comparisons', 'table_threshold_cells', 'table_final_V_triples', 'issues')}), flush=True)
    return 0 if not issues else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--plan-scope',action='store_true',help='Validate a declared legacy/fixture plan, not the full 110-case/550-run contract')
    parser.add_argument('--single-only',action='store_true',help='Require the explicit 60-instance/300-run single-constraint release')
    args = parser.parse_args()
    root = args.results.resolve()
    report = args.report.resolve() if args.report else root / 'report'
    output = args.out.resolve() if args.out else report
    try:
        return audit(root, report, output, args.plan_scope,args.single_only)
    except IncompleteDataset as exc:
        print(f'REFUSED INCOMPLETE DATASET: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
