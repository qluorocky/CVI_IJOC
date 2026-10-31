#!/usr/bin/env python3
"""Write a clearly nonfinal full-grid progress inventory, never numerical tables."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
from collections import Counter
import json


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',required=True,type=Path);ap.add_argument('--out',type=Path);a=ap.parse_args()
    root=a.results.resolve();plan=json.loads((root/'plan.json').read_text());cases=root/'working' if (root/'working').exists() else root
    expected={(s['n'],s['m'],seed) for s in plan['settings'] for seed in plan['seeds']};found={};errors=[]
    for path in cases.glob('*/manifest.json'):
        try:
            meta=json.loads(path.read_text());key=(meta['n'],meta['m'],meta['seed'])
            if key in found:errors.append(f'Duplicate case {key}')
            found[key]=(path,meta)
        except (ValueError,KeyError,OSError) as exc:errors.append(f'{path}: {type(exc).__name__}: {exc}')
    records=[]
    for setting in plan['settings']:
        n,m=setting['n'],setting['m'];selected=[found[(n,m,seed)][1] for seed in plan['seeds'] if (n,m,seed) in found]
        states=Counter(meta.get('state','unknown') for meta in selected);method_stages=Counter(v.get('stage','unknown') for meta in selected for method,v in meta.get('methods',{}).items() if method in plan['methods'])
        records.append(dict(study=setting.get('study','single' if m==1 else 'multi'),n=n,m=m,planned_instances=len(plan['seeds']),started_instances=len(selected),complete_instances=states['complete'],timed_method_stages=sum(method_stages[x] for x in ('timed_complete','complete')),fully_recorded_method_stages=method_stages['complete'],planned_method_stages=len(plan['seeds'])*len(plan['methods']),states=dict(states)))
    completed=sum(r['complete_instances'] for r in records);all_complete=completed==len(expected)
    result=dict(status='record_inventory_complete_not_numerically_audited' if all_complete else 'incomplete_progress_only',updated_utc=datetime.now(timezone.utc).isoformat(),planned_instances=len(expected),complete_instances=completed,planned_method_runs=len(expected)*len(plan['methods']),timed_method_stages=sum(r['timed_method_stages'] for r in records),fully_recorded_method_stages=sum(r['fully_recorded_method_stages'] for r in records),unexpected_cases=sorted(set(found)-expected),read_errors=errors,settings=records,limitation='This inventory reports progress only. It contains no threshold estimates, final table values, or experimental performance claims.')
    out=(a.out or root/'progress_report').resolve();out.mkdir(parents=True,exist_ok=True);(out/'progress_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Full-grid progress inventory','',f'**Status: {result["status"]}. This is not a numerical-results report.**','',f'{completed}/{len(expected)} cases have complete record inventories; {result["timed_method_stages"]}/{result["planned_method_runs"]} method stages have finished timing.','', '| Study | n | m | Started/planned | Complete/planned | Timed method stages |','|---|---:|---:|---:|---:|---:|']
    for r in records:lines.append(f'| {r["study"]} | {r["n"]} | {r["m"]} | {r["started_instances"]}/{r["planned_instances"]} | {r["complete_instances"]}/{r["planned_instances"]} | {r["timed_method_stages"]}/{r["planned_method_stages"]} |')
    lines+=['','Completion of record inventory does not establish valid numerical gaps, threshold attainment, or a passed independent audit. The final reporter refuses missing records and distinguishes invalid diagnostics from observed nonattainment.']
    if errors:lines+=['','Read errors:']+[f'- {error}' for error in errors]
    (out/'progress_inventory.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='settings'}))


if __name__=='__main__':main()
