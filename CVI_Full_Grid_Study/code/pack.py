"""Package only code, validation, and actual measured artifacts."""
import argparse
from pathlib import Path
import zipfile

def package(output):
    root=Path(__file__).resolve().parent
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for p in root.iterdir():
            if p.is_file() and (p.suffix in ['.py','.md','.tex','.txt'] or p.name in ['checks.json','checks.log','robustness_checks.log']):
                z.write(p,root.name+'/'+p.name)
        for p in (root/'tests').rglob('*.py'):
            z.write(p,root.name+'/'+str(p.relative_to(root)))
        measured=root/'measured_small'
        if measured.exists():
            for p in [measured/'plan.json',measured/'progress.json']:
                if p.exists():z.write(p,root.name+'/'+str(p.relative_to(root)))
            for p in (measured/'verified').rglob('*'):
                if p.is_file() and p.suffix in ['.json','.npz','.csv','.md','.tex','.txt','.log','.pdf','.png']:
                    z.write(p,root.name+'/'+str(p.relative_to(root)))
    return output

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True)
    print(package(a.parse_args().out))
