"""One-time full experiment and document pipeline; no fabricated result cells."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results';NOTES=ROOT/'notes';NOTES.mkdir(exist_ok=True)
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',BLIS_NUM_THREADS='1',MPLCONFIGDIR=str(ROOT/'validation/matplotlib_cache'))
ENV['PATH']='/Library/TeX/texbin:'+ENV.get('PATH','')

def update(state,**extra):
    data=dict(state=state,updated_utc=datetime.now(timezone.utc).isoformat(),pipeline_pid=os.getpid(),**extra)
    p=ROOT/'pipeline_status.json';t=p.with_suffix('.json.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
    print(json.dumps(data),flush=True)

def run(args,cwd,logfile):
    with (NOTES/logfile).open('a',buffering=1) as stream:
        subprocess.run(args,cwd=cwd,env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)

def main():
    update('running_full_grid',planned_instances=120,planned_method_runs=600,settings='single n=10,50,100,200,500,1000,m=1; multi n=1000,m=10,50,100,200,500,1000',methods=['proposed','oe_true','oe_low','oe_high','golden_ratio'])
    cmd=[sys.executable,str(ROOT/'code/batch.py'),'--seconds','1000','--diagnostic-seconds','300','--seeds','10','--workers','7','--memory-budget-gib','8','--min-free-disk-gib','12','--heavy-companion-m','0','--out',str(RESULTS)]
    if (RESULTS/'plan.json').exists():cmd.append('--resume')
    run(cmd,ROOT/'code','full_grid_batch.log')
    report=RESULTS/'verified/report';validation=json.loads((report/'validation.json').read_text())
    update('structural_audit',measured_report_status=validation['status'])
    run([sys.executable,str(ROOT/'reporting/audit_saved_records.py'),'--results',str(RESULTS/'verified'),'--temp-dir',str(ROOT/'validation')],ROOT,'full_grid_structural_audit.log')
    update('assembling_measured_article',measured_report_status=validation['status'])
    destination=ROOT/'manuscript_full_grid'
    command=[sys.executable,str(ROOT/'reporting/assemble_manuscript.py'),'--template',str(ROOT.parent/'CVI_Completed_Experiments/manuscript_revised'),'--results',str(RESULTS/'verified'),'--destination',str(destination)]
    if validation['status']!='complete':command.append('--allow-unresolved')
    run(command,ROOT,'full_grid_article_assembly.log')
    update('compiling_measured_article',measured_report_status=validation['status'])
    run(['/bin/sh','build.sh'],destination,'full_grid_article_build.log')
    for i in (1,2):run(['/Library/TeX/texbin/pdflatex','-interaction=nonstopmode','-halt-on-error','numerical_review.tex'],destination,f'full_grid_numerical_build_{i}.log')
    (NOTES/'assembled_pdf_sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (destination/'main.pdf',destination/'numerical_review.pdf')},indent=2)+'\n')
    update('ready_for_review' if validation['status']=='complete' else 'ready_for_review_with_exceptions',instances=validation['instances'],method_runs=validation['method_runs'],report_status=validation['status'],article=str(destination/'main.pdf'),numerical_report=str(destination/'numerical_review.pdf'),note='Rendered PDFs require final visual/statistical review. Unresolved diagnostics or method failures remain explicit; no old values or estimates are inserted.')

if __name__=='__main__':
    try:main()
    except Exception as error:
        update('needs_attention',error_type=type(error).__name__,message=str(error));traceback.print_exc();raise
