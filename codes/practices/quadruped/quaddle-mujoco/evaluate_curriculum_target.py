"""Screen every saved checkpoint on the unchanged original target terrain."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import argparse,json,time
from pathlib import Path
from scenario_matrix import run_case


def job(directory):
    return dict(checkpoint=str(directory),result=run_case('rough','walk',policy_path=directory,
        output=directory/'target_original'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arms',nargs='+',default=['rough_g995','rough_g999']);parser.add_argument('--report',default='target_screen.json');args=parser.parse_args()
    root=Path('outputs/curriculum');completed=set();records=[]
    with ProcessPoolExecutor(2,mp_context=mp.get_context('spawn')) as pool:
        while True:
            pending=sorted(p.parent for arm in args.arms for p in (root/arm).glob('level*/metadata.json') if str(p.parent) not in completed)
            for record in pool.map(job,pending):
                records.append(record);completed.add(record['checkpoint'])
                (root/args.report).write_text(json.dumps(records,indent=2)+'\n')
                r=record['result'];print(record['checkpoint'],r['passed'],r['forward_m'],[k for k,v in r['checks'].items() if not v],flush=True)
            statuses=[]
            for arm in args.arms:
                p=root/arm/'progress.json'
                try: statuses.append(json.loads(p.read_text())['status'])
                except (FileNotFoundError,json.JSONDecodeError):statuses.append('training')
            if all(s in ('finished','pilot_budget_exhausted_without_gate_pass') for s in statuses):
                if not pending:break
            time.sleep(5)
