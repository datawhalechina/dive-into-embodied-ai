"""Show original-policy versus selected trained candidate on unchanged target geometry."""
import argparse,json
from pathlib import Path
from scenario_matrix import run_case


def score(item):
    rs=[x['result'] for x in item['evaluations'] if x['scale']==1.]
    safety=['upright','no_body_contact','joint_speed','heading']
    return (sum(r['passed'] for r in rs),sum(all(r['checks'][k] for k in safety) for r in rs),
        sum(sum(r['checks'].values()) for r in rs),sum(r['forward_m'] for r in rs)/len(rs))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['prior','trained','all'],default='all');a=p.parse_args()
    root=Path('outputs/practice_20261110')
    for scene in ['stairs_up','rough']:
        variants=[]
        if a.part in ('prior','all'):variants.append(('prior',root/scene/'prior_policy'))
        if a.part in ('trained','all'):
            report=json.loads((root/scene/'summary.json').read_text())
            if report['status']!='finished':raise ValueError('Wait for the bounded training run to finish')
            selected=max(report['history'],key=score)
            (root/scene/'selection.json').write_text(json.dumps(dict(checkpoint=selected['checkpoint'],score=score(selected),
                note='Best of this run trained candidates on three original-terrain offsets; does not establish task completion'),indent=2)+'\n')
            variants.append(('trained',Path(selected['checkpoint'])))
        for label,checkpoint in variants:
            r=run_case(scene,'walk',policy_path=checkpoint,output=root/'videos'/label,video=True)
            print(scene,label,r['passed'],r['forward_m'],flush=True)
