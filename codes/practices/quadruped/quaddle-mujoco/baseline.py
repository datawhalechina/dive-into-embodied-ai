"""Replay the frozen per-scenario Quaddle simulation baseline (not one universal policy)."""
import argparse
import hashlib
import json
from pathlib import Path
from scenario_matrix import run_case
from curriculum_rl import curriculum_scene
from command_interaction import replay
from graded_rough import graded_rough

ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT/'reference/baseline-v0.1.json'

def verify(manifest):
    failures=[]
    for name,expected in manifest['sha256'].items():
        path=ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            failures.append(name)
    if failures:raise ValueError('Baseline content mismatch (or Git LFS pointers): '+', '.join(failures))

def run_entry(label,spec,output,video=False):
    config=json.loads((ROOT/spec['config']).read_text()) if spec.get('config') else None
    if label=='interaction':return replay(output/label,video=video,config=config)
    kwargs={}
    if spec.get('geometry')=='stairs_2mm':kwargs['scene_definition']=curriculum_scene('stairs_up',2)
    if spec.get('geometry')=='graded_rough':kwargs['scene_definition']=graded_rough(seed=spec['seed'])
    if spec.get('policy'):kwargs['policy_path']=ROOT/spec['policy']
    return run_case(spec['scene'],'walk',seconds=spec.get('seconds',20.),output=output/label,video=video,config_override=config,**kwargs)

def main():
    manifest=json.loads(MANIFEST.read_text())
    p=argparse.ArgumentParser(description=__doc__)
    available={**manifest['cases'],**manifest.get('experimental_cases',{})}
    p.add_argument('--case',choices=['all',*available],default='all')
    p.add_argument('--video',action='store_true')
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/baseline-v0.1')
    args=p.parse_args();verify(manifest)
    if args.verify_only:
        print('Baseline source/model/config/policy hashes PASS');return
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    selected=manifest['cases'] if args.case=='all' else {args.case:available[args.case]}
    for label,spec in selected.items():
        r=run_entry(label,spec,args.output,args.video)
        records.append(dict(label=label,result=r))
        print(label,'PASS' if r['passed'] else 'FAIL',flush=True)
    (args.output/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
    if not all(r['result']['passed'] for r in records):raise SystemExit(1)

if __name__=='__main__':main()
