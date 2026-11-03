"""Score packaged ProtoMotions clips with the standalone C++ evaluator.

Retiming samples the original trajectory at t*speed and scales every velocity
by speed. It never edits source data. Each cache includes the entire clip.
"""
import argparse, concurrent.futures, hashlib, json, math, os, struct, subprocess
from pathlib import Path
import numpy as np
import torch,yaml
from protomotions.components.motion_lib import MotionLib

def cache(lib, mid, speed, dt, future, dest):
    if not 0 < speed <= 1: raise ValueError('speed must be in (0,1]')
    length=float(lib.motion_lengths[mid])/speed
    k=math.ceil(length/dt)+future+2
    t=(torch.arange(k,dtype=torch.float64)*dt*speed).clamp(0,float(lib.motion_lengths[mid])).float()
    with torch.no_grad():state=lib.get_motion_state(torch.full((k,),mid,dtype=torch.long),t)
    arrays=[state.rigid_body_pos,state.rigid_body_rot,state.rigid_body_vel*speed,
            state.rigid_body_ang_vel*speed,state.dof_pos,state.dof_vel*speed]
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('wb') as f:
        f.write(b'PMC1'+struct.pack('<iiifff',k,24,69,speed/float(lib.motion_dt[mid]),dt,length))
        for a in arrays:
            if not torch.isfinite(a).all():raise ValueError('Non-finite reference')
            f.write(a.contiguous().numpy().astype('<f4').tobytes())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,type=Path);p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--model',required=True,type=Path);p.add_argument('--binary',required=True,type=Path)
    p.add_argument('--mjcf',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--ids',help='Comma separated motion IDs; default Fighting and Kimodo')
    p.add_argument('--speeds',default='1');p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();torch.set_num_threads(1)
    lib=MotionLib.empty('cpu')
    for key,value in torch.load(a.dataset,map_location='cpu',weights_only=False,mmap=True).items():setattr(lib,key,value)
    lib.get_motion_state_use_blend=True
    entries=json.loads(a.manifest.read_text())['motions']
    ids=[int(i) for i in a.ids.split(',')] if a.ids else [e['id'] for e in entries if e['group'] in ('fighting','kimodo')]
    cfg=yaml.safe_load(a.model.with_suffix('.yaml').read_text());dt=cfg['timing']['control_dt'];future=max(cfg['motion']['future_step_indices'])
    a.output.mkdir(parents=True,exist_ok=True)
    def digest(path):
        with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    provenance={str(path.resolve()):digest(path) for path in (a.dataset,a.manifest,a.model,a.model.with_suffix('.yaml'),a.binary,a.mjcf)}
    metadata=a.output/'provenance.json'
    if metadata.exists() and json.loads(metadata.read_text())!=provenance:
        raise ValueError('Inputs changed; choose a new output directory to avoid reusing stale scores')
    metadata.write_text(json.dumps(provenance,indent=2)+'\n')
    def run(mid,speed,pmc,report):
        if report.exists():
            try:
                cached=json.loads(report.read_text())
                if cached.get('complete') and cached.get('motion_id')==mid and cached.get('speed')==speed:
                    return cached
            except (OSError,ValueError):
                pass  # Interrupted/incomplete metadata is retried, never scored as failure.

        cmd=[str(a.binary),'--onnx',str(a.model),'--motion-cache',str(pmc),'--mjcf',str(a.mjcf),'--eval-output',str(report),'--tracking-log-every','0']
        with report.with_suffix('.log').open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=max(120,4*float(lib.motion_lengths[mid])/speed))
        if result.returncode:raise RuntimeError(f'Evaluation exit {result.returncode}: {report.with_suffix(".log")}')
        r=json.loads(report.read_text());r.update(motion_id=mid,speed=speed,name=entries[mid]['name'])
        if not r['complete']:raise RuntimeError(f'Incomplete: {report}')
        tmp=report.with_suffix('.json.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(report);return r
    jobs=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for mid in ids:
            for speed in map(float,a.speeds.split(',')):
                stem=f'{mid}_s{speed:g}';pmc=a.output/'cache'/f'{stem}.pmc';report=a.output/f'{stem}.json'
                if not pmc.exists():cache(lib,mid,speed,dt,future,pmc)
                jobs.append(pool.submit(run,mid,speed,pmc,report))
        rows=[];errors=[]
        for i,f in enumerate(concurrent.futures.as_completed(jobs),1):
            try:r=f.result()
            except Exception as exc:
                errors.append(str(exc));print(str(exc),flush=True);continue
            rows.append(r)
            print(f'{i}/{len(jobs)} ID={r["motion_id"]} speed={r["speed"]} pass={r["passed"]} score={r["score"]:.1f} error={r["mean_position_error_m"]:.3f}',flush=True)
    if errors:raise RuntimeError('Some evaluations failed; rerun to retry missing reports: '+str(errors))
    (a.output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda r:(r['motion_id'],r['speed'])),indent=2)+'\n')
if __name__=='__main__':main()
