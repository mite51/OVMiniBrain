"""Score a .motion animation (optionally at several playback rates) without editing it."""
import argparse,json,subprocess
from pathlib import Path
import torch,yaml,protomotions
from protomotions.components.motion_lib import MotionLib,MotionLibConfig
from evaluate_library import cache

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',required=True,type=Path)
    p.add_argument('--model',required=True,type=Path,help='Exported unified_pipeline.onnx')
    p.add_argument('--output',required=True,type=Path,help='Directory for caches, JSON assessments and logs')
    p.add_argument('--speeds',default='1',help='Comma-separated rates; e.g. 1,.85,.7,.55')
    p.add_argument('--binary',type=Path,default=Path(__file__).resolve().parents[1]/'build/ovminibrain')
    p.add_argument('--mjcf',type=Path,default=Path(protomotions.__file__).resolve().parent/'data/assets/mjcf/smpl_humanoid.xml')
    a=p.parse_args();torch.set_num_threads(1)
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Choose an empty output directory to preserve existing assessments')
    a.output.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load(a.model.with_suffix('.yaml').read_text())
    lib=MotionLib(MotionLibConfig(motion_file=str(a.motion),get_motion_state_use_blend=True),device='cpu')
    if lib.num_motions()!=1:raise ValueError('Use evaluate_library.py for a library containing multiple clips')
    rows=[]
    for speed in map(float,a.speeds.split(',')):
        pmc=a.output/f'speed_{speed:g}.pmc';report=pmc.with_suffix('.json')
        cache(lib,0,speed,cfg['timing']['control_dt'],max(cfg['motion']['future_step_indices']),pmc)
        with pmc.with_suffix('.log').open('w') as log:
            subprocess.run([str(a.binary),'--onnx',str(a.model),'--motion-cache',str(pmc),'--mjcf',str(a.mjcf),'--eval-output',str(report),'--tracking-log-every','0'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(120,4*float(lib.motion_lengths[0])/speed))
        r=json.loads(report.read_text());r.update(source_motion=str(a.motion.resolve()),speed=speed);report.write_text(json.dumps(r,indent=2)+'\n');rows.append(r)
        print(f"{speed:g}x: score {r['score']:.1f}/100, {'PASS' if r['passed'] else 'FAIL'}, mean error {r['mean_position_error_m']:.3f} m; {r['assessment']}",flush=True)
    (a.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('Scores are specific to this model and standalone PhysX. Validate in the training simulator before deleting data.')
if __name__=='__main__':main()
