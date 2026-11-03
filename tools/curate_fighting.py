"""Stage, apply, or restore reviewed animation repairs with verified originals.

Decisions are keyed by canonical motion ID and contain action=keep/retime/remove,
speed, and assessment evidence. This tool never decides that failure alone makes
an animation physically impossible. Only Fighting aliases can be changed.
"""
import argparse,copy,hashlib,json,os,pickle,shutil
from pathlib import Path
import torch,yaml
from protomotions.training.weekend.data import FIELDS
from protomotions.simulator.base_simulator.simulator_state import StateConversion

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def stage(a):
    torch.set_num_threads(2)
    root=a.source.resolve()
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Stage into an empty directory; existing operations must remain immutable')
    a.output.mkdir(parents=True,exist_ok=True)
    backups={r['source']:r for r in json.loads(a.backup_manifest.read_text())}
    entries={e['id']:e for e in json.loads(a.manifest.read_text())['motions']}
    decisions={r['id']:r for r in json.loads(a.decisions.read_text())['decisions'] if r['action']!='keep'}
    operations=[];aliases={};raw_changes={}
    def record(dest,staged=None):
        dest=Path(dest).resolve();dest.relative_to(root)
        if dest.exists():
            b=backups[str(dest)]
            assert sha(dest)==b['sha256']==sha(b['backup']),f'Backup mismatch: {dest}'
        else:b=None
        operations.append(dict(destination=str(dest),staged=str(staged.resolve()) if staged else None,
            before_sha256=b['sha256'] if b else None,after_sha256=sha(staged) if staged else None,backup=b['backup'] if b else None))
    for mid,d in decisions.items():
        e=entries[mid];assert e['group']=='fighting'
        assert d['action'] in ('retime','remove')
        if d['action']=='retime':assert 0<float(d['speed'])<1
        for alias in e['aliases']:
            if '#' in alias:
                filename,index=alias.rsplit('#',1);aliases.setdefault(filename,{})[int(index)]=d
            else:raw_changes[alias]=d
    if any(d['action']=='remove' for d in raw_changes.values()):
        # Also inspect unchanged packages for references to removed raw files.
        for name in ('amass_smpl_train_fighting.pt','amass_smpl_train_just_fighting.pt'):
            if (root/name).exists():aliases.setdefault(str(root/name),{})
    # Existing raw clips retain all their auxiliary metadata and pose samples.
    for filename,d in raw_changes.items():
        source=Path(filename);source.relative_to(root/'Fighting')
        if d['action']=='remove':record(source);continue
        raw=torch.load(source,map_location='cpu',weights_only=False)
        raw['fps']=1/float(torch.tensor(entries[d['id']]['dt']/d['speed'],dtype=torch.float32))
        for k in ('rigid_body_vel','rigid_body_ang_vel','dof_vel'):raw[k]=raw[k]*d['speed']
        dest=a.output/'files'/source.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);torch.save(raw,dest);record(source,dest)
    for filename,changes in aliases.items():
        source=Path(filename);source.relative_to(root)
        assert source.name in ('amass_smpl_train_fighting.pt','amass_smpl_train_just_fighting.pt'),source
        data=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
        ypath=source.with_suffix('.yaml');yb=yaml.safe_load(ypath.read_text());ym=yb['motions']
        count=len(data['motion_files']);assert len(ym)==count
        keep=[i for i in range(count) if changes.get(i,{}).get('action')!='remove']
        idx=torch.tensor(keep);frame_idx=torch.cat([torch.arange(int(data['length_starts'][i]),int(data['length_starts'][i]+data['motion_num_frames'][i])) for i in keep])
        out={}
        frame_fields=set(FIELDS)|{'lrs'}
        for k,v in data.items():
            if k in frame_fields and v is not None:out[k]=v.index_select(0,frame_idx)
            elif torch.is_tensor(v) and v.shape and v.shape[0]==count:out[k]=v.index_select(0,idx)
            elif k=='motion_files':out[k]=tuple(v[i] for i in keep)
            else:out[k]=v
        out['length_starts']=out['motion_num_frames'].cumsum(0)-out['motion_num_frames']
        new_yaml=[];names=list(out['motion_files'])
        for j,i in enumerate(keep):
            item=copy.deepcopy(ym[i]);item['idx']=j;d=changes.get(i)
            if d:
                speed=d['speed'];s=int(out['length_starts'][j]);n=int(out['motion_num_frames'][j]);sl=slice(s,s+n)
                out['motion_dt'][j]/=speed;out['motion_lengths'][j]=(n-1)*out['motion_dt'][j]
                for k in ('gvs','gavs','dvs'):out[k][sl]*=speed
                # A packaged variant need not match the raw clip sharing its name.
                # Give its repaired version an explicit source so YAML rebuilding
                # cannot silently select a different export.
                basename=Path(entries[d['id']]['name'].replace('\\','/')).name
                relative=Path('Fighting/RepairedVersions')/f"{d['id']}_{basename}"
                dest=a.output/'files'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                raw={field:out[k][sl].clone() for k,field in FIELDS.items()}
                raw.update(fps=1/float(out['motion_dt'][j]),state_conversion=StateConversion.COMMON)
                if out.get('lrs') is not None:raw['local_rigid_body_rot']=out['lrs'][sl].clone()
                if not dest.exists():torch.save(raw,dest);record(root/relative,dest)
                names[j]=str(relative)
                item.update(file=str(relative),fps=raw['fps'],sub_motions=[{'timings':{'start':0.,'end':float(out['motion_lengths'][j])}}])
            elif raw_changes.get(str(root/item['file']),{}).get('action')=='remove':
                # A kept package can contain a working export of the same name
                # as a rejected loose file. Preserve that distinct source.
                origin=f"{source}#{i}"
                retained=next(e for e in entries.values() if origin in e['aliases'])
                relative=Path('Fighting/RetainedVersions')/f"{retained['id']}_{Path(item['file']).name}"
                dest=a.output/'files'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                start=int(out['length_starts'][j]);n=int(out['motion_num_frames'][j]);sl=slice(start,start+n)
                raw={field:out[k][sl].clone() for k,field in FIELDS.items()}
                raw.update(fps=1/float(out['motion_dt'][j]),state_conversion=StateConversion.COMMON)
                if out.get('lrs') is not None:raw['local_rigid_body_rot']=out['lrs'][sl].clone()
                if not dest.exists():torch.save(raw,dest);record(root/relative,dest)
                item.update(file=str(relative),fps=raw['fps'],sub_motions=[{'timings':{'start':0.,'end':float(out['motion_lengths'][j])}}])
            new_yaml.append(item)
        out['motion_files']=tuple(names)
        # Non-Fighting clips and all kept pose samples are exactly unchanged.
        for j,i in enumerate(keep):
            old=slice(int(data['length_starts'][i]),int(data['length_starts'][i]+data['motion_num_frames'][i]));new=slice(int(out['length_starts'][j]),int(out['length_starts'][j]+out['motion_num_frames'][j]))
            for k in frame_fields:
                if data.get(k) is None:continue
                if i in changes and k in ('gvs','gavs','dvs'):continue
                assert torch.equal(data[k][old],out[k][new]),(source,i,k)
        dest=a.output/'files'/source.name;torch.save(out,dest);record(source,dest)
        dest=dest.with_suffix('.yaml');yb['motions']=new_yaml;dest.write_text(yaml.safe_dump(yb,sort_keys=False));record(ypath,dest)
        print('Staged',source.name,count,'->',len(keep),flush=True)
        del out,data
    # Remove rejected actions from the extracted FBX intermediates, too. Tag
    # repairs so the retargeter can apply their time scale on regeneration.
    for source in sorted((root/'Fighting').rglob('*.pkl')):
        with source.open('rb') as f:data=pickle.load(f)
        changed=False
        for name in list(data.get('actions',{})):
            raw=source.parent.parent/(name+'.motion');d=raw_changes.get(str(raw))
            if not d:continue
            if d['action']=='remove':del data['actions'][name]
            else:data['actions'][name]['protomotions_retime_speed']=d['speed']
            changed=True
        if changed:
            if not data['actions']:record(source);continue
            dest=a.output/'files'/source.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('wb') as f:pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
            record(source,dest)
    assert len({o['destination'] for o in operations})==len(operations)
    write_json(a.output/'operations.json',dict(source_root=str(root),decisions_sha256=sha(a.decisions),operations=operations))
    print('Ready:',len(operations),'file operations. Original backups verified.')
def apply(a,restore=False):
    plan=json.loads(a.operations.read_text());root=Path(plan['source_root']).resolve()
    # Verify every input before changing anything. Reapplying a completed operation is safe.
    for op in plan['operations']:
        path=Path(op['destination']);path.resolve().relative_to(root)
        current=sha(path) if path.exists() else None
        assert current in (op['before_sha256'],op['after_sha256']),f'Source changed since staging: {path}'
        if op['backup']:assert sha(op['backup'])==op['before_sha256']
        if op['staged']:assert sha(op['staged'])==op['after_sha256']
    completed=[]
    for op in plan['operations']:
        path=Path(op['destination']);src=op['backup'] if restore else op['staged'];expected=op['before_sha256'] if restore else op['after_sha256']
        if path.exists() and sha(path)==expected:completed.append(str(path));continue
        if src:
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.curation.tmp');shutil.copy2(src,tmp);assert sha(tmp)==expected;tmp.replace(path)
        elif path.exists():path.unlink()
        completed.append(str(path));write_json(a.operations.with_name('restore_receipt.json' if restore else 'apply_receipt.json'),dict(completed=completed))
    print('Restored' if restore else 'Applied',len(completed),'verified operations')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('stage')
    for name in ('source','decisions','manifest','backup-manifest','output'):q.add_argument('--'+name,type=Path,required=True)
    for mode in ('apply','restore'):sub.add_parser(mode).add_argument('--operations',type=Path,required=True)
    a=p.parse_args();stage(a) if a.mode=='stage' else apply(a,restore=a.mode=='restore')
