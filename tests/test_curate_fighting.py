"""Exercise deletion, a same-name retained export, and verified restoration."""
import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch,yaml

class CurationTests(unittest.TestCase):
    def test_retained_variant_and_restore(self):
        tool=Path(__file__).resolve().parents[1]/'tools/curate_fighting.py'
        spec=importlib.util.spec_from_file_location('curate',tool);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as temp:
            r=Path(temp);src=r/'source';(src/'Fighting').mkdir(parents=True)
            raw=src/'Fighting/clip.motion';raw.write_bytes(b'original raw animation')
            package=src/'amass_smpl_train_fighting.pt';yp=package.with_suffix('.yaml')
            fields={'gts':torch.ones(3,24,3),'grs':torch.zeros(3,24,4),'gvs':torch.zeros(3,24,3),'gavs':torch.zeros(3,24,3),'dps':torch.zeros(3,69),'dvs':torch.zeros(3,69),'contacts':torch.zeros(3,24,dtype=torch.bool)}
            fields['grs'][:,:,3]=1
            fields.update(length_starts=torch.tensor([0]),motion_num_frames=torch.tensor([3]),motion_dt=torch.tensor([.1]),motion_lengths=torch.tensor([.2]),motion_weights=torch.ones(1),motion_files=('Fighting/clip.motion',),lrs=None)
            torch.save(fields,package);yp.write_text(yaml.safe_dump({'motions':[{'file':'Fighting/clip.motion','fps':10,'idx':0}]}))
            backup=[]
            for p in (raw,package,yp):
                dest=r/'backup'/p.relative_to(src);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
                backup.append(dict(source=str(p),backup=str(dest),sha256=m.sha(p)))
            bm=r/'backup.json';bm.write_text(json.dumps(backup))
            manifest=r/'manifest.json';manifest.write_text(json.dumps({'motions':[{'id':0,'group':'fighting','name':'Fighting/clip.motion','aliases':[str(raw)]},{'id':1,'group':'fighting','name':'Fighting/clip.motion','aliases':[str(package)+'#0']}]}))
            dec=r/'decisions.json';dec.write_text(json.dumps({'decisions':[{'id':0,'action':'remove','speed':None}]}))
            output=r/'stage';m.stage(SimpleNamespace(source=src,output=output,backup_manifest=bm,manifest=manifest,decisions=dec))
            args=SimpleNamespace(operations=output/'operations.json');m.apply(args)
            self.assertFalse(raw.exists());retained=src/'Fighting/RetainedVersions/1_clip.motion';self.assertTrue(retained.exists())
            self.assertEqual(yaml.safe_load(yp.read_text())['motions'][0]['file'],str(retained.relative_to(src)))
            m.apply(args,restore=True);self.assertFalse(retained.exists())
            for b in backup:self.assertEqual(m.sha(b['source']),b['sha256'])
if __name__=='__main__':unittest.main()
