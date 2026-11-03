"""Run with: python -m unittest discover -s tests -p 'test_evaluation.py'."""
import json,subprocess,tempfile,unittest
from pathlib import Path

class EvaluationTests(unittest.TestCase):
    def test_metrics_failure_and_incomplete_clip(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);binary=d/'test'
            subprocess.run(['c++','-std=c++17','-I'+str(root/'src'),str(root/'tests/test_evaluation.cpp'),'-o',str(binary)],check=True)
            subprocess.run([str(binary),str(d/'reports')],check=True)
            reports={k:json.loads((d/'reports'/f'{k}.json').read_text()) for k in ('pass','fail','incomplete')}
            a,b,c=reports['pass'],reports['fail'],reports['incomplete']
            self.assertTrue(a['passed']);self.assertAlmostEqual(a['score'],100/1.2,places=5)
            self.assertAlmostEqual(a['mean_position_error_m'],.1,places=6)
            self.assertEqual(a['mean_rotation_error_rad'],0)
            self.assertEqual(a['model'],'quote"\\\nmodel')
            self.assertEqual(len(a['limbs']),5)
            self.assertFalse(b['passed']);self.assertEqual(b['first_failure_s'],1)
            self.assertFalse(c['complete']);self.assertFalse(c['passed'])
            self.assertEqual(a['reference_kinematics']['peak_body_speed_m_s'],0)
if __name__=='__main__':unittest.main()
