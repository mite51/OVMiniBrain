"""Regression checks for the symlink ZIP failure and safe archive publication."""
import importlib.util
from pathlib import Path
import os
import stat
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('package', Path(__file__).resolve().parents[1] / 'tools/package.py')
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tree = self.root / 'tree'
        self.tree.mkdir()
        self.output = self.root / 'app.zip'

    def link(self, path, target):
        try:
            path.symlink_to(target)
        except OSError as error:
            self.skipTest(f'Symlink creation unavailable: {error}')

    def test_library_alias_is_a_regular_file_with_target_contents(self):
        (self.tree / 'library.so.1.0').write_bytes(b'library contents')
        self.link(self.tree / 'library.so.1', 'library.so.1.0')
        self.link(self.tree / 'library.so', 'library.so.1')
        package.write_zip(self.tree, self.output, 'App')
        with zipfile.ZipFile(self.output) as archive:
            for entry in archive.infolist():
                self.assertTrue(stat.S_ISREG(entry.external_attr >> 16))
                self.assertEqual(archive.read(entry), b'library contents')

    def test_broken_link_does_not_replace_existing_archive(self):
        self.output.write_bytes(b'previous archive')
        self.link(self.tree / 'broken.so', 'missing.so')
        with self.assertRaises(ValueError):
            package.write_zip(self.tree, self.output, 'App')
        self.assertEqual(self.output.read_bytes(), b'previous archive')

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'FIFO requires POSIX')
    def test_special_file_is_rejected_without_hanging(self):
        os.mkfifo(self.tree / 'pipe')
        with self.assertRaises(ValueError):
            package.write_zip(self.tree, self.output, 'App')
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
