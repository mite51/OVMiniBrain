#!/usr/bin/env python3
"""Create a runtime ZIP for this OS, or a source ZIP for both supported OSes."""
import argparse
from pathlib import Path
import platform
import stat
import subprocess
import sys
import tempfile
import zipfile


SOURCE_ENTRIES = (
    'CMakeLists.txt', 'README.md', 'ARCHITECTURE.md', '.gitignore', '.github',
    'run_motion.sh', 'run_motion.cmd', 'run_motion.ps1',
    'src', 'tests', 'tools', 'third_party', 'assets', 'models', 'models_0',
    'motion_cache', 'deps',
)
SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', 'physx-windows'}


def add_tree(archive, path, name):
    """Dereference file links; refuse special files and directory links."""
    if path.is_symlink() and path.is_dir():
        raise ValueError(f'Directory symlink cannot be packaged: {path}')
    if path.is_dir():
        for child in sorted(path.iterdir()):
            if child.name not in SKIP_DIRS:
                add_tree(archive, child, f'{name}/{child.name}')
        return
    if not path.is_file():
        raise ValueError(f'Not a regular file (or broken symlink): {path}')
    # ZipFile.write follows a file symlink and stores the target's regular mode.
    archive.write(path, name)


def write_zip(tree, output, name, entries=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete, validated archive; retain an old ZIP on failure.
    with tempfile.TemporaryDirectory(prefix='ovminibrain-zip-', dir=output.parent) as tmp:
        pending = Path(tmp) / output.name
        with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for entry in entries or sorted(p.name for p in tree.iterdir()):
                add_tree(archive, tree / entry, f'{name}/{entry}')
        with zipfile.ZipFile(pending) as archive:
            for entry in archive.infolist():
                if not stat.S_ISREG(entry.external_attr >> 16):
                    raise ValueError(f'Non-regular ZIP entry: {entry.filename}')
            bad = archive.testzip()
            if bad:
                raise ValueError(f'Corrupt ZIP entry: {bad}')
        pending.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='store_true', help='Package sources, models and dependencies; no local build cache')
    parser.add_argument('--build-dir', type=Path, help='CMake build directory (runtime packages)')
    parser.add_argument('--output', type=Path, help='Destination ZIP')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    system = platform.system().lower()
    if system not in ('linux', 'windows') and not args.source:
        parser.error('Runtime packages must be built on Linux or Windows')
    name = 'OVMiniBrain-source' if args.source else f'OVMiniBrain-{system}-x64'
    output = (args.output or root / 'dist' / f'{name}.zip').resolve()
    if args.source:
        # In particular, do not recursively include the output in deps/assets/etc.
        for entry in SOURCE_ENTRIES:
            if output.is_relative_to(root / entry):
                parser.error('Place the output outside the source entries, e.g. in dist/')
        write_zip(root, output, name, SOURCE_ENTRIES)
    else:
        build = (args.build_dir or root / ('build-windows' if system == 'windows' else 'build')).resolve()
        with tempfile.TemporaryDirectory(prefix='ovminibrain-runtime-') as tmp:
            stage = Path(tmp)
            subprocess.run(['cmake', '--install', str(build), '--config', 'Release',
                            '--component', 'Runtime', '--prefix', str(stage)], check=True)
            exe = stage / 'bin' / ('ovminibrain.exe' if system == 'windows' else 'ovminibrain')
            if not exe.is_file():
                raise ValueError('Install did not produce the expected platform executable')
            write_zip(stage, output, name)
    print(f'Ready to share: {output}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
