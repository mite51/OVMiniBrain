#!/usr/bin/env python3
"""Build the bundled CPU PhysX sources and viewer with Visual Studio 2022."""
import argparse
from pathlib import Path
import subprocess
import sys


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--ort-root', type=Path, help='Unpacked Windows x64 ONNX Runtime SDK (offline builds)')
    args = parser.parse_args()
    if sys.platform != 'win32':
        parser.error('Run this helper on Windows with Visual Studio 2022 C++ build tools installed.')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    try:
        import jinja2  # noqa: F401
    except ImportError:
        parser.error('Install the GL loader build dependency: python -m pip install "Jinja2>=2.7,<4"')

    root = Path(__file__).resolve().parents[1]
    physx = root / 'deps/PhysX/physx'
    sdk = root / 'deps/physx-windows'
    physx_build = root / 'build-physx-windows'
    build = root / 'build-windows'
    generator = ('-G', 'Visual Studio 17 2022', '-A', 'x64', '-T', 'v143')
    run('cmake', '-S', physx / 'compiler/public', '-B', physx_build, *generator,
        f'-DPHYSX_ROOT_DIR={physx.as_posix()}', '-DTARGET_BUILD_PLATFORM=windows',
        f'-DPX_OUTPUT_LIB_DIR={sdk.as_posix()}', f'-DPX_OUTPUT_BIN_DIR={sdk.as_posix()}',
        '-DPX_OUTPUT_ARCH=x86', '-DPX_BUILDSNIPPETS=OFF', '-DPX_BUILDPVDRUNTIME=OFF',
        '-DPX_GENERATE_GPU_PROJECTS=OFF', '-DPX_GENERATE_STATIC_LIBRARIES=OFF',
        '-DNV_USE_STATIC_WINCRT=ON', '-DNV_USE_DEBUG_WINCRT=ON',
        '-DCMAKE_POLICY_DEFAULT_CMP0091=NEW', '-DCMAKE_POLICY_VERSION_MINIMUM=3.5')
    run('cmake', '--build', physx_build, '--config', 'release', '--parallel', args.jobs,
        '--target', 'PhysX', 'PhysXCommon', 'PhysXFoundation', 'PhysXCooking', 'PhysXExtensions', 'PhysXPvdSDK')
    extra = []
    if args.ort_root:
        extra.append(f'-DORT_ROOT={args.ort_root.resolve().as_posix()}')
    run('cmake', '-S', root, '-B', build, *generator,
        f'-DPHYSX_ROOT={physx.as_posix()}',
        f'-DPHYSX_BIN_DIR={(sdk / "bin/win.x86_64.vc143.mt").as_posix()}',
        f'-DPython_EXECUTABLE={Path(sys.executable).as_posix()}',
        '-DCMAKE_POLICY_VERSION_MINIMUM=3.5', *extra)
    run('cmake', '--build', build, '--config', 'Release', '--parallel', args.jobs)
    run('ctest', '--test-dir', build, '-C', 'Release', '--output-on-failure')
    print('Build and checks passed. Run run_motion.cmd, or package with:')
    print('python tools/package.py --build-dir build-windows')


if __name__ == '__main__':
    try:
        main()
    except (OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
