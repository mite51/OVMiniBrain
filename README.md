# OVMiniBrain

[![Build and package](https://github.com/mite51/OVMiniBrain/actions/workflows/build.yml/badge.svg)](https://github.com/mite51/OVMiniBrain/actions/workflows/build.yml)

[GitHub repository](https://github.com/mite51/OVMiniBrain) ·
[Windows and Linux builds](https://github.com/mite51/OVMiniBrain/actions/workflows/build.yml)

Native C++ / PhysX / ONNX Runtime viewer for the hierarchical SMPL controller.
Python and Isaac Sim are not needed to run a packaged viewer. Physics and ONNX
inference run on the CPU; the viewer needs an OpenGL 3.3 capable graphics driver.

## Run a shared package

Extract the **entire ZIP** before launching. Use the package matching your OS:
`OVMiniBrain-linux-x64.zip` or `OVMiniBrain-windows-x64.zip`. A Linux executable
and its `.so` libraries cannot run natively on Windows; Windows needs an `.exe`
and matching `.dll` libraries built on Windows.

Linux (x86-64):

```bash
cd /path/to/OVMiniBrain-linux-x64
bash run_motion.sh idle
bash run_motion.sh left_jab --headless --max-steps 60
bash run_motion.sh left_jab --num-characters 4
```

Windows (x64), from Command Prompt or PowerShell:

```powershell
.\run_motion.cmd idle
.\run_motion.cmd left_jab --headless --max-steps 60
.\run_motion.cmd left_jab --num-characters 4
```

Double-click `run_motion.cmd` to show the idle clip. Both launchers accept a
`.pmc` file path in place of a clip name, `--model-dir DIR` for another model,
and `OVMINIBRAIN_MODEL_DIR` as a default model directory. Quote paths containing
spaces. For now, extract Windows packages to a path using ASCII characters;
the native asset readers have not been validated with Unicode Windows paths.
The launchers locate the app and assets relative to themselves, so you can move
the extracted folder. Headless mode does not need a display server.

Linux binaries built here target Ubuntu 24.04 x86-64 and its system C/C++ runtime;
rebuild from the source ZIP on older distributions. Rendering uses X11 (including
XWayland on Wayland desktops). Windows builds bundle the release Visual C++ runtime
when CMake finds it in Visual Studio; if a runtime DLL is missing, install the
[x64 Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe),
as required by [ONNX Runtime](https://onnxruntime.ai/docs/install/).

The window title is **OVMiniBrain - Hierarchical SMPL**. Space pauses, Right Arrow
advances one tick while paused, R resets the camera, and Escape closes the window.
The reference and simulated character are both drawn. Clips restart from their
initial state when looping. Default stamina is 1.0, matching this checkpoint's
training setting. To test weaker joints, use `--stamina-range 0.2 1.0`.
This checkpoint has no opponent/projectile observations; it is a motion tracker
trained on a corpus that includes melee animations.

## Build and share

Use Python 3.9+ and CMake 3.24+. Python and Jinja2 are build/packaging tools only.
The source ZIP includes dependency sources and Linux SDK libraries, and excludes
machine-specific build caches, Git history, and validation output. It preserves
the current model files on disk, including uncommitted model updates.

Linux also needs a C++17 compiler and OpenGL/X11 development packages. On Ubuntu:

```bash
sudo apt install build-essential cmake python3 python3-jinja2 libgl1-mesa-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
./tools/build_linux.sh
python3 tools/package.py
python3 tools/package.py --source
```

Windows needs Visual Studio 2022 (or Build Tools) with **Desktop development with
C++**, the v143 x64 toolset, and a Windows SDK. Put CMake and Python on PATH:

```powershell
python -m pip install "Jinja2>=2.7,<4"
.\tools\build_windows.cmd
python tools/package.py --build-dir build-windows
```

The Windows helper builds the included PhysX 5.6.1 CPU sources, builds the app,
and runs the four headless C++ checks. Its first configuration downloads the
Windows x64 ONNX Runtime 1.26.0 SDK. For offline builds, supply an unpacked SDK:
`tools\build_windows.cmd --ort-root "C:\SDKs\onnxruntime-win-x64-1.26.0"`.
You can also place that SDK under `deps/onnxruntime-win-x64/` before creating the
source ZIP. Dependency sources for GLFW, GLM, GLAD, pugixml and yaml-cpp are
already included. Linux builds use the included SDKs without downloading.
A GitHub Actions workflow is included to build, run headless checks, test an
extracted package in a path with spaces, and upload runtime ZIPs on both OSes.
It runs on pushes and pull requests, or can be started with **Run workflow** on
the [build workflow page](https://github.com/mite51/OVMiniBrain/actions/workflows/build.yml).
To download a build, open a successful run and select its Windows or Linux
artifact. GitHub wraps the runtime ZIP inside an artifact ZIP; extract both layers
before launching. Downloading Actions artifacts requires signing in to GitHub.
Both helpers build Release. The bundled Linux PhysX libraries are Release-only.

Packages are written under `dist/`. Send the runtime ZIP for someone who just
wants to run the viewer. Send `OVMiniBrain-source.zip` for someone who will build
it, including on Windows. A Windows runtime ZIP must be built and checked on
Windows; adding a Windows launcher does not convert the Linux binary.

The packaging helper installs only the app, current unified model, motion
clips, robot asset, launcher, docs, and dependency notices. ONNX Runtime library
aliases are copied as **ordinary files**, preserving the names required by the
Linux loader. ZIP creation also dereferences file symlinks, rejects broken links
and special files, checks archive integrity, and excludes build caches from the
source archive. Use this helper instead of zipping the entire working folder.

## Which model

`models/smpl/unified_pipeline.onnx` exports the **complete hierarchy** from
`ProtoMotions/results/smpl_hierarchical_v3_all_melee/last.ckpt`:
epoch **16,600**, **2,175,795,200** environment steps, exported November 3, 2026.
The source was snapshotted under the run's `onnx_exports/2026-11-03_last_epoch16600/checkpoint/`
so further training cannot change this export. The previous epoch-7000 model remains
in `models_0/smpl`. Both main-brain and mini-brain weights
come from this checkpoint. The standalone pretraining file is not consulted.

The graph contains observation construction, analytical subtree loads, both
mini-brain evaluations, the main policy, and action processing. It returns
`actions`, `joint_pos_targets`, `stiffness_targets`, and `damping_targets`.
`actions` are the raw normalized actor means used for action history;
`joint_pos_targets = pi * clamp(actions, -1, 1)` are the actual radian drive targets.
The viewer sends the latter directly to PhysX, with no extra tanh or scaling.

The sidecar specifies body/DOF order, gains, inputs, output shapes, and saved
training timing: **30 Hz policy, 120 Hz physics, four substeps**. Analytical loads
and joint features are computed inside ONNX from live physics state. The new C++
inputs are current DOF positions/velocities and future DOF positions/velocities.
Local reference DOFs are copied from the cache without world-space transforms.
Stamina is supplied even though this is not a projectile/fighting-policy graph.

`models/smpl/shared_minibrain.onnx` is the optional **local controller only**:
input `joint_features [N,23,34]`, outputs `joint_pos_targets [N,69]` and
`delta_u [N,23,3]`. It infers the desired pose from current pose plus the wrapped
target error. Unavailable axes hold their current angle. Its PD output is in
radians and does not include the whole-body policy or final joint-range clamp.
Feature layout is documented in [ARCHITECTURE.md](ARCHITECTURE.md).

## Export models and cache motions

```bash
./tools/export_policy.sh
```

Re-exporting requires the ProtoMotions Python environment. The helper defaults to
`/home/jwylie/anaconda3/envs/protomotion_isaaclab_12/bin/python`; override
`PROTOMOTIONS_ROOT` and `PROTOMOTIONS_PYTHON` if needed. The exporter is
`ProtoMotions/deployment/export_minibrain_onnx.py`. An optional first argument to
the helper selects another hierarchical checkpoint with adjacent saved configs.

To cache another source animation, run `tools/dump_motion_cache.py` from the
ProtoMotions environment with `--motion-file`, `--yaml models/smpl/unified_pipeline.yaml`
and `--out motion_cache/name.pmc`, using absolute paths when running elsewhere.

## Validation of the November 3 export

- Full ONNX parity: 9 varied batch/stamina cases plus 43 actual C++ idle ticks.
  See `models/smpl/export_validation.json` for the measured errors and hashes.
- Standalone mini-brain checks cover batches 1/2/4 and locked axes. Its weights
  are unchanged, as expected from frozen mini-brain training.
- Both included clips passed complete standalone PhysX evaluations. Evidence is
  saved separately in `validation/2026-11-03_epoch16600/`.
- The original viewer's 19 Python tests and four C++ checks are documented in
  `validation/`; the C++ executable was not changed for this new export.

These checks establish export fidelity and a small standalone PhysX smoke test;
they do not reproduce the full IsaacLab corpus benchmark.
