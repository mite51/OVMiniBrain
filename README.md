# OVMiniBrain

Native C++ / PhysX / ONNX Runtime viewer for the hierarchical SMPL controller.
Based on `/home/jwylie/Dev/OVProtomotionsCpp`, with its own source, executable,
models, motion caches, robot asset, and copied build dependencies. The original
viewer is unchanged. Python and Isaac Sim are not needed to run the viewer.

## Run

```bash
cd /home/jwylie/Dev/OVMiniBrain
./run_motion.sh idle
./run_motion.sh left_jab
./run_motion.sh left_jab --headless
./run_motion.sh left_jab --num-characters 4
./run_motion.sh /absolute/path/to/another.pmc
# Previous epoch-7000 model:
./run_motion.sh left_jab --model-dir ./models_0/smpl
```

The window title is **OVMiniBrain - Hierarchical SMPL**. Space pauses, Right Arrow
advances one tick while paused, R resets the camera, and Escape closes the window.
The reference and simulated character are both drawn. Clips restart from their
initial state when looping. Default stamina is 1.0, matching this checkpoint's
training setting. To test weaker joints, use `--stamina-range 0.2 1.0`.
This checkpoint has no opponent/projectile observations; it is a motion tracker
trained on a corpus that includes melee animations.

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

## Build and export

```bash
./tools/build_linux.sh
./tools/export_policy.sh
```

The build uses the copied PhysX 5.6.1 libraries and cached dependency sources
under `deps/`; it does not fetch dependencies. Requires a C++17 toolchain, CMake,
Python for GL loader generation, and the system OpenGL/X11 development libraries.
The executable is `build/ovminibrain`. Runtime ONNX libraries are in `lib/`.

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
