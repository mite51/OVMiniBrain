---
name: ovminibrain-prepare-animation
description: >-
  Prepare, retarget, cache, and validate animations (.fbx or .motion to .pmc)
  for the OVMiniBrain C++ PhysX SMPL viewer across any machine or OS. Use when
  adding new animations to OVMiniBrain, dumping .motion files to .pmc motion
  caches, or retargeting FBX clips to SMPL for OVMiniBrain.
---

# Prepare Animations for OVMiniBrain

This skill covers the end-to-end workflow for adding an animation to **OVMiniBrain** (`.fbx` -> `.motion` -> `.pmc`, or directly `.motion` -> `.pmc`) without relying on machine-specific paths.

## 1. Environment & Path Discovery (Machine-Agnostic)

Never hardcode user home directories or Conda installation paths. Resolve these four variables dynamically before running commands:

1. **`PROTOMOTIONS_ROOT`** (ProtoMotions repository root):
   - Use `${PROTOMOTIONS_ROOT}` if set.
   - Otherwise, locate the repo containing `protomotions/components/motion_lib.py` and `data/scripts/retarget_fbx_to_smpl.py` (check `pwd` or sibling directories).
2. **`OVMINIBRAIN_ROOT`** (OVMiniBrain repository root):
   - Use `${OVMINIBRAIN_ROOT}` if set.
   - Otherwise, locate the repo containing `tools/dump_motion_cache.py`, `models/smpl/unified_pipeline.yaml`, and `motion_cache/` (often a sibling directory `../OVMiniBrain`).
3. **`PROTO_PYTHON`** (Python interpreter with `torch`, `pyyaml`, and `protomotions` importable):
   - Check `${PROTOMOTIONS_PYTHON}` first.
   - Next, check if `python` / `python3` on `PATH` can run `python -c "import torch, yaml, protomotions"`.
   - Next, search Conda/virtual environments (`conda info --envs` or `~/anaconda3/envs/*/bin/python`, `~/miniconda3/envs/*/bin/python`, `~/miniforge3/envs/*/bin/python`) for an environment matching `*protomotion*` or `*isaaclab*`.
4. **`ANIM_SOURCE`** (Input `.motion` or `.fbx` file):
   - If the user gives a clip name (e.g. `getup_facedown`) rather than a full path, search `${PROTOMOTIONS_ROOT}`, sibling data directories (e.g. `../ProtomotionsAnimData`), or user-specified folders for `<name>.motion` or `<name>.fbx`.

```bash
# Portable discovery snippet (Linux / macOS / Git Bash)
PROTOMOTIONS_ROOT="${PROTOMOTIONS_ROOT:-$(pwd)}"
OVMINIBRAIN_ROOT="${OVMINIBRAIN_ROOT:-$(cd "$PROTOMOTIONS_ROOT/../OVMiniBrain" 2>/dev/null && pwd)}"

find_proto_python() {
  if [[ -n "${PROTOMOTIONS_PYTHON:-}" && -x "${PROTOMOTIONS_PYTHON}" ]]; then
    echo "$PROTOMOTIONS_PYTHON"; return
  fi
  for cand in python3 python \
              "$HOME"/anaconda3/envs/*protomotion*/bin/python \
              "$HOME"/miniconda3/envs/*protomotion*/bin/python \
              "$HOME"/miniforge3/envs/*protomotion*/bin/python \
              "$HOME"/.conda/envs/*protomotion*/bin/python; do
    if command -v "$cand" >/dev/null 2>&1 && \
       PYTHONPATH="$PROTOMOTIONS_ROOT" "$cand" -c "import torch, yaml, protomotions" >/dev/null 2>&1; then
      command -v "$cand"; return
    fi
  done
  return 1
}
PROTO_PYTHON="$(find_proto_python)"
```

---

## 2. Step A (Optional): Retarget `.fbx` to ProtoMotions `.motion`

Skip to **Step B** if a `.motion` file already exists for the clip.

If starting from an `.fbx` file, run `data/scripts/retarget_fbx_to_smpl.py` from `PROTOMOTIONS_ROOT`.

### Discovering Blender
`retarget_fbx_to_smpl.py` invokes Blender in background mode (`--background --factory-startup`) to extract bone transforms and requires `numpy` inside Blender's bundled Python:
- Check if `blender` on `PATH` has `numpy`: `blender --background --factory-startup --python-expr "import numpy"`.
- If system Blender lacks `numpy` or is not on `PATH`, search standard locations (`$BLENDER_EXE`, `~/Downloads/blender-*/blender`, `/opt/blender*/blender`, `/Applications/Blender.app/Contents/MacOS/Blender`, `C:\Program Files\Blender Foundation\Blender *\blender.exe`) and pass `--blender-exe "$BLENDER_EXE"`, or reuse an existing `_extracted/<name>.pkl` with `--extracted-pkl`.

### Choosing Retargeting Flags by Motion Type
- **Standard upright clips (walks, punches, kicks, idle):**
  - Use `--height-mode global` (default).
- **Getup / laydown clips (e.g. `getup_facedown`, `getup_faceup`):**
  - Use `--height-mode last_frame` (for getups ending in standing) or `--height-mode first_frame` (for laydowns starting in standing). The height anchor automatically enforces MJCF collision-geometry clearance (`>= 2 mm` above `z=0`) across all frames so lying torso/pelvis/ankle collision boxes never penetrate the floor.
  - **Do not** pass `--auto-correct` on clips that start lying down (`--auto-correct` assumes frame 0's pelvis is upright and facing `+X`).
  - If the animation FBX does not contain a T-pose bind pose (e.g., Kimodo getup clips), pass `--rest-fbx /path/to/tpose.fbx`.
- **Non-T-pose bind rigs (e.g. Unreal Manny A-pose):**
  - Pass `--fix-rest-pose`.

```bash
PYTHONPATH="$PROTOMOTIONS_ROOT" "$PROTO_PYTHON" \
  "$PROTOMOTIONS_ROOT/data/scripts/retarget_fbx_to_smpl.py" \
  "$INPUT_FBX" \
  --output-dir "$MOTION_OUT_DIR" \
  --smpl-xml "$PROTOMOTIONS_ROOT/protomotions/data/assets/mjcf/smpl_humanoid.xml" \
  --height-mode last_frame \
  ${REST_FBX:+--rest-fbx "$REST_FBX"} \
  ${BLENDER_EXE:+--blender-exe "$BLENDER_EXE"}
```

---

## 3. Step B: Dump `.motion` to OVMiniBrain `.pmc` Cache

OVMiniBrain reads precomputed binary motion caches (`PMC1` format, `.pmc`) sampled at integer multiples of the exported model's `control_dt` and covering the model's `future_step_indices` lookahead.

Run `tools/dump_motion_cache.py` using `PROTO_PYTHON` with `PYTHONPATH="$PROTOMOTIONS_ROOT"`, passing the active model sidecar YAML (`models/smpl/unified_pipeline.yaml`) so `control_dt` and `future_step_indices` match the deployed ONNX policy:

```bash
CLIP_NAME="getup_facedown" # desired clip name in OVMiniBrain
MOTION_FILE="/path/to/${CLIP_NAME}.motion"

PYTHONPATH="$PROTOMOTIONS_ROOT" "$PROTO_PYTHON" \
  "$OVMINIBRAIN_ROOT/tools/dump_motion_cache.py" \
  --motion-file "$MOTION_FILE" \
  --yaml "$OVMINIBRAIN_ROOT/models/smpl/unified_pipeline.yaml" \
  --out "$OVMINIBRAIN_ROOT/motion_cache/${CLIP_NAME}.pmc"
```

Verify the output line confirms `B=24 bodies, D=69 dofs` and a positive `K` tick count and `length`.

---

## 4. Step C: Verify in OVMiniBrain

Both `run_motion.sh` (Linux) and `run_motion.cmd` / `run_motion.ps1` (Windows) automatically discover any `<clip>.pmc` placed in `$OVMINIBRAIN_ROOT/motion_cache/`. No launcher script edits are required.

Run a headless PhysX simulation check across the full clip duration (at least `K` steps) to confirm the cache loads cleanly and the policy tracks the animation:

**Linux:**
```bash
bash "$OVMINIBRAIN_ROOT/run_motion.sh" "$CLIP_NAME" --headless --max-steps 150
```

**Windows (PowerShell / CMD):**
```powershell
& "$env:OVMINIBRAIN_ROOT\run_motion.cmd" $CLIP_NAME --headless --max-steps 150
```

To record a full standalone evaluation report (score, pass/fail, and mean body position error in metres):
```bash
"$OVMINIBRAIN_ROOT/build/ovminibrain" \
  --onnx "$OVMINIBRAIN_ROOT/models/smpl/unified_pipeline.onnx" \
  --yaml "$OVMINIBRAIN_ROOT/models/smpl/unified_pipeline.yaml" \
  --motion-cache "$OVMINIBRAIN_ROOT/motion_cache/${CLIP_NAME}.pmc" \
  --mjcf "$OVMINIBRAIN_ROOT/assets/smpl_humanoid.xml" \
  --eval-output "/tmp/${CLIP_NAME}_eval.json"
```

If distribution archives under `$OVMINIBRAIN_ROOT/dist/` need to include the new animation, rebuild the packages with:
```bash
python3 "$OVMINIBRAIN_ROOT/tools/package.py"
python3 "$OVMINIBRAIN_ROOT/tools/package.py" --source
```
