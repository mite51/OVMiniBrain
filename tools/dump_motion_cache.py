"""Dump a ProtoMotions .motion clip to a flat binary cache for the C++ app.

The C++ inference loop only ever queries the motion at integer multiples of
``control_dt`` (``t = k*dt`` for the current state, ``t = (k+index)*dt`` for the
mimic future). So sampling ``MotionLib.get_motion_state`` at exactly those
times yields a cache that is byte-identical to running MotionLib live -- no
interpolation in C++, zero parity risk.

Everything is written in SIMULATION coordinates (Z-up, xyzw quats), with NO
re-rooting, so it matches what the policy / obs builder consumed in training.

Run from C:\\Git\\ProtoMotions with the protomotions / env_isaaclab conda env:

    python C:\\Git\\OVProtomotionsCpp\\tools\\dump_motion_cache.py ^
        --motion-file C:\\Git\\ProtomotionsAnimData\\Fighting\\KB_Movement\\KB_WalkRight45.motion ^
        --yaml        C:\\Git\\OVProtomotions\\models\\new_model.yaml ^
        --out         C:\\Git\\OVProtomotionsCpp\\motion_cache\\KB_WalkRight45.pmc

Binary format (little-endian), magic "PMC1":

    char[4]  magic            = "PMC1"
    int32    num_ticks K
    int32    num_bodies B     (24)
    int32    num_dofs   D      (69)
    float32  fps
    float32  control_dt
    float32  length_seconds
    float32  body_pos     [K, B, 3]
    float32  body_rot     [K, B, 4]   (xyzw)
    float32  body_vel     [K, B, 3]   (COM, world)
    float32  body_ang_vel [K, B, 3]
    float32  dof_pos      [K, D]
    float32  dof_vel      [K, D]

Tick k corresponds to motion time clamped to [0, length]; ticks past the end
repeat the final clamped frame, matching MotionLib's clamp in get_future.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def _load_control_dt(yaml_path: Path | None, override: float | None) -> float:
    if override is not None:
        return float(override)
    if yaml_path is None:
        return 1.0 / 30.0
    with open(yaml_path, "r", encoding="utf-8") as f:
        blob = yaml.safe_load(f) or {}
    timing = blob.get("timing") or {}
    if timing.get("control_dt") is not None:
        return float(timing["control_dt"])
    if blob.get("dt") is not None:
        return float(blob["dt"])
    return 1.0 / 30.0


def _max_future_index(yaml_path: Path | None) -> int:
    if yaml_path is None:
        return 1
    with open(yaml_path, "r", encoding="utf-8") as f:
        blob = yaml.safe_load(f) or {}
    motion = blob.get("motion") or {}
    timing = (blob.get("timing") or {}).get("training") or {}
    fsi = motion.get("future_step_indices") or timing.get("future_step_indices") or [1]
    return int(max(int(x) for x in fsi))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion-file", required=True, type=Path)
    ap.add_argument("--yaml", type=Path, default=None,
                    help="Sidecar YAML to read control_dt + future_step_indices.")
    ap.add_argument("--control-dt", type=float, default=None,
                    help="Override control_dt (seconds). Falls back to YAML.")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from protomotions.components.motion_lib import MotionLib, MotionLibConfig

    control_dt = _load_control_dt(args.yaml, args.control_dt)
    max_future = _max_future_index(args.yaml)

    cfg = MotionLibConfig(
        motion_file=str(args.motion_file),
        get_motion_state_use_blend=True,
    )
    lib = MotionLib(cfg, device="cpu")
    if lib.num_motions() == 0:
        print(f"ERROR: 0 motions loaded from {args.motion_file}", file=sys.stderr)
        return 2

    motion_id = 0
    length = float(lib.motion_lengths[motion_id].item())
    fps = 1.0 / float(lib.motion_dt[motion_id].item())

    # Enough ticks that the last current tick plus the furthest future
    # look-ahead are covered; times are clamped to [0, length] so trailing
    # ticks just repeat the final frame (matching get_future's clamp).
    num_ticks = int(math.ceil(length / control_dt)) + max_future + 2

    ks = np.arange(num_ticks, dtype=np.float64)
    times = np.clip(ks * control_dt, 0.0, length).astype(np.float32)

    ids = torch.full((num_ticks,), motion_id, dtype=torch.long)
    ts = torch.from_numpy(times)

    with torch.no_grad():
        rs = lib.get_motion_state(ids, ts)

    def npy(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32, copy=False)

    body_pos = npy(rs.rigid_body_pos)      # (K, B, 3)
    body_rot = npy(rs.rigid_body_rot)      # (K, B, 4) xyzw
    body_vel = npy(rs.rigid_body_vel)      # (K, B, 3)
    body_ang = npy(rs.rigid_body_ang_vel)  # (K, B, 3)
    dof_pos = npy(rs.dof_pos)              # (K, D)
    dof_vel = npy(rs.dof_vel)              # (K, D)

    K, B = body_pos.shape[:2]
    D = dof_pos.shape[1]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(b"PMC1")
        f.write(struct.pack("<iii", K, B, D))
        f.write(struct.pack("<fff", float(fps), float(control_dt), float(length)))
        for arr in (body_pos, body_rot, body_vel, body_ang, dof_pos, dof_vel):
            f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())

    print(f"Wrote {args.out}  (K={K} ticks, B={B} bodies, D={D} dofs, "
          f"fps={fps:.3f}, control_dt={control_dt:.5f}, length={length:.3f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
