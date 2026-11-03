# Hierarchical mini-brain controller versus default Mimic

This describes the implementation and the exported **hierarchical v3 all-melee,
epoch-7000** checkpoint. Some formulas and feature descriptions in the older
experiment report are stale; the Python source and saved checkpoint govern this export.

## What changes

| Aspect | Default Mimic / flat MLP | Hierarchical mini-brain controller |
|---|---|---|
| Whole-body network | Six 1024-unit ReLU layers produce all 69 actions | Same-size main trunk produces 69 bounded reference-pose adjustments |
| Local motor control | Learned within the whole-body actor | One shared 34→128→128→3 network applied to each of 23 joints |
| Meaning of main output | Normalized PD command | Desired pose correction, bounded to ±1.8 radians per axis |
| Local output | No separate local module | PD target = adjusted desired angle + bounded local correction (±0.75 radians) |
| Loads | Must be inferred from observations | Explicit subtree gravity torque, angular momentum, mass, and effective inertia |
| Final action transform | Default tanh followed by PD scale/offset | Normalize radian target, then clamp to [-1,1] and map back to radians |
| Training | Whole-body PPO actor and critic | Pretrain the local network; freeze it during whole-body PPO |
| Coordination | Whole-body network | Whole-body network plus analytically aggregated subtree loads |
| Runtime cadence | Policy tick followed by physics substeps | Both networks run at the same 30 Hz tick; PD holds targets across four 120 Hz physics substeps |

Extended observations can add collision, contact, or stamina information to a
flat policy without changing this fundamental action path. The hierarchy changes
what the main network predicts and adds a learned local controller downstream.
This saved actor uses whole-body state, future poses, previous actions, subtree
loads, and the local controller's proposed correction. Although its environment
config contains collision/stamina observation components, the main MLP does not
consume the collision-primitives or separate stamina-observation components.
Stamina does enter the local joint features.

## One policy tick

1. Compute subtree load summaries from body poses, velocities, masses, topology,
   and saved center-of-mass offsets, in each joint's parent frame.
2. Build 34 features for every joint using its current state and the first future
   reference frame (one 30 Hz tick ahead in this checkpoint).
3. Evaluate the shared mini-brain on the unadjusted reference. Its 69 residual
   offsets are exposed as `minibrain_base_cmd` to the main brain.
4. The main brain predicts `delta_q = 1.8 * tanh(raw_main_output) * active_mask`.
   Set `q_des = q_ref + delta_q` and update the wrapped target-error features.
5. Evaluate the **same shared mini-brain again** on this adjusted target. Its
   output is `u_pd = q_des + 0.75 * tanh(MLP_output + physics_basis) * active_mask`.
   Inactive axes instead hold the current joint angle.
6. Normalize by the saved PD scale (pi here), clamp, convert back to radians,
   and send the final targets to PhysX.

`q_ref` in the actor is reconstructed as current angle plus wrapped reference
error, so it is the locally equivalent representation of the reference angle.
Only the desired pose is adjusted by the main brain; desired joint velocities
continue to come from the reference motion.

The physics basis has learned per-axis weights for velocity lead, gravity
compensation, subtree-momentum compensation, and position-error lead. These
terms and the learned MLP residual enter the same bounded correction. At zero
initialization both heads are zero, so the mini-brain starts at `u_pd = q_des`;
it does **not** start at the analytical PD-lead baseline.

## Shared, modular, and stateless

There are 23 evaluations of one weight-shared network, batched as `[N*23,34]`.
There are not 23 separately trained parameter sets. Joint differences enter via
state, load, gains, joint limits, and stamina. The local network uses shared
running input normalization, LayerNorm, and SiLU. The inference path has **21,903
parameters**; the complete mini-brain training module has **27,926**, including
value and dynamics heads. The main MLP has **6,605,893 parameters**.

There is no explicit neural message passing, attention, recurrence, or online
adaptation inside the mini-brain. Subtree loads carry physical information about
connected limbs; the main brain provides global balance and coordination.
Changing the shared weights affects every joint. Per-joint adapters or separate
calibration weights would require an additional implementation.

The value head supports per-joint PPO pretraining; the one-step dynamics head
supports short-horizon surrogate training. Neither is needed by the deployed
deterministic controller and both are pruned from ONNX. Whole-body training uses
the frozen mini-brain, four critic-warmup epochs, and an episode-length curriculum
from 300 to 1000 steps over 60 epochs in this checkpoint.

## Local ONNX feature contract

Each row of `joint_features [N,23,34]` follows the non-root SMPL body/DOF order:

| Columns (zero-based, end exclusive) | Meaning |
|---|---|
| 0:3 | Current joint angles |
| 3:6 | Current joint velocities |
| 6:9 | Wrapped desired-minus-current angles |
| 9:12 | Desired-minus-current velocities |
| 12:15 | Unit gravity direction in parent frame |
| 15:18 | Subtree gravity torque × 0.01 |
| 18:21 | Subtree angular momentum × 0.1 |
| 21:22 | Subtree mass × 0.05 |
| 22:23 | Effective inertia × 0.5 |
| 23:26 | Velocity prediction error |
| 26:29 | Axis stamina/availability in [0,1] |
| 29:34 | kp/1000, kd/100, effort/500, mean limit span/pi, minimum distance to limit/pi |

The current whole-body observation factory supplies zero velocity-prediction
error and expands one body-stamina value to its three axes. Independent axis
lockouts are supported by the local network API/standalone ONNX; the integrated
graph does not expose an independent axis-lockout input. Its zero-stamina mask
holds the command at the current angle; physically locking a joint would also
require changing the simulator constraint.

## Source map

- `ProtoMotions/examples/experiments/mimic/mlp.py`: default flat model.
- `ProtoMotions/protomotions/agents/ppo/hierarchical_model.py`: two mini-brain
  evaluations, main residual pose, normalization, and training schedule.
- `ProtoMotions/protomotions/agents/common/minibrain.py`: shared local network and physics basis.
- `ProtoMotions/protomotions/envs/obs/joint_load_obs.py`: analytical loads and 34-feature layout.
- `ProtoMotions/protomotions/envs/component_factories.py`: actual observation bindings.
- `ProtoMotions/deployment/export_minibrain_onnx.py`: reproducible complete and local ONNX exports.
