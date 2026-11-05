#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${PROTOMOTIONS_ROOT:-/home/jwylie/Dev/ProtoMotions}"
python_bin="${PROTOMOTIONS_PYTHON:-/home/jwylie/anaconda3/envs/protomotion_isaaclab_12/bin/python}"
checkpoint="${1:-$source_dir/results/smpl_hierarchical_v4_all_melee/last.ckpt}"
cd "$source_dir"
exec "$python_bin" deployment/export_minibrain_onnx.py --checkpoint "$checkpoint" --output "$repo_dir/models/smpl"
