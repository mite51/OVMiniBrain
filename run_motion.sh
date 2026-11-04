#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
motion=idle
if (( $# > 0 )) && [[ "$1" != --* ]]; then motion="$1"; shift; fi
if [[ -f "$repo_dir/motion_cache/$motion.pmc" ]]; then
  cache="$repo_dir/motion_cache/$motion.pmc"
elif [[ "$motion" == *.pmc && -f "$motion" ]]; then
  cache="$(realpath -- "$motion")"
else
  available="$(ls -1 "$repo_dir/motion_cache"/*.pmc 2>/dev/null | xargs -n1 basename -s .pmc | paste -sd'|' -)"
  echo "Usage: $0 [$available|clip.pmc] [--headless] [--model-dir DIR] [viewer options]" >&2
  exit 2
fi
model_dir="${OVMINIBRAIN_MODEL_DIR:-$repo_dir/models/smpl}"
render=(--render)
options=()
while (( $# > 0 )); do
  case "$1" in
    --headless) render=(); shift ;;
    --model-dir)
      if (( $# < 2 )); then echo "--model-dir requires a directory" >&2; exit 2; fi
      model_dir="$2"; shift 2 ;;
    *) options+=("$1"); shift ;;
  esac
done
model_dir="$(realpath -e -- "$model_dir")"
for name in unified_pipeline.onnx unified_pipeline.yaml; do
  if [[ ! -f "$model_dir/$name" ]]; then echo "Missing model file: $model_dir/$name" >&2; exit 2; fi
done
echo "[OVMiniBrain] Model: $model_dir/unified_pipeline.onnx"
# PhysX may create a local session file; keep it within this project.
cd "$repo_dir"
exec "$repo_dir/build/ovminibrain" \
  --onnx "$model_dir/unified_pipeline.onnx" \
  --yaml "$model_dir/unified_pipeline.yaml" \
  --motion-cache "$cache" --mjcf "$repo_dir/assets/smpl_humanoid.xml" \
  "${render[@]}" "${options[@]}"
