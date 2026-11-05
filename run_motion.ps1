# Use run_motion.cmd for the same CLI syntax as run_motion.sh.
$ErrorActionPreference = 'Stop'
try {
    $root = $PSScriptRoot
    $motion = 'idle'
    $remaining = @($args)
    if ($remaining.Count -gt 0 -and -not $remaining[0].StartsWith('--')) {
        $motion = $remaining[0]
        $remaining = @($remaining | Select-Object -Skip 1)
    }
    $cache = Join-Path $root "motion_cache/$motion.pmc"
    if (-not (Test-Path -LiteralPath $cache -PathType Leaf)) {
        if ($motion.EndsWith('.pmc') -and (Test-Path -LiteralPath $motion -PathType Leaf)) {
            $cache = (Resolve-Path -LiteralPath $motion).Path
        } else {
            throw "Unknown motion '$motion'. Use a name from motion_cache or a path to a .pmc file."
        }
    }
    $model = Join-Path $root 'models/smpl'
    if ($env:OVMINIBRAIN_MODEL_DIR) { $model = $env:OVMINIBRAIN_MODEL_DIR }
    $render = @('--render')
    $options = @()
    for ($i = 0; $i -lt $remaining.Count; $i++) {
        switch ($remaining[$i]) {
            '--headless' { $render = @() }
            '--model-dir' {
                $i++
                if ($i -ge $remaining.Count) { throw '--model-dir requires a directory' }
                $model = $remaining[$i]
            }
            default { $options += $remaining[$i] }
        }
    }
    $model = (Resolve-Path -LiteralPath $model).Path
    foreach ($name in @('unified_pipeline.onnx', 'unified_pipeline.yaml')) {
        if (-not (Test-Path -LiteralPath (Join-Path $model $name) -PathType Leaf)) {
            throw "Missing model file: $model/$name"
        }
    }
    $exe = Join-Path $root 'bin/ovminibrain.exe'
    if (-not (Test-Path -LiteralPath $exe)) { $exe = Join-Path $root 'build-windows/Release/ovminibrain.exe' }
    if (-not (Test-Path -LiteralPath $exe)) { throw 'Viewer not built. Run tools/build_windows.cmd first.' }
    Write-Host "[OVMiniBrain] Model: $model/unified_pipeline.onnx"
    Push-Location -LiteralPath $root
    try {
        & $exe --onnx (Join-Path $model 'unified_pipeline.onnx') `
            --yaml (Join-Path $model 'unified_pipeline.yaml') `
            --motion-cache $cache --mjcf (Join-Path $root 'assets/smpl_humanoid.xml') @render @options
        $result = $LASTEXITCODE
    } finally { Pop-Location }
    exit $result
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 2
}
