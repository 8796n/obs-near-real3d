# Install obs-near-real3d into the per-user OBS plugin folder (no admin required).
$ErrorActionPreference = "Stop"
$SRC = $PSScriptRoot
$rtBin  = "$SRC\deps\onnxruntime\bin"
$dll    = "$SRC\build\obs-near-real3d.dll"
$ortDll = "$rtBin\onnxruntime.dll"
$dmlDll = "$rtBin\DirectML.dll"
$effect = "$SRC\data\near-real3d.effect"
$mdl    = "$SRC\models\depth_anything_v2_small.onnx"  # active model (FP16 by default)

# Validate EVERY required file up front (before writing anything), so a partial
# or unloadable plugin tree is never left behind when a dependency is missing.
# onnxruntime.dll is a load-time dependency: without it OBS can't load the plugin.
$required = [ordered]@{
  "plugin DLL (run build.ps1)"                       = $dll
  "onnxruntime.dll (run setup.ps1)"                  = $ortDll
  "DirectML.dll (run setup.ps1)"                     = $dmlDll
  "SBS effect"                                       = $effect
  "depth model (download into models\, see README)" = $mdl
}
$missing = $required.GetEnumerator() | Where-Object { -not (Test-Path $_.Value) }
if ($missing) {
  throw ("install aborted - missing required files:`n" +
    (($missing | ForEach-Object { "  - $($_.Key): $($_.Value)" }) -join "`n"))
}

# On Windows OBS scans %PROGRAMDATA%\obs-studio\plugins\<module>\bin\64bit
# (frontend/widgets/OBSBasic.cpp: GetProgramDataPath + "/bin/64bit"), NOT %APPDATA%.
$root = Join-Path $env:ProgramData "obs-studio\plugins\obs-near-real3d"
$bin  = Join-Path $root "bin\64bit"
$data = Join-Path $root "data"
New-Item -ItemType Directory -Force $bin, $data | Out-Null

Copy-Item $dll $bin -Force                         # plugin
Copy-Item $ortDll $bin -Force                      # ONNX Runtime
Copy-Item $dmlDll $bin -Force                      # DirectML
Copy-Item "$SRC\data\*" $data -Recurse -Force      # effect(s) + locale (obs_module_file)
Copy-Item $mdl $data -Force                        # depth model

Write-Host "installed -> $bin" -ForegroundColor Green
Get-ChildItem $bin | Select-Object Name, Length
