# One-shot setup of build dependencies for obs-near-real3d (reproduces deps/).
# Safe to re-run; skips steps whose outputs already exist.
$ErrorActionPreference = "Stop"
$SRC = $PSScriptRoot
$VS  = "C:\Program Files\Microsoft Visual Studio\18\Community"
$VCV = "$VS\VC\Auxiliary\Build\vcvars64.bat"

# 1. libobs headers @ the installed OBS version --------------------------------
$obsVer = (Get-Item "C:\Program Files\obs-studio\bin\64bit\obs64.exe").VersionInfo.ProductVersion
if (-not (Test-Path "$SRC\deps\obs-studio\libobs\obs-module.h")) {
  Write-Host "[1/4] cloning obs-studio headers @ $obsVer"
  git clone --depth 1 --branch $obsVer --single-branch `
    https://github.com/obsproject/obs-studio.git "$SRC\deps\obs-studio"
} else { Write-Host "[1/4] obs headers present" }

# 2. obs.lib import library from the installed obs.dll -------------------------
if (-not (Test-Path "$SRC\deps\obslib\obs.lib")) {
  Write-Host "[2/4] generating obs.lib from obs.dll"
  $dll = "C:\Program Files\obs-studio\bin\64bit\obs.dll"
  $tmp = "$env:TEMP\obs_exports.txt"
  cmd /c "`"$VCV`" >nul 2>&1 && dumpbin /exports `"$dll`"" > $tmp 2>&1
  $names = foreach ($l in Get-Content $tmp) {
    if ($l -match '^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)') { $matches[1] }
  }
  $names = $names | Where-Object { $_ -and $_ -ne '=' } | Sort-Object -Unique
  New-Item -ItemType Directory -Force "$SRC\deps\obslib" | Out-Null
  @("EXPORTS") + $names | Set-Content "$SRC\deps\obslib\obs.def" -Encoding ascii
  cmd /c "`"$VCV`" >nul 2>&1 && lib /def:`"$SRC\deps\obslib\obs.def`" /out:`"$SRC\deps\obslib\obs.lib`" /machine:x64 /nologo"
} else { Write-Host "[2/4] obs.lib present" }

# 3. ONNX Runtime (DirectML) ---------------------------------------------------
# Verify the runtime DLLs too, not just the import lib: onnxruntime.dll is a
# load-time dependency of the plugin and DirectML.dll is needed at runtime.
$ortRoot = "$SRC\deps\onnxruntime"
if ((-not (Test-Path "$ortRoot\lib\onnxruntime.lib")) -or
    (-not (Test-Path "$ortRoot\include\onnxruntime_cxx_api.h")) -or
    (-not (Test-Path "$ortRoot\bin\onnxruntime.dll")) -or
    (-not (Test-Path "$ortRoot\bin\DirectML.dll"))) {
  Write-Host "[3/4] fetching ONNX Runtime (DirectML)"
  & "$SRC\deps\get_onnxruntime.ps1"
} else { Write-Host "[3/4] onnxruntime present" }

# 4. Depth model (prebuilt artifact, shipped with releases) --------------------
# Depth Anything V2 Small exported to ONNX (448x252, 16:9, FP16). Place the file at
# models\depth_anything_v2_small.onnx (download it from the GitHub release).
if (Test-Path "$SRC\models\depth_anything_v2_small.onnx") {
  Write-Host "[4/4] depth model present"
} else {
  Write-Warning "[4/4] depth model missing: put 'depth_anything_v2_small.onnx' in '$SRC\models\' (download from the GitHub release). Deps are otherwise ready."
}

Write-Host "`nsetup done. Now: .\build.ps1 ; .\install.ps1" -ForegroundColor Green
