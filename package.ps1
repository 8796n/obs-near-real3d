# Stage the built plugin + runtimes + model into the OBS plugin layout, then
# produce a portable ZIP and (if Inno Setup is present) an installer .exe.
#
# The DLL is identical for both variants: the plugin reads its inference dims from
# the bundled model's input shape, so a variant only swaps which .onnx ships as
# data\depth_anything_v2_small.onnx.
#   -Variant full  (default): 448x252 model  (models\depth_anything_v2_small.onnx)
#   -Variant lite           : 392x224 model  (models\dav2_392x224.onnx) -- ~1.4x
#                             faster on weak iGPUs at near-full depth quality. (224x126
#                             was tried but DA-V2 loses subject structure below ~336
#                             tokens; 392x224=448 tokens is the usable lower bound.)
#                             Export it with:
#                               python tools\export_onnx.py --width 392 --height 224 `
#                                 --out models\dav2_392x224.onnx
#
# Run build.ps1 first (so build\obs-near-real3d.dll exists). Works locally and
# in CI. Usage:  .\package.ps1 -Version 0.3.1            # full
#                .\package.ps1 -Version 0.3.1 -Variant lite
param(
  [string]$Version = "0.5.0-dev",
  [ValidateSet("full", "lite")][string]$Variant = "full"
)
$ErrorActionPreference = "Stop"
$SRC   = $PSScriptRoot
$DIST  = Join-Path $SRC "dist"
$ROOT  = Join-Path $DIST "stage\obs-near-real3d"   # zip/installer root folder
$bin   = Join-Path $ROOT "bin\64bit"
$data  = Join-Path $ROOT "data"
# Per-variant model source + artifact name suffix. The runtime always loads the
# same filename, so the chosen model is copied in under that fixed name.
$suffix = if ($Variant -eq "lite") { "-lite" } else { "" }
$modelSrc = if ($Variant -eq "lite") {
  Join-Path $SRC "models\dav2_392x224.onnx"
} else {
  Join-Path $SRC "models\depth_anything_v2_small.onnx"
}

$dll = Join-Path $SRC "build\obs-near-real3d.dll"
if (-not (Test-Path $dll)) { throw "build first (build\obs-near-real3d.dll missing)" }

# Clean only the staging tree, not the whole dist: building both variants
# (full then lite) in sequence must not wipe the first variant's zip/installer.
Remove-Item (Join-Path $DIST "stage") -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $bin, $data | Out-Null

# plugin + runtime DLLs (loader finds them next to the plugin)
Copy-Item $dll $bin -Force
foreach ($d in @("onnxruntime.dll", "DirectML.dll")) {
  $p = Join-Path $SRC "deps\onnxruntime\bin\$d"
  if (-not (Test-Path $p)) { throw "missing runtime DLL: $p (run setup.ps1)" }
  Copy-Item $p $bin -Force
}

# effect + locale (everything under data\) + the depth model. The model is copied
# under the fixed runtime name regardless of variant (the plugin loads it by name
# and reads its dims from the model itself).
Copy-Item (Join-Path $SRC "data\*") $data -Recurse -Force
if (-not (Test-Path $modelSrc)) {
  throw "model missing: $modelSrc (variant '$Variant'). For 'lite', export it first: " +
        "python tools\export_onnx.py --width 392 --height 224 --out models\dav2_392x224.onnx"
}
Copy-Item $modelSrc (Join-Path $data "depth_anything_v2_small.onnx") -Force
Write-Host "[model] $Variant -> $(Split-Path $modelSrc -Leaf)" -ForegroundColor Cyan

# docs / licenses alongside the plugin
foreach ($f in @("LICENSE", "THIRD_PARTY_LICENSES", "README.md")) {
  Copy-Item (Join-Path $SRC $f) $ROOT -Force
}
# full third-party license texts (Apache-2.0 model / ONNX Runtime MIT / DirectML)
$licSrc = Join-Path $SRC "licenses"
$reqLic = @("Apache-2.0.txt", "OnnxRuntime-MIT.txt", "DirectML-LICENSE.txt",
            "DirectML-ThirdPartyNotices.txt")
foreach ($l in $reqLic) {
  if (-not (Test-Path (Join-Path $licSrc $l))) {
    throw "licenses/$l missing - all bundled third-party license texts are required in the package"
  }
}
Copy-Item $licSrc $ROOT -Recurse -Force

# ----- ZIP (root contains obs-near-real3d\...) -----
$zip = Join-Path $DIST "near-real3d-$Version$suffix-windows-x64.zip"
Compress-Archive -Path $ROOT -DestinationPath $zip -Force
Write-Host "zip      -> $zip" -ForegroundColor Green

# ----- Installer (Inno Setup) -----
$iscc = @(
  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
  "C:\Program Files\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($iscc) {
  # Lite variant: define `Lite` so the .iss adds the -lite suffix/label (full = no define).
  $isccArgs = @("/DAppVersion=$Version")
  if ($Variant -eq "lite") { $isccArgs += "/DLite" }
  & $iscc @isccArgs (Join-Path $SRC "installer\obs-near-real3d.iss")
  if ($LASTEXITCODE -ne 0) { throw "ISCC failed ($LASTEXITCODE)" }
  Write-Host "installer-> $(Get-ChildItem $DIST -Filter "*$suffix-installer.exe" | Select-Object -Expand FullName)" -ForegroundColor Green
} else {
  Write-Warning "Inno Setup (ISCC.exe) not found - skipped installer (ZIP still built). Install Inno Setup 6 to build the .exe."
}

Write-Host "`n=== artifacts ===" -ForegroundColor Green
Get-ChildItem $DIST -File | Select-Object Name, Length
