# Stage the built plugin + runtimes + model into the OBS plugin layout, then
# produce a portable ZIP and (if Inno Setup is present) an installer .exe.
#
# Run build.ps1 first (so build\obs-near-real3d.dll exists). Works locally and
# in CI. Usage:  .\package.ps1 -Version 0.1.5
param([string]$Version = "0.1.0-dev")
$ErrorActionPreference = "Stop"
$SRC   = $PSScriptRoot
$DIST  = Join-Path $SRC "dist"
$ROOT  = Join-Path $DIST "stage\obs-near-real3d"   # zip/installer root folder
$bin   = Join-Path $ROOT "bin\64bit"
$data  = Join-Path $ROOT "data"

$dll = Join-Path $SRC "build\obs-near-real3d.dll"
if (-not (Test-Path $dll)) { throw "build first (build\obs-near-real3d.dll missing)" }

Remove-Item $DIST -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $bin, $data | Out-Null

# plugin + runtime DLLs (loader finds them next to the plugin)
Copy-Item $dll $bin -Force
foreach ($d in @("onnxruntime.dll", "DirectML.dll")) {
  $p = Join-Path $SRC "deps\onnxruntime\bin\$d"
  if (-not (Test-Path $p)) { throw "missing runtime DLL: $p (run setup.ps1)" }
  Copy-Item $p $bin -Force
}

# effect + locale (everything under data\) + the depth model
Copy-Item (Join-Path $SRC "data\*") $data -Recurse -Force
$mdl = Join-Path $SRC "models\depth_anything_v2_small.onnx"
if (-not (Test-Path $mdl)) { throw "model missing: $mdl" }
Copy-Item $mdl $data -Force

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
$zip = Join-Path $DIST "near-real3d-$Version-windows-x64.zip"
Compress-Archive -Path $ROOT -DestinationPath $zip -Force
Write-Host "zip      -> $zip" -ForegroundColor Green

# ----- Installer (Inno Setup) -----
$iscc = @(
  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
  "C:\Program Files\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($iscc) {
  & $iscc "/DAppVersion=$Version" (Join-Path $SRC "installer\obs-near-real3d.iss")
  if ($LASTEXITCODE -ne 0) { throw "ISCC failed ($LASTEXITCODE)" }
  Write-Host "installer-> $(Get-ChildItem $DIST -Filter *installer.exe | Select-Object -Expand FullName)" -ForegroundColor Green
} else {
  Write-Warning "Inno Setup (ISCC.exe) not found - skipped installer (ZIP still built). Install Inno Setup 6 to build the .exe."
}

Write-Host "`n=== artifacts ===" -ForegroundColor Green
Get-ChildItem $DIST -File | Select-Object Name, Length
