# Fetch ONNX Runtime (DirectML EP): headers + import lib + runtime DLLs.
# PowerShell-only (no Python). Lays files out under deps/onnxruntime/{include,lib,bin}.
# Versions are pinned for reproducible builds; override via env if needed:
#   $env:ORT_VERSION / $env:DIRECTML_VERSION
$ErrorActionPreference = "Stop"
$HERE = $PSScriptRoot
$OUT  = Join-Path $HERE "onnxruntime"
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Get-Nupkg([string]$pkg, [string]$ver) {
  $url = "https://api.nuget.org/v3-flatcontainer/$pkg/$ver/$pkg.$ver.nupkg"
  $tmp = Join-Path $env:TEMP "$pkg.$ver.nupkg"
  Write-Host "  downloading $pkg $ver"
  Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing
  return $tmp
}

# picks: array of @{ match='<lowercase substring>'; dest='<subdir>' }
function Expand-Picks([string]$nupkg, $picks) {
  $zip = [System.IO.Compression.ZipFile]::OpenRead($nupkg)
  try {
    foreach ($e in $zip.Entries) {
      if ($e.FullName.EndsWith('/')) { continue }
      $ln = $e.FullName.ToLower()
      foreach ($p in $picks) {
        if ($ln.Contains($p.match)) {
          $dstDir = Join-Path $OUT $p.dest
          New-Item -ItemType Directory -Force $dstDir | Out-Null
          $dst = Join-Path $dstDir ([System.IO.Path]::GetFileName($e.FullName))
          [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, $dst, $true)
          Write-Host "    $($e.FullName) -> $($p.dest)/"
          break
        }
      }
    }
  } finally { $zip.Dispose() }
}

# Start from a clean tree so a stale DLL (e.g. an old DirectML.dll from a
# previous version) can't survive and be re-used after a version change.
Remove-Item $OUT -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $OUT | Out-Null

$ortVer = if ($env:ORT_VERSION) { $env:ORT_VERSION } else { "1.24.4" }
$nupkg = Get-Nupkg "microsoft.ml.onnxruntime.directml" $ortVer
Expand-Picks $nupkg @(
  @{ match = "build/native/include/";                    dest = "include" },
  @{ match = "runtimes/win-x64/native/onnxruntime.lib";  dest = "lib" },
  @{ match = "runtimes/win-x64/native/onnxruntime.dll";  dest = "bin" },
  @{ match = "runtimes/win-x64/native/onnxruntime.pdb";  dest = "bin" },
  @{ match = "runtimes/win-x64/native/directml.dll";     dest = "bin" }
)
Remove-Item $nupkg -Force -ErrorAction SilentlyContinue
"onnxruntime.directml $ortVer" | Set-Content (Join-Path $OUT "VERSION.txt")

if (-not (Test-Path (Join-Path $OUT "bin\DirectML.dll"))) {
  Write-Host "  DirectML.dll not in ORT pkg -> fetching Microsoft.AI.DirectML"
  $dmlVer = if ($env:DIRECTML_VERSION) { $env:DIRECTML_VERSION } else { "1.15.4" }
  $dnupkg = Get-Nupkg "microsoft.ai.directml" $dmlVer
  Expand-Picks $dnupkg @( @{ match = "bin/x64-win/directml.dll"; dest = "bin" } )
  Remove-Item $dnupkg -Force -ErrorAction SilentlyContinue
  "directml $dmlVer" | Add-Content (Join-Path $OUT "VERSION.txt")
}

# sanity: the files the build + install need must exist
foreach ($f in @("lib\onnxruntime.lib", "bin\onnxruntime.dll", "bin\DirectML.dll",
                 "include\onnxruntime_cxx_api.h")) {
  if (-not (Test-Path (Join-Path $OUT $f))) { throw "ONNX Runtime fetch incomplete: missing $f" }
}
Write-Host "ONNX Runtime ready under $OUT"
