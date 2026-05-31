# Configure + build obs-near-real3d with the VS18-bundled CMake/Ninja + MSVC.
$ErrorActionPreference = "Stop"
$VS    = "C:\Program Files\Microsoft Visual Studio\18\Community"
$VCV   = "$VS\VC\Auxiliary\Build\vcvars64.bat"
$CMAKE = "$VS\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$NINJA = "$VS\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$SRC   = $PSScriptRoot
$BUILD = "$SRC\build"

# NOTE: $cfg/$bld run inside `cmd /c` below — do NOT prefix with PowerShell's `&`
# call operator. cmd executes a quoted path directly; a leading `&` errors with
# "& was unexpected at this time."
$cfg = "`"$CMAKE`" -G Ninja -S `"$SRC`" -B `"$BUILD`" " +
       "-DCMAKE_MAKE_PROGRAM=`"$NINJA`" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl " +
       "-DCMAKE_BUILD_TYPE=Release"
$bld = "`"$CMAKE`" --build `"$BUILD`""

cmd /c "`"$VCV`" && $cfg && $bld"
if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }
Write-Host "`n=== built artifacts ===" -ForegroundColor Green
Get-ChildItem "$BUILD\*.dll" | Select-Object FullName, Length
