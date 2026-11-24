param([string]$Arch = "")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

if ([string]::IsNullOrWhiteSpace($Arch)) {
    $cc = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1).Trim()
    if ($cc -match "^(\d+)\.(\d+)$") { $Arch = "sm_$($Matches[1])$($Matches[2])" }
    else { $Arch = "sm_86" }
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vcvars = $null
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($vsPath) {
        $cand = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $cand) { $vcvars = $cand }
    }
}
if (-not $vcvars) {
    $cand = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $cand) { $vcvars = $cand }
}
if (-not $vcvars) { throw "vcvars64.bat not found" }

$tmp = Join-Path $env:TEMP "win_cuda_lbm_build"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
Copy-Item (Join-Path $Root "lbm_cuda.cu") (Join-Path $tmp "lbm_cuda.cu") -Force

$bat = Join-Path $tmp "build.bat"
@"
@echo off
call "$vcvars"
if errorlevel 1 exit /b 1
cd /d "$tmp"
nvcc -O3 --use_fast_math -arch=$Arch -shared -Xcompiler "/O2 /MD /utf-8" -o lbm_cuda.dll lbm_cuda.cu
exit /b %errorlevel%
"@ | Set-Content -Path $bat -Encoding ASCII

& cmd.exe /c "`"$bat`""
if ($LASTEXITCODE -ne 0) { throw "nvcc failed: $LASTEXITCODE" }

Copy-Item (Join-Path $tmp "lbm_cuda.dll") (Join-Path $Root "lbm_cuda.dll") -Force
Write-Host "Built lbm_cuda.dll ($Arch)" -ForegroundColor Green
