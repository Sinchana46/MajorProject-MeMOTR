@echo off
REM Build MeMOTR Deformable Attention CUDA ops on Windows
REM Requires: VS 2022 Community + CUDA 13.0 toolkit
REM
REM Uses %CONDA_PREFIX%\python.exe so this works for any user/env,
REM and %~dp0 to derive the project root from the script location.

set PROJECT_ROOT=%~dp0
REM Strip trailing backslash
if "%PROJECT_ROOT:~-1%"=="\" set PROJECT_ROOT=%PROJECT_ROOT:~0,-1%

echo [1] Setting up Visual Studio 2022 x64 environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo ERROR: vcvars64.bat failed
    exit /b 1
)

echo [2] Setting CUDA 13.0 as toolkit...
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;%PATH%
set TORCH_CUDA_ARCH_LIST=8.6

echo [3] Verifying cl.exe and nvcc are accessible...
where cl
where nvcc

echo [4] Building MultiScaleDeformableAttention CUDA extension...
cd /d "%PROJECT_ROOT%\MeMOTR\models\ops"

REM Use CONDA_PREFIX if set, otherwise fall back to 'python' on PATH
if defined CONDA_PREFIX (
    "%CONDA_PREFIX%\python.exe" setup.py build_ext --inplace
) else (
    python setup.py build_ext --inplace
)

if errorlevel 1 (
    echo BUILD FAILED -- see output above
    exit /b 1
) else (
    echo BUILD SUCCEEDED
)
