@echo off
REM Minimal build script for the Deformable Attention CUDA ops.
REM Uses %~dp0 for project root and %CONDA_PREFIX% for the Python interpreter.

set PROJECT_ROOT=%~dp0
if "%PROJECT_ROOT:~-1%"=="\" set PROJECT_ROOT=%PROJECT_ROOT:~0,-1%

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set TORCH_CUDA_ARCH_LIST=8.6
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;%PATH%
set DISTUTILS_USE_SDK=1

cd /d "%PROJECT_ROOT%\MeMOTR\models\ops"

if defined CONDA_PREFIX (
    "%CONDA_PREFIX%\python.exe" setup.py build_ext --inplace
) else (
    python setup.py build_ext --inplace
)
