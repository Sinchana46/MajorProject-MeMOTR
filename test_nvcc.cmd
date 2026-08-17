@echo off
REM Dry-run nvcc compilation to verify CUDA toolkit + MSVC can see each other.
REM Uses %~dp0 for project root so it works regardless of working directory.

set PROJECT_ROOT=%~dp0
if "%PROJECT_ROOT:~-1%"=="\" set PROJECT_ROOT=%PROJECT_ROOT:~0,-1%

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set DISTUTILS_USE_SDK=1

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe" ^
    -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64" ^
    --dry-run -c "%PROJECT_ROOT%\MeMOTR\models\ops\src\cuda\ms_deform_attn_cuda.cu" -o nul 2>&1
