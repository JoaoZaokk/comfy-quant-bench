@echo off
REM Carrega o ambiente do MSVC (sem isto `cl.exe` nao existe no PATH e o torch aborta em
REM `where cl`) e o CUDA 13.2. `CUTLASS_INCLUDE` tem de apontar para o `include/` do CUTLASS,
REM que NAO fica neste repo -- ver o docstring de roda.py.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "DISTUTILS_USE_SDK=1"
set CUDA_VISIBLE_DEVICES=0
F:\COMFY_PORTABLE\python_embeded\python.exe -s "%~dp0roda.py" %*
