@echo off
rem Piloto do cnnlab-tcc: 64 genomas aleatorios, fidelidade 5 x 30 epocas,
rem ruido da sentinela e analise. Leva de 4 a 5 horas de GPU.
rem Pode fechar e abrir de novo: retoma de onde parou e pula o que ja terminou.
rem Requer a variavel CNNLAB_TCC_CIFAR_ROOT com a pasta do CIFAR-100.
rem A pasta das execucoes vem de CNNLAB_TCC_RUNS_ROOT (padrao: Documentos\cnnlab-tcc-runs).
setlocal
set REPO=%~dp0..\..
set PY=%REPO%\.venv\Scripts\python.exe
if "%CNNLAB_TCC_RUNS_ROOT%"=="" set CNNLAB_TCC_RUNS_ROOT=%USERPROFILE%\Documents\cnnlab-tcc-runs
if "%CNNLAB_TCC_CIFAR_ROOT%"=="" (echo [erro] defina CNNLAB_TCC_CIFAR_ROOT com a pasta do CIFAR-100 & pause & exit /b 1)
set DIR=%CNNLAB_TCC_RUNS_ROOT%\piloto
echo Piloto em %DIR%
"%PY%" -m cnnlab.tcc.operation.cli piloto run --dir "%DIR%" --cifar-root "%CNNLAB_TCC_CIFAR_ROOT%" --repo-root "%REPO%"
echo.
"%PY%" -m cnnlab.tcc.operation.cli piloto status --dir "%DIR%"
pause
