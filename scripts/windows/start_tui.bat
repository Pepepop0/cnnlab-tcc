@echo off
rem Abre a TUI do cnnlab-tcc.
rem   start_tui.bat              nova run, recorte FP32
rem   start_tui.bat fp16         nova run, FP16 por bloco (1 geracao, ate 15 min de GPU)
rem   start_tui.bat continuar    reabre a ultima run para operar
rem   start_tui.bat leitura      reabre a ultima run em somente leitura
rem   start_tui.bat <pasta>      abre uma pasta de run em somente leitura
rem   start_tui.bat fila         fila de runs planejadas e modo sono (aba Fila)
if "%~1"=="" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\cnnlab.ps1" -Interface tui
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\cnnlab.ps1" -Interface tui -Mode "%~1"
)
if errorlevel 1 pause
