@echo off
rem Abre a pagina de leitura do cnnlab-tcc no navegador.
rem   start_streamlit.bat           ultima run aberta pelo start_tui.bat
rem   start_streamlit.bat <pasta>   uma pasta de run especifica
rem   start_streamlit.bat fila      pagina das runs da fila e do sono
if "%~1"=="" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\cnnlab.ps1" -Interface view
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\cnnlab.ps1" -Interface view -Mode "%~1"
)
if errorlevel 1 pause
