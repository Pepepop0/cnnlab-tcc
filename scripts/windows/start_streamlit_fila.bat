@echo off
rem Abre a pagina de leitura com todas as runs da fila e do sono.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\cnnlab.ps1" -Interface view -Mode fila
if errorlevel 1 pause
