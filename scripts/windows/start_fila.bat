@echo off
rem Abre a TUI na fila de runs (aba Fila: planejar runs, modo sono, Rodar fila, Parar tudo).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\cnnlab.ps1" -Interface tui -Mode fila
if errorlevel 1 pause
