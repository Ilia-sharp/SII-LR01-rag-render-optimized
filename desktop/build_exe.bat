@echo off
chcp 65001 >nul
title Сборка RAG-Demo.exe
cd /d "%~dp0.."

echo ============================================================
echo   Сборка RAG-Demo.exe
echo   Нужен Python 3.10+ и интернет. Занимает 10-20 минут.
echo   Повторять не нужно: exe потом работает на любом ПК офлайн.
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [ОШИБКА] Python не найден в PATH.
  echo Установите Python 3.10+ с python.org и отметьте "Add python.exe to PATH".
  pause
  exit /b 1
)

echo [1/5] Создаю виртуальное окружение сборки...
if not exist ".venv-build" python -m venv .venv-build
call .venv-build\Scripts\activate.bat

echo [2/5] Ставлю PyTorch (CPU, нужен только для экспорта модели)...
python -m pip install -q -U pip
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fail

echo [3/5] Ставлю остальные зависимости сборки...
pip install -q -r desktop\requirements-build.txt
if errorlevel 1 goto :fail

echo [4/5] Экспортирую модель в ONNX и строю индекс по data\ ...
python desktop\prepare_bundle.py
if errorlevel 1 goto :fail

echo [5/5] Собираю exe...
pyinstaller --noconfirm --clean desktop\rag_demo.spec
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo   ГОТОВО: dist\RAG-Demo.exe
echo   Скопируйте этот файл на флешку — он самодостаточный.
echo ============================================================
dir /b dist
pause
exit /b 0

:fail
echo.
echo [ОШИБКА] Сборка прервана. Скопируйте текст выше — по нему видно шаг.
pause
exit /b 1
