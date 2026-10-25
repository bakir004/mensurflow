@echo off
setlocal

rem Resolve paths relative to this script, not the caller's working directory.
set "PROJECT_DIR=%~dp0"
set "BUILD_DIR=%PROJECT_DIR%build-auto"

cmake -S "%PROJECT_DIR%." -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config Release --target run_benchmarks --parallel
if errorlevel 1 exit /b 1

exit /b 0
