@echo off
rem Windows shortcut: configure a Release build and run both API benchmarks.
rem Build all targets and run CTest separately for the correctness suite.
setlocal

rem Resolve paths relative to this script, not the caller's working directory.
set "PROJECT_DIR=%~dp0"
set "BUILD_DIR=%PROJECT_DIR%build-auto"

rem Detect optional GPU toolchains and generate build instructions.
cmake -S "%PROJECT_DIR%." -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config Release --target run_benchmarks --parallel
if errorlevel 1 exit /b 1

exit /b 0
