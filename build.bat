@echo off
rem Windows shortcut: configure a Release build and run both API benchmarks.
rem Correctness tests are separate: build all targets, then run ctest in build-auto.
rem Keep script variables local to this batch invocation.
setlocal

rem Resolve paths relative to this script, not the caller's working directory.
set "PROJECT_DIR=%~dp0"
set "BUILD_DIR=%PROJECT_DIR%build-auto"

rem Detect available toolchains and generate build instructions in build-auto.
cmake -S "%PROJECT_DIR%." -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1

rem Build both benchmarks and execute them; stop if configuration or build fails.
cmake --build "%BUILD_DIR%" --config Release --target run_benchmarks --parallel
if errorlevel 1 exit /b 1

exit /b 0
