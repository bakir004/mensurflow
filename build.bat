@echo off
rem Windows shortcut: configure a Release build, run correctness tests, and run benchmarks.
rem OpenCL headers and the ICD loader are fetched automatically when not installed.
setlocal

rem Resolve paths relative to this script, not the caller's working directory.
set "PROJECT_DIR=%~dp0"
set "BUILD_DIR=%PROJECT_DIR%build-auto"

rem Install tracked hooks for Git clones; preserve existing custom hook paths.
rem Source archives without Git metadata do not need hook installation.
if exist "%PROJECT_DIR%.git" (
    call :install_hooks
    if errorlevel 1 exit /b 1
)

rem Detect optional GPU toolchains and generate build instructions.
cmake -S "%PROJECT_DIR%." -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config Release --parallel
if errorlevel 1 exit /b 1

ctest --test-dir "%BUILD_DIR%" -C Release --output-on-failure
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config Release --target run_benchmarks
if errorlevel 1 exit /b 1

exit /b 0

:install_hooks
set "HOOKS_PATH="
for /f "delims=" %%H in ('git -C "%PROJECT_DIR%." config --get core.hooksPath') do set "HOOKS_PATH=%%H"
if defined HOOKS_PATH if not "%HOOKS_PATH%"==".githooks" (
    echo Keeping custom core.hooksPath; integrate .githooks/pre-commit manually.
    exit /b 0
)
git -C "%PROJECT_DIR%." config --local core.hooksPath .githooks
if errorlevel 1 exit /b 1
echo Git pre-commit formatting hook enabled (.githooks).
exit /b 0
