@REM Powershell scripts have to be signed, batch scripts don't
@REM and piping powershell commands to the powershell.exe from batch is apparently ok
echo Expand-Archive common.zip -DestinationPath . -Force | powershell.exe
echo Expand-Archive %1.zip -DestinationPath . -Force | powershell.exe

@REM Don't show variable manipulation in console
@echo off
@REM Write all arguments without the first one into REMAINING_VARIABLES
set REMAINING_VARIABLES=
shift
:loop1
if "%1"=="" goto after_loop
set REMAINING_VARIABLES=%REMAINING_VARIABLES% %1
shift
goto loop1

:after_loop
@echo on

python -m venv venv
call .\venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python.exe -m %REMAINING_VARIABLES%
