echo Expand-Archive common.zip -DestinationPath . -Force | powershell.exe
echo Expand-Archive %1.zip -DestinationPath . -Force | powershell.exe

@REM Write all arguments without the first one into RESTVAR
set RESTVAR=
shift
:loop1
if "%1"=="" goto after_loop
set RESTVAR=%RESTVAR% %1
shift
goto loop1

:after_loop

python.exe %RESTVAR%