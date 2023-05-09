ECHO OFF
SET ExecutePeriod=1
SETLOCAL EnableDelayedExpansion

:loop
  cls
  echo %time%
  timeout /t %ExecutePeriod% > nul
goto loop
