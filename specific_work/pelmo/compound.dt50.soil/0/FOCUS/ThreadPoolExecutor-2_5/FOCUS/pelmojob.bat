echo off
C:
cd "C:\FOCUS_PELMO.664\FOCUS"
cd 1570024231206253073.run\Winter_-_cereals.run\Hamburg_-_(H).run
if exist *.plm del *.plm
PELMO500.EXE
if exist PELMO500.EXE del PELMO500.EXE
