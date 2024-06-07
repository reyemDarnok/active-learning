#!/usr/bin/env bash

pyinstaller.exe --onefile \
    --add-data 'templates;.' --add-data 'FOCUS.zip;.' --add-binary 'PELMO500.exe;.' \
    --collect-submodules focusStepsDatatypes \
    --clean \
    psm.py