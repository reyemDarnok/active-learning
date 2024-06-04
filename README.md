# eEnSa-FocusSteps-Pelmo

This Repository aims to automate interacting with FOCUS PELMO

## Components

There are three relevant scripts, psm_runner.py, psm_creator.py and psm.py

### psm_creator.py

The psm_creator has as its task creating the .psm input files for Pelmo.
This will combine every available compound with every available GAP.

#### Arguments

|Name|Effect|Default|
|--------|----|------|
|-c, --compound-file|Where to find the JSON files that define the source compounds. May be either a single file or a directory. If it is a directory, all .json files in it will be assumed to be compound files.|None|
|-g, --gap-file|Where to find the JSON files that define the source GAPs. May be either a single file or a directory. If it is a directory, all .json files in it will be assumed to be gap files.|None|
|-o, --output-dir|The directory to write the results to. Each .psm file in it will be named *compound*-*GAP*-*timing*.psm|output|

#### Errors

The only known cause for errors are invalid input files, which simply cause the program to crash while outputting the relevant parsing error.

### psm_runner.py

When this program is given a list of psm files, it executes using PELMO. Note that this program assumes that PELMO is already installed on the machine and does not bundle it.

#### Arguments

|Name|Effect|Default|
|--------|----|------|
|-p, --psm-files|Either a single psm file or a directory of psm files. Each file given will be run. There are no interactions between psm files.|None|
|-w, --working-dir|Where to place the data PELMO uses while calculating.|cwd / pelmofiles|
|-f, --focus-dir|Where to find PELMO Scenario data. There is bundled scenario data and you are unlikely to need to change it.|cwd / FOCUS.zip|
|-e, --pelmo-exe|Where to find the PELMO CLI executable. The default value corresponds to the default installation location for PELMO|C:\FOCUS_PELMO.664\PELMO500.exe|
|-c, --crop|Which crops to run. Can be specified multiple times if multiple crops are desired|All crops|
|-s, --scenario|Which Scenario to run. Can be specified multiple times if multiple scenarios are desired. If a given Scenario is not defined for a selected Crop it will be silently skipped for that crop|
|-t, --threads|How many threads to use for running PELMO in parallel|cpu_count -1|

#### Errors

If given invalid input files, this program will crash with the relevant parsing error.

PELMO does not accept some timings for some crop and scenario combinations. If those are defined, PELMO will exit with an error which will be written to the program log and this combination will be skipped in the output.

### psm.py

psm.py is a combination of the previous programs and is what you will usually want to use. It starts from compound and gap definitions and outputs PELMO PECs.

#### Arguments

|Name|Effect|Default|
|--------|----|------|
|-c, --compound-file|Where to find the JSON files that define the source compounds. May be either a single file or a directory. If it is a directory, all .json files in it will be assumed to be compound files.|None|
|-g, --gap-file|Where to find the JSON files that define the source GAPs. May be either a single file or a directory. If it is a directory, all .json files in it will be assumed to be gap files.|None|
|-w, --work-dir|Where to place the files PELMO uses for calculations|cwd / pelmofiles|
|-o, --output-dir|The directory to write the results to. Each .psm file in it will be named *compound*-*GAP*-*timing*.psm|output|
|-e, --pelmo-exe|Where to find the PELMO CLI executable. The default value corresponds to the default installation location for PELMO|C:\FOCUS_PELMO.664\PELMO500.exe|
|--crop|Note that this option has no short form. Which crops to run. Can be specified multiple times if multiple crops are desired|All crops|
|-s, --scenario|Which Scenario to run. Can be specified multiple times if multiple scenarios are desired. If a given Scenario is not defined for a selected Crop it will be silently skipped for that crop|
|-t, --threads|How many threads to use for running PELMO in parallel|cpu_count -1|
|-f, --focus|Where to find PELMO Scenario data. There is bundled scenario data and you are unlikely to need to change it.|cwd / FOCUS.zip|



#### Errors

If given invalid input files, this program will crash with the relevant parsing error.

PELMO does not accept some timings for some crop and scenario combinations. If those are defined, PELMO will exit with an error which will be written to the program log and this combination will be skipped in the output.

## File Formats

This project uses several file formats for different purposes.

### Input

#### Compound

Compounds are represented in JSON format as a nested map of keys to values or other maps.
A full example can be found in `./examples/compound.json`

#### GAP

GAPS are represented in JSON format as a nested map of keys to values or other maps.
A full example can be found in `./examples/gap.json`

#### PSM

PSM files are not in any standard format as their format is defined by PELMO, which predates these standard formats (PELMO: 1991, XML: 1998).

In a psm file regions are defined by tags, with the opening tag being `<Name_of_Region>` and the closing tag `<END Name_of_region>`. Comments are available as well, simply defined as tags that have no closing element. Comments also cannot span multiple lines and not ending a comment tag before the end of the line is not a syntax error, merely bad form.

In these Regions Lists are typically defined by the line starting with a space, while level of nesting is indicated by multiple spaces. Values with different meaning on a single line are further seperated by whitespace (any amount).

Example psm files can be found in `./examples`

### Output

#### PSM

PSM files are not in any standard format as their format is defined by PELMO, which predates these standard formats (PELMO: 1991, XML: 1998).

In a psm file regions are defined by tags, with the opening tag being `<Name_of_Region>` and the closing tag `<END Name_of_region>`. Comments are available as well, simply defined as tags that have no closing element. Comments also cannot span multiple lines and not ending a comment tag before the end of the line is not a syntax error, merely bad form.

In these Regions Lists are typically defined by the line starting with a space, while level of nesting is indicated by multiple spaces. Values with different meaning on a single line are further seperated by whitespace (any amount).

Example psm files can be found in `./examples`

#### Final output

The final output file is a combination of psm, crop and scenario information with their PECs attached in JSON format. An example output file can be found in `./examples/output.json`