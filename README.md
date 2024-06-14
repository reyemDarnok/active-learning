# eEnSa-FocusSteps-Pelmo

This Repository aims to automate interacting with FOCUS PELMO

## PELMO

Pelmo is a european groundwater model. It takes proprietary inputs and delivers proprietary outputs

### PELMO Input

Pelmo takes requires three files in the working directory, `pelmo.inp`, `input.dat` and a `*.psm` file, which in turn reference several other files by relative path

#### 2 Levels above the working directory

The scenario files, that is the `.cli`, `.soi` and `.crp` files for the run have to be here.

#### In the working directory

`pelmo.inp` lists all other required files (the files mentioned in the previous section and the `.psm` file), `input.dat` provides lookup values for the crop states and the `*.psm` file is the primary input.

#### The psm file

As Pelmo predates the creation of even XML its input is sadly no standard format. In this format comments are begun with `<` and end with the end of the line or with a `>`.

In terms of actual content, each line without leading whitespace is a new section, with each space at the start of the line indicating one level of nesting of lists. Every block that starts with whitespace also is one section. Inside one line, the elements of the section are separated with whitespace of arbitrary length. Which sections appear in which order is best discovered by inspecting the `FOCUS_defaults.psm` included with PELMO which is well commented.

### PELMO Output

Pelmos outputfiles are all written in the current working directory and end in `.PLM`. Taken together they are about 2.5 MB in size, which is why this project deletes them after extracting the interesting information from them.

#### PLM format

PLM is a mixture of a space seperated tables, typically the interesting data, with one such table per year of simulation and key value pairs, which differ in formatting from file to file slightly. While these segments sadly do not have a delimiter between them, they all start with a heading followed by a line of dashes, which can be used to split the PLM files in parsable chunks. One thing to keep in mind however, is that the seperation of columns sadly is not perfect in these files. Some Columns are paired as value (variation_of_value) and depending on the value, that space before the bracket may or may not be there, which has to be taken into consideration when parsing.

#### Finding the PEC

To determine the PEC two files are relevant: `WASSER.PLM` and `CHEM.PLM`, listing the water volume and compound masses respectively. When calculating the PEC one needs to parse each year segment for the line with compartment 21 and extract the relevant column and calculate that years PEC. Then, the first 6 years of warmup have to be excluded and the 80% percentile of the remaining values has to be taken, which is the final result.

## Components

The primary scripts to execute are found in `code/focusStepsPelmo/pelmo` and are `creator.py`, `local.py`, `runner.py`, `remote_bhpc.py` and `scan.py`

### scan.py

`scan.py` is intended to cover parameter matrixes and takes as input a template input and which range it should use for different parameters. Each part of this matrix can then either be simply written out or directly be calculated on the local machine or the BHPC.

### local.py

This script runs the defined runs on the local machine. To keep the machine still usable, it uses one less thread than the machine has cores, as reported by pythons cpu_count(). It is not required to install PELMO to run this script, as it uses its own bundled executable, regardless of whether is installed or not.

### remote_bhpc.py

This script runs the defined runs on the BHPC. This primarily makes sense for larger runs as a single PELMO run requires about 30 CPU seconds and a bhpc instance requires about 10 minutes to start and is billed for at least an hour. For larger runs however the BHPC can start several 96 core machines, which will greatly reduce the calculation time over smaller instances and, as long as the time remains above one our, reduce the cost as larger ec2 instances have less proportional overhead while maintaining the same cost per core.

### Other Scripts

While the other scripts can be directly run from the command line and this may be usefull for testing, they are primarily intended for invocation by the other three scripts.

