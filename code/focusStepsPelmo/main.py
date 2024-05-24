#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import List
from zipfile import ZipFile
from enum import Enum, auto
import subprocess
from dataclasses import dataclass
from jinja2 import Environment, PackageLoader, select_autoescape
import logging
logging.basicConfig(level=logging.DEBUG)

jinja_env = Environment(loader=PackageLoader("main"), autoescape=select_autoescape())

@dataclass
class Datarow():
    number: int
    DT50: float
    koc: float
    freundlich: float
    rate: float

    def __init__(self, rowstring: str):
        number, DT50, koc, freundlich, rate = rowstring.split(',')
        self.number = int(number)
        self.DT50 = float(DT50)
        self.koc = float(koc)
        self.freundlich = float(freundlich)
        self.rate = float(rate)



class Scenario(Enum):
    Chat = 0
    Hamb = auto()
    Joki = auto()
    Krems = auto()
    Okeh = auto()
    Piac = auto()
    Port = auto()
    Sevi = auto()
    Thiv = auto()

def main():
    global args 
    args = parse_args()
    args.working_dir.mkdir(exist_ok=True, parents=True)
    logging.debug(f'Parsed Arguments as {args}')

    
    logging.debug('Starting to read data file')
    data = []
    with args.data_file.open() as f: 
        data = f.readlines()
    logging.info(f'Datafile read, {len(data)} entries found')

    logging.debug('Begin extracting the FOCUS zip {args.focus_zip} to the working directory {args.working_dir}')
    with ZipFile(args.focus_zip) as zip:
        zip.extractall(path=args.working_dir)
    logging.info('Extracted the FOCUS zip to the working directory')
    

    with Path(args.output_file).open('w') as out_file:
        for row in data[args.start_row:args.end_row]:
            datarow = Datarow(row)
            psm_file_path = Path("FOCUSPELMO664.PSM")
            psm_file_path = fill_psm_data(psm_file_path, datarow)
            start_time = datetime.now()
            gap_dir = args.working_dir / Path('FOCUS', 'GAP.run')
            run_pelmo(psm_file_path, gap_dir)
            total_time = datetime.now() - start_time
            print(total_time)

            try:
                (args.working_dir / 'FOCUS' / 'GAP.run' / 'GAP-WCEREALS.out').unlink()
            except IOError:
                pass

            get_results(args.working_dir / 'FOCUS' / 'GAP.run' / 'WCEREALS.run')
            scenarios = parse_pelmo_output(args.working_dir / 'FOCUS' / 'GAP.run' / 'GAP-WCEREALS.out')

            out_file.writelines(','.join(row, scenario, scenarios[scenario.value]) for scenario in Scenario)

def generate_commit():
    pass

def concat_results():
    pass

def get_results(crop_dir: Path):

    subprocess.run([args.working_dir / 'parsePELMOout.exe', crop_dir.name], cwd=crop_dir.parent)

def parse_pelmo_output(output: Path) -> List[str]:
    with output.open() as file:
        return [[x for x in row.split('\t') if x][-1] for row in file.readlines()[14:23]]
            

def fill_psm_data(psm_file: Path, row: Datarow) -> Path:
    psm_template = jinja_env.get_template('FOCUSPELMO664.PSM')
    psm_file.write_text(psm_template.render(datarow=row, plant_uptake_factor=0))

def run_pelmo(psm_file: Path, gap_dir: Path):
    for inp_file in Path(gap_dir).rglob('*.inp'):
        
        for old_plm_file in Path(inp_file.parent).glob('*.p?m'):
            old_plm_file.unlink()

        (inp_file.parent / 'FOCUSPELMO664.PSM').write_text(psm_file.read_text())

        subprocess.run([args.working_dir / 'PELMO500.EXE'], cwd=inp_file.parent)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-s', '--start-row', type=int, help='The start of the rows to include', default=0)
    parser.add_argument('-e', '--end-row', type=int, help='The end of the rows to include', default=-1)
    parser.add_argument('-w', '--working-dir', type=Path, default=Path.cwd(), help='The directory to use for expanding the PELMO file structure. Defaults to the current working directory')
    parser.add_argument('-d', '--data-file', type=Path, default=Path('data.csv'), help='The file for input data. Defaults to "data.csv"')
    parser.add_argument('-f', '--focus-zip', type=Path, default=Path('FOCUS.zip'), help='The zip file of the FOCUS Model. Defaults to "FOCUS.zip"')
    parser.add_argument('-o', '--output-file', type=Path, default=Path.cwd() / 'results.csv', help='The Path to the output file. Defaults to the current working directory / results.csv')
    return parser.parse_args()

if __name__ == '__main__':
    main()

