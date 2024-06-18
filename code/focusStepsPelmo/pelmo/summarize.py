

import csv
from dataclasses import asdict
import functools
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Generator, Iterable, List, Optional, Union
sys.path += [str(Path(__file__).parent.parent)]

from ioTypes.pelmo import PECResult
from util.conversions import EnhancedJSONEncoder
from pelmo.runner import PelmoResult
from ioTypes.compound import Compound
from ioTypes.gap import GAP

def rebuild_scatterd_to_file(file: Path, parent: Path, output_format: Optional[str] = None, glob_pattern: str = "output.json", psm_root = Path.cwd()):
    write_results_to_file(rebuild_scattered_output(parent, glob_pattern, psm_root), file, output_format)

def rebuild_output_to_file(file: Path, source: Union[Path, List[PelmoResult]], output_format: Optional[str], psm_root = Path.cwd()):
    write_results_to_file(rebuild_output(source, psm_root), file, output_format)

def write_results_to_file(results: Iterable[PECResult], file: Path, format: Optional[str] = None):
    if format == None:
        format = file.suffix[1:]
    if format == 'json':
        with file.with_suffix('.json').open('w') as fp:
            results = list(results)
            json.dump(results, fp, cls=EnhancedJSONEncoder)
    elif format == 'csv':
        with file.with_suffix('.csv').open('w', newline='') as fp:
            writer = csv.writer(fp,)
            header = ["molarMass", "waterSolubility", "dt50", "koc", "freundlich", "plant_uptake","bbch", "rate", "crop", "scenario", "pec"]
            writer.writerow(header)
            def to_list(r: PECResult) -> List[Any]:
                return [r.compound.molarMass, r.compound.waterSolubility, r.compound.degradation.system, r.compound.sorption.koc, r.compound.sorption.freundlich,
                        r.gap.application.timing.bbch_state, r.gap.application.rate, r.crop, r.scenario,
                        r.pec]
            writer.writerows(to_list(r) for r in results)
    else:
        raise ValueError("Could not infer format, please specify explicitly")

def rebuild_scattered_output(parent: Path, glob_pattern: str = "output.json", psm_root = Path.cwd()) -> Generator[PECResult, None, None]:
    logger = logging.getLogger()
    logger.debug("Iterating over output files %s", list(parent.rglob(glob_pattern)))
    for file in parent.rglob(glob_pattern):
        for output in rebuild_output(file, psm_root):
            yield output

def rebuild_output(source: Union[Path, List[PelmoResult]], psm_root = Path.cwd()) -> Generator[PECResult, None, None]:
    logger = logging.getLogger()
    if isinstance(source, Path):
        with source.open() as fp:
            outputs = json.load(fp)
        outputs = [PelmoResult(**item) for item in outputs]
    else:
        outputs = source 
    for output in outputs:
        psm_file = psm_root / output.psm
        with psm_file.open() as psm:
            psm.readline()
            input_data_locations = json.loads(psm.readline())
        compound_file = Path(input_data_locations['compound'])
        gap_file = Path(input_data_locations['gap'])
        logger.debug({"compound_file": compound_file, "gap_file": gap_file})
        compound = get_compound(compound_file)
        gap = get_gap(gap_file)
        yield PECResult(compound=compound, gap=gap, crop=output.crop, scenario=output.scenario, pec=output.pec)

@functools.lru_cache(maxsize=None)
def get_compound(file: Path) -> Compound:
    with file.open() as fp:
        content = json.load(fp)
    return Compound(**content)

@functools.lru_cache(maxsize=None)
def get_gap(file: Path) -> GAP:
    with file.open() as fp:
        content = json.load(fp)
    return GAP(**content)
