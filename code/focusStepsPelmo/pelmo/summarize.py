

from dataclasses import asdict
import functools
import json
import logging
from pathlib import Path
import sys
from typing import Dict, Generator, List, Union
sys.path += [str(Path(__file__).parent.parent)]

from util.conversions import EnhancedJSONEncoder
from pelmo.runner import PelmoResult
from inputTypes.compound import Compound
from inputTypes.gap import GAP

def rebuild_scattered_output(parent: Path, glob_pattern: str = "output.json", psm_root = Path.cwd()) -> Generator[Dict[str, Union[float, str, GAP, Compound]], None, None]:
    logger = logging.getLogger()
    logger.debug("Iterating over output files %s", list(parent.rglob(glob_pattern)))
    for file in parent.rglob(glob_pattern):
        for output in rebuild_output(file, psm_root):
            yield output

def rebuild_output(source: Union[Path, List[PelmoResult]], psm_root = Path.cwd()) -> Generator[Dict[str, Union[float, str, GAP, Compound]], None, None]:
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
        yield {"compound": compound, "gap": gap, **asdict(output)}

@functools.lru_cache(maxsize=None)
def get_compound(file: Path):
    with file.open() as fp:
        content = json.load(fp)
    return Compound(**content)

@functools.lru_cache(maxsize=None)
def get_gap(file: Path):
    with file.open() as fp:
        content = json.load(fp)
    return GAP(**content)
