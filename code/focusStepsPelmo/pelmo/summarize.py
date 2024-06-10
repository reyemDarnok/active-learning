

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
from focusStepsDatatypes.compound import Substance
from focusStepsDatatypes.gap import GAP

def rebuild_scattered_output(parent: Path, glob_pattern: str = "output.json") -> Generator[Dict[str, Union[float, str, GAP, Substance]], None, None]:
    for file in parent.rglob(glob_pattern):
        for output in rebuild_output(file):
            yield output

def rebuild_output(source: Union[Path, List[PelmoResult]]) -> Generator[Dict[str, Union[float, str, GAP, Substance]], None, None]:
    logger = logging.getLogger()
    if isinstance(source, Path):
        outputs = json.reads(source.read_text())
    else:
        outputs = source
    for output in outputs:
        psm_file = Path(output.psm)
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
    return Substance(**content)

@functools.lru_cache(maxsize=None)
def get_gap(file: Path):
    with file.open() as fp:
        content = json.load(fp)
    return GAP(**content)
