from dataclasses import dataclass
import json
from pathlib import Path
from typing import Generator

from ..util.datastructures import TypeCorrecting

from .compound import Compound
from .gap import GAP


@dataclass(frozen=True)
class Combination(TypeCorrecting):
    """A dataclass combining a gap and a compound definition"""
    gap: GAP
    compound: Compound

    def _asdict(self):
        # noinspection PyProtectedMember
        return {'gap': self.gap._asdict(), 'compound': self.compound}

    @staticmethod
    def from_path(path: Path) -> Generator['Combination', None, None]:
        if path.is_dir():
            for file in path.iterdir():
                yield from Combination.from_file(file)
        else:
            yield from Combination.from_file(path)

    @staticmethod
    def from_file(file: Path) -> Generator['Combination', None, None]:
        if file.suffix == '.json':
            with file.open() as f:
                json_content = json.load(f)
                try:
                    yield from (Combination(**element) for element in json_content)
                except TypeError:
                    yield Combination(**json_content)
        if file.suffix == '.xlsx':
            for compound in Compound.from_excel(file):
                for gap in GAP.from_excel(file):
                    yield Combination(gap, compound)
