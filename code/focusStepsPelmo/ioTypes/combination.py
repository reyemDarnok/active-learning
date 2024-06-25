from dataclasses import dataclass

from util.datastructures import TypeCorrecting
from util.conversions import map_to_class

from .compound import Compound
from .gap import GAP


@dataclass(frozen = True)
class Combination(TypeCorrecting):
    """A dataclass combining a gap and a compound definition"""
    gap: GAP
    compound: Compound

    def _asdict(self):
        return {'gap': self.gap._asdict(), 'compound': self.compound}