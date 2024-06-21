from dataclasses import dataclass

from util.conversions import map_to_class

from .compound import Compound
from .gap import GAP


@dataclass(frozen = True)
class Combination:
    gap: GAP
    compound: Compound

    def __post_init__(self):
        object.__setattr__(self, 'gap', map_to_class(self.gap, GAP))
        object.__setattr__(self, 'compound', map_to_class(self.compound, Compound))

    def _asdict(self):
        return {'gap': self.gap._asdict(), 'compound': self.compound}