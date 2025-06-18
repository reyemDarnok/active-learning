"""A file describing combination objects"""
import json
from collections import UserList
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, FrozenSet, Generator, Dict

from focusStepsPelmo.ioTypes.compound import Compound
from focusStepsPelmo.ioTypes.gap import GAP
from focusStepsPelmo.ioTypes.scenario import Scenario
from focusStepsPelmo.util.datastructures import TypeCorrecting


@dataclass(frozen=True)
class Combination(TypeCorrecting):
    """A dataclass combining a gap and a compound definition"""
    gap: GAP
    compound: Compound
    scenarios: frozenset[Scenario]

    def __post_init__(self):
        super().__post_init__()
        if not self.scenarios.intersection(self.gap.defined_scenarios):
            raise ValueError(f"No overlap between Combinatino scenarios {self.scenarios} and scenarios defined for crop {self.gap.defined_scenarios}")

    def asdict(self) -> Dict[str, FrozenSet[Scenario] | Dict[str, Any]]:
        """Represent self as a dictionary - the dataclasses.asdict method chokes on NamedTuples in gap"""
        return {"gap": self.gap.asdict(),
                "compound": asdict(self.compound),
                "scenarios": self.scenarios}

    @staticmethod
    def from_path(path: Path) -> Generator['Combination', None, None]:
        """Create Combinations from a path
        :param path: The path to search for combination files
        :return: A Generator that lazily creates all Combinations in path"""
        if path.is_dir():
            for file in path.iterdir():
                yield from Combination.from_file(file)
        else:
            yield from Combination.from_file(path)

    @staticmethod
    def from_file(file: Path) -> Generator['Combination', None, None]:
        """Create Combinations from a single file. This may create multiple Combinations as many formats support
        multiple objects per file
        :param file: The file to parse
        :return: The Combinations in the file"""
        if file.suffix == '.json':
            with file.open() as f:
                json_content = json.load(f)
                if isinstance(json_content, (list, UserList)):
                    yield from (Combination(**element) for element in json_content) # type: ignore
                else:
                    yield Combination(**json_content)
        if file.suffix == '.xlsx':
            for compound in Compound.from_excel(file):
                for gap in GAP.from_excel(file):
                    yield Combination(gap, compound, scenarios=frozenset(Scenario))
