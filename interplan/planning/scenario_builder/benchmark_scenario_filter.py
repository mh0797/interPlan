from dataclasses import dataclass
from typing import Dict, List

from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

from interplan.planning.scenario_builder.scenario_utils import (
    ModificationsSerializableDictionary as ModDict,
)


@dataclass(frozen=True)
class BenchmarkScenarioFilter(ScenarioFilter):
    """
    Inherits from ScenarioFilter and adds the modifications to the scenarios as well as a list of valid tokens
    """
    modifications: Dict = None

