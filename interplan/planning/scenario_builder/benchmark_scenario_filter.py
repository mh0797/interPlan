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

    only_in_benchmark_scenarios: bool = (
        True  # Only include tokens-modifications from the list of valid tokens
    )
    valid_tokens: Dict = None
    modifications: Dict = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if not self.scenario_tokens:
            self.load_all_valid_benchmark_scenarios()
        else:
            self.replace_asterisks()

    def load_all_valid_benchmark_scenarios(self):
        self.scenario_tokens.clear()
        self.modifications["scenario_specifics"].clear()
        for token in self.valid_tokens:
            self.scenario_tokens.append(token)
            self.modifications["scenario_specifics"].append(
                self.get_all_scenario_specifics(token)
            )

    def get_all_scenario_specifics(self, token) -> List:
        scenario_specific_modifications = []

        for goal in self.valid_tokens[token]["goal"]:
            for density in self.valid_tokens[token]["density"]:
                for observation in self.valid_tokens[token][
                    "observation"
                ]:  # TODO add agent variations
                    scenario_specific_modifications.append(
                        "g" + goal + "d" + density + "o" + observation
                    )

        # Add special scenarios
        for special_scenario in self.valid_tokens[token]["special_scenario"]:
            scenario_specific_modifications.append("s" + special_scenario)

        return scenario_specific_modifications

    def replace_asterisks(self):
        """
        Function to replace modifications with asterisks into multiple tokens
        for the different valid possibilities of each modification type. For example:
        [feb82815b92a5512-g*] would be replaced to [feb82815b92a5512-gl, feb82815b92a5512-gs, feb82815b92a5512-gr]
        """

        # Check that amount of agents don't include an asterisk since amount of agents is an percentage 
        assert "a*" not in self.modifications["scenario_specifics"], "Amount of Agents doesn't support asterisk as value"

        for i, o_token_mods in enumerate(
            self.modifications["scenario_specifics"]
        ):  # o_tokens_mods = Original token modifications e.g grdh
            j = 0
            while j < len(o_token_mods):
                if "*" not in o_token_mods[j]:
                    j += 1
                    continue
                for k, letter in enumerate(o_token_mods[j]):
                    if k % 2 == 1 and letter == "*":
                        for option in self.valid_tokens[self.scenario_tokens[i]][
                            ModDict.get_name_of_mod(o_token_mods[j][k - 1])
                        ]:
                            self.modifications["scenario_specifics"][i].append(
                                o_token_mods[j][:k] + option + o_token_mods[j][k + 1 :]
                            )
                self.modifications["scenario_specifics"][i].pop(j)
