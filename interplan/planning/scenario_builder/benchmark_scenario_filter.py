from dataclasses import dataclass
from typing import Dict

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

    def get_all_scenario_specifics(self, token) -> list:
        scenario_specific_modifications = []

        for goal_option in self.valid_tokens[token]["goal"]:
            for density_option in self.valid_tokens[token]["density"]:
                for observation_option in self.valid_tokens[token][
                    "observation"
                ]:  # TODO add agent variations
                    scenario_specific_modifications.append(
                        "g"
                        + goal_option
                        + "d"
                        + density_option
                        + "o"
                        + observation_option
                    )

        return scenario_specific_modifications

    def replace_asterisks(self):
        """
        Function to replace modifications with asterisks into multiple tokens
        for the different valid possibilities of each modification type. For example:
        [feb82815b92a5512-g*] would be replaced to [feb82815b92a5512-gl, feb82815b92a5512-gs, feb82815b92a5512-gr]
        """

        for i, o_token_mods in enumerate(
            self.modifications["scenario_specifics"]
        ):  # Original token
            j = 0
            while j < len(o_token_mods):
                if "*" not in o_token_mods[j]:
                    j += 1
                    continue
                for k, letter in enumerate(o_token_mods[j]):
                    if k % 2 == 1 and letter == "*":
                        assert (
                            o_token_mods[j][k - 1] != "a"
                        ), "Amount of Agents doesn't support asterisk as value"
                        for option in self.valid_tokens[self.scenario_tokens[i]][
                            ModDict.get_name_of_mod(o_token_mods[j][k - 1])
                        ]:
                            self.modifications["scenario_specifics"][i].append(
                                o_token_mods[j][:k] + option + o_token_mods[j][k + 1 :]
                            )
                self.modifications["scenario_specifics"][i].pop(j)
