from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from interplan.planning.scenario_builder.scenario_utils import ModificationsSerializableDictionary as ModDict


@dataclass(frozen=True)
class BenchmarkScenarioFilter(ScenarioFilter):
    """
    Collection of filters used to construct scenarios from a database for training/simulation.
    """

    # List of scenario types to include:
    scenario_types: Optional[List[str]]
    # List of scenarios to include in the form of (log_name, token):
    scenario_tokens: Optional[List[Sequence[str]]]

    # Filter scenarios by log names:
    log_names: Optional[List[str]]
    # Filter scenarios by map names:
    map_names: Optional[List[str]]

    # Number of scenarios per type:
    num_scenarios_per_type: Optional[int]
    # Limit total scenarios (float = fraction, int = num):
    limit_total_scenarios: Optional[Union[int, float]]
    # Threshold for the interval of time between scenario initial lidar timestamps in seconds:
    timestamp_threshold_s: Optional[float]

    # Inclusive minimum threshold for total distance covered (meters, frame-by-frame) by the ego center
    #   for scenario to be kept:
    ego_displacement_minimum_m: Optional[float]

    # Whether to expand multi-sample scenarios to multiple single-sample scenarios.
    expand_scenarios: bool
    # Whether to remove scenarios where the mission goal is invalid:
    remove_invalid_goals: bool
    # Whether to shuffle the scenarios:
    shuffle: bool

    # Exclusive threshold that the ego's speed must rise above (meters per second) for scenario to be kept:
    ego_start_speed_threshold: Optional[float] = None
    # Inclusive threshold that the ego's speed must fall below (meters per second) for scenario to be kept:
    ego_stop_speed_threshold: Optional[float] = None
    # Value at or below which a cross-threshold speed change between two timepoints should be ignored as noise:
    speed_noise_tolerance: Optional[float] = None

    # Path to a json file containing a Set of lidarpc tokens from a Nuplan DB that we want our scenarios to contain:
    token_set_path: Optional[Path] = None

    # A threshold in [0, 1].
    # If 1, a scenario must ONLY contain lidarpc tokens in the set at token_set_path (see above)
    #   in order to pass the filter.
    # If in [0, 1), scenarios will pass only if the fraction of their lidarpc tokens contained in the set
    #   is strictly greater than the threshold below:
    fraction_in_token_set_threshold: Optional[float] = None

    # Radius around ego to check for the presence of on-route route lane segments
    # Uses a VectorMap to collect lane segments and route status
    # Used to filter out scenarios with no route
    ego_route_radius: Optional[float] = None

    only_in_benchmark_scenarios: bool = True             # Only include tokens-modifications from the list of valid tokens
    valid_tokens: dict = None
    modifications: list = None

    def __post_init__(self) -> None:
        """Sanitize class attributes."""
        if self.num_scenarios_per_type is not None:
            assert 0 < self.num_scenarios_per_type, "num_scenarios_per_type should be a positive integer"

        if isinstance(self.limit_total_scenarios, float):
            assert 0.0 < self.limit_total_scenarios <= 1.0, "limit_total_scenarios should be in (0, 1] when float"
        elif isinstance(self.limit_total_scenarios, int):
            assert 0 < self.limit_total_scenarios, "limit_total_scenarios should be positive when integer"    
        
        if not self.scenario_tokens: self.get_all_possible_scenarios()
        else:
            for i, o_token_mods in enumerate(self.modifications["scenario_specifics"]): # Original token
                j = 0
                while j < len(o_token_mods):
                    if "*" not in o_token_mods[j]: 
                        j+= 1
                        continue
                    for k, letter in enumerate(o_token_mods[j]):
                        if k%2 == 1 and letter == "*":
                            for option in self.valid_tokens[self.scenario_tokens[i]][ModDict.get_name_of_mod(o_token_mods[j][k-1])]:
                                self.modifications["scenario_specifics"][i].append(o_token_mods[j][:k]+option+o_token_mods[j][k+1:])
                    self.modifications["scenario_specifics"][i].pop(j)

    def get_all_possible_scenarios(self):
        self.scenario_tokens.clear()
        self.modifications["scenario_specifics"].clear()
        for token in self.valid_tokens:
            self.scenario_tokens.append(token)        
            self.modifications["scenario_specifics"].append(self.get_all_scenario_specifics(token))
                    

    def get_all_scenario_specifics(self, token) -> list:
        scenario_specific_modifications = []

        for goal_option in self.valid_tokens[token]["goal"]:
            for density_option in self.valid_tokens[token]["density"]: 
                for observation_option in self.valid_tokens[token]["observation"]: # TODO add agent variations
                    scenario_specific_modifications.append("g"+goal_option+"d"+density_option+"o"+observation_option)
                
        return scenario_specific_modifications

