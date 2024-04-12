import logging
from collections import UserDict
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple, Type, Union, cast

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_scenarios_from_db
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import (
    GetScenariosFromDbFileParams,
    ScenarioDict,
    get_scenarios_from_log_file,
    scenario_dict_to_list,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    ScenarioMapping,
    absolute_path_to_log_name,
    download_file_if_necessary,
    ScenarioExtractionInfo,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool, worker_map

from interplan.planning.scenario_builder.nuplan_db.modified_nuplan_scenario import (
    ModifiedNuPlanScenario,
)
from interplan.planning.scenario_builder.scenario_utils import (
    ModificationsSerializableDictionary as ModDict,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GetScenariosFromDbFileParams:
    """
    A convenience class for holding all the parameters to get_scenarios_from_log_file
    """

    # The root folder for the db file (e.g. "/data/sets/nuplan")
    data_root: str

    # The absolute path log file to query
    # e.g. /data/sets/nuplan-v1.1/splits/mini/2021.10.11.08.31.07_veh-50_01750_01948.db
    log_file_absolute_path: str

    # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    expand_scenarios: bool

    # The root directory for maps (e.g. "/data/sets/nuplan/maps")
    map_root: str

    # The map version to load (e.g. "1.0")
    map_version: str

    # The ScenarioMapping to pass to the constructed scenarios.
    scenario_mapping: ScenarioMapping

    # The ego vehicle parameters to pass to the constructed scenarios.
    vehicle_parameters: VehicleParameters

    # If provided, the tokens on which to filter.
    filter_tokens: Optional[List[str]]

    # If provided, the scenario types on which to filter.
    filter_types: Optional[List[str]]

    # If provided, the map names on which to filter (e.g. "[us-nv-las-vegas-strip, us-ma-boston]")
    filter_map_names: Optional[List[str]]

    # The root directory for sensor blobs (e.g. "/data/sets/nuplan/nuplan-v1.1/sensor_blobs")
    sensor_root: str

    # If provided, whether to remove scenarios without a valid mission goal.
    remove_invalid_goals: bool = False

    # If provided, the scenario will contain camera data on the anchor lidar_pc.
    include_cameras: bool = False

    # Verbosity, provides download progression
    verbose: bool = False

    # Modifications done to the original NuPlan scenarios
    modifications: dict = None


class NuPlanModifiedScenarioBuilder(NuPlanScenarioBuilder):
    """Builder class for constructing modified nuPlan scenarios for training and simulation."""

    def get_scenarios(
        self, scenario_filter: ScenarioFilter, worker: WorkerPool
    ) -> List[AbstractScenario]:
        """Implemented. See interface."""
        # Create scenario dictionary and series of filters to apply
        scenario_dict = self._create_scenarios(scenario_filter, worker)
        filter_wrappers = self._create_filter_wrappers(scenario_filter, worker)

        # Apply filtering strategy sequentially to the scenario dictionary
        for filter_wrapper in filter_wrappers:
            scenario_dict = filter_wrapper.run(scenario_dict)

        scenario_list = scenario_dict_to_list(scenario_dict, shuffle=scenario_filter.shuffle)  # type: ignore

        return scenario_list

    def _create_scenarios(
        self, scenario_filter: ScenarioFilter, worker: WorkerPool
    ) -> ScenarioDict:
        """
        Creates a scenario dictionary with scenario type as key and list of scenarios for each type.
        :param scenario_filter: Structure that contains scenario filtering instructions.
        :param worker: Worker pool for concurrent scenario processing.
        :return: Constructed scenario dictionary.
        """
        allowable_log_names = (
            set(scenario_filter.log_names)
            if scenario_filter.log_names is not None
            else None
        )

        assert len(scenario_filter.modifications["scenario_specifics"]) == len(
            scenario_filter.scenario_tokens
        ), (
            "len(scenario_filter.scenario_tokens) == len(scenario_builder.modifications.scenario_specifics) "
            + "should be true if nuplan_modifications is being used as the Scenario Builder"
        )

        scenarios_modifications_dict = {}

        for idx in range(len(scenario_filter.modifications["scenario_specifics"])):
            for scenario_specific_mod in scenario_filter.modifications[
                "scenario_specifics"
            ][idx]:

                # Create a dictionary with the modifications of a single scenario
                modifications_dict = ModDict({})
                for modification_category in scenario_filter.modifications.keys():
                    if modification_category != "scenario_specifics":
                        modifications_dict.dictionary[
                            modification_category
                        ] = scenario_filter.modifications[modification_category]
                    else:
                        modifications_dict.add_scenario_specifics(scenario_specific_mod)

                # Create and add info of the scenario into a dictionary with multiple scenarios
                if (
                    f"{scenario_filter.scenario_tokens[idx]}"
                    not in scenarios_modifications_dict.keys()
                ):
                    scenarios_modifications_dict[
                        f"{scenario_filter.scenario_tokens[idx]}"
                    ] = []
                scenarios_modifications_dict[
                    f"{scenario_filter.scenario_tokens[idx]}"
                ].append(modifications_dict)

        map_parameters = [
            GetScenariosFromDbFileParams(
                data_root=self._data_root,
                log_file_absolute_path=log_file,
                expand_scenarios=scenario_filter.expand_scenarios,
                map_root=self._map_root,
                map_version=self._map_version,
                scenario_mapping=self._scenario_mapping,
                vehicle_parameters=self._vehicle_parameters,
                filter_tokens=scenario_filter.scenario_tokens,
                filter_types=scenario_filter.scenario_types,
                filter_map_names=scenario_filter.map_names,
                remove_invalid_goals=scenario_filter.remove_invalid_goals,
                sensor_root=self._sensor_root,
                include_cameras=self._include_cameras,
                verbose=self._verbose,
                modifications=scenarios_modifications_dict,
            )
            for log_file in self._db_files
            if (allowable_log_names is None)
            or (absolute_path_to_log_name(log_file) in allowable_log_names)
        ]

        if len(map_parameters) == 0:
            logger.warning(
                "No log files found! This may mean that you need to set your environment, "
                "or that all of your log files got filtered out on this worker."
            )
            return {}

        dicts = worker_map(worker, get_scenarios_from_log_file, map_parameters)

        return self._aggregate_dicts(dicts)


def get_scenarios_from_log_file(
    parameters: List[GetScenariosFromDbFileParams],
) -> List[ScenarioDict]:
    """
    Gets all scenarios from a log file that match the provided parameters.
    :param parameters: The parameters to use for scenario extraction.
    :return: The extracted scenarios.
    """
    output_dict: ScenarioDict = {}
    for parameter in parameters:
        this_dict = get_scenarios_from_db_file(parameter)

        for key in this_dict:
            if key not in output_dict:
                output_dict[key] = this_dict[key]
            else:
                output_dict[key] += this_dict[key]

    return [output_dict]


def get_scenarios_from_db_file(params: GetScenariosFromDbFileParams) -> ScenarioDict:
    """
    Gets all of the scenarios present in a single sqlite db file that match the provided filter parameters.
    :param params: The filter parameters to use.
    :return: A ScenarioDict containing the relevant scenarios.
    """
    local_log_file_absolute_path = download_file_if_necessary(
        params.data_root, params.log_file_absolute_path, params.verbose
    )

    scenario_dict: ScenarioDict = {}
    for row in get_scenarios_from_db(
        local_log_file_absolute_path,
        params.filter_tokens,
        params.filter_types,
        params.filter_map_names,
        not params.remove_invalid_goals,
        params.include_cameras,
    ):
        modifications = params.modifications[f"{row['token'].hex()}"]
        for modification in modifications:
            # Get Scenario Type
            special_scenario_number = modification.dictionary.get("special_scenario")
            traffic_density = modification.dictionary.get("density")
            if special_scenario_number:
                scenario_type = modification.dictionary[
                    "modification_details_dictionary"
                ][row["token"].hex()]["special_scenario"][special_scenario_number][
                    "type"
                ]
            elif traffic_density:
                scenario_type = f"{ModDict.density_modification_character_to_command[traffic_density]}_traffic_density"
            else:
                scenario_type = "standard_modified_nuplan_scenario"

            if scenario_type not in scenario_dict:
                scenario_dict[scenario_type] = []

            # Get Extraction Info
            extraction_info = (
                None
                if params.expand_scenarios
                else params.scenario_mapping.get_extraction_info(scenario_type)
            )

            if extraction_info:
                extraction_info = ScenarioExtractionInfo(
                    scenario_name=extraction_info.scenario_name,
                    scenario_duration=30.0,  # All modified scenarios will be 30 seconds long
                    extraction_offset=extraction_info.extraction_offset,
                    subsample_ratio=extraction_info.subsample_ratio,
                )

            scenario_dict[scenario_type].append(
                ModifiedNuPlanScenario(
                    data_root=params.data_root,
                    log_file_load_path=params.log_file_absolute_path,
                    initial_lidar_token=row["token"].hex(),
                    initial_lidar_timestamp=row["timestamp"],
                    scenario_type=scenario_type,
                    map_root=params.map_root,
                    map_version=params.map_version,
                    map_name=row["map_name"],
                    scenario_extraction_info=extraction_info,
                    ego_vehicle_parameters=params.vehicle_parameters,
                    sensor_root=params.sensor_root,
                    modification=modification,
                )
            )

    return scenario_dict
