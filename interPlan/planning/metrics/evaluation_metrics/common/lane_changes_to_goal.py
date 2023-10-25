import logging
from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np
import numpy.typing as npt

from nuplan.common.maps.abstract_map_objects import GraphEdgeMapObject
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from nuplan.common.maps.nuplan_map.lane_connector import NuPlanLaneConnector
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.route_extractor import (
    CornersGraphEdgeMapObject,
    extract_corners_route,
    get_common_or_connected_route_objs_of_corners,
    get_outgoing_edges_obj_dict,
    get_route,
    get_route_simplified,
    get_timestamps_in_common_or_connected_route_objs,
)
from nuplan.planning.metrics.utils.state_extractors import extract_ego_center, extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory

logger = logging.getLogger(__name__)


class LaneChangesToGoalStatistics(MetricBase):
    """Statistics on lane changes required to get to goal at the end"""

    def __init__(self, name: str, category: str) -> None:
        """
        Initializes the LaneChangesToGoalStatistics class
        :param name: Metric name
        :param category: Metric category
        """
        super().__init__(name=name, category=category)

        # Store to re-use in high-level metrics
        self.ego_last_lane: List[List[Optional[GraphEdgeMapObject]]] = []
        self.results: List[MetricStatistics] = []

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        # Return between 1.0 and 0.0 where 1 is that ego go to lane goal and 0 didn't move from initial lane
        return max(0.0 , 1.0 - self.number_of_lane_changes_to_goal/self.initial_number_of_lane_changes_to_goal)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the lane changes to go metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the amount of lane changed required at the end of the scenario to get to the goal.
        """

        # Get the list of lane or lane_connectors associated to ego at each time instance
        ego_route: list([NuPlanLane]) = get_route(history.map_api, extract_ego_center(history.extract_ego_state))
        ego_simplified_route = [element[0] for element in get_route_simplified(ego_route)]
        ego_simplified_route_roadblock_ids = [lane.get_roadblock_id() for lane in ego_simplified_route]

        # Get the list of lane or lane_connectors associated to expert at each time instance
        expert_route = get_route(history.map_api, extract_ego_center(scenario.get_modified_expert_trayectory))
        expert_simplified_route = [element[0] for element in get_route_simplified(expert_route)]
        expert_simplified_route_roadblock_ids = [lane.get_roadblock_id() for lane in expert_simplified_route]
        
        # If ego didn't finish at the same roadblock as the expert find the lane it drove on route
        if ego_simplified_route_roadblock_ids[-1] != expert_simplified_route_roadblock_ids[-1] or \
            (ego_simplified_route_roadblock_ids[-1] == expert_simplified_route_roadblock_ids[-1] and \
             isinstance(ego_simplified_route[-1], NuPlanLaneConnector)):
            for index, id in list(enumerate(expert_simplified_route_roadblock_ids))[::-1]:
                if id in ego_simplified_route_roadblock_ids and isinstance(expert_simplified_route[index], NuPlanLane):
                    last_expert_lane = expert_simplified_route[index] # Last_expert_lane in roadblock droven by ego
                    # Find last ego lane on route 
                    for index, id_ego in list(enumerate(ego_simplified_route_roadblock_ids))[::-1]:
                        if id == id_ego:
                            last_ego_lane_on_route = ego_simplified_route[index]
                            break
                    else: AssertionError("Ego only drove by NuPlanLaneConnectors no calculation of this metric possible") #TODO
                    break
            else: AssertionError("Route is completely NuPlanLaneConnectors no calculation of this metric possible") #TODO
            assert "last_ego_lane_on_route" in locals(), "Something is wrong: Ego was never on route"
        else: 
            last_expert_lane = expert_simplified_route[-1]
            last_ego_lane_on_route = ego_simplified_route[-1]

        self.initial_number_of_lane_changes_to_goal = abs(ego_simplified_route[0].index - expert_simplified_route[0].index)
        self.number_of_lane_changes_to_goal = abs(last_expert_lane.index - last_ego_lane_on_route.index)

        did_ego_got_to_lane_goal = True if self.number_of_lane_changes_to_goal == 0 else False        
        
        metric_statistics = [
            Statistic(
                name=f"number_of_{self.name}",
                unit=MetricStatisticsType.COUNT.unit,
                value=self.number_of_lane_changes_to_goal,
                type=MetricStatisticsType.COUNT,
            ),
            Statistic(
                name=f"did_ego_got_to_goal_lane",
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=did_ego_got_to_lane_goal,
                type=MetricStatisticsType.BOOLEAN,
            ),
        ]

        results: List[MetricStatistics] = self._construct_metric_results(
            metric_statistics=metric_statistics, time_series=None, scenario=scenario
        )

        self.results = results

        return results