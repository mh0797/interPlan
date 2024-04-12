import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import numpy.typing as npt
from nuplan.common.maps.abstract_map_objects import (
    GraphEdgeMapObject,
    LaneGraphEdgeMapObject,
    RoadBlockGraphEdgeMapObject,
)
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from nuplan.common.maps.nuplan_map.lane_connector import NuPlanLaneConnector
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import (
    MetricStatistics,
    MetricStatisticsType,
    Statistic,
    TimeSeries,
)
from nuplan.planning.metrics.utils.route_extractor import (
    CornersGraphEdgeMapObject,
    extract_corners_route,
    get_common_or_connected_route_objs_of_corners,
    get_outgoing_edges_obj_dict,
    get_route,
    get_route_simplified,
    get_timestamps_in_common_or_connected_route_objs,
)
from nuplan.planning.metrics.utils.state_extractors import (
    extract_ego_center,
    extract_ego_time_point,
)
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
        if self.initial_number_of_lane_changes_to_goal == 0:
            if self.number_of_lane_changes_to_goal == 0:
                return 1.0
            else:
                return 0.5
            
        if not self.initial_number_of_lane_changes_to_goal: return 0.2607
        return max(
            0.0,
            (1.0
            - self.number_of_lane_changes_to_goal
            / self.initial_number_of_lane_changes_to_goal),
        )

    def compute(
        self, history: SimulationHistory, scenario: AbstractScenario
    ) -> List[MetricStatistics]:
        """
        Returns the lane changes to go metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the amount of lane changed required at the end of the scenario to get to the goal.
        """

        # Get the list of lane or lane_connectors associated to expert at each time instance
        expert_centers = extract_ego_center(
            scenario.get_ego_state_at_iteration(iteration)
            for iteration in range(len(history.data))
        )
        expert_route = get_route(history.map_api, expert_centers)
        expert_simplified_route = [
            element[0] for element in get_route_simplified(expert_route)
        ]
        expert_simplified_route_lane_ids = [lane.id for lane in expert_simplified_route]
        expert_simplified_route_roadblock_ids = [
            lane.get_roadblock_id() for lane in expert_simplified_route
        ]

        # Get the list of lane or lane_connectors associated to ego at each time instance
        ego_route: list([NuPlanLane]) = get_route(
            history.map_api, extract_ego_center(history.extract_ego_state)
        )
        ego_simplified_route = [
            lane
            for element in get_route_simplified(ego_route)
            for lane in element
            if len(element) == 1 or lane.id in expert_simplified_route_lane_ids
        ]

        # Ego always starts where expert starts to avoid errors
        if ego_simplified_route[0] != expert_simplified_route[0]:
            ego_simplified_route.insert(0, expert_simplified_route[0])

        ego_simplified_route_roadblock_ids = [
            lane.get_roadblock_id() for lane in ego_simplified_route
        ]

        # If ego didn't finish at the same roadblock as the expert find the lane it drove on route
        if (
            ego_simplified_route_roadblock_ids[-1]
            != expert_simplified_route_roadblock_ids[-1]
        ):
            for index, id in list(enumerate(expert_simplified_route_roadblock_ids))[
                ::-1
            ]:
                if id in ego_simplified_route_roadblock_ids:
                    # Last_expert_lane in roadblock droven by ego
                    last_expert_lane = expert_simplified_route[index]
                    # Find last ego lane on route
                    for index, id_ego in list(
                        enumerate(ego_simplified_route_roadblock_ids)
                    )[::-1]:
                        if id == id_ego:
                            last_ego_lane_on_route = ego_simplified_route[index]
                            break
                    else:
                        AssertionError(
                            "Ego only drove along NuPlanLaneConnectors no calculation of this metric possible"
                        )
                    break

            assert (
                "last_ego_lane_on_route" in locals()
            ), "Something is wrong: Ego was never on route"
        else:
            last_expert_lane = expert_simplified_route[-1]
            last_ego_lane_on_route = ego_simplified_route[-1]

        # We define an offset that is 1 in case expert did lane change in first time step
        if expert_route[0][0].get_roadblock_id() == expert_route[1][0].get_roadblock_id():
            offset = 1
        else:
            offset = 0

        self.initial_number_of_lane_changes_to_goal = (
            (
                self.get_number_of_lane_changes_to_goal(
                    ego_route[0][0], expert_route[0 + offset][0]
                )
            )
            if len(expert_simplified_route) > 1
            else 0
        )

        self.number_of_lane_changes_to_goal = self.get_number_of_lane_changes_to_goal(
            last_expert_lane, last_ego_lane_on_route
        )

        did_ego_got_to_lane_goal = (
            True if self.number_of_lane_changes_to_goal == 0 else False
        )

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

    def get_number_of_lane_changes_to_goal(
        self,
        current_lane: LaneGraphEdgeMapObject,
        goal_lane: LaneGraphEdgeMapObject,
    ) -> Dict[str, int]:
        """Copied and modified from llm_feature_builder should be imported once available"""

        # If current and objective lanes are the same then number of lane changes is 0
        if current_lane.id == goal_lane.id:
            return 0
        lane_changes_for_lanes_in_roadbloack = {
            l.id: 0 for l in current_lane.parent.interior_edges if l.id == goal_lane.id
        }
        queue = [
            (l, 0)
            for l in current_lane.parent.interior_edges
            if l.id in lane_changes_for_lanes_in_roadbloack.keys()
        ]
        while queue:
            lane, lane_changes = queue.pop(0)
            if lane.id not in lane_changes_for_lanes_in_roadbloack.keys():
                lane_changes_for_lanes_in_roadbloack.update({lane.id: lane_changes})
            # get adjacent lanes, append if not in visited and not in queue
            for adj_lane in self.get_adjacent_lanes(lane):
                if (
                    adj_lane and adj_lane.id == current_lane.id
                ):  # This doesn't work 21.11
                    return lane_changes + 1
                elif (
                    adj_lane is not None
                    and adj_lane.id not in lane_changes_for_lanes_in_roadbloack.keys()
                    and adj_lane.id not in [item[0] for item in queue]
                ):
                    queue.append((adj_lane, lane_changes + 1))

    def get_adjacent_lanes(
        self,
        current_lane: LaneGraphEdgeMapObject,
    ) -> Tuple[LaneGraphEdgeMapObject]:
        """Copied from llm_feature_builder should be imported once available"""

        def _filter_candidates(
            candidate_lanes: List[LaneGraphEdgeMapObject], side: str
        ) -> Union[LaneGraphEdgeMapObject, None]:
            if len(candidate_lanes) == 0:
                return None
            # We know that the candidates start adjacent to each other as the incoming lanes are adjacent
            # Decide which one stays beside the current lane by comparing the distance of the boundaries at the end of the roadblock
            if side == "right":
                fde = [
                    l.left_boundary.discrete_path[-1].distance_to(
                        current_lane.right_boundary.discrete_path[-1]
                    )
                    for l in candidate_lanes
                ]
            else:  # side == "left"
                fde = [
                    l.right_boundary.discrete_path[-1].distance_to(
                        current_lane.left_boundary.discrete_path[-1]
                    )
                    for l in candidate_lanes
                ]
            if min(fde) > 0.1:
                return None
            else:
                return candidate_lanes[fde.index(min(fde))]

        def _get_candidates(
            current_lane: LaneGraphEdgeMapObject, side: str
        ) -> List[LaneGraphEdgeMapObject]:
            idx = 0 if side == "left" else 1
            candidates: List[LaneGraphEdgeMapObject] = []
            if current_lane.adjacent_edges[idx] is not None:
                candidates.append(current_lane.adjacent_edges[idx])
            previous_lanes_of_adjacent = current_lane.incoming_edges + [
                lane.adjacent_edges[idx]
                for lane in current_lane.incoming_edges
                if lane.adjacent_edges[idx] is not None
            ]
            candidates.extend(
                [l for p in previous_lanes_of_adjacent for l in p.outgoing_edges]
            )
            return list(set([c for c in candidates if c.id != current_lane.id]))

        # Search for adjacent lanes. An adjacent lane can be obtained
        # 1) by current_lane.adjacent_edges
        # 2) successor of adjacent lane of predecessor lane
        # 3) sucessor of predecessor which is not the current lane

        # left neighbor
        candidates_left = _get_candidates(current_lane, "left")
        left = _filter_candidates(candidate_lanes=candidates_left, side="left")

        # right neighbor
        candidates_right = _get_candidates(current_lane, "right")
        right = _filter_candidates(candidate_lanes=candidates_right, side="right")

        return (left, right)
