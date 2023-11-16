from __future__ import annotations
import numpy as np
from shapely import Point
from functools import cached_property

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from interplan.planning.scenario_builder.scenario_modifier.agents_modifier import AgentsModifier


from typing import Any, List, Optional, Tuple, Type, Dict, Union, cast, Generator

from nuplan.common.geometry.compute import principal_value
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.actor_state.ego_state import EgoState, StateSE2
from nuplan.common.actor_state.state_representation import Point2D


from nuplan.database.nuplan_db.nuplan_db_utils import get_lidarpc_sensor_data
from nuplan.database.nuplan_db.nuplan_scenario_queries import ( 
    get_ego_state_for_lidarpc_token_from_db,
    get_roadblock_ids_for_lidarpc_token_from_db,
    get_mission_goal_for_sensor_data_token_from_db
    )
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    ScenarioExtractionInfo,
    extract_tracked_objects,
)
from nuplan.planning.metrics.utils.state_extractors import extract_ego_center
from nuplan.planning.metrics.utils.route_extractor import (
    get_route,
    get_route_simplified,)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from nuplan.common.maps.nuplan_map.lane_connector import NuPlanLaneConnector
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.path.utils import convert_se2_path_to_progress_path
from interplan.planning.simulation.planner.utils.breadth_first_search_lane_goal import BreadthFirstSearch # TODO new name for this
from interplan.planning.scenario_builder.scenario_utils import ModificationsSerializableDictionary
from nuplan.planning.simulation.observation.idm.idm_agents_builder import get_starting_segment
from nuplan.planning.metrics.utils.route_extractor import get_current_route_objects



class ModifiedNuPlanScenario(NuPlanScenario):

    def __init__(self,
        data_root: str,
        log_file_load_path: str,
        initial_lidar_token: str,
        initial_lidar_timestamp: int,
        scenario_type: str,
        map_root: str,
        map_version: str,
        map_name: str,
        scenario_extraction_info: Optional[ScenarioExtractionInfo],
        ego_vehicle_parameters: VehicleParameters,
        sensor_root: Optional[str] = None,
        modification: dict = None,):

        super().__init__(data_root, log_file_load_path, initial_lidar_token, initial_lidar_timestamp, \
                scenario_type, map_root, map_version, map_name, scenario_extraction_info,
                ego_vehicle_parameters, sensor_root)

        # Set lookup table for this specific token
        lookup_table = modification.pop("lookup_table")
        self.lookup_table = next((elem for elem in lookup_table if elem["token"] == self.token), None)

        self._map_modification_character_to_command = {
            "l": "left",
            "r": "right",
            "s": "straight",
        }
        # Add modification attribute in case the scenario is modified
        # If it is special scenario use the config of it
        special_scenario_number = modification.get("special_scenario")
        new_config = self.lookup_table["special_scenario"][special_scenario_number].get("config") if special_scenario_number else None
        if new_config:
            modification = ModificationsSerializableDictionary(modification)
            modification.reset_scenario_specifics()
            modification.add_scenario_specifics(new_config+f"s{special_scenario_number}")
            self.modification = modification.dictionary
        else: 
            self.modification = modification 
        
        # Get goal location if necessary
        if "goal" in modification:
            command = self._map_modification_character_to_command[self.modification["goal"]]
            if self.lookup_table and self.lookup_table.get("goal") and self.lookup_table["goal"].get(command):
                # lookup table contains a goal location for the current goal modification
                goal_coords = self.lookup_table["goal"][command].split(",")
                self.goal_location = Point2D(x=goal_coords[0], y=goal_coords[1])
            else: self.goal_location = None
        else: self.goal_location = None

        # Initialize agent modifier if necessary
        if "density" in self.modification or "amount_of_agents" in self.modification or "special_scenario" in self.modification:
            self.agents_modifier = AgentsModifier(self.modification, 
                                                self.map_api, 
                                                self._log_file, 
                                                self._lidarpc_tokens, 
                                                self.lookup_table)
            _, self.ego_speed = self._get_initial_tracked_objects_and_ego_speed()
        else: self.agents_modifier = None

        self.expert_route_roadblock_ids, self.expert_route_plan = self._initialize_expert_route_plan()

    def __reduce__(self) -> Tuple[Type[NuPlanScenario], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return (
            self.__class__,
            (
                self._data_root,
                self._log_file_load_path,
                self._initial_lidar_token,
                self._initial_lidar_timestamp,
                self._scenario_type,
                self._map_root,
                self._map_version,
                self._map_name,
                self._scenario_extraction_info,
                self._ego_vehicle_parameters,
                self._sensor_root,
                self.modification
            ),
        )
    
    def get_tracked_objects_at_iteration(
        self,
        iteration: int,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations(), f"Iteration is out of scenario: {iteration}!"
        
        if self.agents_modifier:
            return DetectionsTracks(self.agents_modifier.get_tracked_objects_at_iteration(iteration))
        else:
            return DetectionsTracks(
            extract_tracked_objects(self._lidarpc_tokens[iteration], self._log_file, future_trajectory_sampling)
            )
    
    def _get_initial_tracked_objects_and_ego_speed(self):
        # Get initial Iteration
        # Agent_modifier defines ego speed according to position among spawned agents
        tracked_objects, ego_speed = self.agents_modifier.get_initial_tracked_objects_and_ego_speed()
        return DetectionsTracks(tracked_objects), ego_speed
    
    
    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        if "goal" in self.modification:
            if iteration == 0: 
                initial_ego_state = get_ego_state_for_lidarpc_token_from_db(self._log_file, self._lidarpc_tokens[iteration])
                # Set the new speed that ego should have if it spawn among new spawned agents
                if hasattr(self, "ego_speed"): initial_ego_state.dynamic_car_state.rear_axle_velocity_2d.x = self.ego_speed
                return initial_ego_state
            index = int((len(self.get_modified_expert_trayectory)/self.get_number_of_iterations()) * iteration)
            return self.get_modified_expert_trayectory[index]
            # So that iteration = 0 -> index = 0 ... iteration = self.get_number_of_iterations() -> index = len(modified_expert_trayectory)
        else:
            return super().get_ego_state_at_iteration(iteration)

    @cached_property
    def get_modified_expert_trayectory(self) -> List[EgoState]:

        modified_expert_trayectory = []
        initial_ego_state = self.initial_ego_state

        for edge in self.expert_route_plan:
            progress_baseline_path = convert_se2_path_to_progress_path(edge.baseline_path.discrete_path)
            inital_ego_progress = self.expert_route_plan[0].baseline_path.linestring.project(Point(*initial_ego_state.center.point.array))
            for state in progress_baseline_path:
                # Skip states that are behind ego initial location
                if edge.id == self.expert_route_plan[0].id:
                    state_ego_progress = edge.baseline_path.linestring.project(Point(*state.point))
                    if state_ego_progress < inital_ego_progress:
                        continue
                # Append ego states to modified_expert_trayectory
                modified_expert_trayectory.append(EgoState.build_from_center(state, initial_ego_state.dynamic_car_state.center_velocity_2d,
                                                                                    initial_ego_state.dynamic_car_state.center_acceleration_2d,
                                                                                    initial_ego_state.tire_steering_angle,
                                                                                    initial_ego_state.time_point,
                                                                                    initial_ego_state.car_footprint.vehicle_parameters))
        return modified_expert_trayectory

    def get_route_roadblock_ids(self) -> List[str]:
        return self.expert_route_roadblock_ids
        
    def search_route_to_goal(self, starting_roadblock: RoadBlockGraphEdgeMapObject, goal_lane: LaneGraphEdgeMapObject, 
                             candidate_lane_edge_ids: List[str], lengh_of_search: int):

        # Search for route from multiple starting edges to account for lane change
        route_plan = None
        for start_edge in starting_roadblock.interior_edges:
            # Create graph search to find route
            graph_search = BreadthFirstSearch(start_edge, candidate_lane_edge_ids)
            # Search for the route
            _route_plan, path_found, lane_change = graph_search.search(goal_lane, lengh_of_search)

            if path_found and not lane_change: 
                route_plan = _route_plan
                break
            elif path_found: 
                route_plan = _route_plan
        else: 
            if not route_plan: route_plan = _route_plan
                    
        return route_plan, path_found

    def get_mission_goal(self) -> Optional[StateSE2]:
        
        if "goal" not in self.modification.keys():
            return super().get_mission_goal()
        elif self.goal_location:
            return self.expert_route_plan[-1].baseline_path.get_nearest_pose_from_position(self.goal_location)
        else:
            return self.expert_route_plan[-1].baseline_path.discrete_path[-1]

    def _infer_route_plan_from_command(self, route_length: int, command:str) -> List[str]:
        angles_for_command = {
            "left": np.pi/2,
            "straight": 0,
            "right": -np.pi/2,
        }
        starting_roadblock = get_starting_segment(
            agent=self.initial_ego_state,
            map_api=self.map_api
        )[0].parent

        # greedily build the route by taking the outgoing edge that best matches the command
        current_roadblock: RoadBlockGraphEdgeMapObject = starting_roadblock
        route_roadblocks: List[RoadBlockGraphEdgeMapObject] = [current_roadblock]
        angle_for_command = angles_for_command[command]

        while len(route_roadblocks) < route_length and len(current_roadblock.outgoing_edges) > 0:
            # compare angle of each potential next roadblock to the command
            current_angle = current_roadblock.interior_edges[0].baseline_path.discrete_path[-1].heading
            successor_roadblock_angles = [
                principal_value(
                    rb.interior_edges[0].baseline_path.discrete_path[-1].heading - current_angle
                )
                for rb in current_roadblock.outgoing_edges
            ]
            idx = np.argmin(
                [
                    abs(angle_for_command-angle_of_lane)
                    for angle_of_lane in successor_roadblock_angles
                ]
            )

            # select roadblock which best matches the command
            current_roadblock = current_roadblock.outgoing_edges[idx]
            route_roadblocks.append(current_roadblock)

        # search lane-level route plan
        route_plan, path_found = self.search_route_to_goal(
            starting_roadblock=starting_roadblock,
            goal_lane=current_roadblock.interior_edges[-1],
            candidate_lane_edge_ids=[l.id for roadblock in route_roadblocks for l in roadblock.interior_edges],
            lengh_of_search=len(route_roadblocks)
        )
        assert path_found, f"Could not find a path for the command {command} provided for scenario {self.token}" 
        return  [r.id for r in route_roadblocks], route_plan        

    def _infer_route_plan_from_goal_location(self, goal_location: Point2D) -> List[str]:
        def _infer_goal_lane(map_api: AbstractMap, goal_location: Point2D) -> LaneGraphEdgeMapObject:
            # Find Roadblock that corresponds to the goal
            goal_lane = get_current_route_objects(self.map_api, goal_location)
            assert len(goal_lane) <= 1, \
                f"In scenario with token {self.token} the selected goal {goal_location} cannot be assigned to a single lane"
        
            if goal_lane == None:
                nearest_id, _ = map_api.get_distance_to_nearest_map_object(goal_location, SemanticMapLayer.LANE)
                goal_lane = [map_api.get_map_object(nearest_id, SemanticMapLayer.LANE)]

            return goal_lane[0]

        # Assign ego to a starting roadblock
        starting_roadblock = get_starting_segment(
            agent=self.initial_ego_state,
            map_api=self.map_api
        )[0].parent 

        # Infer the goal lane
        goal_lane = _infer_goal_lane(map_api=self.map_api, goal_location=goal_location)

        # Extract all nearby roadblocks and their lanes 
        proximal_roadblocks = self.map_api.get_proximal_map_objects(
            point=self.initial_ego_state.center,
            radius=300,
            layers=[SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
        )
        proximal_roadblocks = proximal_roadblocks[SemanticMapLayer.ROADBLOCK] + proximal_roadblocks[SemanticMapLayer.ROADBLOCK_CONNECTOR]
        candidate_lane_edge_ids = [
            item.id
            for roadblock in proximal_roadblocks
            for item in roadblock.interior_edges
        ]
        assert goal_lane.id in candidate_lane_edge_ids, f"Goal ({goal_location}) provided for scenario {self.token} is too far away."

        route_plan, path_found = self.search_route_to_goal(
            starting_roadblock=starting_roadblock,
            goal_lane=goal_lane,
            candidate_lane_edge_ids=candidate_lane_edge_ids,
            lengh_of_search=len(proximal_roadblocks)
        )

        assert path_found, f"Could not find a path to the goal {goal_location} provided for scenario {self.token}" 

        return list(set([edge.parent.id for edge in route_plan])), route_plan

    def _initialize_expert_route_plan(self) -> List[LaneGraphEdgeMapObject]:
        if "goal" in self.modification:
            assert self.modification["goal"] in self._map_modification_character_to_command.keys(), (
                f"The letter \"{self.modification['goal']}\" is not an option for goal. \
                Current options are: {self._map_modification_character_to_command.keys()}"
            )
            command = self._map_modification_character_to_command[self.modification["goal"]]
            if self.goal_location:
                return self._infer_route_plan_from_goal_location(
                    goal_location=self.goal_location
                )
            else:
                original_route_length = len(super().get_route_roadblock_ids())
                return self._infer_route_plan_from_command(
                    route_length=original_route_length,
                    command=command
                )
        else:
            # return original roadblocks
            expert_route: list([NuPlanLane]) = get_route(self.map_api, extract_ego_center(self.get_expert_ego_trajectory()))
            route_plan = [element[0] for element in get_route_simplified(expert_route)]
            return super().get_route_roadblock_ids(), route_plan

    @property
    def scenario_name(self) -> str:
        """Inherited, see superclass."""        
        return self.token + "-" + ModificationsSerializableDictionary(self.modification).to_string()
       