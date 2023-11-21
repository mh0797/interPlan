from __future__ import annotations
import numpy as np
from shapely import Point, LineString
from shapely.ops import split
from functools import cached_property
from pandas import Series

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from interplan.planning.scenario_builder.scenario_modifier.agents_modifier import AgentsModifier


from typing import Any, List, Optional, Tuple, Type, Dict

from nuplan.common.geometry.compute import principal_value
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.actor_state.ego_state import EgoState, StateSE2
from nuplan.common.actor_state.state_representation import Point2D

from nuplan.database.nuplan_db.nuplan_scenario_queries import ( 
    get_ego_state_for_lidarpc_token_from_db,
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
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from nuplan.common.maps.nuplan_map.polyline_map_object import NuPlanPolylineMapObject
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
        modification: Optional[ModificationsSerializableDictionary] = None,):

        super().__init__(
            data_root=data_root,
            log_file_load_path=log_file_load_path,
            initial_lidar_token=initial_lidar_token, 
            initial_lidar_timestamp=initial_lidar_timestamp, 
            scenario_type=scenario_type, 
            map_root=map_root, 
            map_version=map_version, 
            map_name=map_name, 
            scenario_extraction_info=scenario_extraction_info,
            ego_vehicle_parameters=ego_vehicle_parameters, 
            sensor_root=sensor_root
        )

        if isinstance(modification, dict):
            modification = ModificationsSerializableDictionary(modification)

        # Set modification details dictionary for this specific token, this may include information like 
        # the coordinates of the multiple goals that can be selected for the scenario or the special config for
        # special scenarios
        mod_details_dict = modification.dictionary.get("modification_details_dictionary")
        self.mod_details = mod_details_dict.get(initial_lidar_token) if mod_details_dict else None

        self._map_modification_character_to_command = {
            "l": "left",
            "r": "right",
            "s": "straight",
        }
        
        # If it is special scenario modify the variable modification, since special scenarios may include their own config parameters
        special_scenario_number = modification.dictionary.get("special_scenario")
        special_scenario_config = self.mod_details["special_scenario"][special_scenario_number].get("config") \
                                    if special_scenario_number else None
        if special_scenario_config:
            modification.reset_scenario_specifics()
            modification.add_scenario_specifics(special_scenario_config+f"s{special_scenario_number}")
        
        # Add modification attribute in case the scenario is modified this contains how the scenario should be change
        # Eg. if it contains "goal":"left" then the new goal of the scenario should go to the left
        self.modification: Dict = modification.dictionary 
        
        # Get goal location if necessary
        if "goal" in modification.dictionary:
            command = self._map_modification_character_to_command[modification.dictionary["goal"]]
            if self.mod_details and self.mod_details.get("goal") and self.mod_details["goal"].get(command):
                # lookup table contains a goal location for the current goal modification
                goal_coords = self.mod_details["goal"][command].split(",")
                self.goal_location = Point2D(x=goal_coords[0], y=goal_coords[1])
            else: 
                self.goal_location = None
        else: 
            self.goal_location = None

        # Initialize agent modifier if necessary
        if modification.augment_agents():
            self.agents_modifier = AgentsModifier(modification.dictionary, 
                self.map_api, 
                self._log_file, 
                self._lidarpc_tokens, 
                self.mod_details
            )
            _, self.modified_initial_ego_speed = self._get_initial_tracked_objects_and_ego_speed()
        else: 
            self.agents_modifier = None
            self.modified_initial_ego_speed = None

        self.expert_route_roadblock_ids, self.expert_route_lane_sequence = self._initialize_expert_route_plan()

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
            return DetectionsTracks(
                self.agents_modifier.get_tracked_objects_at_iteration(iteration)
            )
        else:
            return DetectionsTracks(
            extract_tracked_objects(self._lidarpc_tokens[iteration], self._log_file, future_trajectory_sampling)
            )
    
    def _get_initial_tracked_objects_and_ego_speed(self):
        # Get initial Iteration
        # Agent_modifier defines ego speed according to position among spawned agents
        tracked_objects, modified_ego_speed = self.agents_modifier.get_initial_tracked_objects_and_ego_speed()
        return DetectionsTracks(tracked_objects), modified_ego_speed
    
    
    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        if "goal" in self.modification:
            if iteration == 0: 
                initial_ego_state = get_ego_state_for_lidarpc_token_from_db(self._log_file, self._lidarpc_tokens[iteration])
                # Set the new speed that ego should have if it spawn among new spawned agents
                if self.modified_initial_ego_speed: 
                    initial_ego_state.dynamic_car_state.rear_axle_velocity_2d.x = self.modified_initial_ego_speed
                return initial_ego_state
            
            # So that iteration = 0 -> index = 0 ... iteration = self.get_number_of_iterations() -> index = len(modified_expert_trajectory)
            index = int((len(self.get_modified_expert_trajectory)/self.get_number_of_iterations()) * iteration)
            return EgoState.build_from_center(self.get_modified_expert_trajectory[index],
                self.initial_ego_state.dynamic_car_state.center_velocity_2d,
                self.initial_ego_state.dynamic_car_state.center_acceleration_2d,
                self.initial_ego_state.tire_steering_angle,
                self.initial_ego_state.time_point,
                self.initial_ego_state.car_footprint.vehicle_parameters    
            )     
        else:
            return super().get_ego_state_at_iteration(iteration)

    @cached_property
    def get_modified_expert_trajectory(self) -> List[EgoState]:

        modified_expert_trajectory = []
        initial_ego_state = self.initial_ego_state

        for edge in self.expert_route_lane_sequence:
            # Shorten the initial edge so that it starts from ego intial location
            if edge.id == self.expert_route_lane_sequence[0].id:
                ego_pose_along_line = edge.baseline_path.get_nearest_pose_from_position(self.initial_ego_state.center)
                geometry_collection  = split(edge.baseline_path.linestring, Point(*ego_pose_along_line).buffer(0.0001))
                shortened_linestring = geometry_collection.geoms[-1]
                edge = NuPlanPolylineMapObject(Series({"geometry":shortened_linestring,"fid":"7210"}))
                modified_expert_trajectory.extend(edge.discrete_path)
                continue
            
            modified_expert_trajectory.extend(edge.baseline_path.discrete_path)
        return modified_expert_trajectory

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
            return self.expert_route_lane_sequence[-1].baseline_path.get_nearest_pose_from_position(self.goal_location)
        else:
            return self.expert_route_lane_sequence[-1].baseline_path.discrete_path[-1]

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
        
            if not goal_lane:
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

        return list(dict.fromkeys([edge.parent.id for edge in route_plan])), route_plan

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
       