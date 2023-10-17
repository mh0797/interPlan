from __future__ import annotations
import numpy as np
import json
from functools import cached_property

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from interPlan.planning.scenario_builder.scenario_modifier.agents_modifier import AgentsModifier


from typing import Any, List, Optional, Tuple, Type, Dict, Union, cast, Generator

from nuplan.common.geometry.compute import principal_value
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.actor_state.ego_state import EgoState, StateSE2


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
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from nuplan.common.maps.nuplan_map.lane_connector import NuPlanLaneConnector
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.path.utils import convert_se2_path_to_progress_path
from interPlan.planning.simulation.planner.utils.breadth_first_search_lane_goal import BreadthFirstSearch # TODO new name for this



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

        self._modification = modification # If scenario is modified
        lookup_table = json.load(open(self._modification["lookup_table_path"]))
        try: self.lookup_table = [elem for elem in lookup_table if elem["token"] == self.token][0]
        except: self.lookup_table = None

        self.initial_tracked_objects

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
                self._modification
            ),
        )
    
    def get_tracked_objects_at_iteration(
        self,
        iteration: int,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations(), f"Iteration is out of scenario: {iteration}!"
        
        if "density" in self._modification or "amount_of_agents" in self._modification:
            # if there is modification use agents_modifier
            if iteration == 0:
                tracked_objects = extract_tracked_objects(self._lidarpc_tokens[0], self._log_file, future_trajectory_sampling)  
                agents_modifier = AgentsModifier(tracked_objects, self._modification, self.map_api, 
                                                self._log_file, self._lidarpc_tokens[0], self.lookup_table) #TODO
                self.ego_speed = agents_modifier.ego_speed 
                return DetectionsTracks(TrackedObjects(agents_modifier.tracked_objects))
            else:
                return DetectionsTracks(AgentsModifier.delete_objects(
                extract_tracked_objects(self._lidarpc_tokens[iteration], self._log_file, future_trajectory_sampling), delete_pedestrians=True))
        else:
            return DetectionsTracks(
            extract_tracked_objects(self._lidarpc_tokens[iteration], self._log_file, future_trajectory_sampling)
            )
    
    def get_past_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """ Inherited, see superclass. """
        # TODO: This can be made even more efficient with a batch query
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            tracked_objects = extract_tracked_objects(lidar_pc.token, self._log_file, future_trajectory_sampling)
            #agents_modifier = AgentsModifier(tracked_objects, self._modification, self.map_api, self._log_file, lidar_pc.token)
            #TODO
            yield DetectionsTracks(tracked_objects)
    
    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        if "goal" in self._modification:
            if iteration == 0: 
                initial_ego_state = get_ego_state_for_lidarpc_token_from_db(self._log_file, self._lidarpc_tokens[iteration])
                if hasattr(self, "ego_speed"): initial_ego_state.dynamic_car_state.rear_axle_velocity_2d.x = self.ego_speed
                return initial_ego_state
            if not hasattr(self,"route_plan"): self.get_route_roadblock_ids()
            return self.get_modified_expert_trayectory[int((len(self.get_modified_expert_trayectory)/self.get_number_of_iterations()) * iteration)]
            # So that iteration = 0 -> index = 0 ... iteration = self.get_number_of_iterations() -> index = len(modified_expert_trayectory)

        return get_ego_state_for_lidarpc_token_from_db(self._log_file, self._lidarpc_tokens[iteration])

    @cached_property
    def get_modified_expert_trayectory(self) -> List[EgoState]:
        modified_expert_trayectory = []
        continue_bool = True
        initial_ego_state = self.initial_ego_state
        for edge in self.route_plan:
            progress_baseline_path = convert_se2_path_to_progress_path(edge.baseline_path.discrete_path)
            if edge == self.route_plan[0]:
                new_initial_state = min(progress_baseline_path, key= lambda state: state.distance_to(initial_ego_state.center))
            for state in progress_baseline_path:
                # Skip states that are behind ego 
                if edge.id == self.route_plan[0].id and state == new_initial_state:
                    continue_bool = False
                if continue_bool: continue
                modified_expert_trayectory.append(EgoState.build_from_center(state, initial_ego_state.dynamic_car_state.center_velocity_2d,
                                                                                    initial_ego_state.dynamic_car_state.center_acceleration_2d,
                                                                                    initial_ego_state.tire_steering_angle,
                                                                                    initial_ego_state.time_point,
                                                                                    initial_ego_state.car_footprint.vehicle_parameters))
        return modified_expert_trayectory

    def get_route_roadblock_ids(self) -> List[str]:
        return self._get_route_roadblocks_ids
    
    @cached_property
    def _get_route_roadblocks_ids(self) -> List[str]:
        """Inherited, see superclass."""

        roadblock_ids = get_roadblock_ids_for_lidarpc_token_from_db(self._log_file, self._initial_lidar_token)
        assert roadblock_ids is not None, "Unable to find Roadblock ids for current scenario"
        roadblock_ids = cast(List[str], roadblock_ids)

        # Is there goal modification?
        if "goal" in self._modification:
            # Goal direction
            if self._modification["goal"] == "l":   direction = "left"
            elif self._modification["goal"] == "r": direction = "right"
            elif self._modification["goal"] == "s": direction = "straight"
            else: raise ValueError(f"The letter \"{self._modification['goal']}\" is not an opcion for goal. Current options are: l, r, s")

            # Goal in lookup table?
            if self.lookup_table: 
                self.goal_location = self.lookup_table["goal"][direction] # TODO handle error no direction
            else: self.goal_location = None

            # In case there is already a goal location, convert to float
            if self.goal_location: self.goal_location = [float(number) for number in self.goal_location.split(",")] # TODO simplify
        else:
            self.goal_location = None
            return roadblock_ids
        
        if self.goal_location:

            proximal_roadblocks = self.map_api.get_proximal_map_objects(self.initial_ego_state.center, 300,
                                                                [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR])
            proximal_roadblocks = proximal_roadblocks[SemanticMapLayer.ROADBLOCK] + proximal_roadblocks[SemanticMapLayer.ROADBLOCK_CONNECTOR]
            candidate_lane_edge_ids = []
            find_starting_roadblock = True
            for roadblock in proximal_roadblocks:
                interior_edges = roadblock.interior_edges
                # Adding lanes to candidate lanes (every lane near ego)
                candidate_lane_edge_ids.extend([interior_edge.id for interior_edge in interior_edges])
                # Finding the starting roadblock    
                if find_starting_roadblock and roadblock.contains_point(self.initial_ego_state.center): 
                    starting_roadblock = roadblock
                    find_starting_roadblock = False

            assert starting_roadblock, "Starting roadblock could not be found!"

            # Find Roadblock that corresponds to the goal
            try:
                goal_lane = self.map_api.get_one_map_object(StateSE2(self.goal_location[0], self.goal_location[1], 0),
                                                                SemanticMapLayer.LANE) or \
                                self.map_api.get_one_map_object(StateSE2(self.goal_location[0], self.goal_location[1], 0),
                                                                SemanticMapLayer.LANE_CONNECTOR)
                if goal_lane == None:
                    nearest_id, _ = self.map_api.get_distance_to_nearest_map_object(StateSE2(self.goal_location[0], self.goal_location[1], 0),
                                                                                SemanticMapLayer.LANE)
                    goal_lane = self.map_api.get_map_object(nearest_id, SemanticMapLayer.LANE)
                    
            except AssertionError:
                raise AssertionError(f"In scenario with token {self.token} the selected {direction} goal is located in a intersection"+
                                     " where multiple roadblocks meet")
            
            self.route_plan, path_found = self.search_route_to_goal(starting_roadblock, goal_lane, candidate_lane_edge_ids, len(roadblock_ids))
            
            assert path_found, "Path to the goal not found" 
            #TODO make so that it is able to find a path for the cases where the edge to follow the route is paralel to the edge where ego is (LC)
           
            roadblock_ids = []
            for edge in self.route_plan:
                if edge.parent.id not in roadblock_ids:
                    roadblock_ids.append(edge.parent.id)

            self._roadblock_ids = roadblock_ids
        else:
            for roadblock in roadblock_ids:
                current_roadblock = self.map_api.get_map_object(roadblock, SemanticMapLayer.ROADBLOCK) or \
                                    self.map_api.get_map_object(roadblock, SemanticMapLayer.ROADBLOCK_CONNECTOR)
                initial_roadblock = current_roadblock
                if current_roadblock.contains_point(self.initial_ego_state.center): break
        
            new_roadblocks_ids = [current_roadblock.id]
            candidate_lane_edge_ids = []

            for _ in range(len(roadblock_ids)):
                candidate_lane_edge_ids.extend([lane.id for lane in current_roadblock.interior_edges])
                edges = current_roadblock.outgoing_edges
                if not edges: break
                angles = []
                for edge in edges:
                    angle = principal_value(edge.interior_edges[0].baseline_path.discrete_path[-1].heading)
                    angles.append(angle)

                if self._modification["goal"] == "l":   idx = np.argmin([np.pi - abs(angle) for angle in angles])
                elif self._modification["goal"] == "r": idx = np.argmin([abs(angle) for angle in angles])
                elif self._modification["goal"] == "s": idx = np.argmin([(np.pi/2) - abs(angle) for angle in angles])
                else: raise ValueError(f"The letter \"{self._modification['goal']}\" is not an opcion for goal. Current options are: l, r, s")

                current_roadblock = edges[idx]
                new_roadblocks_ids.append(current_roadblock.id)
            
            self.route_plan, _ = self.search_route_to_goal(initial_roadblock, current_roadblock.interior_edges[0], 
                                                           candidate_lane_edge_ids, len(new_roadblocks_ids))
            self._roadblock_ids = new_roadblocks_ids

        return self._roadblock_ids
    
    def search_route_to_goal(self, starting_roadblock: RoadBlockGraphEdgeMapObject, goal_lane: LaneGraphEdgeMapObject, 
                             candidate_lane_edge_ids: List[str], lengh_of_search: int):

        """ # Delete other the lanes of goal roadblock that don't correspond to goal lane
        lanes_ids_to_delete = [lane.id for lane in goal_lane.parent.interior_edges if lane.id != goal_lane.id]
        candidate_lane_edge_ids = [id for id in candidate_lane_edge_ids if id not in lanes_ids_to_delete] """

        # Search for route from multiple starting edges to account for lane change
        closest_distance_to_goal = 1000000 # Distance to find edge closest to goal
        for start_edge in starting_roadblock.interior_edges:
            # Create graph search to find route
            graph_search = BreadthFirstSearch(start_edge, candidate_lane_edge_ids)
            """ # Modify check goal condition method so that finding the goal counts as an end condition
            graph_search._check_goal_condition = _check_goal_condition.__get__(graph_search, BreadthFirstSearch) """
            # Search for the route
            _route_plan, path_found, lane_change = graph_search.search(goal_lane, lengh_of_search)

            if path_found and not lane_change: 
                route_plan = _route_plan
                break
            elif path_found: 
                route_plan = _route_plan
        else: 
            if "route_plan" not in locals(): route_plan = _route_plan
                    
        return route_plan, path_found

    def get_mission_goal(self) -> Optional[StateSE2]:
        
        if not hasattr(self, "goal_location"):
            self.get_route_roadblock_ids()

        if "goal" not in self._modification.keys():
            """Inherited, see superclass."""
            return get_mission_goal_for_sensor_data_token_from_db(
                self._log_file, get_lidarpc_sensor_data(), self._initial_lidar_token
            )
        elif self.goal_location:
            goal_lane = self.map_api.get_one_map_object(StateSE2(self.goal_location[0], self.goal_location[1], 0),
                                                    SemanticMapLayer.LANE) or \
                        self.map_api.get_one_map_object(StateSE2(self.goal_location[0], self.goal_location[1], 0),
                                                    SemanticMapLayer.LANE_CONNECTOR)
            
            if not goal_lane:
                nearest_id, _ = self.map_api.get_distance_to_nearest_map_object(StateSE2(self.goal_location[0], self.goal_location[1], 0),
                                                                            SemanticMapLayer.LANE)
                goal_lane = self.map_api.get_map_object(nearest_id, SemanticMapLayer.LANE)

            return goal_lane.baseline_path.get_nearest_pose_from_position(StateSE2(self.goal_location[0], self.goal_location[1], 0))
        else:
            last_roadblock_of_route = self.map_api.get_map_object(self.get_route_roadblock_ids()[-1], SemanticMapLayer.ROADBLOCK) or \
                                    self.map_api.get_map_object(self.get_route_roadblock_ids()[-1], SemanticMapLayer.ROADBLOCK_CONNECTOR)
            goal = last_roadblock_of_route.interior_edges[0].baseline_path.discrete_path[-1]
            return goal
        

    @property
    def scenario_name(self) -> str:
        """Inherited, see superclass."""
        string = ""
        if "amount_of_agents" in self._modification: string+=f"a{self._modification['amount_of_agents']}"
        if "density" in self._modification: string+=f"d{self._modification['density']}"
        if "goal" in self._modification: string+=f"g{self._modification['goal']}"
        
        return self.token+"-"+string
    
    @cached_property
    def initial_tracked_objects(self) -> DetectionsTracks:
        """
        Get initial tracked objects
        :return: DetectionsTracks.
        """
        return self.get_tracked_objects_at_iteration(0)

def _check_goal_condition(
        self,
        current_edge: LaneGraphEdgeMapObject,
        target_roadblock: RoadBlockGraphEdgeMapObject,
        depth: int,
        target_depth: int,
    ) -> bool:
        """ Function to Modify the method of the same name in the class BreadthFirstSearch"""
        return current_edge.get_roadblock_id() == target_roadblock.id 

# TODO move to custom util
def get_lane(location: StateSE2, map_api: AbstractMap) -> LaneGraphEdgeMapObject:
    """ 
    Util function to return either Lane or Lane connector
    """
    lane =  map_api.get_one_map_object(location, SemanticMapLayer.LANE) or \
            map_api.get_one_map_object(location, SemanticMapLayer.LANE_CONNECTOR)

    return lane