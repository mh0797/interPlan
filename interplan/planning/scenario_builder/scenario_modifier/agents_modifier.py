from __future__ import annotations
import random
import math
import warnings
from typing import Any, List, Optional, Tuple, Type, Dict, Union
from collections import deque
import numpy as np
from nuplan.common.actor_state.agent_state import AgentState
from shapely import line_merge, MultiLineString, LineString
from shapely.geometry.base import CAP_STYLE
from shapely.geometry import Point
import logging
from enum import Enum


from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from nuplan.common.maps.nuplan_map.polyline_map_object import NuPlanPolylineMapObject
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects, TrackedObjectType
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.ego_state import EgoState

from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMap
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory
from nuplan.planning.simulation.path.interpolated_path import InterpolatedPath
from nuplan.planning.simulation.path.utils import convert_se2_path_to_progress_path
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring
from nuplan.planning.metrics.utils.route_extractor import get_current_route_objects
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import extract_tracked_objects
from nuplan.planning.simulation.observation.idm.idm_agents_builder import get_starting_segment
from interplan.planning.utils.agent_utils import get_agent_constant_velocity_path



from nuplan.database.nuplan_db.nuplan_scenario_queries import get_ego_state_for_lidarpc_token_from_db

logger = logging.getLogger(__name__)


class Behavior(Enum):
    # Class to define behavior or the agents
    DEFAULT = 1
    STOPPED = 2
    CAUTIOUS = 3 # Regardless of the IDM config of the observation, this means that if ego and this agent's paths intersects,
                 # this agent will do a hard break

class Type_of_Vehicle(Enum):
    DEFAULT = 1 # Any vehicle
    BUS = 2 # A long vehicle

class interplanPolylineMapObject(NuPlanPolylineMapObject):
    def __init__(self, polyline: LineString):
        self._polyline = polyline
        self._distance_for_curvature_estimation = 2
        self._distance_for_heading_estimation = 0.5

    def get_nearest_pose_from_progress(self, progress: float) -> StateSE2:
        return self.get_nearest_pose_from_position(self._polyline.line_interpolate_point(progress))

class AgentsModifier():

    def __init__(self,
                modification: dict, 
                map_api: AbstractMap,
                log_file: str,
                token_list: str,
                lookup_table: dict = {},
                ego_route_plan: list = [],
                future_trajectory_sampling: bool = False
                ) -> None:

        if not modification:
            logger.warning("Modification parameter is empty. If no modification is desired"+
                           " please don't use the nuplan modifications scenario builder")
            self.ego_state: EgoState = get_ego_state_for_lidarpc_token_from_db(log_file, token_list[0])
            return 

        random.seed(token_list[0])
        self.token_list = token_list
        self.log_file = log_file
        self.future_trajectory_sampling = future_trajectory_sampling
        self.modification = modification
        self.lookup_table = lookup_table
        self.map_api = map_api
        self.ego_state: EgoState = get_ego_state_for_lidarpc_token_from_db(log_file, token_list[0])
        self.ego_speed = self.ego_state.dynamic_car_state.speed
        self.ego_lane = self.get_ego_lane()
        self.ego_route_plan = ego_route_plan
        self.dmax = modification["decel_max"] if "decel_max" in modification else 2
        self.acomf = modification["accel_max"] if "decel_max" in modification else 1
        self.deleted_agents =  []
        first_tracked_objects = self.get_tracked_objects_from_db_at_iteration(0)
        self.max_track_id = max(first_tracked_objects, key= lambda x: x.metadata.track_id).metadata.track_id if first_tracked_objects else 1
        self.pedestrian_paths = []
        self.cones = []
        self.example_agent = self.get_example_agent()
        self.special_scenario = self.modification["special_scenario"] if "special_scenario" in self.modification else None

    def get_tracked_objects_from_db_at_iteration(self, iteration: int) -> TrackedObjects:

        return extract_tracked_objects(self.token_list[iteration], self.log_file, self.future_trajectory_sampling)
    
    def get_example_agent(self):
        """ Get agent which will serve as a blueprint to spawn agents when there are no deleted agents """

        for it in range(0, 1000):
            tracked_objects = self.get_tracked_objects_from_db_at_iteration(it)
            example_agent = next(iter(tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)), None)
            if example_agent: return example_agent
        
        AssertionError("No Agents in this log file")

    def get_ego_lane(self):
        # Get current route objects
        current_route_objects = get_current_route_objects(self.map_api, self.ego_state.center.point) 
        # Assert that ego is in a lane /-connector
        assert current_route_objects, "Ego is not currently in any drivable surface"
        # If there are multiple lanes under ego, return the one on route
        if len(current_route_objects) == 1: return current_route_objects[0]
        elif hasattr(self, "ego_route_plan"): 
            return [edge for edge in current_route_objects if edge.id in [edge.id for edge in self.ego_route_plan]][0]        
        else: return current_route_objects[0]

    def get_extended_occupancy_map(self) -> STRTreeOccupancyMap:

        map = STRTreeOccupancyMap({})
        for obj in self.tracked_objects:
            if isinstance(obj, Agent):
                if not isinstance(obj, ModifiedAgent):
                    # Snap to baseline
                    route, _ = get_starting_segment(obj, self.map_api)
                    if not route: continue
                    state_on_path = route.baseline_path.get_nearest_pose_from_position(obj.center.point)
                    box_on_baseline = OrientedBox.from_new_pose(
                        obj.box, StateSE2(state_on_path.x, state_on_path.y, state_on_path.heading))  

                    # Create agent with the new box
                    obj = ModifiedAgent.from_new_oriented_box(obj, box_on_baseline)
                # Insert in Map
                map.insert(obj.track_token, path_to_linestring(obj.get_path_to_go(2)).buffer((obj._box.width / 2), cap_style=CAP_STYLE.flat))
            else: 
                map.insert(obj.track_token, obj._box.geometry)
        return map

    def delete_percentage_of_agents(self, percentage):
        tracked_agents = self.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)

        amount_to_delete= int((1.0 - percentage)*len(tracked_agents))
        self.deleted_agents =  random.sample(tracked_agents, amount_to_delete)
        # Filter deleted agents
        self.tracked_objects = TrackedObjects(list(filter(lambda object: object not in self.deleted_agents, self.tracked_objects.tracked_objects)))
        
    def delete_objects(self, delete_pedestrians: bool = False, delete_static_objects: bool = True ) -> TrackedObjects: 
        pedestrians = self.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.PEDESTRIAN)
        vehicles = self.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
        static_objects = [object for object in self.tracked_objects if object not in pedestrians and object not in vehicles]
        objects_to_delete = [object for object in self.tracked_objects if (delete_pedestrians and object in pedestrians)
                                                                    or (delete_static_objects and object in static_objects)]
        self.tracked_objects = TrackedObjects(list(filter(lambda object: object not in objects_to_delete, self.tracked_objects.tracked_objects)))
    
    def spawn_agents(self, density) -> tuple[TrackedObjects, float]:
        """ 
        Spawn agents in pseudo random locations

        """
        locations_speed_to_spawn = [] # Pairs of location / speed to spawn agents
        if density == "h": max_distance_between_agents = int(100/3)
        elif density == "m": max_distance_between_agents = int(100/2)
        elif density == "l": max_distance_between_agents = int(100)
        else: AssertionError("Not a valid value for density")   
        
        lanes_to_spawn = [lane for lane in self.ego_lane.parent.interior_edges]

        # Extend the lanes to spawn agents not only to ego but also to the lanes pointed at in the lookup table
        if self.lookup_table and "roadblock_locations_to_spawn_agents" in self.lookup_table: 
            if not self.special_scenario:
                locations = self.lookup_table["roadblock_locations_to_spawn_agents"] 
            else:
                if "roadblock_locations_to_spawn_agents" in self.lookup_table["special_scenario"][self.special_scenario]:
                    locations = self.lookup_table["special_scenario"][self.special_scenario]["roadblock_locations_to_spawn_agents"]
                else:
                    locations = self.lookup_table["roadblock_locations_to_spawn_agents"] 
            for location in locations:
                location = Point(*location)
                roadblock = self.map_api.get_one_map_object(location, SemanticMapLayer.ROADBLOCK) \
                            or self.map_api.get_one_map_object(location, SemanticMapLayer.ROADBLOCK_CONNECTOR)      
                lanes_to_spawn.extend(roadblock.interior_edges)
        
        for lane in lanes_to_spawn:

            list_of_lanes = self.extend_current_lane(lane) # List of succesive lanes with current lane in center
            polyline = interplanPolylineMapObject(line_merge(MultiLineString([lane.baseline_path._polyline for lane in list_of_lanes])))
            ego_progress = polyline._polyline.project(Point(*self.ego_state.center.serialize()))

            position = 0
            progress_along_line = polyline.length
            while progress_along_line > 0:

                # Get a new random distance to leading car
                distance_to_leading_car = random.randrange(10, max_distance_between_agents)

                progress_along_line -= distance_to_leading_car

                if lane.id == self.ego_lane.id and abs(progress_along_line - ego_progress) < 10:
                    distance_to_leading_car += progress_along_line - ego_progress
                    progress_along_line = ego_progress

                # Decide speed 
                if position == 0:
                    # If it is the first car and the next roadblock is an intersection then speed enough to reduce it
                    if (len(self.ego_lane.parent.outgoing_edges) == 1):
                        speed = lane.speed_limit_mps or 10
                    else:
                        speed = math.sqrt((2/3)*abs(polyline.length - progress_along_line)*self.acomf)
                else:
                    reaction_time = max(random.gauss(0.25,0.05),0.1)
                    alfa = (3/(2*self.dmax))
                    speed_of_leading_car = speed
                    speed = (-reaction_time + math.sqrt(reaction_time**2 + 4*alfa*(alfa*(speed_of_leading_car**2) \
                                                                                +distance_to_leading_car)))/(2*alfa)  
                    speed = min(distance_to_leading_car/2, speed)      
                speed = speed + random.gauss(0,1)
                if lane.speed_limit_mps: speed = min(speed, lane.speed_limit_mps)
                speed = max(speed, 0)

                # Create list of location, speed pairs to spawn agents
                if progress_along_line - ego_progress == 0:
                    self.ego_speed = speed
                else:
                    locations_speed_to_spawn.append([polyline.get_nearest_pose_from_progress(progress_along_line), speed])

                position += 1
            
        for location, speed in locations_speed_to_spawn:
            self.add_agent_in_location(location, speed)

        # Add extra agents in lookup table to list to spawn
        if self.lookup_table and "extra_agents_to_spawn" in self.lookup_table:
            for agent in self.lookup_table["extra_agents_to_spawn"]:
                if len(agent) > 3 and "snap" in agent[3] and agent[3]["snap"]: # Snap to route 
                    lane = get_current_route_objects(self.map_api, StateSE2(*agent[0:2], 0))[0]
                    assert lane, "Trying to spawn an agent outside a road"
                    location = lane.baseline_path.get_nearest_pose_from_position(StateSE2(*agent[0:2],0))
                else: location = StateSE2(*agent[0:2], 0)
                agent_behavior = Behavior.DEFAULT if len(agent) > 3 and agent[3]["behavior"] == "default" else Behavior.CAUTIOUS
                self.add_agent_in_location(location, agent[2], behavior=agent_behavior)

    def extend_current_lane(self, lane) -> list(NuPlanLane):
        """ Returns a list of succesive lanes which expand 50 meters front and backward from ego position projection in the lane"""

        return self._extend_current_lane(lane, forward=False) + self._extend_current_lane(lane)[1:]

    def _extend_current_lane(self, lane: LaneGraphEdgeMapObject, distance = 50, forward: bool = True) -> List:

        """ Returns a list of succesive lanes which expand 50 meters front or backward from ego position projection in the lane"""

        ego_progress = lane.baseline_path.linestring.project(Point(*self.ego_state.center.point.array))
        length_extended = lane.baseline_path.length - ego_progress if forward else ego_progress
        extended_lanes: List[LaneGraphEdgeMapObject] = [lane]

        while length_extended < distance:
            
            candidate_edges = extended_lanes[-1].outgoing_edges if forward else extended_lanes[0].incoming_edges
            selected_candidate_edges = []

            # Intersection handling / it extend until traffic light intersection
            for edge in candidate_edges:
                if edge.has_traffic_lights():
                    break
                else:
                    selected_candidate_edges.append(edge)
            if not selected_candidate_edges:
                break

            # Select edge with the lowest curvature (prefer going straight)
            curvatures = [abs(edge.baseline_path.get_curvature_at_arc_length(0.0)) for edge in selected_candidate_edges]
            new_segment: LaneGraphEdgeMapObject = selected_candidate_edges[np.argmin(curvatures)]
        
            length_extended += new_segment.baseline_path.length
            extended_lanes.insert(len(extended_lanes) if forward else 0, new_segment)

        return extended_lanes

    def add_agent_in_location(self, 
                              location: StateSE2, 
                              speed: float = None, 
                              behavior: Behavior = Behavior.DEFAULT, 
                              type: Type_of_Vehicle = Type_of_Vehicle.DEFAULT) -> None:
    
        # Prefer to spawn agents who where previously deleted
        if self.deleted_agents:
            if type == type.BUS:
                oriented_box = OrientedBox(location, 7.455735206604004, 2.0785977840423584, 1.951879620552063)
                agent = ModifiedAgent.from_new_oriented_box(self.deleted_agents[0], oriented_box)
            else: 
                agent = ModifiedAgent.from_new_pose(self.deleted_agents[0], location) 
            self.deleted_agents.pop(0)
        else:
            if type == type.BUS:
                oriented_box = OrientedBox(location, 7.455735206604004, 2.0785977840423584, 1.951879620552063)
                agent = ModifiedAgent.from_new_oriented_box(self.example_agent, oriented_box)
            else:
                agent = ModifiedAgent.from_new_pose(self.example_agent, location)
            # It's almost improbable that random returns a token which an agent already has
            agent._metadata = SceneObjectMetadata(
                            self.ego_state.scene_object_metadata.timestamp_us,
                            f"{random.randrange(16**16):=16x}",
                            self.max_track_id + 1,
                            f"{random.randrange(16**16):=16x}",
                            "vehicle"
                            )
            self.max_track_id += 1
        
        agent._velocity = self.ego_state.dynamic_car_state.rear_axle_velocity_2d if speed else StateVector2D(speed, 0)
        agent.behavior = behavior
        agent: ModifiedAgent

        # Check for agents that intersects THIS agent's path
        agent_path = path_to_linestring(agent.get_path_to_go(3))
        intersecting_agents = self.get_extended_occupancy_map().intersects(agent_path.buffer((agent._box.width / 2), cap_style=CAP_STYLE.flat))

        if intersecting_agents.size > 0: 
            if behavior != Behavior.STOPPED:
                return
            else:
                # If we are spawning a stopped vehicle over a already spawn vehicle, the other vehicle gets deleted
                for agent_id in intersecting_agents.get_all_ids():
                    self.tracked_objects = TrackedObjects([obj for obj in self.tracked_objects.tracked_objects if obj.track_token != agent_id])
        
        self.tracked_objects.tracked_objects.append(agent)

    def spawn_agents_for_specific_scenarios(self, scenario):

        if scenario == "double_side_crash" :
            searching_for_back = True # We first search for a location behind the ego vehicle since baselinepath starts from behind
            for pose in self.ego_lane.baseline_path.discrete_path:
                if pose.distance_to(self.ego_state.center) > 20:
                    if searching_for_back:
                        back: StateSE2 = pose 
                    else: 
                        front: StateSE2 = pose
                        break
                else: searching_for_back = False

            if "back" in locals(): self.add_agent_in_location(back, "fast")
            else: print("Could not Spawn agent at the back of ego vehicle")
            if "front" in locals(): self.add_agent_in_location(front, "slow")
            else: print("Could not Spawn agent at the front of ego vehicle")

    def spawn_agents_for_special_scenarios(self):
        
        if "stopped_vehicle" in self.lookup_table["special_scenario"][self.special_scenario]:
            for vehicle in self.lookup_table["special_scenario"][self.special_scenario]["stopped_vehicle"]:
                location_to_spawn = StateSE2(*vehicle)
                self.add_agent_in_location(location_to_spawn, 0, Behavior.STOPPED)
        if "stopped_bus" in self.lookup_table["special_scenario"][self.special_scenario]:
            for vehicle in self.lookup_table["special_scenario"][self.special_scenario]["stopped_bus"]:
                location_to_spawn = StateSE2(*vehicle)
                self.add_agent_in_location(location_to_spawn, 0, Behavior.STOPPED, type = Type_of_Vehicle.BUS)
        if "pedestrian" in self.lookup_table["special_scenario"][self.special_scenario] and not self.pedestrian_paths:
            for pedestrian in self.lookup_table["special_scenario"][self.special_scenario]["pedestrian"]:
                self.pedestrian_paths.append([
                    InterpolatedPath(convert_se2_path_to_progress_path([StateSE2(*pedestrian[0:2], 0), StateSE2(*pedestrian[2:-1], 0)])),
                    SceneObjectMetadata(timestamp_us=1624917811949521, 
                                    token=f"{random.randrange(16**16):=16x}", 
                                    track_id=random.randint(100,1000), 
                                    track_token=f"{random.randrange(16**16):=16x}", 
                                    category_name='pedestrian'),
                    pedestrian[4]
                                            ]) 
        self.add_pedestrians_at_iteration(0)
        if "cones" in self.lookup_table["special_scenario"][self.special_scenario] and not self.cones:
            for cone in self.lookup_table["special_scenario"][self.special_scenario]["cones"]:
                self.cones.append([StateSE2(*cone), SceneObjectMetadata(timestamp_us=1624917811949521, 
                                                    token=f"{random.randrange(16**16):=16x}", 
                                                    track_id=random.randint(100,1000), 
                                                    track_token=f"{random.randrange(16**16):=16x}", 
                                                    category_name='pedestrian')])
        self.add_cones()

    def get_tracked_objects_at_iteration(self, iteration: int) -> TrackedObjects: 

        if iteration == 0:
            self.tracked_objects = self.get_tracked_objects_from_db_at_iteration(0)
            # Delete Pedestrians and static objects
            self.delete_objects(delete_pedestrians=True)
            # Filter agents by percentage
            if "amount_of_agents" in self.modification.keys(): 
                self.delete_percentage_of_agents(self.modification["amount_of_agents"])
            # Spawn agents according to density
            elif "density" in self.modification.keys(): 
                self.delete_percentage_of_agents(0)
                self.spawn_agents(density=self.modification["density"])
            # Spawn agents in a specific way 
            elif "specific_scenario" in self.modification.keys(): self.spawn_agents_for_specific_scenarios(self.modification["specific_scenario"])
                                                                                                    #TODO no input
            if "special_scenario" in self.modification.keys(): self.spawn_agents_for_special_scenarios()

            return self.tracked_objects

        # Get Objects from DB
        self.tracked_objects = self.get_tracked_objects_from_db_at_iteration(iteration)
        # Delete Pedestrians from DB
        self.delete_objects(delete_pedestrians=True)
        # Add pedestrians in case it is a special scenario
        self.add_pedestrians_at_iteration(iteration)
        # Add cones in case it is a special scenario
        self.add_cones()

        return TrackedObjects(self.tracked_objects.tracked_objects)
    
    def add_pedestrians_at_iteration(self, iteration):
                                        
        path: InterpolatedPath
        for path, metadata, delay in self.pedestrian_paths:
            progress = path.get_end_progress()*(max(0, iteration - delay)/(150 - delay))
            pedestrian_location = StateSE2(*path.get_state_at_progress(progress).serialize()) 
            self.tracked_objects.tracked_objects.append(Agent(
                TrackedObjectType.PEDESTRIAN, 
                OrientedBox(pedestrian_location, 0.9000033778444835, 0.7913900336773054, 1.789137980584556), 
                StateVector2D(0,0), metadata
                ))
    
    def add_cones(self):
                      
        for location, metadata in self.cones:
            self.tracked_objects.tracked_objects.append(StaticObject(TrackedObjectType.GENERIC_OBJECT, 
                                                                     OrientedBox(location, 0.5, 0.5, 0.5), 
                                                                     metadata
                                                                    )
                                                        )


class ModifiedAgent(Agent):
    def __init__(
        self, tracked_object_type: TrackedObjectType, oriented_box: OrientedBox, 
        velocity: StateVector2D, metadata: SceneObjectMetadata, angular_velocity: float | None = None, 
        predictions: List[PredictedTrajectory] | None = None, past_trajectory: PredictedTrajectory | None = None,
        behavior = Behavior.DEFAULT):

        metadata = ModifiedSceneObjectMetadata.from_scene_object_metadata(metadata, behavior = behavior) # Add behavior to metadata
        super().__init__(tracked_object_type, oriented_box, velocity, metadata, angular_velocity, predictions, past_trajectory)

    @property
    def behavior(self):
        return self.metadata.behavior
    
    @behavior.setter
    def behavior(self, behavior: Behavior):
        self._metadata = ModifiedSceneObjectMetadata.from_scene_object_metadata(self.metadata, behavior = behavior)
        

    def get_path_to_go(self, seconds = 3):
        """ 
        Get trayectory of the agent by doing a constant velocity proyection.
        The path will start from location of the back side of the car in the current position and will end in the location 
        of the front part of the vehicle in the last position droven by the car.
        
        """
        return get_agent_constant_velocity_path(self, seconds)
    
    @classmethod
    def from_new_pose(cls, agent: AgentState, pose: StateSE2) -> AgentState:
        return cls.from_agent_state(super().from_new_pose(agent, pose))
    
    @classmethod
    def from_new_oriented_box(cls, agent: AgentState, box: OrientedBox) -> AgentState:
        return cls.from_agent_state(AgentState(agent.tracked_object_type, 
                                               box, 
                                               agent.velocity, 
                                               agent.metadata, 
                                               agent.angular_velocity
                                               )
                                    )
    


class ModifiedSceneObjectMetadata(SceneObjectMetadata):
    def __init__(self, 
                timestamp_us: int,
                token: str,
                track_id: Optional[int], 
                track_token: Optional[str], 
                category_name, 
                behavior = Behavior.DEFAULT):
        super().__init__(timestamp_us, token, track_id, track_token, category_name)
        self.behavior = behavior

    @classmethod
    def from_scene_object_metadata(cls, 
                                   SOM: SceneObjectMetadata, 
                                   timestamp = None,
                                   token = None,
                                   track_id = None,
                                   track_token = None,
                                   category_name = None,
                                   behavior = None):
        
        timestamp_us = timestamp or SOM.timestamp_us
        token = token or SOM.token
        track_id = track_id or SOM.track_id
        track_token = track_token or SOM.track_token
        category_name = category_name or SOM.category_name
        behavior = behavior or (SOM.behavior if isinstance(SOM, cls) else Behavior.DEFAULT)
        
        return cls(timestamp_us, token, track_id, track_token, category_name, behavior)