from __future__ import annotations
import random
import math
import random
import warnings
from typing import Any, List, Optional, Tuple, Type, Dict, Union
from collections import deque
import numpy as np
from shapely.geometry.base import CAP_STYLE
import logging

from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from nuplan.common.maps.nuplan_map.lane_connector import NuPlanLaneConnector
from nuplan.common.maps.nuplan_map.roadblock import NuPlanRoadBlock
from nuplan.common.maps.nuplan_map.roadblock_connector import NuPlanRoadBlockConnector
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.common.maps.abstract_map_objects import (
    LaneConnector,
    LaneGraphEdgeMapObject,
    PolylineMapObject,
    RoadBlockGraphEdgeMapObject,
    StopLine,
)
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects, TrackedObjectType
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, ProgressStateSE2
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.geometry.transform import rotate_angle

from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import (
    STRTreeOccupancyMap,
    STRTreeOccupancyMapFactory,
)
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory
from nuplan.planning.simulation.path.utils import convert_se2_path_to_progress_path
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring


from nuplan.database.nuplan_db.nuplan_scenario_queries import get_ego_state_for_lidarpc_token_from_db

logger = logging.getLogger(__name__)


class AgentsModifier():

    def __init__(self, 
                tracked_objects: TrackedObjects, 
                modification: dict, 
                map_api: AbstractMap,
                log_file: str,
                token: str,
                lookup_table: dict
                ) -> None:
        
        if not modification:
            logger.warning("Modification parameter is empty. If no modification is desired"+
                           " please don't use the nuplan modifications scenario builder")
            self.tracked_objects = tracked_objects
            self.ego_state: EgoState = get_ego_state_for_lidarpc_token_from_db(log_file, token)
            return 

        random.seed(token)

        self.ego_state: EgoState = get_ego_state_for_lidarpc_token_from_db(log_file, token)
        self.ego_speed = self.ego_state.dynamic_car_state.speed
        self.tracked_objects = tracked_objects
        self.tracked_agents = tracked_objects.get_tracked_objects_of_type(TrackedObjectType(0))
        self.map_api = map_api
        self.occupancy_map = STRTreeOccupancyMap({})
        if lookup_table:
            if "spawn_in_intersection" in lookup_table:
                self.spawn_in_intersection = lookup_table["spawn_in_intersection"] # TODO is this doing something?
            else: self.spawn_in_intersection = False
            if "extra_agents_to_spawn" in lookup_table:
                self.extra_agents_to_spawn = lookup_table["extra_agents_to_spawn"]
            else: self.extra_agents_to_spawn = []
        else:
            self.spawn_in_intersection = False
            self.extra_agents_to_spawn = []

        if "decel_max" in modification:
            self.amax = modification["decel_max"]
            self.acomf = modification["accel_max"] 
        else:
            self.amax = 2
            self.acomf = 1
 
        self.ego_lane: Optional[Union[NuPlanLane,NuPlanLaneConnector]] = map_api.get_one_map_object(
                                                                        self.ego_state.center.point,SemanticMapLayer.LANE) 
        if not self.ego_lane:
            logger.warn("Using the agents modifiers in a scenario that starts in a intersection may cause some errors")
            try:
                self.ego_lane = map_api.get_one_map_object(self.ego_state.center.point,SemanticMapLayer.LANE_CONNECTOR)
            except AssertionError:
                logger.warn("Could not find in which lane is the ego vehicle located, since there" + 
                " are multiple lane connnectors in its location. Try not to modify a scenario which starts in a intersection")
                # This could be solved by including lanes_in_route information in the agents modifier but this is not a priority since
                # the agents modifier wouldn't work properly with an ego in intersection anyways
                # TODO make it work if ego spawns in a lane connector
                self.ego_lane = map_api.get_all_map_objects(self.ego_state.center.point,SemanticMapLayer.LANE_CONNECTOR)[0]
                # Get all map objects and take a random one (WORKAROUND)

        # Filter percentage by percentage of agents
        if "amount_of_agents" in modification.keys(): 
            self.delete_percentage_of_agents(modification["amount_of_agents"])
        # Spawn agents according to density
        elif "density" in modification.keys(): 
            self.delete_percentage_of_agents(0)
            self.spawn_agents(density=modification["density"])
        # Spawn agents in a specific way 
        elif "specific_scenario" in modification.keys(): self.spawn_agents_for_specific_scenarios(modification["specific_scenario"])

    def delete_percentage_of_agents(self, percentage):

        amount_to_delete= int((1.0 - percentage)*len(self.tracked_agents))
        self.deleted_agents =  random.sample(self.tracked_agents, amount_to_delete)
        self.tracked_objects = TrackedObjects(list(filter(lambda object: object not in self.deleted_agents, self.tracked_objects.tracked_objects)))
        self.tracked_objects = self.delete_objects(self.tracked_objects, delete_pedestrians=True)
    
    @staticmethod
    def delete_objects(tracked_objects: TrackedObjects, delete_pedestrians: bool = False, delete_static_objects: bool = True ): 
            pedestrians = tracked_objects.get_tracked_objects_of_type(TrackedObjectType.PEDESTRIAN)
            vehicles = tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
            static_objects = [object for object in tracked_objects if object not in pedestrians and object not in vehicles]
            objects_to_delete = [object for object in tracked_objects if (delete_pedestrians and object in pedestrians)
                                                                        or (delete_static_objects and object in static_objects)]
            return TrackedObjects(list(filter(lambda object: object not in objects_to_delete, tracked_objects.tracked_objects)))
    
    def spawn_agents(self, density) -> tuple[TrackedObjects, float]:
        #----------------------------------------------------------------
        # Spawn agents in pseudo random locations
        #----------------------------------------------------------------

        # Get locations to spawn agents in each lane
        current_roadblock: NuPlanRoadBlock = self.ego_lane.parent
        sibling_lanes: List[NuPlanLane] = current_roadblock.interior_edges

        locations_speed_to_spawn = []
        
        if density == "h": max_distance_between_agents = int(100/3)
        elif density == "m": max_distance_between_agents = int(100/2)
        elif density == "l": max_distance_between_agents = int(100)
        else: AssertionError("Not a valid value for density")

        flag_ego = False
        self.deleted_agents_ids = [agent.metadata.track_token for agent in self.deleted_agents]

        for lane in sibling_lanes:
            
            position = 0
            get_new_rand = True
            rand = 1000000
            discrete_path = []
            
            for selected_lane in self.find_lanes_to_spawn(lane):
                discrete_path = discrete_path + selected_lane.baseline_path.discrete_path
            discrete_path = convert_se2_path_to_progress_path(discrete_path)
            #discrete_path = lane.baseline_path.discrete_path
            discrete_path.reverse()
                            
            tentative_location = discrete_path[0]
            for state in discrete_path:

                # Is this state the ego_state?
                if lane.id == self.ego_lane.id and state.distance_to(self.ego_state.center) < 0.5 and not flag_ego:
                    same_as_ego = True
                    flag_ego = True # Switch to only enter here once
                else: same_as_ego = False

                # Get a new random number
                if get_new_rand: 
                    rand = random.randrange(10, max_distance_between_agents)
                    get_new_rand = False

                distance_to_leading_car = abs(state.progress - tentative_location.progress)
                # Create list of agents with certain distance to each other
                if (distance_to_leading_car > rand and lane.id != self.ego_lane.id) or \
                    (lane.id == self.ego_lane.id and distance_to_leading_car > rand and \
                    state.distance_to(self.ego_state.center) > rand) or same_as_ego:

                    # Decide speed 
                    if position == 0 and (len(current_roadblock.outgoing_edges) == 1):
                        speed = lane.speed_limit_mps or 10
                    elif position == 0 :
                        speed = math.sqrt((2/3)*abs(state.progress - discrete_path[0].progress)*self.acomf)
                    else:
                        distance_to_leading_car = distance_to_leading_car
                        reaction_time = max(random.gauss(0.25,0.05),0.1)
                        alfa = (3/(2*self.amax))
                        speed_of_leading_car = speed
                        speed = (-reaction_time + math.sqrt(reaction_time**2 + 4*alfa*(alfa*(speed_of_leading_car**2) \
                                                                                    +distance_to_leading_car)))/(2*alfa)  
                        speed = min(distance_to_leading_car/2, speed)
                        
                    speed = speed + random.gauss(0,1)
                    if lane.speed_limit_mps: speed = min(speed, lane.speed_limit_mps)
                    speed = max(speed, 0)

                    # Create list of location, speed pairs to spawn agents
                    if not same_as_ego:
                        locations_speed_to_spawn.append([state, speed])
                    else: self.ego_speed = speed
                    tentative_location = state
                    get_new_rand = True
                    position += 1
        
        # Spawn extra agents in lookup table 
        for agent in self.extra_agents_to_spawn:
            lane_id, lane_distance = self.map_api.get_distance_to_nearest_map_object(StateSE2(agent[0],agent[1],0), SemanticMapLayer.LANE)
            lane_connector_id, lane_connector_distance = self.map_api.get_distance_to_nearest_map_object(StateSE2(agent[0],agent[1],0), 
                                                                                                         SemanticMapLayer.LANE_CONNECTOR)
            edge_id = lane_id if lane_distance <= lane_connector_distance else lane_connector_id
            lane = self.map_api.get_map_object(edge_id, SemanticMapLayer.LANE) or \
                   self.map_api.get_map_object(edge_id, SemanticMapLayer.LANE_CONNECTOR)
            location = lane.baseline_path.get_nearest_pose_from_position(StateSE2(agent[0],agent[1],0))
            
            locations_speed_to_spawn.append([location, agent[2]])

        # Add agents in occupancy map
        self.agents_track_tokens_to_spawn = self.deleted_agents_ids.copy()
        if len(locations_speed_to_spawn) > len(self.agents_track_tokens_to_spawn):
            self.agents_track_tokens_to_spawn += list(np.zeros(len(locations_speed_to_spawn)-len(self.deleted_agents_ids)))
            assert len(self.agents_track_tokens_to_spawn) == len(locations_speed_to_spawn)
        for index, id in enumerate(self.agents_track_tokens_to_spawn):
            if id == 0:
                track_token = f"{random.randrange(16**16):=16x}"
                self.agents_track_tokens_to_spawn[index] = track_token
                self.occupancy_map.set(track_token, self.deleted_agents[0]._box.geometry)
            else:
                self.occupancy_map.set(id, self.deleted_agents[0]._box.geometry)
            
        for location, speed in locations_speed_to_spawn:
            self.add_agent_in_location(location, speed)

            
    def find_lanes_to_spawn(self, lane) -> list(NuPlanLane):
        ego_progress_path = convert_se2_path_to_progress_path(lane.baseline_path.discrete_path)
        index_of_ego_along_path = min([[self.ego_state.center.distance_to(state), index] for index, state in enumerate(ego_progress_path)],\
                                        key=lambda element: element[0])[1]
        
        ego_progress = ego_progress_path[index_of_ego_along_path].progress

        lenght_behind_ego = ego_progress

        lanes_to_spawn = deque([lane])

        # Search for roads in the back of the ego vehicle
        while lenght_behind_ego < 50:
            
            incoming_edges = lanes_to_spawn[0].incoming_edges
            selected_incoming_edges = []

            for edge in incoming_edges:
                # Intersection handling
                if edge.has_traffic_lights():
                    break
                # Normal road
                else:
                    selected_incoming_edges.append(edge)
            if not selected_incoming_edges:
                break
            # Select edge with the lowest curvature (prefer going straight)
            curvatures = [abs(edge.baseline_path.get_curvature_at_arc_length(0.0)) for edge in selected_incoming_edges]
            idx = np.argmin(curvatures)
            new_segment: LaneGraphEdgeMapObject = selected_incoming_edges[idx]

            lenght_behind_ego += new_segment.baseline_path.length
            lanes_to_spawn.appendleft(new_segment)


        lenght_in_front_ego = ego_progress_path[-1].progress - ego_progress

        # Search for roads in the front of the ego vehicle
        while lenght_in_front_ego < 50:
            
            outgoing_edges = lanes_to_spawn[-1].outgoing_edges
            selected_outgoing_edges = []

            for edge in outgoing_edges:
                # Intersection handling / it spawn until there is a intersection with traffic lights in front of ego
                if edge.has_traffic_lights():
                    break
                # Normal road
                else:
                    selected_outgoing_edges.append(edge)
            if not selected_outgoing_edges:
                break
            # Select edge with the lowest curvature (prefer going straight)
            curvatures = [abs(edge.baseline_path.get_curvature_at_arc_length(0.0)) for edge in selected_outgoing_edges]
            idx = np.argmin(curvatures)
            new_segment: LaneGraphEdgeMapObject = selected_outgoing_edges[idx]
            
            lenght_in_front_ego += new_segment.baseline_path.length
            lanes_to_spawn.append(new_segment)

        return lanes_to_spawn 

    def add_agent_in_location(self, location: StateSE2, speed: float = None, behavior = "default") -> None:

        location = StateSE2(location.x, location.y, location.heading)
        
        if self.agents_track_tokens_to_spawn[0] in self.deleted_agents_ids:
            if speed == None: self.deleted_agents[0]._velocity = self.ego_state.dynamic_car_state.rear_axle_velocity_2d
            else: self.deleted_agents[0]._velocity = StateVector2D(speed, 0)
            agent = ModifiedAgent.from_agent_state(Agent.from_new_pose(self.deleted_agents[0], location))
            self.deleted_agents.pop(0)
        else: 
            if 'max_id' not in locals(): max_id = max(self.tracked_objects.tracked_objects, key= lambda x: x.metadata.track_id).metadata.track_id
            agent = ModifiedAgent.from_agent_state(Agent.from_new_pose(self.tracked_agents[-1], location))
            agent._metadata = SceneObjectMetadata(
                            self.tracked_objects.tracked_objects[-1].metadata.timestamp_us,
                            f"{random.randrange(16**16):=16x}", max_id+1, self.agents_track_tokens_to_spawn[0],"vehicle")
            if speed == None: agent._velocity = self.ego_state.dynamic_car_state.rear_axle_velocity_2d
            else: agent._velocity = StateVector2D(speed,0)
            max_id += 1 
        self.agents_track_tokens_to_spawn.pop(0)
        if behavior != "default":
            agent._behavior = behavior

        # Check for agents that intersects THIS agent's path
        agent_path = path_to_linestring(agent.get_path_to_go())
        intersecting_agents = self.occupancy_map.intersects(
            agent_path.buffer((agent._box.width / 2), cap_style=CAP_STYLE.flat)
        )
        #assert intersecting_agents.contains(agent.track_token), "Agent's baseline does not intersect the agent itself"

        # Checking if there are agents intersecting THIS agent's baseline.
        # Hence, we are checking for at least 2 intersecting agents.
        if intersecting_agents.size > 0: 
            self.occupancy_map.remove([agent.metadata.track_token])
            return
        
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


class ModifiedAgent(Agent):
    def __init__(
        self, tracked_object_type: TrackedObjectType, oriented_box: OrientedBox, 
        velocity: StateVector2D, metadata: SceneObjectMetadata, angular_velocity: float | None = None, 
        predictions: List[PredictedTrajectory] | None = None, past_trajectory: PredictedTrajectory | None = None, behavior = "default"):
        # TODO behavior intenum
        
        super().__init__(tracked_object_type, oriented_box, velocity, metadata, angular_velocity, predictions, past_trajectory)

        self._behavior = behavior

        """ if metadata.track_token == "06986209f35e5a51": 
            self._velocity.y = 0
            self._velocity.x = 0 """


    def get_path_to_go(self) -> list(StateSE2): # TODO change this to function already in nuplan.simulation.utils.agent_utils
        path = []
        velocity = rotate_angle(StateSE2(self.velocity.x, self.velocity.y, self._box._center.heading) , -self._box._center.heading)
        for i in np.arange(0, 1.6, 0.1): #TODO here is refering to the headaway time, change this value when headawaytime changes
            path.append(
                StateSE2(self._box._center.x + i*velocity.x, self._box._center.y + i*velocity.y,self._box.center.heading)
                )
        return path