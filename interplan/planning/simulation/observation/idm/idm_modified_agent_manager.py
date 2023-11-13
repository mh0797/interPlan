from typing import Dict, List

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist
from shapely.geometry.base import CAP_STYLE
import random

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.geometry.transform import rotate_angle
from nuplan.common.geometry.convert import absolute_to_relative_poses
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import StopLine
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.common.maps.nuplan_map.utils import extract_roadblock_objects
from nuplan.planning.metrics.utils.expert_comparisons import principal_value
from nuplan.planning.simulation.observation.idm.idm_states import IDMLeadAgentState
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.occupancy_map.abstract_occupancy_map import OccupancyMap

from interplan.planning.utils.agent_utils import get_agent_constant_velocity_geometry
from interplan.planning.scenario_builder.scenario_modifier.agents_modifier import Behavior, ModifiedAgent, ModifiedSceneObjectMetadata
from interplan.planning.simulation.observation.idm.modified_idm_agent import IDMAgent

UniqueIDMAgents = Dict[str, IDMAgent]


class IDMAgentManager:
    """IDM smart-agents manager."""

    def __init__(self, agents: UniqueIDMAgents, agent_occupancy: OccupancyMap, map_api: AbstractMap, IDM_agents_behavior: str):
        """
        Constructor for IDMAgentManager.
        :param agents: A dictionary pairing the agent's token to it's IDM representation.
        :param agent_occupancy: An occupancy map describing the spatial relationship between agents.
        :param map_api: AbstractMap API
        """
        self.agents: UniqueIDMAgents = agents
        self.agent_occupancy = agent_occupancy
        self._map_api = map_api
        self.IDM_agents_behavior = IDM_agents_behavior

    def propagate_agents(
        self,
        ego_state: EgoState,
        tspan: float,
        iteration: int,
        traffic_light_status: Dict[TrafficLightStatusType, List[str]],
        open_loop_detections: List[TrackedObject],
        radius: float,
    ) -> None:
        """
        Propagate each active agent forward in time.

        :param ego_state: the ego's current state in the simulation.
        :param tspan: the interval of time to simulate.
        :param iteration: the simulation iteration.
        :param traffic_light_status: {traffic_light_status: lane_connector_ids} A dictionary containing traffic light information.
        :param open_loop_detections: A list of open loop detections the IDM agents should be responsive to.
        :param radius: [m] The radius around the ego state
        """
        self.agent_occupancy.set("ego", ego_state.car_footprint.geometry)
        self.agent_occupancy.set("ego_expanded", get_agent_constant_velocity_geometry(ego_state.agent))
        track_ids = []
        for track in open_loop_detections:
            track_ids.append(track.track_token)
            self.agent_occupancy.insert(track.track_token, track.box.geometry)

        self._filter_agents_out_of_range(ego_state, radius)

        ego_roadblock = extract_roadblock_objects(self._map_api, ego_state.center.point)[0] #TODO whatifconnector

        assert ego_roadblock, "Ego is not in any route"

        for agent_token, agent in self.agents.items():
            if agent.is_active(iteration) and agent.has_valid_path():
                agent_behavior = agent.agent.metadata.behavior if isinstance(agent.agent.metadata, ModifiedSceneObjectMetadata) else Behavior.DEFAULT
                if agent_behavior == Behavior.STOPPED: 
                    #agent.propagate(IDMLeadAgentState(progress=0.0, velocity=0.0, length_rear=0), tspan,)
                    continue

                agent.plan_route(traffic_light_status)
                # Add stop lines into occupancy map if they are impacting the agent
                stop_lines = self._get_relevant_stop_lines(agent, traffic_light_status)
                # Keep track of the stop lines that were inserted. This is to remove them for each agent
                inactive_stop_line_tokens = self._insert_stop_lines_into_occupancy_map(stop_lines)

                # Check for agents that intersects THIS agent's path
                agent_path = path_to_linestring(agent.get_path_to_go())
                intersecting_agents = self.agent_occupancy.intersects(
                    agent_path.buffer((agent.width / 2), cap_style=CAP_STYLE.flat)
                )
                assert intersecting_agents.contains(agent_token), "Agent's baseline does not intersect the agent itself"
                
                agent_lane = [lane for lane in agent.get_route() if lane.contains_point(agent.to_se2().point)][0]

                # According to the IDM behavior setting, the agents will consider the ego expanded path or not
                if intersecting_agents.contains("ego_expanded"):
                    
                    if agent_lane.contains_point(ego_state.center.point):
                        intersecting_agents.remove(["ego_expanded"])
                    elif (ego_roadblock.contains_point(agent.agent.center.point) and 
                        not agent_lane.baseline_path.linestring.intersects(self.agent_occupancy.get("ego_expanded"))):

                        if self.IDM_agents_behavior == "cautious" and \
                        absolute_to_relative_poses([agent.to_se2(), ego_state.center])[1].x < 0:
                            intersecting_agents.remove(["ego_expanded"])
                        elif self.IDM_agents_behavior == "egoist":
                            intersecting_agents.remove(["ego_expanded"])
                        elif self.IDM_agents_behavior == "mixed" and random.randint(0, 1):
                            intersecting_agents.remove(["ego_expanded"])
                    elif self.IDM_agents_behavior == "standard":
                        intersecting_agents.remove(["ego_expanded"])
                    elif self.agent_occupancy.get("ego_expanded").intersects(agent.polygon):
                        intersecting_agents.remove(["ego_expanded"])
                    
                    # TODO check this logic with the aditions of behavior properties to agents

                # Checking if there are agents intersecting THIS agent's baseline.
                # Hence, we are checking for at least 2 intersecting agents.
                if intersecting_agents.size > 1:
                    nearest_id, nearest_agent_polygon, relative_distance = intersecting_agents.get_nearest_entry_to(
                        agent_token
                    )
                    agent_heading = agent.to_se2().heading

                    if "ego" in nearest_id and not agent_behavior == Behavior.CAUTIOUS:
                        ego_velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d
                        longitudinal_velocity = np.hypot(ego_velocity.x, ego_velocity.y)
                        relative_heading = ego_state.rear_axle.heading - agent_heading
                    elif 'stop_line' in nearest_id:
                        longitudinal_velocity = 0.0
                        relative_heading = 0.0
                    elif nearest_id in self.agents:
                        nearest_agent = self.agents[nearest_id]
                        longitudinal_velocity = nearest_agent.velocity
                        relative_heading = nearest_agent.to_se2().heading - agent_heading
                    else:
                        longitudinal_velocity = 0.0
                        relative_heading = 0.0

                    # Wrap angle to [-pi, pi]
                    relative_heading = principal_value(relative_heading)
                    # take the longitudinal component of the projected velocity
                    projected_velocity = rotate_angle(StateSE2(longitudinal_velocity, 0, 0), relative_heading).x

                    # relative_distance already takes the vehicle dimension into account.
                    # Therefore there is no need to pass in the length_rear.
                    length_rear = 0
                else:
                    # Free road case: no leading vehicle
                    projected_velocity = 0.0
                    relative_distance = agent.get_progress_to_go()
                    length_rear = agent.length / 2

                agent.propagate(
                    IDMLeadAgentState(progress=relative_distance, velocity=projected_velocity, length_rear=length_rear),
                    tspan,
                )
                self.agent_occupancy.set(agent_token, agent.projected_footprint)
                self.agent_occupancy.remove(inactive_stop_line_tokens)
        self.agent_occupancy.remove(track_ids)

    def get_active_agents(self, iteration: int, num_samples: int, sampling_time: float) -> DetectionsTracks:
        """
        Returns all agents as DetectionsTracks.
        :param iteration: the current simulation iteration.
        :param num_samples: number of elements to sample.
        :param sampling_time: [s] time interval of sequence to sample from.
        :return: agents as DetectionsTracks.
        """
        return DetectionsTracks(
            TrackedObjects(
                [
                    agent.get_agent_with_planned_trajectory(num_samples, sampling_time)
                    for agent in self.agents.values()
                    if agent.is_active(iteration)
                ]
            )
        )

    def _filter_agents_out_of_range(self, ego_state: EgoState, radius: float = 100) -> None:
        """
        Filter out agents that are out of range.
        :param ego_state: The ego state used as the center of the given radius
        :param radius: [m] The radius around the ego state
        """
        if len(self.agents) == 0:
            return

        agents_list = []
        for agent in self.agents.values():
            if not isinstance(agent, ModifiedAgent) or agent.agent.metadata.behavior == Behavior.DEFAULT:
                agents_list.append(agent.to_se2().point.array)
            else: agents_list.append(ego_state.center.point.array) # If agents are not of type default then they will never be filtered
        agents: npt.NDArray[np.int32] = np.array(agents_list)
        distances = cdist(np.expand_dims(ego_state.center.point.array, axis=0), agents)
        remove_indices = np.argwhere(distances.flatten() > radius)
        remove_tokens = np.array(list(self.agents.keys()))[remove_indices.flatten()]

        # Remove agents which are out of scope
        self.agent_occupancy.remove(remove_tokens)
        for token in remove_tokens:
            self.agents.pop(token)

    def _get_relevant_stop_lines(
        self, agent: IDMAgent, traffic_light_status: Dict[TrafficLightStatusType, List[str]]
    ) -> List[StopLine]:
        """
        Retrieve the stop lines that are affecting the given agent.
        :param agent: The IDM agent of interest.
        :param traffic_light_status: {traffic_light_status: lane_connector_ids} A dictionary containing traffic light information.
        :return: A list of stop lines associated with the given traffic light status.
        """
        relevant_lane_connectors = list(
            {segment.id for segment in agent.get_route()} & set(traffic_light_status[TrafficLightStatusType.RED])
        )
        lane_connectors = [
            self._map_api.get_map_object(lc_id, SemanticMapLayer.LANE_CONNECTOR) for lc_id in relevant_lane_connectors
        ]
        return [stop_line for lc in lane_connectors if lc for stop_line in lc.stop_lines]

    def _insert_stop_lines_into_occupancy_map(self, stop_lines: List[StopLine]) -> List[str]:
        """
        Insert stop lines into the occupancy map.
        :param stop_lines: A list of stop lines to be inserted.
        :return: A list of token corresponding to the inserted stop lines.
        """
        stop_line_tokens: List[str] = []
        for stop_line in stop_lines:
            stop_line_token = f"stop_line_{stop_line.id}"
            if not self.agent_occupancy.contains(stop_line_token):
                self.agent_occupancy.set(stop_line_token, stop_line.polygon)
                stop_line_tokens.append(stop_line_token)

        return stop_line_tokens
