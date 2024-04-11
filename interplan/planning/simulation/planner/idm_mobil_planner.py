import copy
import logging
import math
import warnings
from copy import copy
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import CAP_STYLE
from shapely.ops import unary_union

warnings.filterwarnings(
    "ignore", message="(.|\n)*invalid value encountered in line_locate_point"
)

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObject, SceneObjectMetadata
from nuplan.common.actor_state.state_representation import (
    ProgressStateSE2,
    StateSE2,
    StateVector2D,
    TimePoint,
)
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.geometry.compute import principal_value
from nuplan.common.geometry.transform import rotate_angle, transform
from nuplan.common.maps.abstract_map_objects import (
    LaneGraphEdgeMapObject,
    PolylineMapObject,
    RoadBlockGraphEdgeMapObject,
)
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from nuplan.common.maps.nuplan_map.roadblock_connector import NuPlanRoadBlockConnector
from nuplan.planning.simulation.observation.idm.idm_states import (
    IDMAgentState,
    IDMLeadAgentState,
)
from nuplan.planning.simulation.observation.idm.utils import (
    create_path_from_se2,
    path_to_linestring,
)
from nuplan.planning.simulation.occupancy_map.abstract_occupancy_map import OccupancyMap
from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import (
    STRTreeOccupancyMap,
)
from nuplan.planning.simulation.path.interpolated_path import InterpolatedPath
from nuplan.planning.simulation.path.utils import (
    convert_se2_path_to_progress_path,
    trim_path,
)
from nuplan.planning.simulation.planner.abstract_idm_planner import AbstractIDMPlanner
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.planner.utils.breadth_first_search import (
    BreadthFirstSearch,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.predicted_trajectory import (
    PredictedTrajectory,
)

from interplan.planning.scenario_builder.scenario_modifier.agents_modifier import (
    ModifiedSceneObjectMetadata,
)
from interplan.planning.utils.agent_utils import get_agent_constant_velocity_geometry

UniqueObjects = Dict[str, SceneObject]

logger = logging.getLogger(__name__)


class IDMMobilPlanner(AbstractIDMPlanner):
    """
    The IDM planner is composed of two parts:
        1. Path planner that constructs a route to the same road block as the goal pose.
        2. IDM policy controller to control the longitudinal movement of the ego along the planned route.
    """

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def __init__(
        self,
        target_velocity: float,
        min_gap_to_lead_agent: float,
        headway_time: float,
        accel_max: float,
        decel_max: float,
        planned_trajectory_samples: int,
        planned_trajectory_sample_interval: float,
        occupancy_map_radius: float,
        politeness_factor: float,
        changing_threshold: float,
        push_lane_change: bool,
    ):
        """
        Constructor for IDMPlanner
        :param target_velocity: [m/s] Desired velocity in free traffic.
        :param min_gap_to_lead_agent: [m] Minimum relative distance to lead vehicle.
        :param headway_time: [s] Desired time headway. The minimum possible time to the vehicle in front.
        :param accel_max: [m/s^2] maximum acceleration.
        :param decel_max: [m/s^2] maximum deceleration (positive value).
        :param planned_trajectory_samples: number of elements to sample for the planned trajectory.
        :param planned_trajectory_sample_interval: [s] time interval of sequence to sample from.
        :param occupancy_map_radius: [m] The range around the ego to add objects to be considered.
        """
        super(IDMMobilPlanner, self).__init__(
            target_velocity,
            min_gap_to_lead_agent,
            headway_time,
            accel_max,
            decel_max,
            planned_trajectory_samples,
            planned_trajectory_sample_interval,
            occupancy_map_radius,
        )

        self.politeness_factor = politeness_factor
        self.changing_threshold = changing_threshold
        self.push_lane_change = push_lane_change

        self._initialized = False

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._map_api = initialization.map_api
        self._goal = initialization.mission_goal
        self._initialize_route_plan(initialization.route_roadblock_ids)
        self._initialized = False
        self.false_vehicle = False

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """Inherited, see superclass."""

        self.update_observations(current_input)

        # Ego current state
        ego_state, observations = current_input.history.current_state

        if not self._initialized:
            self._initialize_ego_path(ego_state)
            self._initialized = True

        # Create occupancy map
        occupancy_map, unique_observations = self._construct_occupancy_map(
            ego_state, observations
        )

        # Traffic light handling
        traffic_light_data = current_input.traffic_light_data
        self._annotate_occupancy_map(traffic_light_data, occupancy_map)

        return self._get_planned_trajectory(
            ego_state, occupancy_map, unique_observations
        )

    def update_observations(self, current_input):
        """
        Takes the vehicles in the observation of current_input and change:
            * The metadata of the vehicles so that it has the current timestamp
            * The past trajectory attribute so to include the past waypoints

        This is done to be able to calculate acceleration of agents
        """

        ego_state, observations = current_input.history.current_state

        previous_ego_state = current_input.history.ego_states[-2]
        previous_observations = current_input.history.observations[-2]
        current_states = observations.tracked_objects.tracked_objects
        previous_states = previous_observations.tracked_objects.tracked_objects

        for idx, current_state in enumerate(current_states):
            if current_state.tracked_object_type not in [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN]:
                continue
            previous_state: Agent = next(
                iter(
                    [
                        past
                        for past in previous_states
                        if past.track_token == current_state.track_token
                    ]
                ),
                None,
            )
            if not previous_state:
                previous_state = current_state

            metadata = ModifiedSceneObjectMetadata.from_scene_object_metadata(
                previous_state.metadata, timestamp=TimePoint(ego_state.time_us)
            )
            past_trajectory = PredictedTrajectory(
                1,
                [
                    Waypoint(
                        TimePoint(previous_ego_state.time_us),
                        previous_state.box,
                        previous_state.velocity,
                    ),
                    Waypoint(
                        current_state._initial_time_stamp,
                        current_state.box,
                        current_state.velocity,
                    ),
                ],
            )
            current_input.history.observations[-1].tracked_objects.tracked_objects[
                idx
            ].past_trajectory = past_trajectory
            current_input.history.observations[-1].tracked_objects.tracked_objects[
                idx
            ]._metadata = metadata

    def _initialize_ego_path(self, ego_state: EgoState) -> None:
        """
        Initializes the ego path from the ground truth driven trajectory
        :param ego_state: The ego state at the start of the scenario.
        """
        route_plan, _ = self._breadth_first_search(ego_state)
        ego_speed = ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()
        speed_limit = route_plan[0].speed_limit_mps or self._policy.target_velocity
        self._policy.target_velocity = (
            speed_limit if speed_limit > ego_speed else ego_speed
        )
        discrete_path = []
        for edge in route_plan:
            discrete_path.extend(edge.baseline_path.discrete_path)
        self._ego_path = create_path_from_se2(discrete_path)
        self._ego_path_linestring = path_to_linestring(discrete_path)
        self.objective_lane = route_plan[0]

    def _initialize_lane_change(
        self,
        desired_ego_state: EgoState,
        current_ego_state: EgoState,
        current_ego_idm_state: IDMAgentState,
        occupancy_map: OccupancyMap,
        unique_observations: UniqueObjects,
    ) -> None:
        """
        Initializes the ego path to perform a lane change
        :param ego_state: The ego state at the start of the scenario.
        """
        route_plan, _ = self._breadth_first_search(desired_ego_state)
        ego_speed = (
            current_ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()
        )
        speed_limit = route_plan[0].speed_limit_mps or self._policy.target_velocity
        self._policy.target_velocity = (
            speed_limit if speed_limit > ego_speed else ego_speed
        )
        discrete_path = []

        # -------------------------------------------------------------
        # Adding points between current ego state and desired ego state
        # -------------------------------------------------------------
        current_p_state = ProgressStateSE2(
            current_ego_state.center.x,
            current_ego_state.center.y,
            current_ego_state.center.heading,
            0,
        )
        desired_p_state = ProgressStateSE2(
            desired_ego_state.center.x
            - 0.5 * np.sign(current_ego_state.center.x - desired_ego_state.center.x),
            desired_ego_state.center.y,
            desired_ego_state.center.heading,
            0,
        )
        
        #TODO: add parameter to config
        # seconds while ego will continue its path until its path change lanes
        threshold_to_change = 0.5 
        start_manouver_progress = (
            current_ego_idm_state.progress
            + current_ego_state.dynamic_car_state.center_velocity_2d.x
            * threshold_to_change
        )
        start_manouver_progress = min(
            self._ego_path.get_end_progress(), start_manouver_progress
        )
        start_manouver_point = self._ego_path.get_state_at_progress(
            start_manouver_progress
        )
        SM_DPS_angle = principal_value(
            math.atan2(
                desired_p_state.y - start_manouver_point.y,
                desired_p_state.x - start_manouver_point.x,
            )
        )
        start_manouver_p_state = ProgressStateSE2(
            start_manouver_point.x,
            start_manouver_point.y,
            SM_DPS_angle,
            current_ego_state.dynamic_car_state.center_velocity_2d.x
            * threshold_to_change,
        )
        # Change desired_p_state progress
        desired_p_state.progress = (
            start_manouver_p_state.distance_to(desired_ego_state.center)
            + start_manouver_p_state.progress
        )
        # Add the points from this newly created path to the discrete path
        path_from_current_to_desired = InterpolatedPath(
            [current_p_state, start_manouver_p_state, desired_p_state]
        )

        # If there are stopped vehicles in lane change's path then don't do lane change
        intersecting_agents = occupancy_map.intersects(
            self._get_expanded_ego_path_in_AL(
                current_ego_state, current_ego_idm_state, path_from_current_to_desired
            )
        )
        if intersecting_agents.size > 0 and any(
            [
                unique_observations[agent_id].velocity.magnitude() < 1
                for agent_id in intersecting_agents.get_all_ids()
                if not agent_id[:9] == "red_light"
            ]
        ):
            return False

        for progress in np.arange(0.5, desired_p_state.progress, 0.5):
            discrete_path.append(
                path_from_current_to_desired.get_state_at_progress(progress)
            )

        # Cut initial edge to start from ego_vehicle position
        initial_edge = convert_se2_path_to_progress_path(
            route_plan[0].baseline_path.discrete_path
        )
        initial_edge_linestring = path_to_linestring(initial_edge)
        progress_along_initial_edge = initial_edge_linestring.project(
            Point(*desired_ego_state.center.point.array)
        )
        initial_edge = [
            StateSE2(state.x, state.y, state.heading)
            for state in initial_edge
            if state.progress >= progress_along_initial_edge
        ]

        # Add initial edge to discrete path
        discrete_path.extend(initial_edge)

        for edge in route_plan[1:]:
            discrete_path.extend(edge.baseline_path.discrete_path)
        self._ego_path = create_path_from_se2(discrete_path)
        self._ego_path_linestring = path_to_linestring(discrete_path)

        return True

    def _get_starting_edge(self, ego_state: EgoState) -> LaneGraphEdgeMapObject:
        """
        Get the starting edge based on ego state. If a lane graph object does not contain the ego state then
        the closest one is taken instead.
        :param ego_state: Current ego state.
        :return: The curent LaneGraphEdgeMapObject.
        """
        assert (
            self._route_roadblocks is not None
        ), "_route_roadblocks has not yet been initialized. Please call the initialize() function first!"
        assert (
            len(self._route_roadblocks) >= 2
        ), "_route_roadblocks should have at least 2 elements!"

        starting_edge = None
        closest_distance = math.inf

        # Check for edges in route roadblocks
        for edge in [
            lane
            for roadblock in self._route_roadblocks
            for lane in roadblock.interior_edges
        ]:
            if edge.contains_point(ego_state.center):
                starting_edge = edge
                break

            # In case the ego does not start on a road block
            distance = edge.polygon.distance(ego_state.car_footprint.geometry)
            if distance < closest_distance:
                starting_edge = edge
                closest_distance = distance

        assert starting_edge, "Starting edge for IDM path planning could not be found!"
        return starting_edge

    def _breadth_first_search(
        self, ego_state: EgoState
    ) -> Tuple[List[LaneGraphEdgeMapObject], bool]:
        """
        Performs iterative breath first search to find a route to the target roadblock.
        :param ego_state: Current ego state.
        :return:
            - A route starting from the given start edge
            - A bool indicating if the route is successfully found. Successful means that there exists a path
              from the start edge to an edge contained in the end roadblock. If unsuccessful a longest route is given.
        """
        assert (
            self._route_roadblocks is not None
        ), "_route_roadblocks has not yet been initialized. Please call the initialize() function first!"
        assert (
            self._candidate_lane_edge_ids is not None
        ), "_candidate_lane_edge_ids has not yet been initialized. Please call the initialize() function first!"

        starting_edge = self._get_starting_edge(ego_state)
        graph_search = BreadthFirstSearch(starting_edge, self._candidate_lane_edge_ids)
        # Target depth needs to be offset by one if the starting edge belongs to the second roadblock in the list
        offset = (
            1 if starting_edge.get_roadblock_id() == self._route_roadblocks[1].id else 0
        )
        route_plan, path_found = graph_search.search(
            self._route_roadblocks[-1], len(self._route_roadblocks[offset:])
        )

        """ if not path_found:
            logger.warning(
                "IDMPlanner could not find valid path to the target roadblock. Using longest route found instead"
            ) """

        return route_plan, path_found

    def _get_planned_trajectory(
        self,
        ego_state: EgoState,
        occupancy_map: OccupancyMap,
        unique_observations: UniqueObjects,
    ) -> InterpolatedTrajectory:
        """
        Plan a trajectory w.r.t. the occupancy map.
        :param ego_state: EgoState at current iteration.
        :param occupancy_map: OccupancyMap containing all objects in the scene.
        :param unique_observations: A mapping between the object token and the object itself.
        :return: A trajectory representing the predicted ego's position in future.
        """
        assert (
            self._ego_path_linestring
        ), "_ego_path_linestring has not yet been initialized. Please call the initialize() function first!"

        # Extract ego IDM state
        ego_progress = self._ego_path_linestring.project(
            Point(*ego_state.center.point.array)
        )
        ego_idm_state = IDMAgentState(
            progress=ego_progress,
            velocity=ego_state.dynamic_car_state.center_velocity_2d.x,
        )
        vehicle_parameters = ego_state.car_footprint.vehicle_parameters

        # Initialize planned trajectory with current state
        current_time_point = ego_state.time_point
        projected_ego_state = self._idm_state_to_ego_state(
            ego_idm_state, current_time_point, vehicle_parameters
        )
        planned_trajectory: List[EgoState] = [projected_ego_state]

        current_lane = self._get_starting_edge(ego_state)

        # When crossing from one lane to outgoing lane -> update objective lane
        if self.objective_lane.id not in [
            lane.id for lane in current_lane.parent.interior_edges
        ]:
            self.objective_lane = current_lane

        # False vehicle that incentivize the vehicle braking to make a lane change ------------
        if self.false_vehicle:
            false_vehicle_progress = min(
                ego_idm_state.progress + 10, self._ego_path.get_end_progress()
            )
            false_vehicle_state = self._ego_path.get_state_at_progress(
                false_vehicle_progress
            )
            false_vehicle_state = StateSE2(
                false_vehicle_state.x,
                false_vehicle_state.y,
                false_vehicle_state.heading,
            )  # Progress state isn't hashable
            false_vehicle_in_lane_progress = (
                current_lane.baseline_path.linestring.line_locate_point(
                    Point(*false_vehicle_state)
                )
            )
            meters_until_intersection = (
                current_lane.baseline_path.length - false_vehicle_in_lane_progress
            )
            false_vehicle_velocity = StateVector2D(
                (meters_until_intersection / current_lane.baseline_path.length)
                * self._policy.target_velocity,
                0,
            )
            ego_OB = ego_state.car_footprint
            false_vehicle_OB = OrientedBox(
                false_vehicle_state, ego_OB.length, ego_OB.width, ego_OB.height
            )
            occupancy_map.insert("false_vehicle", false_vehicle_OB.geometry)
            unique_observations["false_vehicle"] = Agent(
                TrackedObjectType.VEHICLE,
                false_vehicle_OB,
                false_vehicle_velocity,
                ego_state.scene_object_metadata,
            )

        # ----------------------------------------------
        # LANE CHANGE
        # ----------------------------------------------

        # Project ego state to adjacent lanes
        adjacent_lanes = current_lane.adjacent_edges
        navigation_benefit_of_changing_lane = [-1, -1]

        if (
            isinstance(current_lane, NuPlanLane)
            and len(current_lane.parent.interior_edges) > 1
        ):
            assert adjacent_lanes[0] or adjacent_lanes[1], "Did not find starting edge"

        for right_bool, adjacent_lane in enumerate(
            adjacent_lanes
        ):  # Right bool: 0-> Left / 1-> Right
            if not adjacent_lane:
                continue

            # Condicitons to consider lane change
            position_in_baseline = (
                self.objective_lane.baseline_path.get_nearest_pose_from_position(
                    ego_state.center
                )
            )
            is_lane_change_happening = (
                position_in_baseline.distance_to(ego_state.center) > 0.5
                or principal_value(
                    abs(position_in_baseline.heading - ego_state.center.heading)
                )
                > 0.1
            )
            navigation_benefit_of_changing_lane[right_bool] = (
                self._distances_to_lane_on_route[current_lane.parent.id][
                    current_lane.index - 1
                ]
                - self._distances_to_lane_on_route[current_lane.parent.id][
                    adjacent_lane.index - 1
                ]
            )
            does_adjacent_lane_leads_to_route = (
                not navigation_benefit_of_changing_lane[right_bool] < 0
            )

            if (
                isinstance(current_lane, NuPlanLane)
                and not is_lane_change_happening
                and does_adjacent_lane_leads_to_route
            ):
                if "incentive_criterion" not in locals():
                    # Initializations -------------------------------------------------------------
                    incentive_criterion = [
                        -1000,
                        -1000,
                    ]  # [Left lane change, Right lane change]
                    # Calculate ac
                    leading_agent = self._get_leading_object(
                        ego_idm_state, ego_state, occupancy_map, unique_observations
                    )
                    new_idm_state = self._policy.solve_forward_euler_idm_policy(
                        IDMAgentState(0, ego_idm_state.velocity),
                        leading_agent,
                        self._planned_trajectory_sample_interval,
                    )
                    ac = (
                        new_idm_state.velocity - ego_idm_state.velocity
                    ) / self._planned_trajectory_sample_interval
                    ac_tilde = [None, None]
                    an = [None, None]
                    an_tilde = [None, None]
                    ego_state_in_adjacent = [
                        None,
                        None,
                    ]  # Ego state if ego would be in adjacent lane with the same progress
                    ego_idm_state_in_adjacent = [None, None]

                    # Calculate parameters in current ego lane: ao and ao_tilde -----------------------------------

                    # Get leading agent of ego vehicle
                    leading_agent = self._get_leading_object(
                        ego_idm_state, ego_state, occupancy_map, unique_observations
                    )
                    # Get follower agent of ego vehicle
                    follower_idm_state, follower_state = self._get_follower_agent(
                        ego_state, occupancy_map, unique_observations
                    )

                    if follower_state:
                        # Since the progress of both follower and leader are based on distance with respect to ego vehicle
                        leading_agent.progress += follower_idm_state.progress
                        # Use IDM equation with follower as the target vehicle
                        new_follower_idm_state = (
                            self._policy.solve_forward_euler_idm_policy(
                                IDMAgentState(0, follower_state.velocity.magnitude()),
                                leading_agent,
                                self._planned_trajectory_sample_interval,
                            )
                        )

                        ao_tilde = (
                            new_follower_idm_state.velocity
                            - follower_idm_state.velocity
                        ) / self._planned_trajectory_sample_interval

                        ao = (
                            follower_idm_state.velocity
                            - follower_state.previous_state.velocity.magnitude()
                        ) / (
                            ego_state.time_us
                            - follower_state.previous_state.time_point.time_s
                        )
                        # Using ego state to retrieve time information since when the observations get updated, the timestamp info
                        # of the metadata doesn't get updated
                    else:
                        ao = 0
                        ao_tilde = 0

                # Ego States in adjacent
                ego_state_in_adjacent[
                    right_bool
                ]: StateSE2 = adjacent_lane.baseline_path.get_nearest_pose_from_position(
                    projected_ego_state.center
                )
                ego_progress_in_adjacent = (
                    adjacent_lane.baseline_path.linestring.project(
                        Point(*ego_state_in_adjacent[right_bool].point.array)
                    )
                )
                ego_idm_state_in_adjacent[right_bool] = IDMAgentState(
                    ego_progress_in_adjacent,
                    ego_state.dynamic_car_state.center_velocity_2d.x,
                )

                ego_state_in_adjacent[right_bool] = EgoState.build_from_center(
                    center=ego_state_in_adjacent[right_bool],
                    center_velocity_2d=StateVector2D(
                        ego_idm_state_in_adjacent[right_bool].velocity, 0
                    ),
                    center_acceleration_2d=ego_state.dynamic_car_state.center_acceleration_2d,
                    tire_steering_angle=0.0,
                    time_point=current_time_point,
                    vehicle_parameters=vehicle_parameters,
                )
                leading_adjacent_agent = self._get_leading_object_in_AL(
                    ego_idm_state_in_adjacent[right_bool],
                    ego_state_in_adjacent[right_bool],
                    occupancy_map,
                    unique_observations,
                    adjacent_lane,
                )

                # If ego agent is coliding if it changes now, then automatically incentive criterion = 0
                if leading_adjacent_agent.progress == 0:
                    incentive_criterion[right_bool] = 0
                    continue
                elif (
                    leading_adjacent_agent.progress < 1
                ):  # Calculate new IDM step if relative distance < 1 m
                    new_idm_state = IDMAgentState(
                        ego_idm_state.progress,
                        ego_idm_state.velocity
                        + self._planned_trajectory_sample_interval
                        * -self._policy._decel_max,
                    )
                else:  # Normal IDM calculation
                    new_idm_state = self._policy.solve_forward_euler_idm_policy(
                        IDMAgentState(0, ego_idm_state.velocity),
                        leading_adjacent_agent,
                        self._planned_trajectory_sample_interval,
                    )

                ac_tilde[right_bool] = (
                    new_idm_state.velocity - ego_idm_state.velocity
                ) / self._planned_trajectory_sample_interval

                # Calculate incentive criterion before politeness
                incentive_criterion[right_bool] = ac_tilde[right_bool] - ac

                # Getting the follower vehicle in adjacent lane
                follower_idm_state, follower_state = self._get_follower_agent(
                    ego_state_in_adjacent[right_bool],
                    occupancy_map,
                    unique_observations,
                )
                if follower_idm_state:
                    # State of follower in left lane if ego vehicle change lanes
                    new_follower_IDM_state = (
                        self._policy.solve_forward_euler_idm_policy(
                            IDMAgentState(0, follower_idm_state.velocity),
                            IDMLeadAgentState(
                                follower_idm_state.progress, ego_idm_state.velocity, 0
                            ),
                            self._planned_trajectory_sample_interval,
                        )
                    )

                    an_tilde[right_bool] = (
                        new_follower_IDM_state.velocity - follower_idm_state.velocity
                    ) / self._planned_trajectory_sample_interval

                    # Acceleration of follower vehicle in adjacent lane without lane change
                    an[right_bool] = (
                        follower_idm_state.velocity
                        - follower_state.previous_state.velocity.magnitude()
                    ) / (
                        ego_state.time_us
                        - follower_state.previous_state.time_point.time_s
                    )
                    # Using ego state to retrieve time information since when the observations get updated, the timestamp info
                    # of the metadata doesn't get updated
                else:
                    an_tilde[right_bool] = 0
                    an[right_bool] = 0

                # Calculate complete Incentive Criterion
                incentive_criterion[right_bool] += self.politeness_factor * (
                    an_tilde[right_bool] - an[right_bool] + ao_tilde - ao
                )
            elif does_adjacent_lane_leads_to_route:
                self.false_vehicle = False

        if "incentive_criterion" in locals():
            max_incentive_criterion = max(
                enumerate(incentive_criterion), key=lambda x: x[1]
            )
            if max_incentive_criterion[1] >= self.changing_threshold:
                self.objective_lane = adjacent_lanes[max_incentive_criterion[0]]
                desired_ego_state_in_adjacent = self._get_ego_state_after_lane_change(
                    ego_state_in_adjacent[max_incentive_criterion[0]],
                    ego_idm_state_in_adjacent[max_incentive_criterion[0]],
                    adjacent_lanes[max_incentive_criterion[0]],
                )
                success = self._initialize_lane_change(
                    desired_ego_state_in_adjacent,
                    ego_state,
                    ego_idm_state,
                    occupancy_map,
                    unique_observations,
                )
                if success:
                    ego_idm_state.progress = 0
                    planned_trajectory = [ego_state]
                    self.false_vehicle = False
            elif (
                navigation_benefit_of_changing_lane[max_incentive_criterion[0]] > 0
                and not is_lane_change_happening
                and self.push_lane_change
            ):
                # If you have to do a lane change but you can't: slow down the vehicle -------------
                self.false_vehicle = True

        # Propagate planned trajectory for set number of samples
        for _ in range(self._planned_trajectory_samples):
            # Propagate IDM state w.r.t. selected leading agent
            leading_agent = self._get_leading_object(
                ego_idm_state, ego_state, occupancy_map, unique_observations
            )
            self._propagate(
                ego_idm_state, leading_agent, self._planned_trajectory_sample_interval
            )

            # Convert IDM state back to EgoState
            current_time_point += TimePoint(
                int(self._planned_trajectory_sample_interval * 1e6)
            )
            ego_state = self._idm_state_to_ego_state(
                ego_idm_state, current_time_point, vehicle_parameters
            )

            planned_trajectory.append(ego_state)

        return InterpolatedTrajectory(planned_trajectory)

    def _get_ego_state_after_lane_change(
        self,
        ego_state_in_adjacent: EgoState,
        ego_idm_state_in_adjacent: IDMAgentState,
        adjacent_lane: NuPlanLane,
    ):
        """
        TODO
        """
        distance_to_change_lane = max(
            ego_idm_state_in_adjacent.velocity * 1, 1
        )  # TODO add to config
        # Calculate desired ego state after lane change
        adjacent_lane_path = InterpolatedPath(
            convert_se2_path_to_progress_path(adjacent_lane.baseline_path.discrete_path)
        )
        desired_progress = ego_idm_state_in_adjacent.progress + distance_to_change_lane
        desired_progress = min(desired_progress, adjacent_lane_path.get_end_progress())
        desired_progress_state = adjacent_lane_path.get_state_at_progress(
            desired_progress
        )
        desired_ego_state_in_adjacent = EgoState.build_from_center(
            center=StateSE2(
                desired_progress_state.x,
                desired_progress_state.y,
                desired_progress_state.heading,
            ),
            center_velocity_2d=StateVector2D(ego_idm_state_in_adjacent.velocity, 0),
            center_acceleration_2d=ego_state_in_adjacent.dynamic_car_state.center_acceleration_2d,
            tire_steering_angle=0.0,
            time_point=ego_state_in_adjacent.time_point,
            vehicle_parameters=ego_state_in_adjacent.car_footprint.vehicle_parameters,
        )
        return desired_ego_state_in_adjacent

    def _get_ego_path_in_adjacent_lane(
        self, lane: LaneGraphEdgeMapObject, ego_idm_state: IDMAgentState
    ) -> InterpolatedPath:
        """
        Do a graph search to return a ego path beginning from adjacent lane.
        :return: An interpolated path representing the ego's path.
        """
        graph_search = BreadthFirstSearch(
            lane.incoming_edges[0], self._candidate_lane_edge_ids
        )
        # Target depth needs to be offset by one if the starting edge belongs to the second roadblock in the list
        offset = 1 if lane.get_roadblock_id() == self._route_roadblocks[1].id else 0
        route_plan, _ = graph_search.search(
            self._route_roadblocks[-1], len(self._route_roadblocks[offset:])
        )
        discrete_adjacent_ego_path = []
        for lane_in_route in route_plan:
            discrete_adjacent_ego_path.extend(lane_in_route.baseline_path.discrete_path)
        # Adjacent ego path starts from previous lane so its lenghts has to be added to ego idm state
        ego_idm_state.progress += lane.incoming_edges[0].baseline_path.length
        return InterpolatedPath(
            convert_se2_path_to_progress_path(discrete_adjacent_ego_path)
        )

    def _get_expanded_ego_path_in_AL(
        self,
        ego_state: EgoState,
        ego_idm_state: IDMAgentState,
        adjacent_ego_path: InterpolatedPath,
    ) -> Polygon:
        """
        Returns the ego's expanded path in a adjacent lane as a Polygon.
        :return: A polygon representing the ego's path.
        """

        ego_footprint = ego_state.car_footprint
        path_to_go = trim_path(
            adjacent_ego_path,
            max(
                adjacent_ego_path.get_start_progress(),
                min(ego_idm_state.progress, adjacent_ego_path.get_end_progress()),
            ),
            max(
                adjacent_ego_path.get_start_progress(),
                min(
                    ego_idm_state.progress
                    + abs(self._policy.target_velocity) * self._planned_horizon,
                    adjacent_ego_path.get_end_progress(),
                ),
            ),
        )
        expanded_path = path_to_linestring(path_to_go).buffer(
            (ego_footprint.width / 2), cap_style=CAP_STYLE.square
        )
        return unary_union([expanded_path, ego_state.car_footprint.geometry])

    def _get_leading_object_in_AL(
        self,
        ego_idm_state: IDMAgentState,
        ego_state: EgoState,
        occupancy_map: OccupancyMap,
        unique_observations: UniqueObjects,
        lane: LaneGraphEdgeMapObject,
    ) -> IDMLeadAgentState:
        """
        Get the most suitable leading object based on the occupancy map.
        :param ego_idm_state: The ego's IDM state at current iteration.
        :param ego_state: EgoState at current iteration.
        :param occupancy_map: OccupancyMap containing all objects in the scene.
        :param unique_observations: A mapping between the object token and the object itself.
        """
        ego_idm_state = copy(ego_idm_state)

        adjacent_ego_path = self._get_ego_path_in_adjacent_lane(lane, ego_idm_state)
        expanded_ego_path_in_adjacent_lane = self._get_expanded_ego_path_in_AL(
            ego_state, ego_idm_state, adjacent_ego_path
        )
        intersecting_agents = occupancy_map.intersects(
            expanded_ego_path_in_adjacent_lane
        )
        # Check if there are agents intersecting the ego's baseline
        if intersecting_agents.size > 0:
            # Extract closest object
            intersecting_agents.insert(
                self._ego_token, ego_state.car_footprint.geometry
            )
            (
                nearest_id,
                nearest_agent_polygon,
                relative_distance,
            ) = intersecting_agents.get_nearest_entry_to(self._ego_token)

            # Red light at intersection
            if self._red_light_token in nearest_id:
                return self._get_red_light_leading_idm_state(relative_distance)

            # An agent is the leading agent
            return self._get_leading_idm_agent(
                ego_state, unique_observations[nearest_id], relative_distance
            )

        else:
            # No leading agent
            return self._get_free_road_leading_idm_state_in_adjacent(
                ego_state, ego_idm_state, adjacent_ego_path
            )

    def _get_free_road_leading_idm_state_in_adjacent(
        self,
        ego_state: EgoState,
        ego_idm_state: IDMAgentState,
        adjacent_ego_path: InterpolatedPath,
    ) -> IDMLeadAgentState:
        """
        Returns a lead IDM agent state when there is no leading agent.
        :return: A IDM lead agents state.
        """
        projected_velocity = 0.0
        relative_distance = (
            adjacent_ego_path.get_end_progress() - ego_idm_state.progress
        )
        length_rear = ego_state.car_footprint.length / 2
        return IDMLeadAgentState(
            progress=relative_distance,
            velocity=projected_velocity,
            length_rear=length_rear,
        )

    def _get_follower_agent(
        self,
        ego_state: EgoState,
        occupancy_map: OccupancyMap,
        unique_observations: UniqueObjects,
    ) -> Tuple[IDMLeadAgentState, Agent]:
        """
        Get the most suitable leading object based on the occupancy map.
        :param ego_idm_state: The ego's IDM state at current iteration.
        :param ego_state: EgoState at current iteration.
        :param occupancy_map: OccupancyMap containing all objects in the scene.
        :param unique_observations: A mapping between the object token and the object itself.
        """
        extended_occupancy_map = STRTreeOccupancyMap({})
        # Only take into account agents that are not false vehicle because false vehicle should never be behind of ego
        for obs in [
            obs
            for id, obs in unique_observations.items()
            if isinstance(obs, Agent) and id != "false_vehicle"
        ]:
            extended_occupancy_map.set(
                obs.track_token, get_agent_constant_velocity_geometry(obs)
            )

        intersecting_agents = extended_occupancy_map.intersects(
            ego_state.car_footprint.geometry
        )

        for agent_id in intersecting_agents.get_all_ids():
            intersecting_agents.set(agent_id, occupancy_map.get(agent_id))

        # Check if there are agents intersecting the ego's baseline
        if intersecting_agents.size > 0:
            # Extract closest object
            intersecting_agents.insert(
                self._ego_token, ego_state.car_footprint.geometry
            )
            (
                nearest_id,
                nearest_agent_polygon,
                relative_distance,
            ) = intersecting_agents.get_nearest_entry_to(self._ego_token)

            # An agent is the following agent
            return [
                self._get_follower_idm_agent(
                    ego_state, unique_observations[nearest_id], relative_distance
                ),
                unique_observations[nearest_id],
            ]

        else:
            # No Follower agent
            return None, None

    def _get_follower_idm_agent(
        self, ego_state: EgoState, agent: SceneObject, relative_distance: float
    ) -> IDMLeadAgentState:
        """
        Returns a follower IDM agent state that represents another static and dynamic agent.
        :param agent: A scene object.
        :param relative_distance: [m] The relative distance from the scene object to the ego.
        :return: A IDM lead agents state
        """
        if isinstance(agent, Agent):
            # Dynamic object
            longitudinal_velocity = agent.velocity.magnitude()
            # Wrap angle to [-pi, pi]
            relative_heading = principal_value(
                agent.center.heading - ego_state.center.heading
            )
            projected_velocity = transform(
                StateSE2(longitudinal_velocity, 0, 0),
                StateSE2(0, 0, relative_heading).as_matrix(),
            ).x
        else:
            # Static object
            projected_velocity = 0.0

        return IDMAgentState(progress=relative_distance, velocity=projected_velocity)

    def _initialize_route_plan(self, route_roadblock_ids: List[str]) -> None:
        """
        Initializes the route plan with roadblocks.
        :param route_roadblock_ids: A list of roadblock ids that make up the ego's route
        """
        assert (
            self._map_api
        ), "_map_api has not yet been initialized. Please call the initialize() function first!"
        self._route_roadblocks = []
        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(
                id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )
            self._route_roadblocks.append(block)

        self._candidate_lane_edge_ids = []

        # Dictionary for each roadblock on route which contains an array with lenght of the number of lanes of the roadblock
        # and each value of the same position of the lane represent the number of lanes to change to get to a lane on route
        self._distances_to_lane_on_route = {}

        for block in self._route_roadblocks:
            if block:
                lanes_on_route = []
                if not isinstance(block, NuPlanRoadBlockConnector):
                    block.interior_edges.sort(key=lambda x: x.index)
                for edge in block.interior_edges:
                    self._candidate_lane_edge_ids.append(edge.id)
                    for out in edge.outgoing_edges:
                        if out.parent.id in route_roadblock_ids:
                            lanes_on_route.append(1)
                            break
                    else:
                        if (
                            block.id == self._route_roadblocks[-1].id and not self._goal
                        ) or (
                            block.id == self._route_roadblocks[-1].id
                            and edge.contains_point(self._goal)
                        ):
                            lanes_on_route.append(1)
                        else:
                            lanes_on_route.append(0)

                if isinstance(block, NuPlanRoadBlockConnector):
                    continue

                # [1 1 0 0 0] -> [0 0 1 2 3]  / [0 0 1 0 0] -> [2 1 0 1 2]
                if not all(v == 0 for v in lanes_on_route):
                    indices_of_lanes_on_route = np.where(lanes_on_route)
                    indices_of_lanes_on_route = indices_of_lanes_on_route[0]
                    min_IOLOR = min(list(indices_of_lanes_on_route))
                    max_IOLOR = max(list(indices_of_lanes_on_route))
                    for idx, edge in enumerate(
                        block.interior_edges
                    ): 
                        if idx < min_IOLOR:
                            lanes_on_route[idx] = min_IOLOR - idx
                        elif idx > max_IOLOR:
                            lanes_on_route[idx] = idx - max_IOLOR
                        elif idx in indices_of_lanes_on_route:
                            lanes_on_route[idx] = 0

                self._distances_to_lane_on_route[block.id] = lanes_on_route

        assert not [
            edge
            for edge in self._route_roadblocks[-1].outgoing_edges
            if edge in self._route_roadblocks
        ]
        # Propagate route info
        for roadblock in reversed(self._route_roadblocks[:-1]):
            # TODO: doesn't consider lane connectors since mobil doesn't change in lane connectors anyways
            if roadblock.id in self._distances_to_lane_on_route and all(
                v == 0 for v in self._distances_to_lane_on_route[roadblock.id]
            ):
                distances_to_route = self._distances_to_lane_on_route[roadblock.id]
                for lane in roadblock.interior_edges:
                    distances_to_route[lane.index - 1] = min(
                        [
                            self._distances_to_lane_on_route[
                                out.outgoing_edges[0].get_roadblock_id()
                            ][out.outgoing_edges[0].index - 1]
                            if out.outgoing_edges[0].get_roadblock_id()
                            in self._distances_to_lane_on_route
                            else 0
                            for out in lane.outgoing_edges
                        ]
                    )

                self._distances_to_lane_on_route[roadblock.id] = distances_to_route

        assert (
            self._route_roadblocks
        ), "Cannot create route plan. No roadblocks were extracted from the given route_roadblock_ids!"

    def _annotate_occupancy_map(
        self,
        traffic_light_data: List[TrafficLightStatusData],
        occupancy_map: OccupancyMap,
    ) -> None:
        """
        Add red light lane connectors on the route plan to the occupancy map. Note: the function works inline, hence,
        the occupancy map will be modified in this function.
        :param traffic_light_data: A list of all available traffic status data.
        :param occupancy_map: The occupancy map to be annotated.
        """
        assert (
            self._map_api
        ), "_map_api has not yet been initialized. Please call the initialize() function first!"
        assert (
            self._candidate_lane_edge_ids is not None
        ), "_candidate_lane_edge_ids has not yet been initialized. Please call the initialize() function first!"
        for data in traffic_light_data:
            if data.status == TrafficLightStatusType.RED and (
                self._map_api.get_map_object(
                    str(data.lane_connector_id), SemanticMapLayer.LANE_CONNECTOR
                ).parent
                in self._route_roadblocks
                or self._map_api.get_map_object(
                    str(data.lane_connector_id), SemanticMapLayer.LANE_CONNECTOR
                )
                .incoming_edges[0]
                .parent
                in self._route_roadblocks
            ):
                id_ = str(data.lane_connector_id)
                lane_conn = self._map_api.get_map_object(
                    id_, SemanticMapLayer.LANE_CONNECTOR
                )
                occupancy_map.insert(
                    f"{self._red_light_token}_{id_}", lane_conn.polygon
                )

        for id in occupancy_map.get_all_ids():
            if id.split("_")[0] == "red":
                t_light_lane_c = self._map_api.get_map_object(
                    id.split("_")[2], SemanticMapLayer.LANE_CONNECTOR
                )
                paralel_lanes_to_t_light = t_light_lane_c.incoming_edges[
                    0
                ].outgoing_edges

                if t_light_lane_c.parent not in self._route_roadblocks and len(
                    [
                        True
                        for edge in paralel_lanes_to_t_light
                        if "red_light_" + edge.id in occupancy_map.get_all_ids()
                    ]
                ) < len(paralel_lanes_to_t_light):
                    occupancy_map.remove(["red_light_" + t_light_lane_c.id])
