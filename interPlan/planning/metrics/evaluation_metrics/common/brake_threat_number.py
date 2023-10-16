from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Any

import numpy as np
import numpy.typing as npt
from sympy import symbols, solve

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox, in_collision
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import (
    Collisions,
    EgoAtFaultCollisionStatistics,
)
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.state_extractors import extract_ego_time_point, extract_ego_velocity
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind
from nuplan.planning.simulation.observation.observation_type import Observation

# Typing for trajectory of tracks pose, speed and box
TRACKS_POSE_SPEED_BOX = Tuple[
    List[npt.NDArray[np.float64]], List[npt.NDArray[np.float64]], List[npt.NDArray[Any]]
]


@dataclass
class EgoTracksDisplacementInfo:
    """
    Relevent pose, displacement information for ego and tracks supporting time to collision calculations.
    """

    ego_pose: npt.NDArray[np.float64]  # x, y, heading
    ego_box: OrientedBox
    ego_dx: float
    ego_dy: float
    tracks_dxy: npt.NDArray[np.float64]  # <num_tracks, dx, dy>


def extract_tracks_info_excluding_collided_tracks(
    ego_states: List[EgoState],
    ego_timestamps: npt.NDArray[np.int64],
    observations: List[Observation],
    all_collisions: List[Collisions],
    timestamps_in_common_or_connected_route_objs: List[int],
    map_api: AbstractMap,
) -> TRACKS_POSE_SPEED_BOX:
    """
    Extracts arrays of tracks pose, speed and oriented box for TTC: all lead and cross tracks, plus lateral tracks if ego is in
    between lanes or in nondrivable area or in intersection.

    :param ego_states: A list of ego states
    :param ego_timestamps: Array of times in time_us
    :param observations: A list of observations
    :param all_collisions: List of all collisions in the history
    :param timestamps_in_common_or_connected_route_objs: List of timestamps where ego is in same or connected
        lanes/lane connectors
    :param map_api: map api.
    :return: A tuple of lists of arrays of tracks pose, speed and represented box at each timestep.
    """
    collided_track_ids: Set[str] = set()

    history_tracks_poses: List[npt.NDArray[np.float64]] = []
    history_tracks_speed: List[npt.NDArray[np.float64]] = []
    history_tracks_boxes: List[npt.NDArray[Any]] = []
    history_track_tokens: List[npt.NDArray[Any]] = []

    collision_time_dict = {
        collision.timestamp: list(collision.collisions_id_data.keys()) for collision in all_collisions
    }

    for index, (ego_state, timestamp, observation) in enumerate(zip(ego_states, ego_timestamps, observations)):

        # Add the collided tracks at this timestamp to be excluded from later timestamps
        collided_track_ids = collided_track_ids.union(set(collision_time_dict.get(timestamp, [])))

        # Check if ego is in between lanes or between/in nondrivable area, in this case we consider
        # lead/cross and lateral tracks for TTC, otherwise only lead/cross tracks are considered in TTC.
        ego_not_in_common_or_connected_route_objs = timestamp not in timestamps_in_common_or_connected_route_objs

        tracked_objects = [
            tracked_object
            for tracked_object in observation.tracked_objects
            if tracked_object.track_token not in collided_track_ids
            and (
                is_agent_ahead(ego_state.rear_axle, tracked_object.center)
                or (
                    (
                        ego_not_in_common_or_connected_route_objs
                        or map_api.is_in_layer(ego_state.rear_axle, layer=SemanticMapLayer.INTERSECTION)
                    )
                    and not is_agent_behind(ego_state.rear_axle, tracked_object.center)
                )
            )
        ]

        poses: List[npt.NDArray[np.float64]] = [
            np.array([*tracked_object.center], dtype=np.float64) for tracked_object in tracked_objects
        ]

        speeds: List[npt.NDArray[np.float64]] = [
            np.array(tracked_object.velocity.magnitude(), dtype=np.float64) if isinstance(tracked_object, Agent) else 0  # type: ignore
            for tracked_object in tracked_objects
        ]

        boxes: List[OrientedBox] = [tracked_object.box for tracked_object in tracked_objects]

        track_tokens: List[str] = [tracked_object.track_token for tracked_object in tracked_objects] # TODO delete

        history_tracks_poses.append(np.array(poses))
        history_tracks_speed.append(np.array(speeds))
        history_tracks_boxes.append(np.array(boxes))
        history_track_tokens.append(track_tokens)

    return history_tracks_poses, history_tracks_speed, history_tracks_boxes, history_track_tokens


def _get_elongated_box_length(length: float, dx: float, dy: float, time_step_size: float, time_horizon: float) -> float:
    """
    Helper to find the length of an elongated box projected up to a given time horizon.
    :param length: The length of the OrientedBox.
    :param dx: Movement in x axis in global frame at each time_step_size.
    :param dy: Movement in y axis in global frame at each time_step_size.
    :param time_step_size: [s] Step size for the propagation of collision agents.
    :param time_horizon: [s] Time horizon for collision checking.
    :return: Length of elonated box up to time horizon.
    """
    return float(length + np.hypot(dx * time_horizon / time_step_size, dy * time_horizon / time_step_size))


def _get_ego_tracks_displacement_info(
    ego_state: EgoState,
    ego_speed: npt.NDArray[np.float64],
    tracks_poses: npt.NDArray[np.float64],
    tracks_speed: npt.NDArray[np.float64],
    time_step_size: float,
) -> EgoTracksDisplacementInfo:
    """
    Helper function for compute_time_to_collision. Gets relevent pose, displacement values for TTC calculations.
    :param ego_state: Ego state.
    :param ego_speed: Ego speed.
    :param tracks_poses: Array of tracks poses.
    :param tracks_speed: Array of tracks speeds.
    :param time_step_size: [s] Step size for the propagation of collision agents.
    :return: Relevent pose, displacement information for ego and tracks supporting time to collision calculations.
    """
    ego_pose: npt.NDArray[np.float64] = np.array([*ego_state.center], dtype=np.float64)
    ego_box = ego_state.car_footprint.oriented_box

    # Find ego movements in the global frame
    ego_dx = np.cos(ego_pose[2]) * ego_speed * time_step_size
    ego_dy = np.sin(ego_pose[2]) * ego_speed * time_step_size

    # Find tracks' movements in the global frame, assume all tracks also follow the bicycle dynamic model
    tracks_dxy: npt.NDArray[np.float64] = np.array(
        [
            np.cos(tracks_poses[:, 2]) * tracks_speed * time_step_size,
            np.sin(tracks_poses[:, 2]) * tracks_speed * time_step_size,
        ],
        dtype=np.float64,
    ).T

    return EgoTracksDisplacementInfo(ego_pose, ego_box, ego_dx, ego_dy, tracks_dxy)


def _get_relevant_tracks(
    ego_pose: npt.NDArray[np.float64],
    ego_box: OrientedBox,
    ego_dx: float,
    ego_dy: float,
    tracks_poses: npt.NDArray[np.float64],
    tracks_boxes: List[OrientedBox],
    tracks_dxy: npt.NDArray[np.float64],
    time_step_size: float,
    time_horizon: float,
) -> npt.NDArray[np.int64]:
    """
    Find relevant tracks affecting time to collision, determined by overlapping boxes elongated according to current
      movement.
    :param ego_pose: Ego pose.
    :param ego_box: Oriented box of ego.
    :param ego_dx: Movement in x axis in global frame at each time_step_size.
    :param ego_dy: Movement in y axis in global frame at each time_step_size.
    :param tracks_poses: Pose for each track.
    :param tracks_boxes: Oriented box for each track.
    :param tracks_dxy: Tracks' movements in the global frame
    :param time_step_size: [s] Step size for the propagation of collision agents.
    :param time_horizon: [s] Time horizon for collision checking.
    :return: Indices for tracks revlevant to time to collision calculations.
    """
    # Find the center of elongated boxes if ego and tracks continue their movement with
    # the same speed and heading
    ego_elongated_box_center_pose: npt.NDArray[np.float64] = np.array(
        [
            (time_horizon / time_step_size) / 2 * ego_dx + ego_pose[0],
            (time_horizon / time_step_size) / 2 * ego_dy + ego_pose[1],
            ego_pose[2],
        ],
        dtype=np.float64,
    )

    ego_elongated_box = OrientedBox(
        StateSE2(*ego_elongated_box_center_pose),
        _get_elongated_box_length(ego_box.length, ego_dx, ego_dy, time_step_size, time_horizon),
        ego_box.width,
        ego_box.height,
    )

    # Project tracks poses up to the time_horizon
    tracks_elongated_box_center_poses: npt.NDArray[np.float64] = np.concatenate(
        (
            (time_horizon / time_step_size) / 2 * tracks_dxy + tracks_poses[:, :2],
            tracks_poses[:, 2].reshape(-1, 1),
        ),
        axis=1,
    )
    # Find the convex hulls including tracks initial and projected corners
    tracks_elongated_boxes = [
        OrientedBox(
            StateSE2(*track_elongated_box_center_pose),
            _get_elongated_box_length(track_box.length, track_dxy[0], track_dxy[1], time_step_size, time_horizon),
            track_box.width,
            track_box.height,
        )
        for track_box, track_dxy, track_elongated_box_center_pose in zip(
            tracks_boxes, tracks_dxy, tracks_elongated_box_center_poses
        )
    ]

    # Find relevant tracks for which the elongated box overlaps with ego elongated box
    relevant_tracks_mask = np.where(
        [in_collision(ego_elongated_box, track_elongated_box) for track_elongated_box in tracks_elongated_boxes]
    )[0]

    return relevant_tracks_mask


def _get_index_of_observation_in_collision(
    timestamp: int,
    ego_state: EgoState,
    ego_speed: npt.NDArray[np.float64],
    tracks_poses: npt.NDArray[np.float64],
    tracks_speed: npt.NDArray[np.float64],
    tracks_boxes: List[OrientedBox],
    timestamps_at_fault_collisions: List[int],
    time_step_size: float,
    time_horizon: float,
    stopped_speed_threshold: float,
) -> int:
    """
    Helper function for compute_time_to_collision. Computes time to collision value at given timestamp.
    :param timestamp: Time in time_us.
    :param ego_state: Ego state.
    :param ego_speed: Ego speed.
    :param tracks_poses: Pose for each track.
    :param tracks_speed: Array of tracks speeds.
    :param tracks_boxes: Oriented box for each track.
    :param timestamps_at_fault_collisions: List of timestamps corresponding to at-fault-collisions in the history.
    :param time_step_size: [s] Step size for the propagation of collision agents.
    :param time_horizon: [s] Time horizon for collision checking.
    :param stopped_speed_threshold: Threshold for 0 speed due to noise.
    :return: Computed time to collision if available, otherwise None.
    """
    _tracks_poses = tracks_poses.copy() # So that track poses don't get modified here

    ego_in_at_fault_collision = timestamp in timestamps_at_fault_collisions
    # TODO Set TTC to 0 if ego is in an at-fault collision
    if ego_in_at_fault_collision:
        return None

    # Remain default if we don't have any agents or ego is stopped
    if len(tracks_poses) == 0 or ego_speed <= stopped_speed_threshold:
        return None

    displacement_info = _get_ego_tracks_displacement_info(
        ego_state, ego_speed, tracks_poses, tracks_speed, time_step_size
    )
    relevant_tracks_mask = _get_relevant_tracks(
        displacement_info.ego_pose,
        displacement_info.ego_box,
        displacement_info.ego_dx,
        displacement_info.ego_dy,
        tracks_poses,
        tracks_boxes,
        displacement_info.tracks_dxy,
        time_step_size,
        time_horizon,
    )

    # If there is no relevant track affecting TTC, remain default
    if not len(relevant_tracks_mask):
        return None

    # Find TTC for relevant tracks by projecting ego and tracks boxes with time_step_size
    for _ in np.arange(time_step_size, time_horizon, time_step_size):
        # project ego's center pose and footprint with a fixed speed
        displacement_info.ego_pose[:2] += (displacement_info.ego_dx, displacement_info.ego_dy)
        projected_ego_box = OrientedBox.from_new_pose(
            displacement_info.ego_box, StateSE2(*(displacement_info.ego_pose))
        )
        # project tracks's center pose and footprint with a fixed speed
        _tracks_poses[:, :2] += displacement_info.tracks_dxy
        for index, (track_box, track_pose) in enumerate(zip(tracks_boxes[relevant_tracks_mask], _tracks_poses[relevant_tracks_mask])):
            projected_track_box = OrientedBox.from_new_pose(track_box, StateSE2(*track_pose))
            if in_collision(projected_ego_box, projected_track_box):
                return relevant_tracks_mask[index]

    return None


def compute_time_to_collision(
    ego_states: List[EgoState],
    ego_timestamps: npt.NDArray[np.int64],
    observations: List[Observation],
    timestamps_in_common_or_connected_route_objs: List[int],
    all_collisions: List[Collisions],
    timestamps_at_fault_collisions: List[int],
    map_api: AbstractMap,
    time_step_size: float,
    time_horizon: float,
    stopped_speed_threshold: float = 5e-03,
) -> npt.NDArray[np.float64]:
    """
    Computes an estimate of the minimal time to collision with other agents. Ego and agents are projected
    with constant velocity until there is a collision or the maximal time window is reached.
    :param ego_states: A list of ego states.
    :param ego_timestamps: Array of times in time_us.
    :param observations: Observations to consider collisions with ego states.
    :param timestamps_in_common_or_connected_route_objs: List of timestamps where ego is in same or connected
    lanes/lane connectors.
    :param all_collisions: List of all collisions in the history.
    :param timestamps_at_fault_collisions: List of timestamps corresponding to at-fault-collisions in the history.
    :param map_api: Map to consider.
    :param time_step_size: [s] Step size for the propagation of collision agents.
    :param time_horizon: [s] Time horizon for collision checking.
    :param stopped_speed_threshold: Threshold for 0 speed due to noise.
    :return: The minimal TTC for each sample, inf if no collision is found within the projection horizon.
    """
    # Extract speed of ego from ego states.
    ego_velocities = extract_ego_velocity(ego_states)

    # Extract tracks info, collided tracks are removed from the analysis after the first timestamp of the collision
    (history_tracks_poses, history_tracks_speed, history_tracks_boxes, history_tracks_indexes) = extract_tracks_info_excluding_collided_tracks(
        ego_states,
        ego_timestamps,
        observations,
        all_collisions,
        timestamps_in_common_or_connected_route_objs,
        map_api,
    )

    # Default TTC to be inf
    brake_threat_number: npt.NDArray[np.float64] = np.asarray([np.inf] * len(ego_states), dtype=np.float64)
    brake_threat_number[0] = 0 # There is no acceleration information in first time step so BTN in first frame is 0 TODO

    for timestamp_index, (timestamp, ego_state, ego_speed, tracks_poses, tracks_speed, tracks_boxes,) in enumerate(
        zip(
            ego_timestamps,
            ego_states,
            ego_velocities,
            history_tracks_poses,
            history_tracks_speed,
            history_tracks_boxes,
        )
    ):
        observation_index = _get_index_of_observation_in_collision(
            timestamp,
            ego_state,
            ego_speed,
            tracks_poses,
            tracks_speed,
            tracks_boxes,
            timestamps_at_fault_collisions,
            time_step_size,
            time_horizon,
            stopped_speed_threshold,
        )
        #track_index = history_tracks_indexes[observation_index]
        ego_state: EgoState # TODO delete
        if timestamp_index and observation_index is not None:
            # target agent in previous timestamp for acceleration calculation purposes
            try:
                target_in_previous_TS: Agent = next(agent for agent in observations[timestamp_index-1].tracked_objects.tracked_objects if 
                                            agent.track_token == history_tracks_indexes[timestamp_index][observation_index])
            except StopIteration: 
                # If this agent didn't exist in previous iteration we assume it hasn't move so prev_agent=current_agent
                target_in_previous_TS: Agent = next(agent for agent in observations[timestamp_index].tracked_objects.tracked_objects if 
                                            agent.track_token == history_tracks_indexes[timestamp_index][observation_index])
            range_to_target_0 = ego_state.center.distance_to(StateSE2(*tracks_poses[observation_index]))
            target_stopped = tracks_speed[observation_index] < 0.01
            target_accelerating = timestamp_index and history_tracks_speed[timestamp_index][observation_index] - \
                target_in_previous_TS.velocity.magnitude() < 0.01
            target_aligned_with_ego = abs(ego_state.center.heading - tracks_poses[observation_index][2]) < 0.4 
            if (target_stopped and not target_accelerating) or not target_aligned_with_ego:
                required_acceleration = -(ego_speed**2)/(2*(range_to_target_0 + ego_state.car_footprint.half_length))
                brake_threat_number[timestamp_index] = required_acceleration / -3.0 # TODO
            else:
                time = symbols("time")
                jmin = -10
                target_speed = tracks_speed[observation_index] # Speed of the target agent
                target_acceleration =  (target_speed - target_in_previous_TS.velocity.magnitude()) / \
                    ((ego_timestamps[timestamp_index] - ego_timestamps[timestamp_index - 1])*10**-6)
                ego_acceleration = ego_state.dynamic_car_state.acceleration
                range_to_target = range_to_target_0 + (target_speed - ego_speed) * time + \
                                    (((target_acceleration - ego_acceleration) * time**2 ) / 2) - jmin * time**3 / 6
                range_rate_to_target = (target_speed - ego_speed) + ((target_acceleration - ego_acceleration) * time) - jmin * time ** 2 / 2
                host_accel = ego_acceleration + jmin * time
                accel_to_avoid_colission = target_acceleration - (range_rate_to_target**2 / (2*range_to_target))
                t_x = solve(host_accel - accel_to_avoid_colission)
                for number in t_x:
                    try: 
                        float_n = float(number)
                        if float_n > 0:
                            t_x = float_n
                            break
                    except: continue
                if type(t_x) == float:
                    a_rec = ego_acceleration + t_x * jmin
                    brake_threat_number[timestamp_index] = a_rec / -3
                else: # Target stops before coliding
                    r_t = - target_speed**2 / ( 2 * target_acceleration)
                    range_to_target = range_to_target_0 + r_t - ego_speed * time - ((ego_acceleration * time**2 ) / 2) - (jmin * time ** 3) / 6
                    range_rate_to_target = - ego_speed - ego_acceleration * time - jmin * time ** 2 / 2
                    accel_to_avoid_colission = target_acceleration - (range_rate_to_target**2 / (2*range_to_target))
                    t_x = solve(host_accel - accel_to_avoid_colission)
                    for number in t_x:
                        try: 
                            float_n = float(number)
                            if float_n > 0:
                                t_x = float_n
                                break
                        except: continue
                    a_rec = ego_acceleration + t_x * jmin
                    brake_threat_number[timestamp_index] = a_rec / -3

    return brake_threat_number


class BrakeThreatNumberStatistics(MetricBase):
    """
    Ego time to collision metric, reports the minimal time for a projected collision if agents proceed with
    zero acceleration.
    """

    def __init__(
        self,
        name: str,
        category: str,
        ego_lane_change_metric: EgoLaneChangeStatistics,
        no_ego_at_fault_collisions_metric: EgoAtFaultCollisionStatistics,
        time_step_size: float,
        time_horizon: float,
        least_min_ttc: float,
        metric_score_unit: Optional[str] = None,
    ):
        """
        Initializes the TimeToCollisionStatistics class
        :param name: Metric name
        :param category: Metric category
        :param ego_lane_change_metric: Lane chang metric computed prior to calling the current metric
        :param no_ego_at_fault_collisions_metric: Ego at fault collisions computed prior to the current metric
        :param time_step_size: [s] Step size for the propagation of collision agents
        :param time_horizon: [s] Time horizon for collision checking
        :param least_min_ttc: minimum desired TTC.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)

        self._time_step_size = time_step_size
        self._time_horizon = time_horizon
        self._least_min_ttc = least_min_ttc

        # Initialize lower level metrics
        self._ego_lane_change_metric = ego_lane_change_metric
        self._no_ego_at_fault_collisions_metric = no_ego_at_fault_collisions_metric

        # save to load in higher level metrics
        self.results: List[MetricStatistics] = []

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        # Return 1.0 if time_to_collision_bound is True, otherwise 0
        return float(metric_statistics[-1].value)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the time to collision statistics
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the time to collision metric
        """
        # Load pre-calculated timestamps from ego_lane_change_metric
        timestamps_in_common_or_connected_route_objs: List[
            int
        ] = self._ego_lane_change_metric.timestamps_in_common_or_connected_route_objs

        # Load pre-calculated results from ego_at_fault_collision metric
        assert (
            self._no_ego_at_fault_collisions_metric.results
        ), "no_ego_at_fault_collisions metric must be run prior to calling {}".format(self.name)

        all_collisions = self._no_ego_at_fault_collisions_metric.all_collisions
        timestamps_at_fault_collisions = self._no_ego_at_fault_collisions_metric.timestamps_at_fault_collisions

        # Extract states of ego from history.
        ego_states = history.extract_ego_state

        # Extract ego timepoints
        ego_timestamps = extract_ego_time_point(ego_states)

        # Extract observation from history.
        observations = [sample.observation for sample in history.data]

        brake_threat_number = compute_time_to_collision(
            ego_states,
            ego_timestamps,
            observations,
            timestamps_in_common_or_connected_route_objs,
            all_collisions,
            timestamps_at_fault_collisions,
            history.map_api,
            self._time_step_size,
            self._time_horizon,
        )

        brake_threat_number_relevant = [number for number in brake_threat_number if number != np.inf]
        brake_threat_number_mean = sum(brake_threat_number_relevant)/len(brake_threat_number_relevant)

        brake_threat_number = [number if number != np.inf else 0 for number in brake_threat_number]


        #time_to_collision_within_bounds = self._least_min_ttc < np.array(time_to_collision, dtype=np.float64)
        time_series = TimeSeries(
            unit='BTN_under_' + f'{self._time_horizon}' + '_seconds [s]',
            time_stamps=list(ego_timestamps),
            values=list(brake_threat_number),
        )
        metric_statistics = [
            Statistic(
                name='max_BTN',
                unit="",
                value=np.max(brake_threat_number),
                type=MetricStatisticsType.MAX,
            ),
            Statistic(
                name=f'{self.name}',
                unit="",
                value=brake_threat_number_mean,
                type=MetricStatisticsType.MEAN,
            ),
        ]
        # Save to load in high level metrics
        self.results = self._construct_metric_results(
            metric_statistics=metric_statistics,
            time_series=time_series,
            scenario=scenario,
            metric_score_unit=self.metric_score_unit,
        )

        return self.results
