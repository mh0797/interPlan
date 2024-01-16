from typing import List, Optional
from shapely import Point

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.ego_progress_along_expert_route import (
    EgoProgressAlongExpertRouteStatistics,
)
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring


class EgoSortsConstructionZoneStatistics(MetricBase):
    """
    Check if ego trajectory is making progress along expert route more than a minimum required progress.
    """

    def __init__(
        self,
        name: str,
        category: str,
        ego_progress_along_expert_route_metric: EgoProgressAlongExpertRouteStatistics,
        metric_score_unit: Optional[str] = None,
    ) -> None:
        """
        Initializes the EgoSortsConstructionZoneStatistics class
        :param name: Metric name
        :param category: Metric category
        :param ego_progress_along_expert_route_metric: Ego progress along expert route metric
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)

        # Initialize lower level metrics
        self._ego_progress_along_expert_route_metric = ego_progress_along_expert_route_metric

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[0].value)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the ego_is_making_progress metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the estimated metric.
        """
        # Get Ego Expert trajectory
        ego_expert_linestring = path_to_linestring([state.waypoint for state in scenario.get_expert_ego_trajectory()])

        # Get cones location
        cones_observations = \
            scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.TRAFFIC_CONE)

        min_progress = min(ego_expert_linestring.project(Point(*cone.center)) for cone in cones_observations) \
            / ego_expert_linestring.length
            
        # Load ego_progress_along_expert_route ratio
        ego_sorts_construction_zone = (
            self._ego_progress_along_expert_route_metric.results[0].statistics[-1].value >= min_progress
        )
        statistics = [
            Statistic(
                name='ego_sorts_construction_zone',
                unit='boolean',
                value=ego_sorts_construction_zone,
                type=MetricStatisticsType.BOOLEAN,
            )
        ]

        results = self._construct_metric_results(
            metric_statistics=statistics, time_series=None, scenario=scenario, metric_score_unit=self.metric_score_unit
        )

        return results  # type: ignore
