from __future__ import annotations

import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pandas

from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.metrics.aggregator.weighted_average_metric_aggregator import WeightedAverageMetricAggregator
from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame

logger = logging.getLogger(__name__)
metric_aggregator_dict_column = Dict[str, Dict[str, Any]]


class InterplanWeightedAverageMetricAggregator(WeightedAverageMetricAggregator):
    """Metric aggregator by implementing weighted sum."""

    def __init__(
        self,
        name: str,
        metric_weights: Dict[str, float],
        file_name: str,
        aggregator_save_path: Path,
        multiple_metrics: List[str],
        scenario_type_weights: Dict[str, int],
        challenge_name: Optional[str] = None,
    ):
        super().__init__(name, metric_weights, file_name, aggregator_save_path, multiple_metrics, challenge_name)
        self.scenario_type_weights = scenario_type_weights

    def _group_final_score_metric(
        self,
        scenario_type_metric_columns: metric_aggregator_dict_column,
    ) -> metric_aggregator_dict_column:
        """
        Compute a final score based on a group of scenario types.
        :param scenario_type_metric_columns: Scenario type metric columns in the format of {scenario_type:
        {metric_column: value}}.
        :return A dictionary of final score in the format of {'final_score': {metric_column: value}}.
        """
        # Transform to final_score: {}
        final_score_dicts: metric_aggregator_dict_column = defaultdict(lambda: defaultdict(list))
        for scenario_type, columns in scenario_type_metric_columns.items():
            for column_key, column_value in columns.items():
                final_score_dicts['final_score'][column_key].append(column_value)

        final_score_metric_columns: metric_aggregator_dict_column = defaultdict(lambda: defaultdict())
        total_scenarios = sum(final_score_dicts['final_score']['num_scenarios'])
        # Column get only first index value
        common_columns = ['planner_name', 'aggregator_type']
        for final_score_column_name, columns in final_score_dicts.items():
            for key, values in columns.items():
                if key == 'scenario_type':
                    final_score_metric_columns[final_score_column_name][key] = 'final_score'
                elif key == 'log_name':
                    final_score_metric_columns[final_score_column_name][key] = None
                elif key in common_columns:
                    final_score_metric_columns[final_score_column_name][key] = values[0]
                elif key == 'num_scenarios':
                    final_score_metric_columns[final_score_column_name][key] = total_scenarios
                else:
                    available_values: List[float] = []
                    if key == 'score':
                        for value, num_scenario in zip(values, columns['num_scenarios']):
                            if value is not None:
                                available_values.append(value * num_scenario * self.scenario_type_weights[columns["scenario_type"][0]])
                    else:
                        available_values = [value for value in values if value is not None]

                    if not available_values:
                        total_values = None
                    else:
                        available_value_array: npt.NDArray[np.float64] = np.asarray(available_values)
                        total_values = np.sum(available_value_array) / total_scenarios

                    final_score_metric_columns[final_score_column_name][key] = total_values
        return final_score_metric_columns

    