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
        # To get weighted_total_scenarios
        scenario_weights = [self.scenario_type_weights[key] for key in final_score_dicts["final_score"]["scenario_type"]]
        scenario_numbers = final_score_dicts['final_score']['num_scenarios'] 
        weighted_total_scenarios = sum([x * y for x, y in zip(scenario_numbers, scenario_weights)])

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
                        if key == "score":
                            total_values = np.sum(available_value_array) / weighted_total_scenarios
                        elif len(available_value_array) != total_scenarios:
                            total_values = np.sum(available_value_array) / sum([x for x, v in zip(scenario_numbers, values) if v != None])
                        else:
                            total_values = np.sum(available_value_array) / total_scenarios

                    final_score_metric_columns[final_score_column_name][key] = total_values
        return final_score_metric_columns

    def _compute_scenario_score(self, scenario_metric_columns: metric_aggregator_dict_column) -> None:
        """
        Compute scenario scores.
        :param scenario_metric_columns: Scenario metric column in the format of {scenario_names: {metric_column:
        value}}.
        """
        excluded_columns = ['log_name', 'planner_name', 'aggregator_type', 'scenario_type', 'num_scenarios', 'score']
        for scenario_name, columns in scenario_metric_columns.items():
            metric_scores = 0.0
            sum_weights = 0.0
            multiple_factor = 1.0
            for column_key, column_value in columns.items():
                # Skip if column key is excluded or the value is None
                if column_key in excluded_columns or column_value is None:
                    continue
                scenario_multiple_metrics = self._multiple_metrics.get(columns["scenario_type"], self._multiple_metrics["default"])
                if scenario_multiple_metrics and column_key in scenario_multiple_metrics:
                    multiple_factor *= column_value
                else:
                    weight = self._get_metric_weight(metric_name=column_key)
                    assert column_value is not None, f"Metric: {column_key} value should not be None!"
                    assert weight is not None, f"Metric: {column_key} weight " f"should not be None!"
                    sum_weights += weight
                    metric_scores += weight * column_value
            weighted_average_score = metric_scores / sum_weights if sum_weights else 0.0
            final_score = multiple_factor * weighted_average_score
            scenario_metric_columns[scenario_name]['score'] = final_score

    def __call__(self, metric_dataframes: Dict[str, MetricStatisticsDataFrame]) -> None:
        """
        Run an aggregator to generate an aggregated parquet file.
        :param metric_dataframes: A dictionary of metric name and dataframe.
        """
        # Get all planner names
        planner_names = sorted(
            list(
                {
                    planner_name
                    for metric_statistic_dataframe in metric_dataframes.values()
                    for planner_name in metric_statistic_dataframe.planner_names
                }
            )
        )

        weighted_average_dataframe_columns: Dict[str, List[Any]] = dict()
        for planner_name in planner_names:

            metric_names = sorted(list(metric_dataframes.keys())) + ['score']
            dataframe_columns: Dict[str, List[Any]] = {
                'scenario': [],
                'log_name': [],
                'scenario_type': [],
                'num_scenarios': [],
                'planner_name': [],
                'aggregator_type': [],
            }
            not_metric_keys_list = list(dataframe_columns.keys())
            metric_name_columns: Dict[str, List[float]] = {metric_name: [] for metric_name in metric_names}
            dataframe_columns.update(metric_name_columns)
            # Group scenario metrics
            scenario_metric_columns = self._group_scenario_metrics(
                metric_dataframes=metric_dataframes, planner_name=planner_name
            )

            # Compute scenario scores
            self._compute_scenario_score(scenario_metric_columns=scenario_metric_columns)
            # Get metric columns based on scenario types
            scenario_type_metric_columns = self._group_scenario_type_metric(
                scenario_metric_columns=scenario_metric_columns
            )

            # Compute a final score based on scenario types
            scenario_type_final_metric_columns = self._group_final_score_metric(
                scenario_type_metric_columns=scenario_type_metric_columns
            )

            # Append scenario type metric columns to scenario metric columns
            scenario_metric_columns.update(scenario_type_metric_columns)

            # Append final_score metric columns to scenario metric columns
            scenario_metric_columns.update(scenario_type_final_metric_columns)

            # Arrange columns into dict format
            for scenario_name, columns in scenario_metric_columns.items():
                dataframe_columns['scenario'].append(scenario_name)
                for key, value in columns.items():
                    dataframe_columns[key].append(value)
            if not weighted_average_dataframe_columns:
                weighted_average_dataframe_columns.update(dataframe_columns)
            else:
                for column_name, value in weighted_average_dataframe_columns.items():
                    value += dataframe_columns[column_name]
        # Replace values that are None in the metric lists, otherwise nuboard crashes when visualizing
        for key, value in weighted_average_dataframe_columns.items():
            if None in value and any(x is not None for x in value) and key not in not_metric_keys_list: 
                weighted_average_dataframe_columns[key] = [x if x is not None else 0.5 for x in value]
        # Convert to pandas dataframe
        self._aggregated_metric_dataframe = pandas.DataFrame(data=weighted_average_dataframe_columns)

        # Save to a parquet file
        self._save_parquet(dataframe=self._aggregated_metric_dataframe, save_path=self._parquet_file)