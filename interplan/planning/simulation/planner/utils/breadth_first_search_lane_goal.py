from collections import deque
from typing import Dict, List, Optional, Tuple

from nuplan.common.maps.abstract_map_objects import (
    LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject)
from nuplan.common.maps.nuplan_map.lane import NuPlanLane


class BreadthFirstSearch:
    """
    A class that performs iterative breadth first search. The class operates on lane level graph search.
    The goal condition is specified to be if the lane can be found at the target roadblock or roadblock connector.
    """

    def __init__(self, start_edge: LaneGraphEdgeMapObject, candidate_lane_edge_ids: List[str]):
        """
        Constructor for the BreadthFirstSearch class.
        :param start_edge: The starting edge for the search
        :param candidate_lane_edge_ids: The candidates lane ids that can be included in the search.
        """
        self._queue = deque([start_edge, None])
        self._parent: Dict[str, Optional[LaneGraphEdgeMapObject]] = dict()
        self._candidate_lane_edge_ids = candidate_lane_edge_ids

    def search(
        self, target_lane: LaneGraphEdgeMapObject, target_depth: int
    ) -> Tuple[List[LaneGraphEdgeMapObject], bool]:
        """
        Performs iterative breadth first search to find a route to the target lane.
        :param target_lane: The target lane the path should end at.
        :param target_depth: The target depth the lane should be at.
        :return:
            - A route starting from the given start edge
            - A bool indicating if the route is successfully found. Successful means that there exists a path
              from the start edge to an edge contained in the end lane. If unsuccessful a longest route is given.
        """
        start_edge = self._queue[0]

        # Initial search states
        path_found: bool = False
        end_edge: LaneGraphEdgeMapObject = start_edge
        end_depth: int = 1
        depth: int = 1
        closest_distance_to_goal: int = 10000
        lane_change = False

        self._parent[start_edge.id + f"_{depth}"] = None

        while self._queue:
            current_edge = self._queue.popleft()

            # Early exit condition
            if self._check_end_condition(depth, target_depth):
                break

            # Depth tracking
            if current_edge is None:
                depth += 1
                self._queue.append(None)
                if self._queue[0] is None:
                    break
                continue

            # Goal condition
            if self._check_goal_condition(current_edge, target_lane):
                end_edge = current_edge
                end_depth = depth
                path_found = True
                lane_change = False
                break
            # If not in goal lane, check if it is in goal roadblock
            elif self._check_goal_roadblock(current_edge, target_lane.parent):
                if isinstance(target_lane, NuPlanLane):
                    distance_to_goal = abs(target_lane.index - current_edge.index)
                    if distance_to_goal < closest_distance_to_goal:
                        closest_distance_to_goal = distance_to_goal
                        self._parent[target_lane.id + f"_{depth}"] = self._parent[current_edge.id + f"_{depth}"]
                        end_edge = target_lane
                        end_depth = depth
                        path_found = True
                        lane_change = True
                    

            # Populate queue
            for next_edge in current_edge.outgoing_edges:
                if next_edge.id in self._candidate_lane_edge_ids:
                    self._queue.append(next_edge)
                    self._parent[next_edge.id + f"_{depth + 1}"] = current_edge
                    end_edge = next_edge
                    end_depth = depth + 1

        return self._construct_path(end_edge, end_depth), path_found, lane_change

    @staticmethod
    def _check_end_condition(depth: int, target_depth: int) -> bool:
        """
        Check if the search should end regardless if the goal condition is met.
        :param depth: The current depth to check.
        :param target_depth: The target depth to check against.
        :return: True if:
            - The current depth exceeds the target depth.
        """
        return depth > target_depth

    @staticmethod
    def _check_goal_condition(
        current_edge: LaneGraphEdgeMapObject,
        target_lane: LaneGraphEdgeMapObject,
    ) -> bool:
        """
        Check if the current edge is at the target lane
        :param current_edge: The edge to check.
        :param target_roadblock: The target roadblock the edge should be contained in.
        :return: True if the lane edge is contain the in the target roadblock at the target depth. False, otherwise.
        """
        return current_edge.id == target_lane.id 
    
    @staticmethod
    def _check_goal_roadblock(
        current_edge: LaneGraphEdgeMapObject,
        target_roadblock: RoadBlockGraphEdgeMapObject,
    ) -> bool:
        """
        Check if the current edge is at the target roadblock.
        :param current_edge: The edge to check.
        :param target_roadblock: The target roadblock the edge should be contained in.
        :return: True if the lane edge is contain the in the target roadblock. False, otherwise.
        """
        return current_edge.get_roadblock_id() == target_roadblock.id

    def _construct_path(self, end_edge: LaneGraphEdgeMapObject, depth: int) -> List[LaneGraphEdgeMapObject]:
        """
        :param end_edge: The end edge to start back propagating back to the start edge.
        :param depth: The depth of the target edge.
        :return: The constructed path as a list of LaneGraphEdgeMapObject
        """
        path = [end_edge]
        while self._parent[end_edge.id + f"_{depth}"] is not None:
            path.append(self._parent[end_edge.id + f"_{depth}"])
            end_edge = self._parent[end_edge.id + f"_{depth}"]
            depth -= 1
        path.reverse()

        return path

