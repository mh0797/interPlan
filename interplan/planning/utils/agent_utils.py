import math
import warnings

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import CAP_STYLE
from shapely.ops import unary_union

warnings.filterwarnings('ignore', message="(.|\n)*invalid value encountered in line_locate_point")


from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.state_representation import (ProgressStateSE2,
                                                            StateSE2)
from nuplan.common.geometry.transform import rotate_angle
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring
from nuplan.planning.simulation.path.utils import \
    convert_se2_path_to_progress_path


def get_agent_constant_velocity_path(agent: Agent, seconds: float = 3) -> list[ProgressStateSE2]:
    back_center_agent_point = StateSE2(agent.center.x + agent.box.half_length*math.cos(agent.center.heading + math.pi),
                                        agent.center.y + agent.box.half_length*math.sin(agent.center.heading + math.pi),
                                        agent.center.heading)
    path = [StateSE2(*back_center_agent_point), StateSE2(*agent.center)]
    rotated_velocity = rotate_angle(StateSE2(agent.velocity.x, agent.velocity.y, agent.center.heading) , -agent.center.heading)
    for i in np.arange(0, seconds + 0.1, 0.1):
        new_agent_center = StateSE2(
            agent.center.x + i*rotated_velocity.x,
            agent.center.y + i*rotated_velocity.y,
            agent.center.heading)
        front_center_agent_point = StateSE2(
            new_agent_center.x + agent.box.half_length*math.cos(new_agent_center.heading),
            new_agent_center.y + agent.box.half_length*math.sin(new_agent_center.heading),
            new_agent_center.heading)
        path.append(
            StateSE2(
                front_center_agent_point.x,
                front_center_agent_point.y,
                front_center_agent_point.heading
            )
        )
    path: ProgressStateSE2 = convert_se2_path_to_progress_path(path)    
    return path

def get_agent_constant_velocity_geometry(agent: Agent) -> Polygon:
    """
    Returns the agent's expanded path (constant velocity, going straight) as a Polygon.
    :return: A polygon representing the agent's path.
    """
    path_to_go = get_agent_constant_velocity_path(agent)            
    expanded_path = path_to_linestring(path_to_go).buffer((agent.box.width / 2), cap_style=CAP_STYLE.square)
    return unary_union([expanded_path, agent.box.geometry])