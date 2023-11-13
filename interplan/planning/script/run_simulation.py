import logging
import os
import pkg_resources
from pathlib import Path

import hydra
from omegaconf import DictConfig

from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.script.run_simulation import (
    run_simulation,
    clean_up_s3_artifacts
)

from interplan.planning.utils.modifications_preprocessing import preprocess_scenario_filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

os.environ["NUPLAN_SIMULATION_ALLOW_ANY_BUILDER"] = "1"

@hydra.main(config_path=pkg_resources.resource_filename("nuplan.planning.script.config", "simulation"), config_name='default_simulation')
def main(cfg: DictConfig) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Calls run_simulation to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    """
    assert cfg.simulation_log_main_path is None, 'Simulation_log_main_path must not be set when running simulation.'
    
    # Proprocess token Names
    preprocess_scenario_filter(cfg)

    # Execute simulation with preconfigured planner(s).
    run_simulation(cfg=cfg)

    if is_s3_path(Path(cfg.output_dir)):
        clean_up_s3_artifacts()


if __name__ == '__main__':
    main()
