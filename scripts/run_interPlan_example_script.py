import hydra
import os
from pathlib import Path
from nuplan.planning.script.run_simulation import main as main_simulation
import warnings
warnings.filterwarnings('ignore', message="(.|\n)*invalid value encountered in line_locate_point")

#-------------------------------------------------
# Run simulation
#-------------------------------------------------
# Location of path with all simulation configs
nuplan_devkit_absolute_path = "/home/USER/nuplan-devkit" # For Example
path_to_devkit = os.path.relpath(nuplan_devkit_absolute_path, Path(__file__).parent.resolve())
config_path = "nuplan/planning/script/config/simulation/"
CONFIG_PATH = os.path.join(path_to_devkit, config_path)
CONFIG_NAME = 'default_simulation'

experiments_path = "/home/USER/scripts/" # For example
SAVE_DIR = Path(experiments_path + "Experiments") 
CHECKPOINT_PDM = Path(experiments_path + "model_checkpoints/checkpoints/pdm_open_checkpoint.ckpt")
CHECKPOINT_GCPGP = Path(experiments_path + "model_checkpoints/checkpoints/gc_pgp_checkpoint.ckpt")
CHECKPOINT_URBANDRIVER = Path(experiments_path + "model_checkpoints/checkpoints/urbandriver_checkpoint.ckpt")



# Select the planner and simulation challenge
PLANNER = ["idm_planner", 'idm_mobil_planner', "pdm_closed_planner", "pdm_open_planner", "urban_driver_planner", "gc_pgp_planner", "gameformer_planner", "dtpp_planner"]
# Options are: [["idm_planner", 'idm_mobil_planner', "pdm_closed_planner", "pdm_open_planner", "urban_driver_planner", "gc_pgp_planner", "gameformer_planner", "dtpp_planner"]]
CHALLENGE = 'default_interplan_benchmark'
DATASET_PARAMS = [
    #"scenario_filter=interplan10", default: benchmark_scenarios
    #"scenario_filter.scenario_tokens=[71f182558ee95100-s0, cd0e827efbe85a8f-s0, cfad48a855765482-s0, 2d62c3139aa95007-s0, c710330e5114501c-s0, c710330e5114501c-s1, 5016a2a4ad1350d6-s0]",

    f"planner.pdm_open_planner.checkpoint_path={CHECKPOINT_PDM}",
    f"planner.urban_driver_planner.checkpoint_path={CHECKPOINT_URBANDRIVER}" ,
    f"planner.gc_pgp_planner.checkpoint_path={CHECKPOINT_GCPGP}", 
    "+model@urban_driver_model=urban_driver_open_loop_model",
    "+model@gc_pgp_model=gc_pgp_model", 
    
    'hydra.searchpath=[\
        \"pkg://interplan.planning.script.config.common\", \
        \"pkg://interplan.planning.script.config.simulation\", \
        \"pkg://interplan.planning.script.experiments\", \
        \"pkg://nuplan_garage.planning.script.config.common\", \
        \"pkg://nuplan_garage.planning.script.config.simulation\",\
        \"pkg://nuplan.planning.script.config.common\", \
        \"pkg://nuplan.planning.script.experiments\"]', 
                ]

# Name of the experiment
EXPERIMENT = 'simulation_simple_experiment'

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=CONFIG_PATH)

# Compose the configuration
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    f'experiment_name={EXPERIMENT}',
    f'group={SAVE_DIR}',
    f'planner={PLANNER}',
    f'+simulation={CHALLENGE}',
    *DATASET_PARAMS,
])


# Run the simulation loop (real-time visualization not yet supported, see next section for visualization)
main_simulation(cfg)



