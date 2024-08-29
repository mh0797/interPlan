EXPERIMENT=sim_interplan_gc_pgp
NUPLAN_EXP_ROOT=~/interplan_workspace/exp/
INTERPLAN_PLUGIN_ROOT=~/interplan_workspace/interplan-plugin

CHECKPOINT_PATH=/path/to/checkpoints/gc_pgp_checkpoint.ckpt

python $INTERPLAN_PLUGIN_ROOT/interplan/planning/script/run_simulation.py \
+simulation=default_interplan_benchmark \
scenario_filter=interplan10 \
planner=ml_planner \
planner.ml_planner.model_config='\${model}' \
planner.ml_planner.checkpoint_path=$CHECKPOINT_PATH \
model=gc_pgp_model \
scenario_filter=interplan10 \
experiment_name=$EXPERIMENT \
hydra.searchpath="[\
pkg://interplan.planning.script.config.simulation,\
pkg://interplan.planning.script.experiments,\
pkg://tuplan_garage.planning.script.config.common,\
pkg://interplan.planning.script.config.common,\
pkg://tuplan_garage.planning.script.config.simulation,\
pkg://nuplan.planning.script.config.common,\
pkg://nuplan.planning.script.config.simulation,\
pkg://nuplan.planning.script.experiments\
]"
