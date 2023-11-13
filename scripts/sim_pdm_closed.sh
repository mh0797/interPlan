EXPERIMENT=sim_interplan_pdm_closed
NUPLAN_EXP_ROOT=/home/aah1si/llm_nuplan/exp/
INTERPLAN_PLUGIN_ROOT=/home/aah1si/llm_nuplan/interplan-plugin

python $INTERPLAN_PLUGIN_ROOT/interplan/planning/script/run_simulation.py \
+simulation=default_interplan_benchmark \
scenario_filter.only_in_benchmark_scenarios=False \
planner=pdm_closed_planner \
experiment_name=$EXPERIMENT \
hydra.searchpath="[\
pkg://interplan.planning.script.config.common,\
pkg://interplan.planning.script.config.simulation,\
pkg://interplan.planning.script.experiments,\
pkg://nuplan_garage.planning.script.config.common,\
pkg://nuplan_garage.planning.script.config.simulation,\
pkg://nuplan.planning.script.config.common,\
pkg://nuplan.planning.script.config.simulation,\
pkg://nuplan.planning.script.experiments\
]"
