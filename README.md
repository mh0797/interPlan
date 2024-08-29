<div id="top" align="center">

<img src="interPlan_logo.png">
<h3 align="center"><a href="https://arxiv.org/pdf/2404.07569">Paper</a></h3>

A challenging **interactive** closed-loop **planning** benchmark

</div>


## 🔥 Highlights 🔥 <a name="highlight"></a>

interPlan is a closed-loop driving benchmark based on the nuPlan dataset and framework. It modifies original nuPlan scenarios by augmenting traffic agents quantitiy and behavior and by modifying the navigation target so that lane-changes become necessary. 

🎥 **The generated scenarios comprise**  🎥:
> - Jaywalkers
> - Overtaking obstacles (with oncoming traffic)
> - Nudging around parked vehicles (no oncoming traffic)
> - Lane changes in low / medium / high traffic density
> - Encountering accident sites
> - Approaching construction zones

interPlan is an easy-to-use plugin for nuPlan. Currently it contains 335 diverse scenarios.
The official interPlan split comprises 80 scenarios, 10 for each of the types mentioned above.
It can easily be extended and adapted to your needs.

> ⭐ **Key features** ⭐
> - 100% compatible with nuPlan models. Just Plug&Play your planner
> - diverse traffic agent policies including conservative, assertive and mixed traffic
> - exhaustive set of sotA baselines available and tested
> - scenario modification interface for nuPlan provides a simple method to adapt and create scenarios for your needs

## News
* **`29 Aug 2024`:** Added more simulation scripts
* **`29 Aug 2024`:** We found a wrong token in our config (used in `f8684ea2cf1c512b-s0`,`f8684ea2cf1c512b-s1`) which is unsolvable. We replaced it (`9e7581ef72155c9f`) and re-evaluated all models. Results remained unchanged since all models still fail in the new scenario.
* **`30 June, 2024`:**  Our paper was accepted at [IROS 2024](https://iros2024-abudhabi.org/)!
* **`12 Apr, 2024`:**  Initial release of interPlan Code.

## Table of Contents
1. [Highlights](#highlight)
2. [Results](#results)
3. [Getting started](#gettingstarted)
4. [License and citation](#licenseandcitation)
5. [Other resources](#otherresources)

## Results <a name="results"></a>

Planning results on the proposed *interPlan* benchmark. We also report the score on the Val14 benchmark for reference (see [tuplan-garage](https://github.com/autonomousvision/tuplan_garage) for details). Please refer to the [paper](https://arxiv.org/abs/2404.07569) for more details.

| **Method**        | **Val14 score**     | **interPlan score** |
|-------------------|--------------|------------|
| [Urban Driver](https://arxiv.org/abs/2109.13333)*  | 50   |  4    |
| [GC-PGP](https://arxiv.org/abs/2302.07753v1)       | 55   | 10    |
| [GameFormer](https://opendrivelab.com/e2ead/AD23Challenge/Track_4_AID.pdf)                                     | 75   | 11    |
| [PDM-Open](https://arxiv.org/abs/2306.07962)       | 54   | 25    |
| [DTPP](https://arxiv.org/pdf/2310.05885.pdf)       | 73   | 25    |
| [IDM](https://arxiv.org/abs/cond-mat/0002177)      | 77   | 31    |
| [IDM+Mobil](https://arxiv.org/abs/cond-mat/0002177)| 75   | 31    |
| [PDM-Closed](https://arxiv.org/abs/2306.07962)     | 92   | 42    |
| HybridLLMPlanner (Llama-7B)                        | -    | 53    |
| LLMWaypointsPlanner (GPT-3.5)                      | -    | 22    |

*Open-loop reimplementation of Urban Driver

<p align="right">(<a href="#top">back to top</a>)</p>

## Getting started <a name="gettingstarted"></a>

1. Install the nuPlan-devkit \
Download and install the nuPlan devkit according to the [instructions ](https://github.com/motional/nuplan-devkit/blob/master/docs/installation.md) \
You also need to download the dataset. Please make sure to read the license agreement before doing so!
2. Install interPlan \
    Clone this repository and install interplan: 
    ```
    git clone https://github.com/mh0797/interPlan.git
    cd interplan-plugin
    pip install -e .
    ```

3. **[Optional]** Install tuplan-garage \
The [tuPlan garage](https://github.com/autonomousvision/tuplan_garage) provides many state-of-the-art planners in a handy plugin. It is 100% compatible with interPlan.

4. You are done! Run your first simulation
    - Use the interplan simulation script
    When simulating a planner in nuPlan, you would usually use the script `nuplan-devkit/planning/script/run_simulation.py` from the nuplan-devkit. 
    To run your planner in interplan, you just have to replace this with the respective script from interplan, i.e., `interplan/planning/script/run_simulation.py`.
    - You can apply all default interPlan overrides by adding `+simulation=default_interplan_benchmark`.
    - Additionally, we provide two different scenario-filters. `scenario_filter=benchmark_scenarios` runs the simulation with all 335 scenarios, whereas `scenario_filter=interplan10` uses 10 scenarios per type. Note that the latter was used in our paper and to generate the results in the table above.
    - Select your planner with an override: `planner=your_planner_name`.  Have a look at the script `scripts/sim_pdm_closed.sh` for an example of how to include a planner that is implemented in tuplan_garage.
    That's it! - You can check your results on the nuboard, just as when you run a simulation in nuPlan.

5. **[Advanced]** Create your own scenarios \
    interPlan builds the scenarios by augmenting nuPlan scenarios. The modifications are defined in hydra config files. That way, they can easily be adapted and extended. You can find the modifications in `/interplan/planning/script/config/common/scenario_filter/modifications/interPlan_modifications.yaml`. 
    Let's have a look at an example modification to understand how it works:
    ```
    2d62c3139aa95007:
        extra_agents_to_spawn:
          - [365393.1, 143807.98, 1.29] 
          - [365391.0, 143805.72, 0.0] 
        roadblock_locations_to_spawn_agents:
        - [365395.63, 143802.64]
        goal:
            left: 365393.96, 143805.68
            right: null
            straight: null
        spawn_in_intersection: true
        special_scenario:
        '0':
            type: overtake_parked_vehicle
            config: dh
            stopped_vehicle:
            - [365393.1, 143807.98, 1.29]
            stopped_bus:
            - [365421.39, 143944.2, 1.74533]
            cones:
            - [664434.93, 3998307.44]
            - [664434.93, 3998297.44]
            - [664434.93, 3998297.44]
            pedestrian: 
            - [365420.6, 143949.66, 365427.59, 143949.83, null] 
    ```
    The modification starts with the token of the original nuPlan scenario (here: `2d62c3139aa95007`) that is going to be modified. Depending on how you load the scenario, a different set of the modifications you specified will be used. For instance, if you load the scenario as `2d62c3139aa95007-dlgloa` (`dl`: traffic density low; `gl`: navigation goal left, `oa`: other traffic agents are assertive), then the modification for goal left will be used.
    Similarly, if you lead the scenario as `2d62c3139aa95007-s0`, the modifications defined in special-scenario 0 will be applied. It is also possible to only add modifications for a special scenario or for the lane-change scenarios.

    **Lane-change scenarios** \
    By default, interPlan will spawn agents in the roadblock that the ego-vehicle is currently driving in. Under `extra_agents_to_spawn` you can include a list of poses (x,y,yaw) where additional agents should be spawned. Additionally, you can specify extra roadblocks that should be considered when spawning agents. To do so, add a point (x,y) that is within the roadblock to the list defined in `roadblock_locations_to_spawn_agents`.
    
    Finally, you can define the goal that should be used for the scenario. This can either be set to a goal location (x,y) or to null, in which case the route will be created by repeatedly applying the respective navigation command. For example, the goal right option will result in a route that makes a right turn at every intersection.

    **Special Scenarios** \
    Each special scenario is defined by an index (here: 0), a type (here: overtake_parked_vehicle) and additional paramters for object spawning. The scenario type has to be defined as it determines which scenario-specific metrics will be used. You can spawn the following objects by defining the following paramters: 
    - stopped vehicles: x,y,yaw
    - stopped bus: x, y, yaw
    - traffic cone: x, y
    - pedestrians: x_start, y_start, x_end, y_end, yaw \
        The pedestrians will move at constant speed for the start to the end-point at a constant yaw





<p align="right">(<a href="#top">back to top</a>)</p>



## License and citation <a name="licenseandcitation"></a>
All assets and code in this repository are under the [Apache 2.0 license](./LICENSE) unless specified otherwise. The nuPlan dataset inherit their own distribution licenses. Please consider citing our paper and project if they help your research.

```BibTeX
@misc{Contributors2024interPlan,
    title={interPlan: A challenging interactive closed-loop planning benchmark},
    author={interPlan Contributors},
    howpublished={\url{https://github.com/mh0797/interPlan}},
    year={2024}
} 
```

```BibTeX
@inproceedings{Hallgarten2024interPlan,
    title = {Can Vehicle Motion Planning Generalize to Realistic Long-tail Scenarios?},
    author = {Marcel Hallgarten and Julian Zapata and Martin Stoll and Katrin Renz and Andreas Zell},
    journal={arXiv preprint arXiv:2404.07569},
    year = {2024}
} 
```

<p align="right">(<a href="#top">back to top</a>)</p>


## Other resources <a name="otherresources"></a>
- [nuPlan devkit](https://github.com/motional/nuplan-devkit/)
- [tuPlan garage](https://github.com/autonomousvision/tuplan_garage)
- [Survey on interactive Prediction and Planning](https://arxiv.org/abs/2308.05731)

<p align="right">(<a href="#top">back to top</a>)</p>

# 
<a href="https://twitter.com/MHallgarten0797" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/Marcel Hallgarten?style=social&color=brightgreen&logo=twitter" />
  </a>
