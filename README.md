# PyTorch Pommerman

This is a PyTorch starting point for experimenting with ideas for the Pommerman competitions (https://www.pommerman.com/)

The reinforcement learning codebase is based upon the work (https://arxiv.org/abs/2211.11940)

It requires the Pommerman `playground` (https://github.com/MultiAgentLearning/playground) to be installed in your Python environment, in addition to any dependencies of `pytorch-a2c-ppo-acktr`.

UPDATE:
```
# install depends of playgroud project
cd playground
conda env create -f env.yml

# install extra depends of pytorch-pommerman-rl
cd pytorch-pommerman-rl
conda activate pommerman
conda env update --file env.yml  --prune

# run the training command
python main_domac_FFA.py --use-gae --env-name PommeFFACompetitionFast-v0 --no-norm --seed 42 --algo a2c --lr-schedule 25000000 --no-vis

```
## Usage

With the spatial feature representation and CNN based models, I've been able to train an agent FFA play that does quite well (> 95% win rate). Without reward shaping, it does not learn to bomb, but it does a great job of evading and letting the other agents blow themselves up.

`python main_domac_FFA.py --use-gae --env-name PommeFFACompetitionFast-v0 --no-norm --seed 42 --algo a2c --lr-schedule 25000000`

Below is a training curve for above command. Note that it shows the training reward (non-deterministic), not evaluation which is higher.

![](imgs/DOMAC_FFA.png)

