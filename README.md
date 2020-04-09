# roam_rl

roam_rl is an implementation of wrapper classes for benchmark RL implementations with the following goals - 
1. Bring different RL implemenents to follow a single interface
2. Setup hyperparameters and all configurable objects through a single configuration file


Module `roam_rl.env` has wrappers for gym's base env classes `Env` & `GoalEnv` and module `roam_rl.baselines.ppo` has wrapper 
class for OpenAI baselines' ppo2 implementation.

If the environment you are using based on a `RobotWorld` (see [roam_robot_worlds](https://github.com/roamlab/roam_robot_worlds)), then you might find it useful to instead inherit from RobotEnv and RobotGoalEnv in `roam_rl/robot_env`. 

`RobotEnv` and `RobotGoalEnv` provide additional templating common used in robot learning.

### Installation 

If you need to setup apt dependencies and install MuJoCo, as root run `make setup`.  

For installing the pip dependencies, activate your virtual environment and then run `make default` or just `make`. If your machine is configured to use NVIDIA GPU you can run `make gpu` instead to make use of the GPU.

### Testing 
For a quick test you can run

`python scripts/baselines/train_ppo.py scripts/baselines/configs/train_ppo_acrobat.cfg`

Install [roam_robot_worlds](https://github.com/roamlab/roam_robot_worlds) to run config files `configs/train_ppo_point_robot_env.cfg` and `configs/train_ppo_point_robot_goal_env.cfg`.
