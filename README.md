# roam_rl
Consists of wrappers for environments and benchmark RL implementations which accomplish two things - 
1. Bring different RL implemenents to follow a single interface
2. Setup hyperparameters and all configurable objects through a single configuration file

`roam_rl/env` has wrappers for gym's base env classes Env & GoalEnv. 
If you are using a RobotWorld (see [roam_robot_worlds](https://github.com/roamlab/roam_robot_worlds)),
then you might find it useful to instead inherit from RobotEnv and RobotGoalEnv in `roam_rl/robot_env`.
RobotEnv/RobotGoalEnv provide additional templating that is commonly used in robot learning.


### Installation 

The installation procedure is as simple as runing the setup file below. And yes, you need sudo access. 

```console 
$ bash setup.sh
```

roam_rl provides wrappers for the following benchmark implementations

1. `baselines` from OpenAI

    * PPO


### Testing 
For a quick test you can run

`python scripts/train_ppo.py scripts/configs/train_ppo_acrobat.cfg`

Install [roam_robot_worlds](https://github.com/roamlab/roam_robot_worlds) if you want to run `train_ppo_point_robot_env.cfg` and `train_ppo_point_robot_goal_env.cfg`. 
