# roam_rl

roam_rl implements helper classes for baselines and also enables use of config files  

Module `roam_rl.baselines.ppo` has wrapper class for OpenAI baselines' ppo2 implementation.

### Installation 

If you need to setup apt dependencies and install MuJoCo, run `sudo make setup`.  

For installing the pip dependencies, activate your virtual environment and then run `make default` or just `make`. If your machine is configured to use NVIDIA GPU you can run `make gpu` instead to make use of the GPU.

### Testing
For a quick test you can run

`python scripts/baselines/train_ppo.py scripts/baselines/configs/train_ppo_acrobat.cfg`