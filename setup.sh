#!/bin/bash

# stop execution on the first command that errors 
set -o errexit 

# This script is a one stop solution to setting up the environment for roam_rl on a machine.

# OpenAI Issues: baselines dependencies have not been setup well at all and here are some known issues
#
# 1) Installation of baselines fails if tensorflow has not been installed. The problem here is that even after listing
# tensorflow as a dependency in the project's setup.py baselines still cannot see that tensorflow has already been
# installed. This has to do with how pip works and there is nothing that we can change about that.
#
# 2) baselines' lists gym as a dependency and so after installation of baselines, gym is also reinstalled from PyPI. The
# issue here is that version available on PyPI is lags behind the latest.
#
# Solution: We install the required version of tensorflow first then install baselines and then finally install gym

sudo apt-get update

# baselines
yes | sudo apt install libopenmpi-dev zlib1g-dev
# baselines requires tensorflow to be manually installed
pip install tensorflow==1.13.2
pip install --force-reinstall git+https://github.com/openai/baselines#egg=baselines

# gym (must be installed after baselines. baselines installation forces install of gym from PyPI)
pip install --force-reinstall git+https://github.com/openai/gym#egg=gym


# roam_utils
pip install --force-reinstall git+https://git@github.com/roamlab/roam_utils@master#egg=roam_utils

# install roam_rl
pip install -e .
