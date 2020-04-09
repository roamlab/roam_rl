# To be used as a base image for your project. In your project's image
# make sure you place your MuJoCo key at /root/.mujoco/

FROM rlworkgroup/garage-base

COPY . /root/code/roam_rl/

WORKDIR /root/code/roam_rl/

RUN make default

CMD /bin/bash

