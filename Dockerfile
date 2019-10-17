FROM roamlab/roam_utils:latest

COPY . /root/roam_rl

WORKDIR /root/roam_rl

RUN sed -e 's/sudo //g' -i setup.sh && sed -i '/roam_utils/d' setup.sh

RUN bash setup.sh
