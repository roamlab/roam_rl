FROM python:3.6

COPY . /root/roam_rl

WORKDIR /root/roam_rl

RUN sed -e 's/sudo //g' -i setup.sh && bash setup.sh

CMD /bin/bash
