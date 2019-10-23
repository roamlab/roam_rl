FROM python:3.6

RUN pip install tensorflow==1.5.0

COPY . /root/roam_rl

WORKDIR /root/roam_rl

RUN sed -e 's/sudo //g' -i setup.sh \
&& sed -i '/tensorflow/d' setup.sh \
&& bash setup.sh

CMD /bin/bash
