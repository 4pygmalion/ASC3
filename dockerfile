FROM ubuntu:20.04

ENV WORKING_DIR container
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /${WORKING_DIR}

COPY ./ASC3 ./ASC3
COPY ./requirements.txt ./
COPY ./data ./data

RUN apt-get -y update && apt-get install -y gcc python3 python3-pip libgl1-mesa-glx libglib2.0-0 git

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

CMD python3 ASC3