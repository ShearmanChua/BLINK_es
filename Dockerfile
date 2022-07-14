FROM python:3.8

RUN apt-get update
RUN apt-get -y install python3-pip vim git
RUN apt-get -y install libfreetype-dev libfreetype6 libfreetype6-dev

RUN pip install -U pip
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir /BLINK_api && mkdir /BLINK_api/src && mkdir /BLINK_api/models && mkdir /BLINK_api/configs && mkdir /BLINK_api/logs
COPY BLINK_api/src/ /BLINK_api/src
WORKDIR /BLINK_api/src

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

CMD ["/bin/bash"]