FROM python:3.10
ENV LANG C.UTF-8
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN  mkdir /root/.pip/
COPY pip.conf /root/.pip/
RUN chmod 0644 /root/.pip/pip.conf  

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
