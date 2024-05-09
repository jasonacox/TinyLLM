# Run monitor.py script
# This script monitors GPU usage and sends the data to an InfluxDB instance
#
# Build `docker build -t gpumonitor .`
#
# docker run -d --name gpumonitor --gpus all \
#     -e INFLUXDB_HOST=localhost \
#     -e INFLUXDB_PORT=8086 \
#     -e INFLUXDB_DBNAME=tinyllm \
#     -e WAIT_TIME=5 \
#     gpumonitor

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install psutil influxdb
ENV INFLUXDB_HOST localhost
ENV INFLUXDB_PORT 8086
ENV INFLUXDB_DBNAME tinyllm
ENV WAIT_TIME 5
COPY monitor.py .
CMD ["python3", "monitor.py"]
