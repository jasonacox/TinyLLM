# Monitoring Dashboard

This sets up a Grafana dashboard to monitor your TinyLLM system. It uses Telegraf to fetch the Prometheus metrics from vLLM and a [monitor.py](monitor.py) script to pull the GPU metrics from the nvidia-smi tool.

![image](https://github.com/jasonacox/TinyLLM/assets/836718/a1389cb0-c3d1-46ec-bec1-1ff3ac412507)
![image](https://github.com/vllm-project/vllm/assets/836718/878b4c99-2707-4907-9847-6521aad30755)

## Setup

Launch InfluxDB, Telegraf and Grafana

```bash
# Run InfluxDB docker container
docker run -d -p 0.0.0.0:8086:8086 --name influxdb \
    -v $PWD/influxdb:/var/lib/influxdb \
    --restart=unless-stopped \
    -e INFLUXDB_DB=tinyllm \
    influxdb:1.8

# Run Telegraf docker container (for vLLM)
docker run -d --name telegraf \
    --user "$(id -u)" \
    -v $PWD/telegraf.conf:/etc/telegraf/telegraf.conf \
    --network host \
    --restart unless-stopped \
    telegraf:1.28.2 \
    --config /etc/telegraf/telegraf.conf \
    --config-directory /etc/telegraf/telegraf.d

# Run Grafana docker container
docker run -d -p 3000:3000 --name grafana \
    --user "$(id -u)" \
    -v $PWD/grafana:/var/lib/grafana \
    --restart=unless-stopped \
    -e GF_AUTH_ANONYMOUS_ENABLED=true \
    grafana/grafana
```

## Monitoring Tool

The monitor.py script will poll local Nvidia GPU and host CPU information and store it in the InfluxDB for graphing in Grafana. The steps below will build a CUDA container that will fetch the metrics every 5 seconds.

Dockerfile to build the container:

```Dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install psutil influxdb
ENV INFLUXDB_HOST localhost
ENV INFLUXDB_PORT 8086
ENV INFLUXDB_DBNAME tinyllm
ENV WAIT_TIME 5
COPY monitor.py .
CMD ["python3", "monitor.py"]
```

Build and Run

```bash
# Build
docker build -t gpumonitor .

# Run 
docker run -d --name gpumonitor --gpus all \
     -e INFLUXDB_HOST=localhost \
     -e INFLUXDB_PORT=8086 \
     -e INFLUXDB_DBNAME=tinyllm \
     -e WAIT_TIME=5 \
     --network host \
     --restart always \
     gpumonitor
```

## Dashboard Setup

Dashboard Setup

1. Go to `http://localhost:3000` and default user/password is admin/admin.
2. Create a data source, select InfluxDB and use URL http://x.x.x.x:8086 (replace with IP address of host), database name `tinyllm` and timeout `5s`.
3. Import dashboard and upload or copy/paste [dashboard.json](dashboard.json). Select InfluxDB as the data source.
