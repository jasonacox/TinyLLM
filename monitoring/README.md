# Monitoring Tools

These are some tools to help monitor your TinyLLM system.

<img width="1319" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/ab24068b-5303-4e82-b05a-ee23e55a7959">

## Setup

Launch InfluxDB and Grafana

```bash
# Run InfluxDB docker container
docker run -d -p 8086:8086 --name influxdb \
    -v $PWD/influxdb:/var/lib/influxdb \
    --restart=unless-stopped \
    -e INFLUXDB_DB=tinyllm \
    influxdb:1.8

# Run Grafana docker container
docker run -d -p 3000:3000 --name grafana \
    --user "$(id -u)" \
    -v $PWD/grafana:/var/lib/grafana \
    --restart=unless-stopped \
    grafana/grafana
```

## Monitoring Tool

The monitor.py script will poll local GPU and CPU information and store it in the InfluxDB for graphing in Grafana.

Dockerfile to build the container:

```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
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
     --restart always \
     gpumonitor
```

## Dashboard Setup

Dashboard Setup

1. Go to `http://localhost:3000` and default user/password is admin/admin.
2. Create a data source, select InfluxDB and use URL http://x.x.x.x:8086 (replace with IP address of host), database name `tinyllm` and timeout `5s`.
3. Import dashboard and select `dashboard.json`.
