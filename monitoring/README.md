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

## Cronjob

The monitor.py script will poll local GPU and CPU information and store it in the InfluxDB for graphing in Grafana.

```bash
# Install dependent libraries
pip install psutil
pip install influxdb

cd ~TinyLLM/monitoring

# Add a task to crontab to run montior every 5 seconds
echo "`crontab -l`" > mycron
echo "*/5 * * * * `which python3` ${PWD}/monitor.py" >> mycron
crontab mycron
rm mycron
```

## Dashboard Setup

Dashboard Setup

1. Go to `http://localhost:3000` and default user/password is admin/admin.
2. Create a data source, select InfluxDB and use URL http://x.x.x.x:8086 (replace with IP address of host), database name `tinyllm` and timeout `5s`.
3. Import dashboard and select `dashboard.json`.
