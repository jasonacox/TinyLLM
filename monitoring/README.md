# Monitoring Tools

These are some tools to help monitor your TinyLLM system.

## Setup

Launch InfluxDB and Grafana

```bash
# Run InfluxDB docker container
docker run -d -p 8086:8086 --name influxdb -v $PWD/influxdb:/var/lib/influxdb --restart=unless-stopped influxdb

# Run Grafana docker container
docker run -d -p 3000:3000 --name grafana -v $PWD/grafana:/var/lib/grafana --restart=unless-stopped grafana/grafana
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