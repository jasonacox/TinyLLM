#!/usr/bin/python3
#
# Gather System and GPU data and store in InfluxDB
#
# Requirements:
#  pip install psutil influxdb
# 
# Author: Jason Cox
# 7 May 2024
#

import subprocess
import psutil
from influxdb import InfluxDBClient
import time
import os
import sys
import signal

BUILD = "0.1"

# Replace these with your InfluxDB server details from environment variables or secrets
host = os.getenv('INFLUXDB_HOST') or 'localhost'
port = int(os.getenv('INFLUXDB_PORT')) or 8086
database = os.getenv('INFLUXDB_DATABASE') or 'tinyllm'  
wait_time = int(os.getenv('WAIT_TIME')) or 5

# Print application header
print(f"System and GPU Monitor v{BUILD}", file=sys.stderr)
sys.stderr.flush()

# Signal handler - Exit on SIGTERM
def sigTermHandler(signum, frame):
    raise SystemExit
signal.signal(signal.SIGTERM, sigTermHandler)

# Connect
client = InfluxDBClient(
    host=host,
    port=port,
    database=database)
# Check connection
if not client:
    print(f" - Connection to InfluxDB {host}:{port} database {database} failed", file=sys.stderr)
    sys.stderr.flush()
    sys.exit(1)
else:
    print(f" - Connection to InfluxDB {host}:{port} database {database} successful", file=sys.stderr)
    sys.stderr.flush()

# Function to run a command and return the output
def getcommand(command):
    try:
        output = subprocess.check_output(command, shell=True, universal_newlines=True)
        return output.strip()
    except subprocess.CalledProcessError as e:
        print("Error executing the command:", e)

print(f" - Monitor started - Looping every {wait_time} seconds.", file=sys.stderr)
sys.stderr.flush()

# Main loop
try:
    while True:
        # Get system metrics
        measurements = {}
        memory_stats = psutil.virtual_memory()
        measurements["memory"] = memory_stats.used
        measurements["cpu"] =  psutil.cpu_percent(interval=1.0)

        # Get GPU metrics
        command = "/usr/bin/nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total --format csv"
        """
        Command output:
        utilization.gpu [%], temperature.gpu, power.draw [W], memory.used [MiB], memory.total [MiB]
        0 %, 32, 31.09 W, 13322 MiB, 16384 MiB
        0 %, 34, 31.33 W, 13288 MiB, 16384 MiB
        0 %, 31, 32.56 W, 13288 MiB, 16384 MiB
        0 %, 33, 33.78 W, 13228 MiB, 16384 MiB
        0 %, 31, 31.07 W, 360 MiB, 16384 MiB
        0 %, 31, 31.31 W, 460 MiB, 16384 MiB
        0 %, 28, 30.82 W, 1294 MiB, 16384 MiB
        """
        nvidia = getcommand(command).split("\n")[1:]
        i = 0
        for gpu in nvidia:
            (util,temp,power,used,total) = gpu.split(",")
            measurements[f"gpupower{i}"] = float(power.replace(" W",""))
            measurements[f"gputemp{i}"] = float(temp)
            measurements[f"gpumemory{i}"] = int(used.replace(" MiB",""))
            measurements[f"gputotalmemory{i}"] = int(total.replace(" MiB",""))
            measurements[f"gpuutil{i}"] = int(util.replace(" %",""))
            i += 1

        # Create payload
        json_body = []

        for name, value in measurements.items():
            data_point = {
                "measurement": name,
                "tags": {},
                "fields": {"value": value}
            }
            json_body.append(data_point)

        # Send to InfluxDB
        r = client.write_points(json_body)
        client.close()

        # Wait
        time.sleep(wait_time)

except (KeyboardInterrupt, SystemExit):
    print(" - Monitor stopped by user", file=sys.stderr)
    sys.stderr.flush()
except Exception as e:
    print(f" - Monitor stopped with error: {e}", file=sys.stderr)
    sys.stderr.flush()

print(" - Monitor stopped", file=sys.stderr)
sys.stderr.flush()

