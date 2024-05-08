#!/usr/bin/python3
#
# Gather System and GPU data and store in InfluxDB
#
# Requirements:
#  pip install psutil
#  pip install influxdb
# 
# Author: Jason Cox
# 7 May 2024
#

import subprocess
import psutil
from influxdb import InfluxDBClient
import json

# Replace these with your InfluxDB server details
host = 'localhost'
port = 8086
database = 'tinyllm'

# Connect
client = InfluxDBClient(
    host=host,
    port=port,
    database=database)

# Functions
def getsensors():
    try:
        output = subprocess.check_output(['sensors', '-j'])
        # Parse the JSON output
        temperature_data = json.loads(output.decode())
        return temperature_data
    except Exception as e:
        return str(e)

def getcommand(command):
    try:
        output = subprocess.check_output(command, shell=True, universal_newlines=True)
        # The 'universal_newlines=True' option ensures that the output is returned as a string (text mode).
        return output.strip()
    except subprocess.CalledProcessError as e:
        print("Error executing the command:", e)


# Grab data points
measurements = {}
memory_stats = psutil.virtual_memory()
sen = getsensors()
hd1temp = hd2temp = hd3temp = 0.0

measurements["memory"] = memory_stats.used
measurements["cpu"] =  psutil.cpu_percent(interval=1.0)
command = "nvidia-smi -q -d POWER -i 0 | grep Draw | grep W | awk '{print $4}'"
measurements["gpupower"] =  float(getcommand(command))
command = "nvidia-smi -q -d MEMORY -i 0 | grep Used | head -1 | awk '{print $3}'"
measurements["gpumemory"] =  int(getcommand(command))
command = "nvidia-smi -q -d TEMPERATURE -i 0 | grep Current | head -1 | awk '{print $5}'"
measurements["gputemp"] =  float(getcommand(command))


"""
drivetemp-scsi-1-0 : {'Adapter': 'SCSI adapter', 'temp1': {'temp1_input': 34.0, 'temp1_max': 55.0, 'temp1_min': 14.0, 'temp1_crit': 60.0, 'temp1_lcrit': 10.0, 'temp1_lowest': 27.0, 'temp1_highest': 36.0}}

coretemp-isa-0000 : {'Adapter': 'ISA adapter', 'Package id 0': {'temp1_input': 32.0, 'temp1_max': 84.0, 'temp1_crit': 100.0, 'temp1_crit_alarm': 0.0}, 'Core 0': {'temp2_input': 28.0, 'temp2_max': 84.0, 'temp2_crit': 100.0, 'temp2_crit_alarm': 0.0}, 'Core 1': {'temp3_input': 27.0, 'temp3_max': 84.0, 'temp3_crit': 100.0, 'temp3_crit_alarm': 0.0}, 'Core 2': {'temp4_input': 32.0, 'temp4_max': 84.0, 'temp4_crit': 100.0, 'temp4_crit_alarm': 0.0}, 'Core 3': {'temp5_input': 29.0, 'temp5_max': 84.0, 'temp5_crit': 100.0, 'temp5_crit_alarm': 0.0}}

"""
for i in sen:
    #print(f"{i} : {sen[i]}")
    if i.startswith("drivetemp-scsi-"): #0-0":
        hd = int(i[15]) + 1
        #hd1temp = sen[i]['temp1']['temp1_input']
        name = "hd%dtemp" % hd
        value = sen[i]['temp1']['temp1_input']
        #print(f"{name} = {value}")
        measurements[name] = value
    if i.startswith("coretemp-isa-0000"):
        name = "cputemp"
        value = sen[i]['Package id 0']['temp1_input']
        measurements[name] = value
    

# Create payload
json_body = []

for name, value in measurements.items():
    data_point = {
        "measurement": name,
        "tags": {},
        "fields": {"value": value}
    }
    json_body.append(data_point)

# Debug
#print(json.dumps(json_body, indent=4))
#print("Sending...")

# Send to InfluxDB
r = client.write_points(json_body)
#print(r)
client.close()