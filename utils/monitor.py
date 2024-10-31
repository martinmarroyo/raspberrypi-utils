import psutil
import subprocess
import time
import json
from gpiozero import CPUTemperature
from datetime import datetime as dt

def monitor_system_usage(pid, interval=1):
    metrics = [] # time, cpu, memory, temp
    cpu_temp = CPUTemperature()
    # Start the monitoring loop
    while True:

        try:
            process = psutil.Process(pid)

            if not process.is_running():
                print("Process completed")
                break
        
            cpu_pct = psutil.cpu_percent(interval=0.1)
            mem_pct = psutil.virtual_memory().percent 
            temp_c = cpu_temp.temperature
            timestamp = int(dt.utcnow().timestamp())
            reading = dict(timestamp=timestamp, cpu=cpu_pct, 
                        memory=mem_pct, temperature=temp_c)
            metrics.append(reading)

            time.sleep(interval)

        except (KeyboardInterrupt, psutil.NoSuchProcess):
            print("Process terminated")
            break

    
    return metrics 


def run_chat_demo():
    command = ["python3", "slm_eval_trivia.py"]
    process = subprocess.Popen(command)

    metrics = monitor_system_usage(process.pid, interval=5)

    process.wait()

    with open(f"slm_demo_usage_metrics-{int(dt.utcnow().timestamp())}.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":

    run_chat_demo()
