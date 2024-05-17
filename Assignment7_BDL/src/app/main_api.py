from fastapi import FastAPI, File, UploadFile, Request
import uvicorn
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import io
import sys
import numpy as np
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge
import time
import psutil

# Define Prometheus metrics
API_CALL_COUNTER = Counter("api_call_counter", "Count of API calls", ["client_ip"])
PROCESSING_DURATION_GAUGE = Gauge("processing_duration_gauge", "Duration of API processing", ["client_ip"])
CPU_USAGE_GAUGE = Gauge("cpu_usage_gauge", "CPU usage during processing", ["client_ip"])
MEMORY_USAGE_GAUGE = Gauge("memory_usage_gauge", "Memory usage during processing", ["client_ip"])
NETWORK_BYTES_GAUGE = Gauge("network_bytes_gauge", "Network I/O bytes during processing", ["client_ip"])
NETWORK_RATE_GAUGE = Gauge("network_rate_gauge", "Network I/O rate during processing", ["client_ip"])
TOTAL_RUNTIME_GAUGE = Gauge("total_runtime_gauge", "Total runtime of the API", ["client_ip"])
PER_CHAR_TIME_GAUGE = Gauge("per_char_time_gauge", "Processing time per character", ["client_ip"])

app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Function to simulate digit prediction (replace with actual model prediction)
def simulate_prediction(data):
    normalized_data = np.array(data, dtype=np.float32) / 255.0
    normalized_data = normalized_data.reshape(1, -1)
    return str(np.random.randint(10))

# Resize image to 28x28 pixels
def resize_image(image: Image) -> Image:
    return image.resize((28, 28))

# Calculate processing time per character
def get_processing_time(start: float, length: int) -> float:
    elapsed_time = time.time() - start
    return (elapsed_time / length) * 1e6

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

@app.post('/predict')
async def predict_digit_api(request: Request, upload_file: UploadFile = File(...)):
    start_time = time.time()
    contents = await upload_file.read()
    client_ip = request.client.host

    # Log API usage
    API_CALL_COUNTER.labels(client_ip=client_ip).inc()

    # Process image
    image = Image.open(io.BytesIO(contents)).convert('L')
    image = resize_image(image)
    image_array = np.array(image)
    input_length = len(image_array)

    # Collect system metrics
    cpu_usage = psutil.cpu_percent(interval=1)
    CPU_USAGE_GAUGE.labels(client_ip=client_ip).set(cpu_usage)

    memory_info = psutil.virtual_memory()
    MEMORY_USAGE_GAUGE.labels(client_ip=client_ip).set(memory_info.percent)

    net_io = psutil.net_io_counters()
    total_network_bytes = net_io.bytes_sent + net_io.bytes_recv
    NETWORK_BYTES_GAUGE.labels(client_ip=client_ip).set(total_network_bytes)
    NETWORK_RATE_GAUGE.labels(client_ip=client_ip).set(total_network_bytes / (time.time() - start_time))

    # Flatten image for prediction
    flat_data = image_array.flatten().tolist()
    prediction = simulate_prediction(flat_data)

    # Log processing time
    processing_time = get_processing_time(start_time, len(flat_data))
    PROCESSING_DURATION_GAUGE.labels(client_ip=client_ip).set(processing_time)

    # Log API runtime and T/L time
    total_runtime = time.time() - start_time
    TOTAL_RUNTIME_GAUGE.labels(client_ip=client_ip).set(total_runtime)
    per_char_time = total_runtime / input_length
    PER_CHAR_TIME_GAUGE.labels(client_ip=client_ip).set(per_char_time)

    return {"predicted_digit": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
