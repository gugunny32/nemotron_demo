import requests
import os
import time
from pathlib import Path
import base64
import io
from PIL import Image
import getpass

# nv-ingest imports
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient

# API Key Setup
# We need an NVIDIA API Key for the Remote NIMs (Extraction & Generation).
if not os.environ.get("NVIDIA_API_KEY"):
    print("[ACTION REQUIRED] Please enter your NVIDIA API Key (starts with 'nvapi-').")
    os.environ["NVIDIA_API_KEY"] = getpass.getpass("Enter NVIDIA API Key: ")
else:
    print("[INFO] NVIDIA API Key detected in environment.")

# 1. Download Sample PDF (World Bank Peru 2017)
PDF_URL = "https://documents1.worldbank.org/curated/en/484531533637705178/pdf/129273-WP-PUBLIC-Peru-2017.pdf"
PDF_PATH = "129273-WP-PUBLIC-Peru-2017.pdf"

if not Path(PDF_PATH).exists():
    print(f"[INFO] Downloading {PDF_PATH}...")
    response = requests.get(PDF_URL)
    with open(PDF_PATH, "wb") as f:
        f.write(response.content)

# 2. Start nv-ingest in Library Mode
# This spins up the pipeline locally but uses remote NIMs by default for inference.
print("[INFO] Starting Ingestion Pipeline (Library Mode)...")
run_pipeline(
    block=False,
    disable_dynamic_scaling=True,
    run_in_subprocess=True,
    quiet=True
)
time.sleep(15) # Warmup

client = NvIngestClient(
    message_client_allocator=SimpleClient,
    message_client_port=7671, # Default LibMode port
    message_client_hostname="localhost"
)

print("[INFO] Submitting job...")

# 4. Extract Content
ingestor = (
    Ingestor(client=client)
    .files([PDF_PATH])
    .extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,  # Triggers remote YOLOX for chart crops
        extract_images=True, # Focus on charts/tables for VLM
        extract_method="pdfium", # CPU-based text extraction
        table_output_format="markdown"
    )
)

job_results = ingestor.ingest()
extracted_data = job_results[0]
with open("data.txt", "a") as f:
    for i, data in enumerate(extracted_data):
        f.write(f"{i}: {data}\n")
print(f"[SUCCESS] Extracted {len(extracted_data)} items from document.")