import requests
import os
import time
from pathlib import Path
import base64
import io
from PIL import Image

# nv-ingest imports
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient

# Local NIM Endpoint Setup
def configure_local_nim_endpoints():
    endpoint_defaults = {
        "OCR_HTTP_ENDPOINT": "http://localhost:8009",
        "OCR_INFER_PROTOCOL": "http",
        "OCR_MODEL_NAME": "paddle",
        "YOLOX_HTTP_ENDPOINT": "http://localhost:8000",
        "YOLOX_INFER_PROTOCOL": "http",
        "YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT": "http://localhost:8006",
        "YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL": "http",
        "YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT": "http://localhost:8003",
        "YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL": "http",
    }

    for key, value in endpoint_defaults.items():
        os.environ.setdefault(key, value)

    print("[INFO] Using local NIM endpoints for extraction.")


configure_local_nim_endpoints()

if os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY"):
    print("[INFO] NGC/NVIDIA API key detected (optional for some local NIM images).")
else:
    print("[INFO] No API key detected. This is fine if your local NIM containers do not require auth.")

# 1. Download Sample PDF (World Bank Peru 2017)
PDF_URL = "https://documents1.worldbank.org/curated/en/484531533637705178/pdf/129273-WP-PUBLIC-Peru-2017.pdf"
PDF_PATH = "129273-WP-PUBLIC-Peru-2017.pdf"

if not Path(PDF_PATH).exists():
    print(f"[INFO] Downloading {PDF_PATH}...")
    response = requests.get(PDF_URL)
    with open(PDF_PATH, "wb") as f:
        f.write(response.content)

# 2. Start nv-ingest in Library Mode
# Pipeline runs locally and picks endpoint values from environment variables.
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