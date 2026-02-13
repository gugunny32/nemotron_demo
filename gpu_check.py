import sys
import logging
import getpass
import torch

# Setup Logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("nv_ingest").setLevel(logging.INFO)

print(f"[INFO] Python Version: {sys.version.split()[0]}")
# Setup Compute Device
# We target the local GPU for the Embedding and Reranking VLMs.
device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    try:
         for index in range(0,2):
            props = torch.cuda.get_device_properties(index)
            print(f"[INFO] Local Inference Device: {device} ({props.name})")
            print(f"[INFO] VRAM Available: {props.total_memory / 1024**3:.2f} GB")
    except Exception as e:
        print(f"[ERROR] Could not access GPU: {e}")
else:
    print("[WARNING] CUDA not available. Defaulting to CPU (Slow!).")