import gc
import torch

# drop references to tensors/models you don't need
# del model
# del some_tensor

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # helps clean up inter-process cached blocks sometimes