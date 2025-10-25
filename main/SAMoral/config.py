import torch

device_id = 0 
torch.cuda.set_device(device_id)


device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
