import torch

# Monkey patch torch.load to use MPS or CPU if map_location is not specified
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        # Determine the appropriate device (MPS for Mac, else CPU)
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        kwargs['map_location'] = torch.device(device)
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Optional: Add a print statement to confirm the patch is applied when this module is imported.
# print("torch.load has been patched by Chatterbox patch_utils.")