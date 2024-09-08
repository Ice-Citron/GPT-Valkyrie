import torch
from Implementation import GPT, GPTConfig  # Import your model classes

# Load the checkpoint
checkpoint_path = "./checkpoint_19072.pt"  # Adjust the path as needed
checkpoint = torch.load(checkpoint_path)

# Create a model instance
config = GPTConfig()  # Adjust parameters as needed
model = GPT(config)

# If the model was compiled, try to access the original module
if hasattr(model, '_orig_mod'):
    model = model._orig_mod

# If it was wrapped in DDP, unwrap it
raw_model = model.module if hasattr(model, 'module') else model

# Load the state dict
raw_model.load_state_dict(checkpoint['model'])

# Now we can proceed with examining the model
print(raw_model)

# Print all state dict keys
for key in raw_model.state_dict().keys():
    print(key)

# Print the shape of each parameter
for name, param in raw_model.named_parameters():
    print(f"{name}: {param.shape}")
