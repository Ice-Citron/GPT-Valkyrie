import torch
from transformers import GPT2LMHeadModel, GPT2Config


# Load the checkpoint
checkpoint_path = "./checkpoint_19072.pt"  # Adjust the path as needed
checkpoint = torch.load(checkpoint_path)

# Get the state dict
state_dict = checkpoint['model']

# Create a new state dict with modified keys and transposed weights where necessary
new_state_dict = {}
for k, v in state_dict.items():
    # Remove '_orig_mod.' prefix
    new_k = k.replace('_orig_mod.', '')
    
    # Change 'gain' to 'weight' for layer norm
    if '.gain' in new_k:
        new_k = new_k.replace('.gain', '.weight')
    
    # Transpose weights for attention and MLP layers
    if any(x in new_k for x in ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']):
        new_state_dict[new_k] = v.t()
    else:
        new_state_dict[new_k] = v

# Create a GPT2Config with the correct vocab size
config = GPT2Config(
    vocab_size=50304,  # Use your model's vocab size
    n_positions=1024,  # Adjust if your max sequence length is different
    n_embd=768,        # Embedding dimension
    n_layer=12,        # Number of layers
    n_head=12          # Number of attention heads
)

# Create a new GPT2LMHeadModel
model = GPT2LMHeadModel(config)

# Load our modified state dict
model.load_state_dict(new_state_dict, strict=True)

# Save the model
output_dir = "./hf_gpt2_model"
model.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")