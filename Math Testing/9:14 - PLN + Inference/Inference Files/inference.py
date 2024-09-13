import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
import tiktoken

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

# Load model architecture
model = GPT(GPTConfig(vocab_size=50304))  # Initialize with your config

# Load checkpoint
checkpoint_dir = "./checkpoint_80.pt"
checkpoint = torch.load(checkpoint_dir, map_location=torch.device('cpu'))

# Remove the prefix from the state dict if necessary
def remove_prefix_from_state_dict(state_dict, prefix='_orig_mod.'):
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}

if 'model' in checkpoint:
    checkpoint['model'] = remove_prefix_from_state_dict(checkpoint['model'])
else:
    checkpoint = remove_prefix_from_state_dict(checkpoint)

# Load the modified state dict into your model
model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint, strict=False)

# Set model to evaluation mode
model.eval()

# Generation parameters
num_return_sequences = 1
max_length = 50
top_k = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Prepare input
prompt = "Hello, I'm a language model,"
tokens = enc.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)

# Generate text
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42)

with torch.no_grad():
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(xgen)  # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :]  # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the sequence
        xgen = torch.cat((xgen, xcol), dim=1)

# Print the generated text
for i in range(num_return_sequences):
    tokens = xgen[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"Sample {i}: {decoded}")