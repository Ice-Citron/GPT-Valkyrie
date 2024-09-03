import torch
from transformers import GPT2Config, GPT2LMHeadModel
from model_definition import Head, MultiHeadAttention, FeedFoward, Block, GPTLanguageModel
from model_definition import vocab_size, n_embd, n_head, n_layer, block_size


# Load the trained model
trained_model = torch.load('./full_model.pth', map_location=torch.device('cpu'))

# Create a GPT2Config that matches your model's architecture
config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=block_size,
    n_embd=n_embd,
    n_layer=n_layer,
    n_head=n_head
)

# Create a new GPT2LMHeadModel with this config
hf_model = GPT2LMHeadModel(config)

# Copy weights from trained model to Hugging Face model
hf_model.transformer.wte.weight.data = trained_model.token_embedding_table.weight.data.contiguous()
hf_model.transformer.wpe.weight.data = trained_model.position_embedding_table.weight.data.contiguous()

for i, block in enumerate(trained_model.blocks):
    # Combine key, query, value weights
    combined = torch.cat([
        block.sa.heads[0].key.weight.data,
        block.sa.heads[0].query.weight.data,
        block.sa.heads[0].value.weight.data
    ], dim=0).t().contiguous()
    hf_model.transformer.h[i].attn.c_attn.weight.data = combined

    hf_model.transformer.h[i].attn.c_proj.weight.data = block.sa.proj.weight.data.t().contiguous()
    hf_model.transformer.h[i].mlp.c_fc.weight.data = block.ffwd.net[0].weight.data.t().contiguous()
    hf_model.transformer.h[i].mlp.c_proj.weight.data = block.ffwd.net[2].weight.data.t().contiguous()

hf_model.lm_head.weight.data = trained_model.lm_head.weight.data.contiguous()

# Save the model
hf_model.save_pretrained("./hf_model")

print("Model converted and saved successfully!")