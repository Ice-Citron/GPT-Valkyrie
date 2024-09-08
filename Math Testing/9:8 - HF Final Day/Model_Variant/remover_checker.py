import os
import json
from transformers import GPT2LMHeadModel, GPT2Config
import torch

def check_model(model_path):
    print(f"Checking model: {model_path}")
    
    # Load the model and config
    model = GPT2LMHeadModel.from_pretrained(model_path)
    config = GPT2Config.from_pretrained(model_path)
    
    # Check config
    print("Checking config:")
    print(f"  name_or_path: {config.name_or_path}")
    print(f"  model_type: {config.model_type}")
    
    # Check normalization layers
    print("Checking normalization layers:")
    for i, block in enumerate(model.transformer.h):
        print(f"  Block {i}:")
        
        # Check ln_1 (before attention)
        ln_1_weight = block.ln_1.weight.data
        ln_1_bias = block.ln_1.bias.data if block.ln_1.bias is not None else None
        print(f"    ln_1: weight_mean={ln_1_weight.mean().item():.4f}, weight_std={ln_1_weight.std().item():.4f}")
        if ln_1_bias is not None:
            print(f"         bias_mean={ln_1_bias.mean().item():.4f}, bias_std={ln_1_bias.std().item():.4f}")
        
        # Check ln_2 (before FFN)
        ln_2_weight = block.ln_2.weight.data
        ln_2_bias = block.ln_2.bias.data if block.ln_2.bias is not None else None
        print(f"    ln_2: weight_mean={ln_2_weight.mean().item():.4f}, weight_std={ln_2_weight.std().item():.4f}")
        if ln_2_bias is not None:
            print(f"         bias_mean={ln_2_bias.mean().item():.4f}, bias_std={ln_2_bias.std().item():.4f}")
    
    # Check final layer norm
    if hasattr(model.transformer, 'ln_f'):
        ln_f_weight = model.transformer.ln_f.weight.data
        ln_f_bias = model.transformer.ln_f.bias.data if model.transformer.ln_f.bias is not None else None
        print("  Final layer norm:")
        print(f"    weight_mean={ln_f_weight.mean().item():.4f}, weight_std={ln_f_weight.std().item():.4f}")
        if ln_f_bias is not None:
            print(f"    bias_mean={ln_f_bias.mean().item():.4f}, bias_std={ln_f_bias.std().item():.4f}")
    
    print("\n")

# List of model paths
model_paths = [
    "shng2025/GPT-Valkyrie_RMSN-124m__AttnOnly__",
    "shng2025/GPT-Valkyrie_LN-124m__FFNonly__",
    "shng2025/GPT-Valkyrie_LN-124m__noNorm__",
    "shng2025/GPT-Valkyrie_LN-124m__AttnOnly__",
    "shng2025/GPT-Valkyrie_RMSN-124m__noNorm__",
    "shng2025/GPT-Valkyrie_RMSN-124m__FFNonly__"
]

# Check each model
for path in model_paths:
    check_model(path)