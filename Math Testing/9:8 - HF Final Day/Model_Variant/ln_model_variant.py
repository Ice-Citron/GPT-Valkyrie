import torch
from transformers import GPT2LMHeadModel, GPT2Config
import os

def create_model_variant(base_model, variant_type):
    model = GPT2LMHeadModel(base_model.config)
    model.load_state_dict(base_model.state_dict())
    
    if variant_type == "noNorm":
        for block in model.transformer.h:
            block.ln_1.weight.data.fill_(1.0)
            block.ln_1.bias.data.zero_()
            block.ln_2.weight.data.fill_(1.0)
            block.ln_2.bias.data.zero_()
    elif variant_type == "AttnOnly":
        for block in model.transformer.h:
            block.ln_2.weight.data.fill_(1.0)
            block.ln_2.bias.data.zero_()
    elif variant_type == "FFNonly":
        for block in model.transformer.h:
            block.ln_1.weight.data.fill_(1.0)
            block.ln_1.bias.data.zero_()

    # Update the model name in the configuration
    model.config.name_or_path = f"shng2025/GPT-Valkyrie_LN-124m__{variant_type}__"
    
    return model

base_model_path = "shng2025/GPT-Valkyrie_LN-124m__baseModel__"  # Changed to LN model
base_model = GPT2LMHeadModel.from_pretrained(base_model_path)

variants = ["noNorm", "AttnOnly", "FFNonly", "baseModel"]

for variant in variants:
    model_variant = create_model_variant(base_model, variant)
    save_path = f"GPT-Valkyrie_LN-124m__{variant}__"  # Changed to LN
    os.makedirs(save_path, exist_ok=True)
    model_variant.save_pretrained(save_path)
    print(f"Saved {variant} model to {save_path}")

def examine_layers(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    print(f"Examining model: {model_path}")
    print("\nNormalization layer details:")
    for i, block in enumerate(model.transformer.h):
        print(f"  Block {i}:")
        print(f"    ln_1 (before attention):")
        print(f"      Weight: mean={block.ln_1.weight.mean().item():.4f}, std={block.ln_1.weight.std().item():.4f}")
        print(f"      Bias: mean={block.ln_1.bias.mean().item():.4f}, std={block.ln_1.bias.std().item():.4f}")
        print(f"    ln_2 (before FFN):")
        print(f"      Weight: mean={block.ln_2.weight.mean().item():.4f}, std={block.ln_2.weight.std().item():.4f}")
        print(f"      Bias: mean={block.ln_2.bias.mean().item():.4f}, std={block.ln_2.bias.std().item():.4f}")
    
    if hasattr(model.transformer, 'ln_f'):
        print("  Final layer norm:")
        print(f"    Weight: mean={model.transformer.ln_f.weight.mean().item():.4f}, std={model.transformer.ln_f.weight.std().item():.4f}")
        print(f"    Bias: mean={model.transformer.ln_f.bias.mean().item():.4f}, std={model.transformer.ln_f.bias.std().item():.4f}")

for variant in variants:
    model_path = f"GPT-Valkyrie_LN-124m__{variant}__"  # Changed to LN
    examine_layers(model_path)
    print("=" * 50)