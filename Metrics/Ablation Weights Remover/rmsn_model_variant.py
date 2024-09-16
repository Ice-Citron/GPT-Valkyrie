import torch
from transformers import GPT2LMHeadModel, GPT2Config
import os
import torch.nn as nn

class IdentityLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)
        
        # Set weights to 1 and bias to 0
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return x  # Identity function

def replace_layer_norm(module, name):
    if isinstance(getattr(module, name), nn.LayerNorm):
        setattr(module, name, IdentityLayerNorm(768))

def create_noNorm_model(base_model):
    model = GPT2LMHeadModel(base_model.config)
    model.load_state_dict(base_model.state_dict())
    
    for block in model.transformer.h:
        replace_layer_norm(block, 'ln_1')
        replace_layer_norm(block, 'ln_2')

    # Ensure the changes are reflected in the state dict
    model.transformer.h = nn.ModuleList(model.transformer.h)

    # Store the variant type in the config
    model.config.variant_type = "noNorm"
    model.config.name_or_path = "shng2025/GPT-Valkyrie_RMSN-124m__noNorm__"
    return model


def create_AttnOnly_model(base_model):
    model = GPT2LMHeadModel(base_model.config)
    model.load_state_dict(base_model.state_dict())
    
    for block in model.transformer.h:
        # replace_layer_norm(block, 'ln_1')
        replace_layer_norm(block, 'ln_2')

    # Ensure the changes are reflected in the state dict
    model.transformer.h = nn.ModuleList(model.transformer.h)

    # Store the variant type in the config
    model.config.variant_type = "AttnOnly"
    model.config.name_or_path = "shng2025/GPT-Valkyrie_RMSN-124m__AttnOnly__"
    return model

def create_FFNonly_model(base_model):
    model = GPT2LMHeadModel(base_model.config)
    model.load_state_dict(base_model.state_dict())
    
    for block in model.transformer.h:
        replace_layer_norm(block, 'ln_1')
        # replace_layer_norm(block, 'ln_2')

    # Ensure the changes are reflected in the state dict
    model.transformer.h = nn.ModuleList(model.transformer.h)

    # Store the variant type in the config
    model.config.variant_type = "FFNonly"
    model.config.name_or_path = "shng2025/GPT-Valkyrie_RMSN-124m__FFNonly__"
    return model


def examine_layers(model, model_name):
    # Create a sample input
    sample_input = torch.randn(1, 1, model.config.n_embd)
    
    print(f"Examining model: {model_name}")
    print(f"Variant type: {getattr(model.config, 'variant_type', 'Not specified')}")
    print("\nNormalization layer details:")
    for i, block in enumerate(model.transformer.h):
        print(f"  Block {i}:")
        
        # ln_1 (before attention)
        ln1_output = block.ln_1(sample_input)
        print(f"    ln_1 (before attention):")
        print(f"      Type: {type(block.ln_1).__name__}")
        print(f"      Input mean: {sample_input.mean().item():.4f}, std: {sample_input.std().item():.4f}")
        print(f"      Output mean: {ln1_output.mean().item():.4f}, std: {ln1_output.std().item():.4f}")
        print(f"      Is identity: {torch.allclose(sample_input, ln1_output, atol=1e-6)}")
        
        # ln_2 (before FFN)
        ln2_output = block.ln_2(sample_input)
        print(f"    ln_2 (before FFN):")
        print(f"      Type: {type(block.ln_2).__name__}")
        print(f"      Input mean: {sample_input.mean().item():.4f}, std: {sample_input.std().item():.4f}")
        print(f"      Output mean: {ln2_output.mean().item():.4f}, std: {ln2_output.std().item():.4f}")
        print(f"      Is identity: {torch.allclose(sample_input, ln2_output, atol=1e-6)}")

    # Final layer norm
    ln_f_output = model.transformer.ln_f(sample_input)
    print(f"  Final layer norm:")
    print(f"    Type: {type(model.transformer.ln_f).__name__}")
    print(f"    Input mean: {sample_input.mean().item():.4f}, std: {sample_input.std().item():.4f}")
    print(f"    Output mean: {ln_f_output.mean().item():.4f}, std: {ln_f_output.std().item():.4f}")
    print(f"    Is identity: {torch.allclose(sample_input, ln_f_output, atol=1e-6)}")

# Load the base model
base_model_path = "shng2025/GPT-Valkyrie_RMSN-124m__baseModel__"
base_model = GPT2LMHeadModel.from_pretrained(base_model_path)

# Create the noNorm variant
noNorm_model = create_noNorm_model(base_model)
AttnOnly_model = create_AttnOnly_model(base_model)
FFNonly_model = create_FFNonly_model(base_model)

# Save the noNorm model
save_path = "GPT-Valkyrie_RMSN-124m__noNorm__"
os.makedirs(save_path, exist_ok=True)
noNorm_model.save_pretrained(save_path)
print(f"Saved noNorm model to {save_path}")

# Save the noNorm model
save_path = "GPT-Valkyrie_RMSN-124m__AttnOnly__"
os.makedirs(save_path, exist_ok=True)
AttnOnly_model.save_pretrained(save_path)
print(f"Saved noNorm model to {save_path}")

# Save the noNorm model
save_path = "GPT-Valkyrie_RMSN-124m__FFNonly__"
os.makedirs(save_path, exist_ok=True)
FFNonly_model.save_pretrained(save_path)
print(f"Saved noNorm model to {save_path}")

# Examine both models
print("\nBase Model:")
examine_layers(base_model, "Base Model")

print("\n" + "="*50 + "\n")

print("noNorm Model:")
examine_layers(noNorm_model, "noNorm Model")

print("\n" + "="*50 + "\n")

print("AttnOnly Model:")
examine_layers(AttnOnly_model, "AttnOnly Model")

print("\n" + "="*50 + "\n")

print("FFNonly Model:")
examine_layers(FFNonly_model, "FFNonly Model")