import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples

from argparse import Namespace

import wandb

import tiktoken
import numpy as np

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from huggingface_hub import Repository, create_branch
import logging
import glob

# -----------------------------------------------------------------------------
os.environ['NUMEXPR_MAX_THREADS'] = '96'

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# -----------------------------------------------------------------------------

# given up
class GroupScaling1D(nn.Module):
    def __init__(self, eps=1e-5, group_num=1):
        super(GroupScaling1D, self).__init__()
        self.eps = eps
        self.group_num = group_num

    def extra_repr(self):
        return f'eps={self.eps}, group={self.group_num}'

    def forward(self, input):
        B, T, C = input.shape
        if self.group_num == 1:
            moment2 = torch.mean(input * input, dim=-1, keepdim=True)
        else:
            Cg = C // self.group_num
            input = input.view(B, T, self.group_num, Cg)
            moment2 = torch.mean(input * input, dim=-1, keepdim=True)
            moment2 = moment2.repeat(1, 1, 1, Cg).view(B, T, C)
        return input / torch.sqrt(moment2 + self.eps)

class SyncPowerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_phi, eps, afwd, abkw, ema_gz, warmup_iters, current_iter, process_group, affine):
        ctx.affine = affine
        ctx.eps = eps
        current_iter = current_iter.item()
        ctx.process_group = process_group
        ctx.abkw = abkw

        B, T, C = x.size()
        x2 = torch.mean(torch.square(x), dim=(0, 1))
        var = x2.view(1, 1, C)

        if current_iter <= warmup_iters:
            y = x * torch.rsqrt(var + eps) # Shape: (B, T, C)
        else:
            y = x * torch.rsqrt(running_phi + eps) # Shape: (B, T, C)

        ctx.save_for_backward(y, var, weight, ema_gz)

        if current_iter < warmup_iters:
            running_phi.copy_(running_phi * (current_iter-1)/current_iter + var/current_iter) # MISTAKE? UNSURE <-- I think its correct, because we want to value for every feature inside dim=-1 calculated across the batch
        running_phi.copy_(afwd*running_phi + (1-afwd)*var) # MISTAKE? UNSURE

        if process_group is not None: # and (current_iter % 100 or current_iter == args.max_steps): # experimental, to try and reduce time spent accessing memory
            torch.distributed.all_reduce(running_phi, op=torch.distributed.ReduceOp.AVG, group=process_group)

        if affine:
            y = weight.view(1, 1, C) * y + bias.view(1, 1, C)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        alpha = ctx.alpha

        y, var, weight, ema_gz = ctx.saved_tensors
        
        if ctx.affine:
            g = grad_output * weight.view(1, 1, -1)
        else:
            g = grad_output

        approx_grad_g = g - (1 - alpha) * ema_gz * y
        ema_gz.add_(torch.mean(approx_grad_g * y, dim=(0, 1), keepdim=True))

        if ctx.process_group is not None: # and (current_iter % 100 or current_iter == args.max_steps): # experimental, to try and reduce time spent accessing memory
            dist.all_reduce(ema_gz, op=dist.ReduceOp.AVG, group=ctx.process_group)

        gx = torch.rsqrt(var + eps) * approx_grad_g

        if ctx.affine:
            grad_weight = torch.sum(grad_output * y, dim=(0, 1))
            grad_bias = torch.sum(grad_output, dim=(0, 1))
            if ctx.process_group is not None:
                dist.all_reduce(grad_weight, op=dist.ReduceOp.AVG, group=ctx.process_group)
                dist.all_reduce(grad_bias, op=dist.ReduceOp.AVG, group=ctx.process_group)
        else:
            grad_weight = None
            grad_bias = None

        return gx, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None

class SyncPowerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-3, alpha_fwd=0.9, alpha_bkw=0.9,
                 affine=True, warmup_iters=715, process_group=None, group_num=1):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.process_group = process_group

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.weight = None
            self.bias = None

        self.register_buffer('running_phi', torch.ones(1, 1, num_features))     # I THINK THIS IS LIKELY THE ISSUE, ITS NOT MEANT TO BE A PARAMETER BUT WE ARE TREATING IT LIKE ONE???
        self.register_buffer('ema_gz', torch.zeros(1, 1, num_features))
        self.register_buffer('iters', torch.ones(1).type(torch.LongTensor))

        self.afwd = alpha_fwd
        self.abkw = alpha_bkw

        self.warmup_iters = warmup_iters
        self.grad_accum_steps = args.gradient_accumulation_steps
        self.accum_count = 0
        self.group_scaling = GroupScaling1D(eps=eps, group_num=group_num)

    def extra_repr(self):
        return f'{self.num_features}, eps={self.eps}, alpha_fwd={self.afwd}, alpha_bkw={self.abkw}, ' \
               f'affine={self.affine}, warmup={self.warmup_iters}'

    def forward(self, input):
        B, T, C = input.size()
        assert C == self.num_features, f"Input features {C} doesn't match num_features {self.num_features}"

        # Apply GroupScaling1D
        input = self.group_scaling(input)

        if self.training:
            self.accum_count += 1
            if self.accum_count >= self.grad_accum_steps:
                self.iters.add_(1)
                self.accum_count = 0

            output = SyncPowerFunction.apply(input, self.weight if self.affine else None, self.bias if self.affine else None,
                                         self.running_phi, self.eps, self.afwd, self.abkw, self.ema_gz, self.warmup_iters, self.iters,
                                         self.process_group, self.affine)
        else:
            # var = self.running_phi
            output = input * torch.rsqrt(self.running_phi + self.eps)
            if self.affine: # if not, do nothing.
                output = self.weight.reshape(1, 1, C) * output + self.bias.reshape(1, 1, C)

        return output # Shape: (B, T, C)

# -----------------------------------------------------------------------------


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = SyncPowerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = SyncPowerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = SyncPowerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "./../edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s] # listing out shards file in the data_root dir
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        # NEW IMPL
        self.current_shard = 0
        self.current_position = self.B * self.T * self.process_rank

        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def set_state(self, state):
        self.current_shard = state['current_shard']
        self.current_position = state['current_position']
        self.tokens = load_tokens(self.shards[self.current_shard])

    def get_state(self):
        return {
            'current_shard': self.current_shard,
            'current_position': self.current_position
        }

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


def setup_logging(project_name, args):
    logger = logging.getLogger(__name__)
    dir_name = "./log"
    os.makedirs(dir_name, exist_ok=True)
    print(f"Directory '{dir_name}' {'already exists' if os.path.exists(dir_name) else 'was created'}.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"log/debug_{ddp_rank}.log"),
            logging.StreamHandler(),
        ],
    )

    if master_process:
        wandb.init(project=project_name, config=args, dir="./../")
        run_name = wandb.run.name
        wandb_id = wandb.run.id
        logger.setLevel(logging.INFO)
        print(f"Starting new run: {run_name}")
    else:
        run_name = ""
        wandb_id = ""
        logger.setLevel(logging.ERROR)

    return logger, run_name, wandb_id

def resume_logging(project_name, run_id, args):
    logger = logging.getLogger(__name__)
    dir_name = "./log"
    os.makedirs(dir_name, exist_ok=True)
    print(f"Directory '{dir_name}' {'already exists' if os.path.exists(dir_name) else 'was created'}.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"log/debug_{ddp_rank}.log"),
            logging.StreamHandler(),
        ],
    )

    if master_process:
        wandb.init(project=project_name, id=run_id, resume="must", config=args, dir='./../')
        run_name = wandb.run.name
        logger.setLevel(logging.INFO)
        print(f"Resuming run: {run_name}")
    else:
        run_name = ""
        logger.setLevel(logging.ERROR)

    return logger, run_name


def log_metrics(metrics):
    if master_process:
        wandb.log(metrics)

def save_checkpoint(model, optimizer, step, val_loss, run_name, train_loader_state, val_loader_state, wandb_id):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'val_loss': val_loss,
        'run_name': run_name,
        'train_loader_state': train_loader_state,
        'val_loader_state': val_loader_state,
        'wandb_id': wandb_id,
    }

    checkpoint_path = os.path.join(log_dir, f"checkpoint_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer, train_loader, val_loader):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    train_loader.set_state(checkpoint['train_loader_state'])
    val_loader.set_state(checkpoint['val_loader_state'])

    return checkpoint['step'], checkpoint['val_loss'], checkpoint['run_name'], checkpoint['wandb_id']

# logging powernorm stats
def log_powernorm_stats(model):
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, SyncPowerNorm):
            running_phi = module.get_buffer('running_phi')
            if running_phi is not None:
                stats[f"{name}/running_phi"] = running_phi.mean().item()
            ema_gz = module.get_buffer('ema_gz')
            if ema_gz is not None:
                stats[f"{name}/ema_gz"] = ema_gz.mean().item()
            iters = module.get_buffer('iters')
            if iters is not None:
                stats[f"{name}/iters"] = iters.item()
    return stats

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


# -----------------------------------------------------------------------------

# GPTesla - 111M param setup in comment. Modification to make lighter training requirement needed
config = {
    "weight_decay": 0.1,
    # "lr_scheduler_type": "cosine",
    "gradient_accumulation_steps": (2**16 * 7) // (56 * 1024 * ddp_world_size),  # total_batch_size // (B * T * ddp_world_size
    "max_eval_steps": 20,
    "seq_length": 1024,

    # New centralised parameters
    "project_name": "shng2025/GPT-Valkyrie_PN-124m",
    "total_batch_size": 2**16 * 7, # temporarily because 6 GPUs  # 2**19, ~0.5M, in number of tokens
    "micro_batch_size": 56,
    "max_lr": 6e-4,
    "min_lr": 6e-5,  # 10% of max_lr // not used, as we are using weight_decay instead
    "warmup_steps": 715,
    "max_steps": 21797, # had to be scaled up after 2001st step, as memory ran out when DDP
    "val_every": 500,           # EVALUATION
    "generate_every": 500,      # EVALUATION
    "hellaswag_every": 500,     # EVALUATION
    "save_every": 2000,           # SAVE CHECKPOINTING   
    "log_dir": "./log",
    "device": "auto",  # "auto", "cpu", "cuda", or "mps"
    "use_compile": True,
    "grad_clip": 1.0,
    "num_return_sequences": 4,
    "max_generate_length": 32,
    "top_k": 50,

    "resume_from_checkpoint": False,
}

args = Namespace(**config)
samples_per_step = torch.cuda.device_count() * args.micro_batch_size

project_name = args.project_name

# Logging - DEPRECATED
if master_process:
    pass
    # run_name, wandb_id = setup_logging(project_name.split("/")[1])
    # print(f"Weights and Biases run name: {run_name}")
    # print(f"Weights and Biases run id  : {wandb_id}")

# -----------------------------------------------------------------------------


# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = args.total_batch_size # 2**19, ~0.5M, in number of tokens
B = args.micro_batch_size # micro batch size
T = args.seq_length # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = True # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = args.max_lr
min_lr = max_lr * args.weight_decay
warmup_steps = args.warmup_steps
max_steps = args.max_steps # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay, learning_rate=args.max_lr, device_type=device_type)

# create the log directory we will write checkpoints to and log to
log_dir = args.log_dir
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

"""
# Initialize HuggingFace repository
if master_process:
    new_branch_name = run_name
    create_branch(project_name, repo_type="model", branch=new_branch_name)
    hf_repo = Repository("./", clone_from=project_name, revision=run_name)
"""

# Training loop
starting_step = 0

if args.resume_from_checkpoint:
    checkpoint_dir = args.log_dir
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    checkpoint_path = latest_checkpoint

    # Use the load_checkpoint function here
    starting_step, val_loss, run_name, wandb_id = load_checkpoint(checkpoint_path, raw_model, optimizer, train_loader, val_loader)
    starting_step += 1 # to make sure step 80 isn't repeated

    logger, run_name = resume_logging(project_name.split("/")[1], wandb_id, args)
    print(f"Resuming from checkpoint: {checkpoint_path}")
    print(f"Weights and Biases run name: {run_name}")
    print(f"Resuming from step: {starting_step}")

    # Initialize HuggingFace repository <-- UNSURE IF NEEDED
    if master_process:
        new_branch_name = run_name
        # create_branch(project_name, repo_type="model", branch=new_branch_name)
        hf_repo = Repository("./", clone_from=project_name, revision=run_name)

    # Local subprocess for git pulling and checking out the newest branch
    if master_process:
        import subprocess
        subprocess.run(["git", "fetch", "origin"])
        subprocess.run(["git", "checkout", new_branch_name])
        subprocess.run(["git", "pull", "origin", new_branch_name])

    if master_process:
        print(f"Resuming from checkpoint at step {starting_step}")
else:
    starting_step = 0
    logger, run_name, wandb_id = setup_logging(project_name.split("/")[1], args)
    print(f"Weights and Biases run name: {run_name}")
    print(f"Weights and Biases run id  : {wandb_id}")

    # Initialize HuggingFace repository
    if master_process:
        new_branch_name = run_name
        create_branch(project_name, repo_type="model", branch=new_branch_name)
        hf_repo = Repository("./", clone_from=project_name, revision=run_name)
    
    if master_process:
        print(f"Starting new run: {run_name}")

# Training Loop
for step in range(starting_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % args.val_every == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = args.max_eval_steps
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        log_metrics({"loss/validation": val_loss_accum.item()})

    # once in a while evaluate hellaswag
    if (step % args.hellaswag_every == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            log_metrics({"hella/swag": acc_norm, "hella/correct norm": num_correct_norm, "hella/num total": num_total})
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % args.generate_every == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = args.num_return_sequences
        max_length = args.max_generate_length
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, args.top_k, dim=-1) # top_k == 50 in this case
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    if (step % args.save_every == 0 or last_step) and step > 0:
        if master_process:
            # Save checkpoint and push to HuggingFace
            checkpoint_path = save_checkpoint(
                raw_model, optimizer, step, val_loss_accum.item(), run_name,
                train_loader.get_state(), val_loader.get_state(), wandb_id
            )
            hf_repo.push_to_hub(commit_message=f"Checkpoint at step {step}")
            print(f"Saved checkpoint and pushed to HuggingFace at step {step}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        
        # Add NaN check here
        if not torch.isnan(loss):
            loss.backward()
        else:
            print(f"NaN loss detected at step {step}, micro_step {micro_step}. Skipping backward.")
            if master_process:
                wandb.alert(
                    title="NaN Loss Detected",
                    text=f"NaN loss detected at step {step}, micro_step {micro_step}. Skipping backward.",
                    level=wandb.AlertLevel.WARN
                )
            break

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # Add another NaN check here before optimizer step
    if not torch.isnan(loss_accum):
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) # grad_clip == 1.0 by default
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
    else:
        print("successfully broke")
        print(f"NaN accumulated loss detected at step {step}. Skipping optimizer step.")
        wandb.alert(
            title="NaN Accumulated Loss Detected",
            text=f"NaN accumulated loss detected at step {step}. Skipping optimizer step.",
            level=wandb.AlertLevel.ERROR
        )
        break

    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        pn_stats = log_powernorm_stats(raw_model)
        log_metrics({
            "lr": lr, # get_lr()
            "samples": step * samples_per_step,
            "steps": step,
            "loss/train": loss_accum.item(),
            # file specific addition
            "global gradient norm": norm,
            "dt": dt,
            "tok per sec": tokens_per_sec,
            **pn_stats  # Unpack the PowerNorm stats into the metrics
        })
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()

# Final save and push
if master_process:
    final_checkpoint_path = checkpoint_path = save_checkpoint(raw_model, optimizer, step, val_loss_accum.item(), run_name, train_loader.get_state(), val_loader.get_state(), wandb_id)
    hf_repo.push_to_hub(commit_message="Final model")
    print("Training completed. Final model pushed to HuggingFace.")