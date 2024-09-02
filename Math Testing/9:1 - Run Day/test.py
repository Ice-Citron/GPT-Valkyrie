# torchrun --standalone --nproc_per_node=4 test.py

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# Global variable to control printing
PRINT_ALL_RANKS = True

def controlled_print(message):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if PRINT_ALL_RANKS or local_rank == 0:
        print(f"Rank {local_rank}: {message}")

class GroupScaling1D(nn.Module):
    def __init__(self, eps=1e-5, group_num=4):
        super(GroupScaling1D, self).__init__()
        self.eps = eps
        self.group_num = group_num

    def extra_repr(self):
        return f'eps={self.eps}, group={self.group_num}'

    def forward(self, input):
        T, B, C = input.shape[0], input.shape[1], input.shape[2]
        Cg = C // self.group_num
        gn_input = input.contiguous().reshape(T, B, self.group_num, Cg)
        moment2 = torch.repeat_interleave(torch.mean(gn_input * gn_input, dim=3, keepdim=True),
            repeats=Cg, dim=-1).contiguous().reshape(T, B, C)
        return input / torch.sqrt(moment2 + self.eps)

class PowerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_phi, eps, afwd, abkw, ema_gz,
                debug, warmup_iters, current_iter, mask_x):
        # Original PowerFunction forward code here
        ctx.eps = eps
        ctx.debug = debug
        current_iter = current_iter.item()
        ctx.current_iter = current_iter
        ctx.warmup_iters = warmup_iters
        ctx.abkw = abkw
        rmax = 1
        N, C, H, W = x.size()
        x2 = (mask_x * mask_x).mean(dim=0)

        var = x2.reshape(1, C, 1, 1)
        if current_iter <= warmup_iters:
            z = x /(var + eps).sqrt() # dividing by sqrt(varience + eps), which is same as dividing by standard deviation
        else:
            z = x /(running_phi + eps).sqrt() # same thing as above, but using running stats instead
            
        y = z
        ctx.save_for_backward(z, var, weight, ema_gz)

        if current_iter < warmup_iters:
            running_phi.copy_(running_phi * (current_iter-1)/current_iter + var.mean(dim=0, keepdim=True)/current_iter) # cumulative moving average
        running_phi.copy_(afwd*running_phi + (1-afwd)*var.mean(dim=0, keepdim=True)) # exponential moving average
        y = weight.reshape(1,C,1,1) * y + bias.reshape(1,C,1,1)

        controlled_print(f"Original Forward - Input mean: {x.mean().item()}, std: {x.std().item()}")
        controlled_print(f"Original Forward - Weight mean: {weight.mean().item()}, std: {weight.std().item()}")
        controlled_print(f"Original Forward - Bias mean: {bias.mean().item()}, std: {bias.std().item()}")
        controlled_print(f"Original Forward - Running phi mean: {running_phi.mean().item()}, std: {running_phi.std().item()}")
        controlled_print(f"Original Forward - Var mean: {var.mean().item()}, std: {var.std().item()}")
        controlled_print(f"Original Forward - Output mean: {y.mean().item()}, std: {y.std().item()}")

        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        debug = ctx.debug
        current_iter = ctx.current_iter
        warmup_iters = ctx.warmup_iters
        abkw = ctx.abkw

        N, C, H, W = grad_output.size()
        z, var, weight, ema_gz = ctx.saved_tensors

        y = z
        g = grad_output * weight.reshape(1, C, 1, 1)
        g = g * 1

        gz = (g * z).mean(dim=3).mean(dim=2).mean(dim=0)

        approx_grad_g = (g - (1 - abkw) * ema_gz * z) # approx function seems to just be using CTX stored tensors
        ema_gz.add_((approx_grad_g * z).mean(dim=3, keepdim=True).mean(dim=2, keepdim=True).mean(dim=0, keepdim=True))

        gx = 1. / torch.sqrt(var + eps) * approx_grad_g # REFER TO NOTES REGARDING BACKPROP DERIVATIVE EQUATION
        grad_weight = (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0)
        grad_bias = grad_output.sum(dim=3).sum(dim=2).sum(dim=0)

        is_sync = hasattr(ctx, 'process_group')
        prefix = "Sync" if is_sync else "Original"
        
        controlled_print(f"{prefix} Backward - Grad output mean: {grad_output.mean().item()}, std: {grad_output.std().item()}")
        controlled_print(f"{prefix} Backward - Grad input mean: {gx.mean().item()}, std: {gx.std().item()}")
        controlled_print(f"{prefix} Backward - Grad weight mean: {grad_weight.mean().item()}, std: {grad_weight.std().item()}")
        controlled_print(f"{prefix} Backward - Grad bias mean: {grad_bias.mean().item()}, std: {grad_bias.std().item()}")

        return gx, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None, None

class MaskPowerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, alpha_fwd=0.9, alpha_bkw=0.9,
                 affine=True, warmup_iters=10000, group_num=1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_phi', torch.ones(1,num_features,1,1))
        self.register_buffer('ema_gz', torch.zeros(1,num_features,1,1))
        self.register_buffer('iters', torch.zeros(1).type(torch.LongTensor))
        self.afwd = alpha_fwd
        self.abkw = alpha_bkw
        self.debug = False
        self.warmup_iters = warmup_iters
        self.gp = GroupScaling1D(group_num=group_num)
        self.group_num = group_num

    def extra_repr(self):
        return '{num_features}, eps={eps}, alpha_fwd={afwd}, alpha_bkw={abkw}, ' \
               'affine={affine}, warmup={warmup_iters}, group_num={group_num}'.format(**self.__dict__)

    def forward(self, input, pad_mask=None, is_encoder=False):
        shaped_input = (len(input.shape) == 2)
        if shaped_input:
            input = input.unsqueeze(0)
        
        if input.dim() == 4:  # N, C, H, W
            N, C, H, W = input.shape
            input = input.permute(2, 3, 0, 1).contiguous().view(H*W, N, C)
        
        T, B, C = input.shape
        input = self.gp(input)

        # construct the mask_input, size to be (BxL) x C: L is the real length here
        if pad_mask is None:
            mask_input = input.clone()
        else:
            # Transpose the bn_mask (B x T -> T x B)
            bn_mask = ~pad_mask
            bn_mask = bn_mask.transpose(0, 1)

        if pad_mask is not None:
            pad_size = (~bn_mask).sum()
            mask_input = input[bn_mask, :]
        else:
            mask_input = input.clone()

        mask_input = mask_input.reshape(-1, self.num_features)

        input = input.permute(1, 2, 0).contiguous()
        input_shape = input.size()
        input = input.reshape(input.size(0), self.num_features, -1)
        input = input.unsqueeze(-1)

        if self.training:
            self.iters.copy_(self.iters + 1)
            output = PowerFunction.apply(input, self.weight, self.bias, self.running_phi, self.eps, \
                        self.afwd, self.abkw, self.ema_gz, self.debug, self.warmup_iters, self.iters, mask_input)
            
        else:
            N, C, H, W = input.size()
            var = self.running_phi
            output = input / (var + self.eps).sqrt()
            output = self.weight.reshape(1,C,1,1) * output + self.bias.reshape(1,C,1,1)

        output = output.reshape(input_shape)
        output = output.permute(2, 0, 1).contiguous()
        # Reshape it.
        if shaped_input:
            output = output.squeeze(0)

        return output


class SyncPowerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_phi, eps, afwd, abkw, ema_gz,
                debug, warmup_iters, current_iter, mask_x, process_group, world_size):
        ctx.eps = eps
        ctx.debug = debug
        current_iter = current_iter.item()
        ctx.current_iter = current_iter
        ctx.warmup_iters = warmup_iters
        ctx.abkw = abkw
        ctx.process_group = process_group
        ctx.world_size = world_size

        N, C, H, W = x.size()
        x2 = (mask_x * mask_x).mean(dim=0)

        var = x2.reshape(1, C, 1, 1)

        # Synchronize var across GPUs
        if process_group is not None:
            dist.all_reduce(var, op=dist.ReduceOp.AVG, group=process_group) # no need to divide, because already averaging

        if current_iter <= warmup_iters:
            z = x / (var + eps).sqrt()
        else:
            z = x / (running_phi + eps).sqrt()

        y = z
        ctx.save_for_backward(z, var, weight, ema_gz)

        if current_iter < warmup_iters:
            running_phi.copy_(running_phi * (current_iter-1)/current_iter + var.mean(dim=0, keepdim=True)/current_iter)
        running_phi.copy_(afwd*running_phi + (1-afwd)*var.mean(dim=0, keepdim=True))

        # Synchronize running_phi across all processes
        if process_group is not None:
            torch.distributed.all_reduce(running_phi, op=torch.distributed.ReduceOp.AVG, group=process_group)

        y = weight.reshape(1,C,1,1) * y + bias.reshape(1,C,1,1)

        controlled_print(f"Sync Forward - Input mean: {x.mean().item()}, std: {x.std().item()}")
        controlled_print(f"Sync Forward - Weight mean: {weight.mean().item()}, std: {weight.std().item()}")
        controlled_print(f"Sync Forward - Bias mean: {bias.mean().item()}, std: {bias.std().item()}")
        controlled_print(f"Sync Forward - Running phi mean: {running_phi.mean().item()}, std: {running_phi.std().item()}")
        controlled_print(f"Sync Forward - Var mean: {var.mean().item()}, std: {var.std().item()}")
        controlled_print(f"Sync Forward - Output mean: {y.mean().item()}, std: {y.std().item()}")

        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        debug = ctx.debug
        current_iter = ctx.current_iter
        warmup_iters = ctx.warmup_iters
        abkw = ctx.abkw

        N, C, H, W = grad_output.size()
        z, var, weight, ema_gz = ctx.saved_tensors

        y = z
        g = grad_output * weight.reshape(1, C, 1, 1)
        g = g * 1

        gz = (g * z).mean(dim=3).mean(dim=2).mean(dim=0)

        approx_grad_g = (g - (1 - abkw) * ema_gz * z)
        ema_gz.add_((approx_grad_g * z).mean(dim=3, keepdim=True).mean(dim=2, keepdim=True).mean(dim=0, keepdim=True))

        if ctx.process_group is not None:
            dist.all_reduce(ema_gz, op=dist.ReduceOp.AVG, group=ctx.process_group)

        gx = 1. / torch.sqrt(var + eps) * approx_grad_g 
        grad_weight = (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0)
        grad_bias = grad_output.sum(dim=3).sum(dim=2).sum(dim=0)

        is_sync = hasattr(ctx, 'process_group')
        prefix = "Sync" if is_sync else "Original"

        if is_sync:
            process_group = ctx.process_group
            world_size = ctx.world_size
            dist.all_reduce(grad_weight, op=dist.ReduceOp.AVG, group=process_group)
            dist.all_reduce(grad_bias, op=dist.ReduceOp.AVG, group=process_group)
        
        controlled_print(f"{prefix} Backward - Grad output mean: {grad_output.mean().item()}, std: {grad_output.std().item()}")
        controlled_print(f"{prefix} Backward - Grad input mean: {gx.mean().item()}, std: {gx.std().item()}")
        controlled_print(f"{prefix} Backward - Grad weight mean: {grad_weight.mean().item()}, std: {grad_weight.std().item()}")
        controlled_print(f"{prefix} Backward - Grad bias mean: {grad_bias.mean().item()}, std: {grad_bias.std().item()}")

        return gx, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None, None

class SyncMaskPowerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, alpha_fwd=0.9, alpha_bkw=0.9,
                 affine=True, warmup_iters=10000, group_num=1, process_group=None):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group) if process_group else 1

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        self.register_buffer('running_phi', torch.ones(1,num_features,1,1))
        self.register_buffer('ema_gz', torch.zeros(1,num_features,1,1))
        self.register_buffer('iters', torch.zeros(1).type(torch.LongTensor))

        # Synchronize buffers
        dist.all_reduce(self.running_phi, op=dist.ReduceOp.AVG, group=self.process_group)
        dist.all_reduce(self.ema_gz, op=dist.ReduceOp.AVG, group=self.process_group)
        # dist.all_reduce(self.iters, op=dist.ReduceOp.MAX, group=self.process_group)

        self.afwd = alpha_fwd
        self.abkw = alpha_bkw

        self.eps = eps
        self.debug = False
        self.warmup_iters = warmup_iters
        self.gp = GroupScaling1D(group_num=group_num)
        self.group_num = group_num

    def extra_repr(self):
        return '{num_features}, eps={eps}, alpha_fwd={afwd}, alpha_bkw={abkw}, ' \
               'affine={affine}, warmup={warmup_iters}, group_num={group_num}'.format(**self.__dict__)

    def forward(self, input, pad_mask=None, is_encoder=False):
        shaped_input = (len(input.shape) == 2)
        if shaped_input:
            input = input.unsqueeze(0)
        
        if input.dim() == 4:  # N, C, H, W
            N, C, H, W = input.shape
            input = input.permute(2, 3, 0, 1).contiguous().view(H*W, N, C)
        
        T, B, C = input.shape
        input = self.gp(input)

        if pad_mask is None:
            mask_input = input.clone()
        else:
            bn_mask = ~pad_mask
            bn_mask = bn_mask.transpose(0, 1)

        if pad_mask is not None:
            pad_size = (~bn_mask).sum()
            mask_input = input[bn_mask, :]
        else:
            mask_input = input.clone()

        mask_input = mask_input.reshape(-1, self.num_features)

        input = input.permute(1, 2, 0).contiguous()
        input_shape = input.size()
        input = input.reshape(input.size(0), self.num_features, -1)
        input = input.unsqueeze(-1)

        if self.training:
            self.iters.copy_(self.iters + 1) # maybe consider syncing this, but unlikely
            output = SyncPowerFunction.apply(input, self.weight, self.bias, self.running_phi, self.eps,
                        self.afwd, self.abkw, self.ema_gz, self.debug, self.warmup_iters, self.iters, mask_input,
                        self.process_group, self.world_size)
        else:
            N, C, H, W = input.size()
            var = self.running_phi
            output = input / (var + self.eps).sqrt()
            output = self.weight.reshape(1,C,1,1) * output + self.bias.reshape(1,C,1,1)

        output = output.reshape(input_shape)
        output = output.permute(2, 0, 1).contiguous()
        if shaped_input:
            output = output.squeeze(0)

        return output

class TestModel(nn.Module):
    def __init__(self, norm_layer):
        super(TestModel, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.norm = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

def run_test(local_rank, world_size):
    dist.init_process_group(backend="nccl", init_method='env://', world_size=world_size, rank=local_rank)

    # Set same seed for all processes
    torch.manual_seed(42)
    
    # Generate same data for all processes
    batch_size = 32
    torch.manual_seed(42)
    data = torch.randn(batch_size, 3, 64, 64).cuda(local_rank)
    
    # Ensure all processes have the same data
    dist.broadcast(data, src=0)

    if local_rank == 0:
        # Run original MaskPowerNorm on single GPU
        torch.manual_seed(42)
        model_original = TestModel(lambda num_features: MaskPowerNorm(num_features)).cuda(local_rank)
        out_original = model_original(data)
        loss_original = out_original.sum()
        loss_original.backward()
        
        controlled_print("Running original PowerNorm on single GPU")
        controlled_print(f"Original output mean: {out_original.mean().item()}")

    # Run SyncMaskPowerNorm on all GPUs
    torch.manual_seed(42)
    model_sync = TestModel(lambda num_features: SyncMaskPowerNorm(num_features, process_group=dist.group.WORLD)).cuda(local_rank)
    model_sync = DDP(model_sync, device_ids=[local_rank])
    
    controlled_print("Running SyncPowerNorm")
    out_sync = model_sync(data)
    controlled_print(f"Sync output mean on rank {local_rank}: {out_sync.mean().item()}")

    loss_sync = out_sync.sum()
    loss_sync.backward()

    if local_rank == 0:
        for (name_o, param_o), (name_s, param_s) in zip(model_original.named_parameters(), model_sync.named_parameters()):
            if param_o.grad is not None and param_s.grad is not None:
                grad_diff = (param_o.grad - param_s.grad).abs().mean().item()
                controlled_print(f"Gradient difference for {name_o}:")
                controlled_print(f"  Original - mean: {param_o.grad.mean().item()}, std: {param_o.grad.std().item()}")
                controlled_print(f"  Sync     - mean: {param_s.grad.mean().item()}, std: {param_s.grad.std().item()}")
                controlled_print(f"  Absolute difference: {grad_diff}")


    dist.barrier()
    dist.destroy_process_group()

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    controlled_print(f"Running on rank {local_rank} of {world_size}")
    run_test(local_rank, world_size)

if __name__ == "__main__":
    main()