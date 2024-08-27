# torchrun --standalone --nproc_per_node=4 test.py

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Function
import os

# Add this global variable
PRINT_ALL_RANKS = False
local_rank = int(os.environ["LOCAL_RANK"])

# Modify the print statements in OriginalSyncBatchNorm and CustomSyncBatchNorm
def controlled_print(message):
    if PRINT_ALL_RANKS or local_rank == 0:
        print(f"Rank {local_rank}: {message}")

class CustomSyncBatchNorm(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        # Ensure input is contiguous
        input = input.contiguous()

        # Calculate dimensions
        N, C, H, W = input.shape
        size = N * H * W

        # Compute mean and variance
        sum_ = input.sum(dim=[0, 2, 3])
        var_sum = (input ** 2).sum(dim=[0, 2, 3])
        
        # Synchronize sum and var_sum across GPUs
        if process_group is not None:
            dist.all_reduce(sum_, op=dist.ReduceOp.SUM, group=process_group)
            dist.all_reduce(var_sum, op=dist.ReduceOp.SUM, group=process_group)
            size *= world_size

        mean = sum_ / size
        var = var_sum / size - mean ** 2

        # Update running stats
        if running_mean is not None:
            running_mean.mul_(1 - momentum).add_(mean * momentum)
        if running_var is not None:
            running_var.mul_(1 - momentum).add_(var * momentum)

        # Normalize
        inv_std = 1 / torch.sqrt(var + eps)
        x_norm = (input - mean.view(1, C, 1, 1)) * inv_std.view(1, C, 1, 1)
        
        # Scale and shift
        out = x_norm * weight.view(1, C, 1, 1) + bias.view(1, C, 1, 1)

        # Save variables for backward
        ctx.save_for_backward(input, weight, mean, inv_std, x_norm)
        ctx.eps = eps
        ctx.size = size
        ctx.process_group = process_group

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, mean, inv_std, x_norm = ctx.saved_tensors
        eps = ctx.eps
        size = ctx.size
        process_group = ctx.process_group

        N, C, H, W = grad_output.shape

        # Compute gradients
        grad_bias = grad_output.sum(dim=[0, 2, 3])
        grad_weight = (grad_output * x_norm).sum(dim=[0, 2, 3])

        grad_x_norm = grad_output * weight.view(1, C, 1, 1)
        grad_var = (-0.5 * inv_std ** 3 * ((input - mean.view(1, C, 1, 1)) * grad_x_norm).sum(dim=[0, 2, 3]))
        grad_mean = (-inv_std.view(1, C, 1, 1) * grad_x_norm).sum(dim=[0, 2, 3])

        grad_input = inv_std.view(1, C, 1, 1) * (grad_x_norm - (2 / size) * (input - mean.view(1, C, 1, 1)) * grad_var.view(1, C, 1, 1) - (1 / size) * grad_mean.view(1, C, 1, 1))

        # Synchronize gradients
        if process_group is not None:
            dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=process_group)
            dist.all_reduce(grad_weight, op=dist.ReduceOp.SUM, group=process_group)
            dist.all_reduce(grad_bias, op=dist.ReduceOp.SUM, group=process_group)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

class OriginalSyncBatchNormModule(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, process_group=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group) if process_group else 1

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input):
        return OriginalSyncBatchNorm.apply(
            input, self.weight, self.bias, self.running_mean, self.running_var,
            self.eps, self.momentum, self.process_group, self.world_size
        )

class CustomSyncBatchNormModule(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, process_group=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group) if process_group else 1

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input):
        return CustomSyncBatchNorm.apply(
            input, self.weight, self.bias, self.running_mean, self.running_var,
            self.eps, self.momentum, self.process_group, self.world_size
        )

class TestModel(nn.Module):
    def __init__(self, norm_layer):
        super(TestModel, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

def run_test(local_rank, world_size):
    dist.init_process_group(backend="nccl", init_method='env://', world_size=world_size, rank=local_rank)
    
    torch.cuda.set_device(local_rank)
    torch.manual_seed(42)
    
    model_original = TestModel(lambda num_features: OriginalSyncBatchNormModule(num_features, process_group=dist.group.WORLD)).cuda(local_rank)
    
    torch.manual_seed(42)
    model_custom = TestModel(lambda num_features: CustomSyncBatchNormModule(num_features, process_group=dist.group.WORLD)).cuda(local_rank)

    model_original = DDP(model_original, device_ids=[local_rank])
    model_custom = DDP(model_custom, device_ids=[local_rank])

    batch_size = 32
    data = torch.randn(batch_size, 3, 64, 64).cuda(local_rank)

    controlled_print(f"Rank {local_rank}: Running original SyncBatchNorm")
    out_original = model_original(data)
    controlled_print(f"Rank {local_rank}: Running custom SyncBatchNorm")
    out_custom = model_custom(data)

    diff = (out_original - out_custom).abs().mean().item()
    controlled_print(f"Rank {local_rank}: Mean absolute difference between outputs: {diff}")

    loss_original = out_original.sum()
    loss_custom = out_custom.sum()

    loss_original.backward()
    loss_custom.backward()

    for (name_o, param_o), (name_c, param_c) in zip(model_original.named_parameters(), model_custom.named_parameters()):
        if param_o.grad is not None and param_c.grad is not None:
            grad_diff = (param_o.grad - param_c.grad).abs().mean().item()
            controlled_print(f"Rank {local_rank}: Gradient difference for {name_o}: {grad_diff}")

    dist.destroy_process_group()

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    controlled_print(f"Running on rank {local_rank} of {world_size}")
    run_test(local_rank, world_size)

if __name__ == "__main__":
    import os
    main()