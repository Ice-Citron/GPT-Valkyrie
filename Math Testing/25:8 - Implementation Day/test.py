import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class SimplifiedSyncBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, input):
        if not input.is_contiguous():
            input = input.contiguous()

        size = input.size()
        batch_size = size[0] * size[2] * size[3]  # N*H*W

        mean = input.mean([0, 2, 3])
        var = input.var([0, 2, 3], unbiased=False)

        print(f"Forward - Local mean: {mean.mean().item()}, var: {var.mean().item()}")

        if dist.is_initialized() and self.training:
            mean_var = torch.stack([mean, var])
            dist.all_reduce(mean_var, op=dist.ReduceOp.SUM)
            mean, var = mean_var / dist.get_world_size()

        print(f"Forward - Synced mean: {mean.mean().item()}, var: {var.mean().item()}")

        if self.training:
            n = batch_size * dist.get_world_size() if dist.is_initialized() else batch_size
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var * n / (n - 1)
        else:
            mean = self.running_mean
            var = self.running_var

        print(f"Forward - Running mean: {self.running_mean.mean().item()}, var: {self.running_var.mean().item()}")

        input_normalized = (input - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        output = self.weight[None, :, None, None] * input_normalized + self.bias[None, :, None, None]

        print(f"Forward - Output mean: {output.mean().item()}, std: {output.std().item()}")

        self.save_for_backward(input_normalized, mean, var)
        return output

    def backward(self, grad_output):
        input_normalized, mean, var = self.saved_tensors
        
        N, C, H, W = input_normalized.shape
        batch_size = N * H * W

        print(f"Backward - Grad output mean: {grad_output.mean().item()}, std: {grad_output.std().item()}")

        grad_input = grad_weight = grad_bias = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * self.weight[None, :, None, None]
            grad_input = grad_input / torch.sqrt(var[None, :, None, None] + self.eps)

        if self.needs_input_grad[1]:
            grad_weight = (grad_output * input_normalized).sum([0, 2, 3])

        if self.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2, 3])

        if dist.is_initialized() and self.training:
            dist.all_reduce(grad_weight, op=dist.ReduceOp.SUM)
            dist.all_reduce(grad_bias, op=dist.ReduceOp.SUM)

        print(f"Backward - Grad input mean: {grad_input.mean().item()}, std: {grad_input.std().item()}")
        print(f"Backward - Grad weight mean: {grad_weight.mean().item()}, std: {grad_weight.std().item()}")
        print(f"Backward - Grad bias mean: {grad_bias.mean().item()}, std: {grad_bias.std().item()}")

        return grad_input, grad_weight, grad_bias, None, None

    def extra_repr(self):
        return f'{self.num_features}, eps={self.eps}, momentum={self.momentum}'

class TestModel(nn.Module):
    def __init__(self, norm_layer):
        super(TestModel, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

def run_test(local_rank, world_size):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method='env://', world_size=world_size, rank=local_rank)

    # Synchronize random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model_builtin = TestModel(nn.SyncBatchNorm).cuda(local_rank)
    model_custom = TestModel(SimplifiedSyncBatchNorm).cuda(local_rank)

    model_builtin = DDP(model_builtin, device_ids=[local_rank])
    model_custom = DDP(model_custom, device_ids=[local_rank])

    batch_size = 32
    data = torch.randn(batch_size, 3, 64, 64).cuda(local_rank)

    # Forward pass
    out_builtin = model_builtin(data)
    out_custom = model_custom(data)

    diff = (out_builtin - out_custom).abs().mean().item()
    print(f"Rank {local_rank}: Output difference: {diff}")

    # Log intermediate values
    print(f"Rank {local_rank}: Builtin BN running_mean: {model_builtin.module.bn.running_mean.mean().item()}")
    print(f"Rank {local_rank}: Builtin BN running_var: {model_builtin.module.bn.running_var.mean().item()}")
    print(f"Rank {local_rank}: Custom BN running_mean: {model_custom.module.bn.running_mean.mean().item()}")
    print(f"Rank {local_rank}: Custom BN running_var: {model_custom.module.bn.running_var.mean().item()}")

    # Backward pass
    loss_builtin = out_builtin.sum()
    loss_custom = out_custom.sum()
    loss_builtin.backward()
    loss_custom.backward()

    for (name_b, param_b), (name_c, param_c) in zip(model_builtin.named_parameters(), model_custom.named_parameters()):
        if param_b.grad is not None and param_c.grad is not None:
            grad_diff = (param_b.grad - param_c.grad).abs().mean().item()
            print(f"Rank {local_rank}: Gradient difference for {name_b}: {grad_diff}")

    dist.destroy_process_group()

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    run_test(local_rank, world_size)

if __name__ == "__main__":
    main()