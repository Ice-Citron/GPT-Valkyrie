# torchrun --standalone --nproc_per_node=4 test.py

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Function

class OriginalSyncBatchNorm(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        if not input.is_contiguous():
            input = input.contiguous()
        if weight is not None:
            weight = weight.contiguous()

        size = int(input.numel() // input.size(1))
        if size == 1 and world_size < 2:
            raise ValueError(f"Expected more than 1 value per channel when training, got input size {size}")

        num_channels = input.shape[1]
        if input.numel() > 0:
            # calculate mean/invstd for input.
            mean, invstd = torch.batch_norm_stats(input, eps)
            print(f"Original Forward - Local mean: {mean.mean().item()}, invstd: {invstd.mean().item()}")

            count = torch.full((1,), input.numel() // input.size(1), dtype=mean.dtype, device=mean.device)
            combined = torch.cat([mean, invstd, count], dim=0)
        else:
            combined = torch.zeros(2 * num_channels + 1, dtype=input.dtype, device=input.device)

        # Use all_gather instead of all_reduce
        combined_list = [torch.empty_like(combined) for _ in range(world_size)]
        dist.all_gather(combined_list, combined, process_group, async_op=False)
        combined = torch.stack(combined_list, dim=0)
        mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)

        # remove stats from empty inputs
        mask = count_all.squeeze(-1) >= 1
        count_all = count_all[mask]
        mean_all = mean_all[mask]
        invstd_all = invstd_all[mask]

        # calculate global mean & invstd
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input, mean_all, invstd_all, running_mean, running_var, momentum, eps, count_all.view(-1)
        )
        print(f"Original Forward - Synced mean: {mean.mean().item()}, invstd: {invstd.mean().item()}")

        ctx.save_for_backward(input, weight, mean, invstd, count_all.to(torch.int32))
        ctx.process_group = process_group
        
        # print(f"Original Forward - Input mean: {input.mean().item()}, std: {input.std().item()}")
        # print(f"Original Forward - Weight mean: {weight.mean().item()}, std: {weight.std().item()}")
        # print(f"Original Forward - Bias mean: {bias.mean().item()}, std: {bias.std().item()}")
        # print(f"Original Forward - Running mean: {running_mean.mean().item()}, std: {running_mean.std().item()}")
        # print(f"Original Forward - Running var: {running_var.mean().item()}, std: {running_var.std().item()}")
    

        # apply element-wise normalization
        out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        print(f"Original Forward - Output mean: {out.mean().item()}, std: {out.std().item()}")
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        print(f"Custom Backward - Grad output mean: {grad_output.mean().item()}, std: {grad_output.std().item()}")
        saved_input, weight, mean, invstd, count_tensor = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        process_group = ctx.process_group

        if saved_input.numel() > 0:
            sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
                grad_output, saved_input, mean, invstd, weight, ctx.needs_input_grad[0],
                ctx.needs_input_grad[1], ctx.needs_input_grad[2]
            )

            if ctx.needs_input_grad[0]:
                combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
                dist.all_reduce(combined, op=dist.ReduceOp.SUM, group=process_group)
                sum_dy, sum_dy_xmu = torch.split(combined, sum_dy.size(0))

                grad_input = torch.batch_norm_backward_elemt(
                    grad_output, saved_input, mean, invstd, weight, sum_dy, sum_dy_xmu, count_tensor
                )

        print(f"Original Backward - Grad input mean: {grad_input.mean().item()}, std: {grad_input.std().item()}")
        print(f"Original Backward - Grad weight mean: {grad_weight.mean().item()}, std: {grad_weight.std().item()}")
        print(f"Original Backward - Grad bias mean: {grad_bias.mean().item()}, std: {grad_bias.std().item()}")

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

class CustomSyncBatchNorm(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        if not input.is_contiguous():
            input = input.contiguous()

        size = input.size()
        batch_size = size[0] * size[2] * size[3]  # N*H*W
        num_channels = input.size(1)

        mean = input.mean([0, 2, 3])
        var = input.var([0, 2, 3], unbiased=False)

        print(f"Custom Forward - Local mean: {mean.mean().item()}, var: {var.mean().item()}")

        if process_group is not None:
            count = torch.full((1,), batch_size, dtype=mean.dtype, device=mean.device)
            combined = torch.cat([mean, var, count])
            dist.all_reduce(combined, op=dist.ReduceOp.SUM, group=process_group)
            mean, var, count = torch.split(combined, [num_channels, num_channels, 1])
            mean /= world_size
            var /= world_size
            count = count.item() / world_size

        print(f"Custom Forward - Synced mean: {mean.mean().item()}, var: {var.mean().item()}")

        if running_mean is not None:
            running_mean.mul_(1 - momentum).add_(mean * momentum)
        if running_var is not None:
            running_var.mul_(1 - momentum).add_(var * momentum * count / (count - 1))

        invstd = torch.rsqrt(var + eps)
        input_normalized = (input - mean[None, :, None, None]) * invstd[None, :, None, None]
        output = weight[None, :, None, None] * input_normalized + bias[None, :, None, None]

        print(f"Custom Forward - Output mean: {output.mean().item()}, std: {output.std().item()}")

        ctx.save_for_backward(input, weight, mean, invstd, torch.tensor(count))
        ctx.eps = eps
        ctx.process_group = process_group

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, mean, invstd, count = ctx.saved_tensors
        eps = ctx.eps
        process_group = ctx.process_group

        print(f"Custom Backward - Grad output mean: {grad_output.mean().item()}, std: {grad_output.std().item()}")

        N, C, H, W = input.shape
        count = count.item()

        grad_input = grad_weight = grad_bias = None

        # Compute gradients
        sum_dy = grad_output.sum(dim=[0, 2, 3])
        sum_dy_xmu = (grad_output * (input - mean[None, :, None, None])).sum(dim=[0, 2, 3])

        if process_group is not None:
            combined = torch.cat([sum_dy, sum_dy_xmu])
            dist.all_reduce(combined, op=dist.ReduceOp.SUM, group=process_group)
            sum_dy, sum_dy_xmu = torch.split(combined, C)

        if ctx.needs_input_grad[0]:
            grad_input = (grad_output - sum_dy[None, :, None, None] / (count * H * W) - 
                          (input - mean[None, :, None, None]) * sum_dy_xmu[None, :, None, None] / 
                          (count * H * W * (1 / invstd ** 2))) * invstd[None, :, None, None] * weight[None, :, None, None]

        if ctx.needs_input_grad[1]:
            grad_weight = (input - mean[None, :, None, None]) * invstd[None, :, None, None] * grad_output
            grad_weight = grad_weight.sum(dim=[0, 2, 3])

        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])

        print(f"Custom Backward - Grad input mean: {grad_input.mean().item()}, std: {grad_input.std().item()}")
        print(f"Custom Backward - Grad weight mean: {grad_weight.mean().item()}, std: {grad_weight.std().item()}")
        print(f"Custom Backward - Grad bias mean: {grad_bias.mean().item()}, std: {grad_bias.std().item()}")

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

    print(f"Rank {local_rank}: Running original SyncBatchNorm")
    out_original = model_original(data)
    print(f"Rank {local_rank}: Running custom SyncBatchNorm")
    out_custom = model_custom(data)

    diff = (out_original - out_custom).abs().mean().item()
    print(f"Rank {local_rank}: Mean absolute difference between outputs: {diff}")

    loss_original = out_original.sum()
    loss_custom = out_custom.sum()

    loss_original.backward()
    loss_custom.backward()

    for (name_o, param_o), (name_c, param_c) in zip(model_original.named_parameters(), model_custom.named_parameters()):
        if param_o.grad is not None and param_c.grad is not None:
            grad_diff = (param_o.grad - param_c.grad).abs().mean().item()
            print(f"Rank {local_rank}: Gradient difference for {name_o}: {grad_diff}")

    dist.destroy_process_group()

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    print(f"Running on rank {local_rank} of {world_size}")
    run_test(local_rank, world_size)

if __name__ == "__main__":
    import os
    main()