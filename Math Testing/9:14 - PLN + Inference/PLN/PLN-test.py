import torch
import torch.nn.functional as F
import torch.distributed as dist


class SyncPowerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, running_phi, ema_gz, eps, afwd, abkw, warmup_iters, current_iter, process_group):
        ctx.eps = eps
        current_iter = current_iter.item()
        ctx.abkw = abkw
        ctx.process_group = process_group

        B, T, C = x.size()
        x2 = torch.mean(torch.square(x), dim=(0, 1))
        var = x2.view(1, 1, C)

        if current_iter <= warmup_iters:
            y = x * torch.rsqrt(var + eps)
        else:
            y = x * torch.rsqrt(running_phi + eps)

        ctx.save_for_backward(y, var, ema_gz)

        if current_iter < warmup_iters:
            running_phi.copy_(running_phi * (current_iter-1)/current_iter + var/current_iter)
        running_phi.copy_(afwd*running_phi + (1-afwd)*var)

        if process_group is not None:
            dist.all_reduce(running_phi, op=dist.ReduceOp.AVG, group=process_group)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        abkw = ctx.abkw

        y, var, ema_gz = ctx.saved_tensors

        approx_grad_g = (grad_output - (1 - abkw) * ema_gz * y)
        ema_gz.add_(torch.mean(approx_grad_g * y, dim=(0, 1), keepdim=True))

        if ctx.process_group is not None:
            dist.all_reduce(ema_gz, op=dist.ReduceOp.AVG, group=ctx.process_group)

        gx = torch.rsqrt(var + eps) * approx_grad_g

        return gx, None, None, None, None, None, None, None, None

class PowerLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, alpha_fwd=0.9, alpha_bkw=0.9, warmup_iters=10000, process_group=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.process_group = process_group
        self.alpha_fwd = alpha_fwd
        self.alpha_bkw = alpha_bkw
        self.warmup_iters = warmup_iters

        # Learnable alpha parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Affine parameters for the final transformation
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # PowerNorm running statistics (as regular tensors, not buffers)
        self.running_phi = nn.Parameter(torch.ones(1, 1, num_features), requires_grad=False)
        self.ema_gz = nn.Parameter(torch.zeros(1, 1, num_features), requires_grad=False)
        self.iters = nn.Parameter(torch.ones(1, dtype=torch.long), requires_grad=False)

        self.grad_accum_steps = args.gradient_accumulation_steps
        self.accum_count = 0

    def extra_repr(self):
        return f'{self.num_features}, eps={self.eps}, alpha_fwd={self.alpha_fwd}, alpha_bkw={self.alpha_bkw}, ' \
               f'warmup={self.warmup_iters}'

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        assert x.size(-1) == self.num_features, f"Input features {x.size(-1)} doesn't match num_features {self.num_features}"

        # 1. Layer Normalization (without affine transformation)
        x_ln = F.layer_norm(x, (self.num_features,), eps=self.eps) # weights and biases are None by default from torch.F

        # 2. PowerNorm (without affine transformation)
        if self.training:
            self.accum_count += 1
            if self.accum_count >= self.grad_accum_steps:
                self.iters.add_(1)
                self.accum_count = 0
                
            x_pn = SyncPowerFunction.apply(
                x, self.running_phi, self.ema_gz, self.eps, self.alpha_fwd, self.alpha_bkw,
                self.warmup_iters, self.iters, self.process_group
            )
        else:
            x_pn = x * torch.rsqrt(self.running_phi + self.eps)

        # 3. Combine LN and PN using learnable alpha
        alpha = torch.clamp(self.alpha, 0, 1)  # Ensure alpha is between 0 and 1
        x_combined = (alpha + self.eps) * x_ln + (1 - (alpha + self.eps)) * x_pn

        # 4. Apply affine transformation (weights and biases)
        return self.weight * x_combined + self.bias
