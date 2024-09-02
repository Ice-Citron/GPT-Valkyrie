import torch
import torch.nn as nn

def print_tensor_info(name, tensor):
    print(f"{name}:")
    print(f"  Tensor: {tensor}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Mean: {tensor.mean().item():.6f}")
    print(f"  Std: {tensor.std().item():.6f}")
    print(f"  Min: {tensor.min().item():.6f}")
    print(f"  Max: {tensor.max().item():.6f}")

class PowerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_phi, eps, afwd, abkw, ema_gz, debug, warmup_iters, current_iter, mask_x):
        print("Forward Pass:")
        print_tensor_info("x", x)
        print_tensor_info("weight", weight)
        print_tensor_info("bias", bias)
        print_tensor_info("running_phi", running_phi)
        print_tensor_info("ema_gz", ema_gz)
        print_tensor_info("mask_x", mask_x)
        print(f"eps: {eps}")
        print(f"afwd: {afwd}")
        print(f"abkw: {abkw}")
        print(f"debug: {debug}")
        print(f"warmup_iters: {warmup_iters}")
        print(f"current_iter: {current_iter}")

        ctx.eps = eps
        ctx.debug = debug
        current_iter = current_iter.item()
        ctx.current_iter = current_iter
        ctx.warmup_iters = warmup_iters
        ctx.abkw = abkw

        B, T, C = x.size()
        x2 = (mask_x * mask_x).mean(dim=1)  # Mean over sequence length
        var = x2.reshape(B, 1, C)

        print_tensor_info("x2", x2)
        print_tensor_info("var", var)

        if current_iter <= warmup_iters:
            z = x / (var + eps).sqrt()
        else:
            z = x / (running_phi + eps).sqrt()

        print_tensor_info("z", z)

        y = z
        ctx.save_for_backward(z, var, weight, ema_gz)

        if current_iter < warmup_iters:
            running_phi.copy_(running_phi * (current_iter-1)/current_iter + var.mean(dim=0, keepdim=True)/current_iter)
        running_phi.copy_(afwd*running_phi + (1-afwd)*var.mean(dim=0, keepdim=True))

        y = weight.reshape(1, 1, C) * y + bias.reshape(1, 1, C)

        print_tensor_info("y (output)", y)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        print("Backward Pass:")
        print_tensor_info("grad_output", grad_output)

        eps = ctx.eps
        abkw = ctx.abkw

        B, T, C = grad_output.size()
        z, var, weight, ema_gz = ctx.saved_tensors

        print_tensor_info("z", z)
        print_tensor_info("var", var)
        print_tensor_info("weight", weight)
        print_tensor_info("ema_gz", ema_gz)

        g = grad_output * weight.reshape(1, 1, C)
        gz = (g * z).mean(dim=1).mean(dim=0)

        print_tensor_info("g", g)
        print_tensor_info("gz", gz)

        approx_grad_g = (g - (1 - abkw) * ema_gz * z)
        ema_gz.add_((approx_grad_g * z).mean(dim=1, keepdim=True).mean(dim=0, keepdim=True))

        print_tensor_info("approx_grad_g", approx_grad_g)
        print_tensor_info("updated ema_gz", ema_gz)

        gx = 1. / torch.sqrt(var + eps) * approx_grad_g 
        grad_weight = (grad_output * z).sum(dim=1).sum(dim=0)
        grad_bias = grad_output.sum(dim=1).sum(dim=0)

        print_tensor_info("gx", gx)
        print_tensor_info("grad_weight", grad_weight)
        print_tensor_info("grad_bias", grad_bias)

        return gx, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None

# Test the implementation
if __name__ == "__main__":
    # Set up test data
    B, T, C = 1, 3, 6  # Batch size, Sequence length, Feature dimension
    x = torch.randn(B, T, C)
    print(x)

    weight = torch.ones(C)
    bias = torch.zeros(C)
    running_phi = torch.ones(1, 1, C)
    eps = 1e-5
    afwd = 0.9
    abkw = 0.9
    ema_gz = torch.zeros(1, 1, C)
    debug = False
    warmup_iters = 1000
    current_iter = torch.tensor(500)
    mask_x = x.clone()

    # Make sure all tensors require grad
    x.requires_grad_()
    weight.requires_grad_()
    bias.requires_grad_()

    # Forward pass
    print("Running forward pass...")
    output = PowerFunction.apply(x, weight, bias, running_phi, eps, afwd, abkw, ema_gz, debug, warmup_iters, current_iter, mask_x)

    # Backward pass
    print("\nRunning backward pass...")
    loss = output.sum()
    loss.backward()

    print("\nFinal gradients:")
    print_tensor_info("x.grad", x.grad)
    print_tensor_info("weight.grad", weight.grad)
    print_tensor_info("bias.grad", bias.grad)