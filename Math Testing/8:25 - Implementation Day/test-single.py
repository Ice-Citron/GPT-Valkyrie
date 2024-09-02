import torch
import torch.nn as nn

class RegularBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.bn.weight.register_hook(lambda grad: print(f"Regular BatchNorm - Grad weight mean: {grad.mean().item()}, std: {grad.std().item()}"))
        self.bn.bias.register_hook(lambda grad: print(f"Regular BatchNorm - Grad bias mean: {grad.mean().item()}, std: {grad.std().item()}"))

    def forward(self, input):
        output = self.bn(input)
        print(f"Regular BatchNorm - Input mean: {input.mean().item()}, std: {input.std().item()}")
        print(f"Regular BatchNorm - Weight mean: {self.bn.weight.mean().item()}, std: {self.bn.weight.std().item()}")
        print(f"Regular BatchNorm - Bias mean: {self.bn.bias.mean().item()}, std: {self.bn.bias.std().item()}")
        print(f"Regular BatchNorm - Running mean: {self.bn.running_mean.mean().item()}, std: {self.bn.running_mean.std().item()}")
        print(f"Regular BatchNorm - Running var: {self.bn.running_var.mean().item()}, std: {self.bn.running_var.std().item()}")
        print(f"Regular BatchNorm - Output mean: {output.mean().item()}, std: {output.std().item()}")
        
        output.register_hook(lambda grad: print(f"Regular BatchNorm - Grad output mean: {grad.mean().item()}, std: {grad.std().item()}"))
        
        return output

class CustomBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        self.weight.register_hook(lambda grad: print(f"Custom BatchNorm - Grad weight mean: {grad.mean().item()}, std: {grad.std().item()}"))
        self.bias.register_hook(lambda grad: print(f"Custom BatchNorm - Grad bias mean: {grad.mean().item()}, std: {grad.std().item()}"))

    def forward(self, input):
        batch_size = input.size(0)
        mean = input.mean([0, 2, 3])
        var = input.var([0, 2, 3], unbiased=False)
        
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()

        input_normalized = (input - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        output = self.weight[None, :, None, None] * input_normalized + self.bias[None, :, None, None]

        print(f"Custom BatchNorm - Input mean: {input.mean().item()}, std: {input.std().item()}")
        print(f"Custom BatchNorm - Weight mean: {self.weight.mean().item()}, std: {self.weight.std().item()}")
        print(f"Custom BatchNorm - Bias mean: {self.bias.mean().item()}, std: {self.bias.std().item()}")
        print(f"Custom BatchNorm - Running mean: {self.running_mean.mean().item()}, std: {self.running_mean.std().item()}")
        print(f"Custom BatchNorm - Running var: {self.running_var.mean().item()}, std: {self.running_var.std().item()}")
        print(f"Custom BatchNorm - Output mean: {output.mean().item()}, std: {output.std().item()}")
        
        output.register_hook(lambda grad: print(f"Custom BatchNorm - Grad output mean: {grad.mean().item()}, std: {grad.std().item()}"))
        
        return output

class TestModel(nn.Module):
    def __init__(self, norm_layer):
        super(TestModel, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

def compare_batchnorms():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(42)
    model_regular = TestModel(RegularBatchNorm).to(device)
    torch.manual_seed(42)
    model_custom = TestModel(CustomBatchNorm).to(device)

    batch_size = 32
    data = torch.randn(batch_size, 3, 64, 64).to(device)

    print("Running Regular BatchNorm")
    out_regular = model_regular(data)
    print("\nRunning Custom BatchNorm")
    out_custom = model_custom(data)

    diff = (out_regular - out_custom).abs().mean().item()
    print(f"\nMean absolute difference between outputs: {diff}")

    loss_regular = out_regular.sum()
    loss_custom = out_custom.sum()

    print("\nBackward pass for Regular BatchNorm")
    loss_regular.backward()
    print("\nBackward pass for Custom BatchNorm")
    loss_custom.backward()

    print("\nGradient differences:")
    for (name_r, param_r), (name_c, param_c) in zip(model_regular.named_parameters(), model_custom.named_parameters()):
        if param_r.grad is not None and param_c.grad is not None:
            grad_diff = (param_r.grad - param_c.grad).abs().mean().item()
            print(f"Gradient difference for {name_r}: {grad_diff}")

if __name__ == "__main__":
    compare_batchnorms()