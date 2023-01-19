from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.utils.benchmark as benchmark

model = resnet18(weights=ResNet18_Weights.DEFAULT)

scripted_module = torch.jit.script(model)

x = torch.randn(1,3,224,224)

print(model(x).shape)

unscripted_top5_indices = torch.topk(model(x),5,dim=1)[1]
scripted_top5_indices = torch.topk(scripted_module(x),5,dim=1)[1]

assert torch.allclose(unscripted_top5_indices, scripted_top5_indices)

def run_nonscripted(x):
    return model(x)

def run_scripted(x):
    return scripted_module(x)

x = torch.randn(1,3,224,224)

t0 = benchmark.Timer(
    stmt='run_nonscripted(x)',
    setup='from __main__ import run_nonscripted',
    globals={'x': x})

t1 = benchmark.Timer(
    stmt='run_scripted(x)',
    setup='from __main__ import run_scripted',
    globals={'x': x})

print(t0.timeit(100))
print(t1.timeit(100))