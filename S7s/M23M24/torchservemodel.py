from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.utils.benchmark as benchmark

model = resnet18(weights=ResNet18_Weights.DEFAULT)

scripted_model = torch.jit.script(model)

scripted_model.save('model_store/deployable_model.pt')