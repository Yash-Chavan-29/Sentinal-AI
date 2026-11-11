import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)

print("Video AI models available:",
      hasattr(torchvision.models, "video"))

print("Action recognition setup is ready!")