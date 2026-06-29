import torch
import torchvision
import cv2
import numpy as np
import matplotlib

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
print("Setup OK")