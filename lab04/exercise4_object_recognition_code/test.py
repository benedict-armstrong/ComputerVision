# load model from runs/4043/last_model.plk

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# use model to on example image

# load image
img = cv2.imread('airplane1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# transform image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img = transform(img)

# load model
model = torch.load('runs/4043/last_model.pkl')

# use model on image
print(model(img.unsqueeze(0)))
