from torchvision import transforms
import torchvision.models as models
import torchvision
from torch import nn

def image_norm():
    mean = (0.4914,0.4882,0.4465)
    std = (0.247,0.243,0.261)
    return mean, std

def transform():
    mean, std = image_norm()
    return transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def create_model():
    model = models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(960, 1280),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(1280, 5)
    )
    return model

def model_desc():
    return "mobilenet_large_v3 1000"

def get_datafolder():
    return "./dataset_1000"
