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
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

def create_model():
    model = models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(2048, 5)
    )
    return model

def model_desc():
    return "resnet152"

def get_datafolder():
    return "./dataset_1000"