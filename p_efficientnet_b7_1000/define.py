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
    model = models.efficientnet_b7(weights=torchvision.models.EfficientNet_B7_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(2560, 4),
    )
    return model

def model_desc():
    return "efficientnet_b7 dataset angry1000 neutral1000 shock1000 smile1000"

def get_datafolder():
    return "./dataset_4_1000"