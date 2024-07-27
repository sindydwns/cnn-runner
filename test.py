# load dataset
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import importlib
import util
from torch import nn
import torch
from glob import glob

test_dir = "dataset_1000/test"
model_types = [
    "p_efficientnet_b0_1000",
    "p_efficientnet_b1_1000",
    "p_efficientnet_b2_1000",
    "p_efficientnet_b3_1000",
    "p_efficientnet_b4_1000",
    "p_efficientnet_b5_1000",
    "p_efficientnet_b6_1000",
    "p_efficientnet_b7_1000",
]

def run(runner: util.Runner, loader):
    targets = []
    results = []
    def func(data, target, result):
        targets.append(target)
        results.append(result)
    criterion = nn.CrossEntropyLoss()
    runner.test(loader, criterion, func)
    targets = torch.cat(targets, dim=0)
    results = torch.cat(results, dim=0)
    return targets.to("cpu"), results.to("cpu")

for model_type in model_types:
    define = importlib.import_module(model_type).define
    transforms = define.transform()
    test_data = ImageFolder(test_dir, transform=transforms)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=True)
    
    model = define.create_model()
    model_desc = define.model_desc()
    
    runner = util.Runner(model, default_dir=model_type)
    
    for model in glob(model_type + "/*.*"):
        if model.endswith(".pth") == False:
            continue
        runner.load(model)
    
        targets, results = run(runner, test_loader)
        acc = ((targets - results) == 0).sum().item() / len(targets)
        print(f"acc: {acc}")
