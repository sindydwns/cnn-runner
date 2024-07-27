# load dataset
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import importlib
import util
from torch import nn
import torch

test_dir = "dataset_1000/test"
model_names = [
    "p_efficientnet_b0_1000/m_0727_1531246419ea66.pth",
    "p_efficientnet_b1_1000/m_0727_153827649ed5a2.pth",
    "p_efficientnet_b2_1000/m_0727_155215bcf4af20.pth",
    "p_efficientnet_b3_1000/m_0727_15561704d95a87.pth",
    "p_efficientnet_b4_1000/m_0727_155439591d5e37.pth",
    "p_efficientnet_b5_1000/m_0727_160040550c5234.pth",
    "p_efficientnet_b6_1000/m_0727_16081892072c81.pth",
    "p_efficientnet_b7_1000/m_0726_17154578038021.pth",
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

for model_name in model_names:
    model_directory_name = model_name.split("/")[0]
    define = importlib.import_module(model_directory_name).define
    transforms = define.transform()
    test_data = ImageFolder(test_dir, transform=transforms)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=True)
    
    model = define.create_model()
    model_desc = define.model_desc()
    
    runner = util.Runner(model, default_dir=model_directory_name)
    runner.load(model_name)
    
    targets, results = run(runner, test_loader)
    acc = ((targets - results) == 0).sum().item() / len(targets)
    print(f"acc: {acc}")
