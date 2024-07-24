from typing import Iterator
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import os
from uuid import uuid4
import csv
import time

empty_file_name = "__________________________"

class Runner(torch.nn.Module):
    def __init__(self, model, device=None, default_dir="store"):
        super(Runner, self).__init__()
        self.model: torch.nn.Module = model
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model.to(device)
        self.writer = SummaryWriter()
        self.c_loss = None
        self.counter = 0
        self.learn = []
        self.base = empty_file_name
        self.default_dir = default_dir
        self.record_file_path = default_dir + "/training_records.csv"
        if not os.path.exists(default_dir):
            os.makedirs(default_dir)

    def _train_one_epoch(self,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_data: Iterator[tuple[torch.Tensor, torch.Tensor]]):
        self.model.train(True)
        loss_total = 0
        for data, target in tqdm(train_data):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            pred: torch.Tensor = self.model(data)
            loss: torch.Tensor = criterion(pred, target)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()
            self.c_loss = loss.item()
        return loss_total / len(train_data)
    
    def run(self,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_data: Iterator[tuple[torch.Tensor, torch.Tensor]],
            test_data: Iterator[tuple[torch.Tensor, torch.Tensor]],
            epochs: int = 1,
            record: dict = {}):
        for epoch in range(epochs):
            loss = self._train_one_epoch(criterion, optimizer, train_data)
            acc = self.test(test_data)
            self.writer.add_scalar('loss', loss, self.counter)
            self.writer.add_scalar('acc', acc, self.counter)
            self.counter += 1
            self.learn.append({
                "epoch": self.counter,
                "criterion": type(criterion).__name__,
                "optimizer": type(optimizer).__name__,
                "loss": math.floor(loss * 100000) / 100000,
                "acc": math.floor(acc * 100000) / 100000,
            } | record)
            print(f"epoch: {epoch + 1}; loss: {loss:.3f}; acc: {acc:.3f}")
            time.sleep(1) # 중간에 멈출 수 있게 위해 약간의 시간을 둠
            
    def test(self,
            iterator: Iterator[tuple[torch.Tensor, torch.Tensor]],
            action: callable = None):
        self.model.train(False)
        with torch.no_grad():
            total = 0
            corr = 0
            for data, target in tqdm(iterator):
                data, target = data.to(self.device), target.to(self.device)
                pred: torch.Tensor = self.model(data)
                result = torch.max(pred, 1)[1]
                if action is not None:
                    action(data, target, result)
                total += len(result)
                corr += (result == target.to(self.device)).sum().item()
            acc = corr / total
        return acc

    def save(self, path=None):
        if path == None:
            path = datetime.now().strftime(f"m_%m%d_%H%M%S_{str(uuid4()).split("-")[0]}.pth")
        torch.save(self.model.state_dict(), self.default_dir + "/" + path)
        with open(self.record_file_path, "a", encoding="utf-8") as file:
            if os.path.getsize(self.record_file_path) == 0:
                 file.write("date,base,tag,model,criterion,optimizer,lr,epoch,loss,acc,memo\n")
            for v in self.learn:
                lst = []
                lst.append(datetime.now().strftime("%m%d_%H%M%S"))   #0
                lst.append(self.base)                               #1
                lst.append(path)                                    #2
                lst.append(str(f'"{v["model"]}"'))                  #3
                lst.append(str(f'{v["criterion"]}'))                #4
                lst.append(str(f'{v["optimizer"]}'))                #5
                lst.append(str(f'{v["lr"]}'))                       #6
                lst.append(str(f'{v["epoch"]:5}'))                  #7
                lst.append(str(f'{v["loss"]:.5f}'))                 #8
                lst.append(str(f'{v["acc"]:.5f}'))                  #9
                lst.append(str(f'"{v["memo"]}"'))                   #10
                file.write(",".join(lst) + "\n")
        self.base = path.split("/")[-1]
        self.learn.clear()
        print(f"saved {path}")
    
    def load(self, path):
        if path is None:
            pth_files = [self.default_dir + "/" + f for f in os.listdir(self.default_dir) if f.endswith(".pth")]
            if not pth_files:
                print("no saved file")
                return
            files = [(os.path.getctime(pth_file), pth_file) for pth_file in pth_files]
            files.sort(key=lambda x: x[0])
            path = files[-1][1]
        m = torch.load(path, map_location=self.device)
        self.model.load_state_dict(m)
        self.base = path.split("/")[-1]
        print(f"loaded {path}")
        
        # update counter
        try:
            self.counter = 0
            with open(self.record_file_path, "r") as file:
                reader = csv.reader(file)
                for row in reader:
                    if self.base != str(row[2]):
                        continue
                    counter = int(row[7])
                    self.counter =  counter if self.counter < counter else self.counter
        except:
            self.counter = 0






############################################################################





from torch.utils.data import Dataset
from PIL import Image
import glob

class CustomDataset(Dataset):
    def __init__(self, patterns=[], transform=None):
        super(CustomDataset, self).__init__()
        self.transform = transform
        self.item_list = []
        self.label_list = []
        
        label = 0
        for pattern in patterns:
            item_list = glob.glob(pattern)
            assert len(item_list) != 0, f"no content: {pattern}"
            self.item_list += item_list
            self.label_list += [label] * len(item_list)
            label += 1
            
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img_path = self.item_list[idx]
        label = self.label_list[idx]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label





############################################################################





import matplotlib.pyplot as plt
import math
def show(img: torch.Tensor):
    if img.ndim == 3:
        temp_img = img.numpy()
        temp_img_tr = temp_img.transpose(1,2,0)
        plt.imshow(temp_img_tr)
    elif img.ndim == 4:
        count = len(img)
        if count == 0:
            return
        col = int(math.sqrt(count))
        row = int(math.ceil(count / col))
        fig = plt.figure(figsize=(10, 10))
        
        idx = 0
        for r in range(row):
            for c in range(col):
                if idx >= len(img):
                    return
                fig.add_subplot(row, col, idx + 1)
                temp_img = img[idx].numpy()
                temp_img_tr = temp_img.transpose(1,2,0)
                plt.imshow(temp_img_tr)
                plt.axis("off")
                idx += 1 
    else:
        assert False, f"show: image type not match ({img.shape})"

import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
def show2(
        img: torch.Tensor,
        title,
        mean=np.array([0.0, 0.0, 0.0]),
        std=np.array([1.0, 1.0, 1.0])):
    if img.ndim == 4:
        img = make_grid(img)
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    plt.imshow(img)
    plt.title(title)
    plt.show()
