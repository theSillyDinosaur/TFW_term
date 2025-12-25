import torch
import torch.nn as nn
import torch.optim as optim
from WavKAN import *
from KAN import *
from tqdm import *
from arithmetic_set import get_dataloader
import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# 固定 random seed（KAN 對 seed 非常敏感）
torch.manual_seed(0)

dir = f"exp{time.time()}"
os.makedirs(dir)

json_logs = []

device = "mps"

model_name = ["KAN", "WavKAN"]
settings = [[784, 128, 10], [784, 128, 128, 10], [784, 256, 10], [784, 256, 256, 10], [784, 512, 10], [784, 512, 512, 10]]
loss_fn = nn.CrossEntropyLoss()
epochs = 100
patience = 10

transform = transforms.Compose([
    transforms.ToTensor(),                # (1, 28, 28), value in [0,1]
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))  # flatten to (784,)
])

trainval_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
n_train = int(len(trainval_dataset) * 0.9)
n_val = len(trainval_dataset) - n_train
train_dataset, val_dataset = random_split(trainval_dataset, [n_train, n_val])
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for model_n in model_name:
    for setting in settings:
        json_log = {"model name": model_n, "layer": setting}

        if model_n == "KAN":
            model = KAN(layers_hidden=setting).to(device)
        elif model_n == "WavKAN":
            model = WavKAN(layers_hidden=setting).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        json_log["total_params"] = total_params

        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )

        best_performance = 0
        train_loss_record = []
        val_performance_record = []


        pbar = tqdm(range(epochs))
        for epoch in pbar:
            train_loss = 0.0
            model.train()
            for xb, yb in train_dataloader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                loss = model.compute_loss(xb, yb, loss_fn)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_dataloader)
            train_loss_record.append(train_loss)

            val_performance = 0.0
            total = 0.0
            model.eval()
            for xb, yb in val_dataloader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                predict = model(xb)
                pred_class = predict.argmax(dim=1)
                correct = (pred_class == yb).sum().item()
                val_performance += correct
                total += yb.size(0)
            val_performance /= total
            val_performance_record.append(val_performance)
            if val_performance > best_performance:
                    best_performance = val_performance
                    best_model = copy.deepcopy(model)
                    torch.save(model.state_dict(), os.path.join(dir, f"{model_n}_{setting}.pt"))
                    epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

            pbar.set_description(
                f"epoch {epoch:4d} | "
                f"train_loss {train_loss:.3e} | "
                f"val_performance {val_performance:.3e}"
            )

        pbar = tqdm(test_dataloader)
        best_model.eval()
        test_performance = 0.0
        total = 0.0
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            predict = model(xb)
            pred_class = predict.argmax(dim=1)
            correct = (pred_class == yb).sum().item()
            val_performance += correct
            total += yb.size(0)
        print(f"test_performance: {test_performance/total:.3e}")
        json_log["test_performance"] = test_performance/total
        json_logs.append(json_log)

        json.dump(json_logs, open(os.path.join(dir, "log.json"), "w"), indent=4)
        np.savez(
            os.path.join(dir, f"{model_n}_{setting}_record.npz"), 
            train_loss_record=np.array(train_loss_record),
            val_performance_record=np.array(val_performance_record),
            test_performance=np.array(test_performance/total)
        )
        plt.figure()
        plt.plot(list(range(epochs)), np.log10(np.array(train_loss_record)))
        plt.title('Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (in log scale)')
        plt.savefig(os.path.join(dir, f"{model_n}_{setting}_train.png"))
        plt.close()
        plt.figure()
        plt.plot(list(range(epochs)), np.log10(np.array(val_performance_record)))
        plt.title('Val Performance')
        plt.xlabel('Epoch')
        plt.ylabel('Performance (mse, in log scale)')
        plt.savefig(os.path.join(dir, f"{model_n}_{setting}_val.png"))
        plt.close()