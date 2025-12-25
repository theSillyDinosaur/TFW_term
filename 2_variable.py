import torch
import torch.nn as nn
import torch.optim as optim
from WavKAN import *
from tqdm import *
from arithmetic_set import get_dataloader
import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import copy

# 固定 random seed（KAN 對 seed 非常敏感）
torch.manual_seed(0)

dir = f"exp{time.time()}"
os.makedirs(dir)

json_logs = []

device = "cpu"

def target_fn(x):
    u = x[:, 0]
    v = x[:, 1]
    return ((u + v) / (1 + u * v)).unsqueeze(1)

settings = [[2, 4, 1], [2, 8, 1], [2, 16, 1], [2, 32, 1], [2, 4, 4, 1], [2, 8, 8, 1], [2, 16, 16, 1], [2, 32, 32, 1], [2, 64, 1], [2, 64, 64, 1], [2, 128, 1], [2, 128, 128, 1], [2, 256, 1], [2, 256, 256, 1]]
loss_fn = nn.MSELoss()
epochs = 25

batch_size = 16
_, train_dataloader = get_dataloader(target_fn, 65536, batch_size=batch_size, device=device)
_, val_dataloader = get_dataloader(target_fn, 1024, batch_size=batch_size, device=device)
_, test_dataloader = get_dataloader(target_fn, 512, batch_size=batch_size, device=device)

for setting in settings:
    json_log = {"layer": setting}

    model = WavKAN(layers_hidden=setting).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    json_log["total_params"] = total_params

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    best_performance = 1
    train_loss_record = []
    val_performance_record = []


    pbar = tqdm(range(epochs))
    for epoch in pbar:
        train_loss = 0.0
        model.train()
        for xb, yb in train_dataloader:
            optimizer.zero_grad()
            loss = model.compute_loss(xb, yb, loss_fn)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        train_loss_record.append(train_loss)

        val_performance = 0.0
        model.eval()
        for xb, yb in val_dataloader:
            optimizer.zero_grad()
            predict = model(xb)
            mse = loss_fn(predict, yb)
            val_performance += mse.item()
        val_performance /= len(val_dataloader)
        val_performance_record.append(val_performance)
        if best_performance > val_performance:
            best_model = copy.deepcopy(model)
            best_performance = mse
            torch.save(model.state_dict(), os.path.join(dir, f"KAN_{setting}.pt"))
            json_log["val_performance"] = val_performance

        pbar.set_description(
            f"epoch {epoch:4d} | "
            f"train_loss {train_loss:.3e} | "
            f"val_performance {val_performance:.3e}"
        )

    pbar = tqdm(test_dataloader)
    model.eval()
    test_performance = 0
    for xb, yb in pbar:
        optimizer.zero_grad()
        predict = best_model(xb)
        mse = loss_fn(predict, yb)
        test_performance += mse.item()
    print(f"test_performance: {test_performance/len(test_dataloader):.3e}")
    json_log["test_performance"] = test_performance/len(test_dataloader)
    json_logs.append(json_log)

    json.dump(json_logs, open(os.path.join(dir, "log.json"), "w"), indent=4)
    np.savez(
        os.path.join(dir, f"KAN_{setting}_record.npz"), 
        train_loss_record=np.array(train_loss_record),
        val_performance_record=np.array(val_performance_record),
        test_performance=np.array(test_performance/len(test_dataloader))
    )
    plt.figure()
    plt.plot(list(range(epochs)), np.log10(np.array(train_loss_record)))
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (in log scale)')
    plt.savefig(os.path.join(dir, f"KAN_{setting}_train.png"))
    plt.close()
    plt.figure()
    plt.plot(list(range(epochs)), np.log10(np.array(val_performance_record)))
    plt.title('Val Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Performance (mse, in log scale)')
    plt.savefig(os.path.join(dir, f"KAN_{setting}_val.png"))
    plt.close()