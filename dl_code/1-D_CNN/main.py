import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from model_cj import Model_CJ
from sklearn.metrics import mean_squared_error

class WaveDataset(Dataset):
    def __init__(self, data, sbp, transform=True):
        self.data = torch.tensor(np.load(data), dtype=torch.float32)
        self.sbp = torch.tensor(np.load(sbp), dtype=torch.float).reshape(-1, 1)

        self.max_val, self.min_val = torch.max(self.data), torch.min(self.data)
        self.range = self.max_val - self.min_val
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.sbp[index]

        if self.transform:
            x = x.reshape(-1)
            first, second = torch.zeros(x.shape[0]), torch.zeros(x.shape[0])
            a, b = x[:-1],x[1:] 
            res = b-a 
            first[1:] = res 
            a, b = first[:-1], first[1:]
            res = b-a
            second[1:]=res
            second = torch.clip((second - self.min_val) / self.range, 0, 1)
            first = torch.clip((first-self.min_val)/self.range, 0, 1)
            x = torch.clip((x-self.min_val)/self.range, 0, 1)
        return torch.stack([x,second,first]), y

    def __len__(self):
        return len(self.data)


dataset_type = "complete_overlap"
print(dataset_type)
train_dataset = WaveDataset(
    data=f"../../final_data/{dataset_type}/X_train.npy",
    sbp=f"../../final_data/{dataset_type}/Y_train.npy",
)

test_dataset = WaveDataset(
    data=f"../../final_data/{dataset_type}/X_test.npy",
    sbp=f"../../final_data/{dataset_type}/Y_test.npy",
)

save_folder = f"../../results/{dataset_type}"
os.makedirs(f"{save_folder}", exist_ok=True)


bs = 4096

train_dataloader = DataLoader(
    train_dataset,
    batch_size=bs,
    num_workers=6,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)  # sampler=weighted_sampler,
test_dataloader = DataLoader(
    test_dataset,
    batch_size=bs,
    num_workers=6,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)

epochs = 400
model = Model_CJ(1)
model.cuda()
optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
min_val_mse = 1000
best_bias, best_std_dev, best_rmse_val = 1000, 1000, 1000


for epoch in range(epochs):
    training_mse_loss, testing_mse_loss_sbp = 0.0, 0.0
    model.train()
    start_time = time.time()
    preds, gt = [], []
    for inputs, labels in train_dataloader:
        inputs, labels= inputs.cuda(), labels.cuda()
        optim.zero_grad()
        with torch.set_grad_enabled(True):
            out = model(inputs)
            loss = F.mse_loss(out, labels) 
            loss.backward()
            optim.step()

        training_mse_loss += loss.item()

    model.eval()

    for inputs, labels  in test_dataloader:
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.set_grad_enabled(False):
            out = model(inputs)
            preds.extend(out.cpu().numpy().reshape(-1))
            gt.extend(labels.cpu().numpy().reshape(-1))
            loss_1 = F.mse_loss(out[:,0].reshape(-1,1), labels)
        testing_mse_loss_sbp += loss_1.item()


    rmse_val = mean_squared_error(gt, preds, squared=False)

    errors = np.asarray(gt) - np.asarray(preds)
    bias = np.mean(errors)
    std_dev = np.std(errors)
    value_sbp  = testing_mse_loss_sbp/len(test_dataloader)
    best_bias, best_std_dev = min(best_bias, abs(bias)), min(best_std_dev, std_dev)

    if value_sbp < min_val_mse:
        min_val_mse = value_sbp
        print(f"SAVING MODEL at {min_val_mse}")
        torch.save(model, f"{save_folder}/best_mse_model.pth")

    if rmse_val < best_rmse_val:
        best_rmse_val = min(best_rmse_val, rmse_val)
        print(f"SAVING MODEL at {best_rmse_val}")
        torch.save(model, f"{save_folder}/best_rmse_model.pth")

    print(f"EPOCH:{epoch} BIAS:{str(round(bias,2))} STD_DEV:{str(round(std_dev, 2))} TRAINING_MSE_LOSS:{round(training_mse_loss/len(train_dataloader),2)} TEST_RMSE:{str(round(rmse_val,2))} TEST_BEST_RMSE:{str(round(best_rmse_val,2))} BEST_BIAS:{str(round(best_bias, 2))} BEST_STD_DEV:{str(round(best_std_dev, 2))} TESTING_MSE_LOSS_SBP:{str(round(value_sbp,2))} MIN_VAL_MSE:{str(round(min_val_mse,2))}")
    
