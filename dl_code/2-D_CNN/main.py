import collections
import os
import time

import librosa
import numpy as np
#import pyCompare
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as f
import torchvision
from torch.utils.data import (DataLoader, Dataset, TensorDataset,
                              WeightedRandomSampler)
from sklearn.metrics import mean_squared_error
class WaveDataset(Dataset):
    def __init__(
        self, data, targets, hop_len=15, win_len=8, max_val=6.5147, min_val=-3, transform=True
    ):
        self.data = torch.tensor(np.load(data), dtype=torch.float32)
        self.targets = torch.tensor(np.load(targets), dtype=torch.float).reshape(-1, 1)
        self.max_val, self.min_val = torch.max(self.data), torch.min(self.data)
        self.range = self.max_val-self.min_val 

        self.hop_len, self.crop_len = hop_len, 1250 // hop_len + 1
        self.win_len = win_len
        self.max_val_spec, self.min_val_spec = max_val, min_val
        self.range_spec = max_val - min_val
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:

            x = x.reshape(-1)
            spec = x - x.mean()
            spec = torch.clip(
                torch.log(
                    torch.abs(
                        torch.stft(
                            spec,
                            n_fft=2048,
                            hop_length=self.hop_len,
                            win_length = self.win_len*125,
                            window=torch.hann_window(self.win_len*125),
                            center=True,
                            pad_mode="reflect",
                            return_complex=True,
                        )
                    )[: self.crop_len]
                ),
                -3,
                100,
            )
            spec = torch.clip((spec - self.min_val_spec) / self.range_spec, 0, 1)
        return torch.stack([spec, spec, spec], dim=0), y

    def __len__(self):
        return len(self.data)
    

dataset_type = "no_overlap"
print(dataset_type)
train_dataset = WaveDataset(
    data=f"../final_data/{dataset_type}/X_train.npy",
    targets=f"../final_data/{dataset_type}/Y_train.npy",
)

test_dataset = WaveDataset(
    data=f"../final_data/{dataset_type}/X_test.npy",
    targets=f"../final_data/{dataset_type}/Y_test.npy",
)

save_folder = f"../results/spec_{dataset_type}"
os.makedirs(f"{save_folder}", exist_ok=True)

bs = 128

train_dataloader = DataLoader(
    train_dataset, batch_size=bs, num_workers=12, shuffle=True,  drop_last=True
) 
test_dataloader = DataLoader(
    test_dataset, batch_size=bs, num_workers=12, shuffle=False, drop_last=False
)

model = torchvision.models.densenet121(pretrained=True)
model.classifier = nn.Linear(1024, 1)  # densenet

optim = torch.optim.AdamW(
    model.parameters(), lr=1e-3
)


epochs = 200
device = "cuda:0"
model.to(device)

min_val_mse = 1000
best_bias, best_std_dev, best_rmse_val = 1000, 1000, 1000



for epoch in range(epochs):
    training_mse_loss, testing_mse_loss = 0.0, 0.0
    model.train()
    start_time = time.time()
    preds, gt = [], []
    for inputs, labels in train_dataloader:
        inputs, labels= inputs.to(device), labels.to(device)
        optim.zero_grad()
        with torch.set_grad_enabled(True):
            out = model(inputs)
            loss = F.mse_loss(out, labels)
            loss.backward()
            optim.step()
        training_mse_loss += loss.item()

    model.eval()

    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.set_grad_enabled(False):
            out = model(inputs)
            preds.extend(out.cpu().numpy().reshape(-1))
            gt.extend(labels.cpu().numpy().reshape(-1))
            loss = F.mse_loss(out, labels)
        testing_mse_loss += loss.item()

    rmse_val = mean_squared_error(gt, preds, squared=False)
    errors = np.asarray(gt) - np.asarray(preds)
    bias = np.mean(errors)
    std_dev = np.std(errors)
    value_sbp  = testing_mse_loss/len(test_dataloader)
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
