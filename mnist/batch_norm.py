import torch
import torch.nn as nn
from random import randint
import torch.optim as optim
import torch.nn.functional as F
import os
import json
from PIL import Image
import torchvision
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

to_tensor = tfs.Compose([tfs.ToImage(),
                         tfs.Grayscale(),
                         tfs.ToDtype(torch.float32, scale=True),
                         tfs.Lambda(lambda _img: _img.ravel())])


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.batch_norm(x)
        x = self.l2(x)
        return x


model = Net(784, 128, 10)

model_state_dict = torch.load('model_dnn_1.tar')

loss_func = nn.CrossEntropyLoss()
optim = optim.Adam(params=model.parameters(), lr=0.01)

epochs = 20


dataset_mnist = torchvision.datasets.MNIST(r'C:\datasets\mnist', download=True, train=True, transform=to_tensor)
d_train, d_val = data.random_split(dataset_mnist, [0.7, 0.3])
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)
train_data_val = data.DataLoader(d_val, batch_size=32, shuffle=False)

loss_lst_val = []
loss_lst = []

for _e in range(epochs):
    model.train()
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=False)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_func(predict, y_train)

        optim.zero_grad()
        loss.backward()
        optim.step()

        lm_count += 1
        loss_mean = 1/lm_count * loss.item() + (1-1/lm_count) * loss_mean

    model.eval()
    Q_val = 0
    count_val = 0

    for x_val, y_val in train_data_val:
        with torch.no_grad():
            p = model(x_val)
            loss = loss_func(p, y_val)
            Q_val += loss.item()
            count_val += 1

    Q_val /= count_val

    loss_lst.append(loss_mean)
    loss_lst_val.append(Q_val)

    print(f' Epoch {_e}/20 | loss_mean={loss_mean:.3f}, Q_val={Q_val:.3f}')

d_test = ImageFolder("dataset/test", transform=to_tensor)
test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)
Q = 0
model.eval()

for x_test, y_test in test_data:
    with torch.no_grad():
        p = model(x_test)
        p = torch.argmax(p, dim=1)
        Q += torch.sum(p == y_test).item()

Q /= len(d_test)
print(Q)

plt.plot(loss_lst)
plt.plot(loss_lst_val)
plt.grid()
plt.show()
