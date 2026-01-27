import torch
import torch.nn as nn
from random import randint
import torch.optim as optim
import torch.nn.functional as F
import os
import json
from PIL import Image
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
from tqdm import tqdm
from torchvision.datasets import ImageFolder

to_tensor = tfs.Compose([tfs.ToImage(),
                         tfs.Grayscale(),
                         tfs.ToDtype(torch.float32, scale=True),
                         tfs.Lambda(lambda _img: _img.ravel())])


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x


model = Net(784, 32, 10)


loss_func = nn.CrossEntropyLoss()
optim = optim.Adam(params=model.parameters(), lr=0.01)

epochs = 2
model.train()

d_train = ImageFolder("dataset/train", transform=to_tensor)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

model_state_dict = {
    'tfs': to_tensor.state_dict(),
    'opt': optim.state_dict(),
    'model': model.state_dict()
}

best_loss = 1e10

for _e in range(epochs):
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

    if best_loss > loss_mean*1.1:
        best_loss = loss_mean
        model_state_dict['model'] = model.state_dict()
        torch.save(model_state_dict, f'model_dnn_{_e}.tar')

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
