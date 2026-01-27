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

to_tensor = tfs.ToImage()


class DigitDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, 'train' if train else 'test')
        self.transform = transform

        with open(os.path.join(path, 'format.json'), 'r') as fp:
            self.format = json.load(fp)

        self.length = 0
        self.files = []
        self.targets = torch.eye(10)

        for _dir, _target in self.format.items():
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path)
            self.length += len(list_files)
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))

    def __getitem__(self, item):
        path_file, target = self.files[item]
        t = self.targets[target]
        img = Image.open(path_file)

        if self.transform:
            img = self.transform(img).ravel().float() / 255.0

        return img, t

    def __len__(self):
        return self.length


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

d_train = DigitDataset("dataset", transform=to_tensor)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

for _e in range(epochs):
    train_tqdm = tqdm(train_data, leave=False)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_func(predict, y_train)

        optim.zero_grad()
        loss.backward()
        optim.step()

d_test = DigitDataset("dataset", train=False, transform=to_tensor)
test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)
Q = 0
model.eval()

for x_test, y_test in test_data:
    with torch.no_grad():
        p = model(x_test)
        p = torch.argmax(p, dim=1)
        y = torch.argmax(y_test, dim=1)
        Q += torch.sum(p == y).item()

Q /= len(d_test)
print(Q)
