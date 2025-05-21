import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import LeNet


data_trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])



train_data = datasets.MNIST(root='./data', train=True, download=True, transform=data_trans)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

model = LeNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

model.train()
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 20 == 0:
            print(f'Epoch [{epoch}/{10}]: Loss: {loss.item():.4f}')
torch.save(model.state_dict(), 'model.pt')











