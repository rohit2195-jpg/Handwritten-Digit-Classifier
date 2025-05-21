import torch
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image

from model import LeNet

data_trans = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def test_dataset(model):
    model.eval()
    test_loss = 0
    correct = 0


    loss_fn = nn.CrossEntropyLoss()

    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=data_trans)
    test = DataLoader(test_data, batch_size=1, shuffle=True)

    with torch.no_grad():
        for batch,(data,target) in enumerate(test):
            output = model(data)
            test_loss += loss_fn(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum()

    print("Average accuracy: " + str(100 * correct.item() / len(test)))


def test_picture(model, path):
    model.eval()



    test_data = Image.open(path).convert('L')
    test = data_trans(test_data).unsqueeze(0)
    with torch.no_grad():
        output = model(test)
        prediction = output.argmax(dim=1, keepdim=True)
        print("Prediction is ", prediction.item())

model = LeNet()
model.load_state_dict(torch.load('model.pt'))
print(test_dataset(model))