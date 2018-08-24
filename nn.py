import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torchvision as tv
import torch.optim as optim


MNIST_data = tv.datasets.MNIST("data/MNIST/", train = True, download = True,
                                transform = T.ToTensor())

train_sampler = D.sampler.SubsetRandomSampler(range(0, int(0.7 * len(MNIST_data))))
test_sampler = D.sampler.SubsetRandomSampler(range(int(0.7*len(MNIST_data)), len(MNIST_data)-1))
trainloader = D.DataLoader(MNIST_data, sampler=train_sampler)
testloader = D.DataLoader(MNIST_data, sampler=test_sampler)


#Hyperparameters
image_height = 28
image_width = 28
in_neurons = image_width * image_height
out_neurons = 10
k_size = 5


class FCModel(nn.Module):
    def __init__(self):
        super(FCModel, self).__init__()
        self.fc1 = nn.Linear(in_neurons, 2 * in_neurons)
        self.fc2 = nn.Linear(2 * in_neurons, 2 * 2 * in_neurons)
        self.fc3 = nn.Linear(2 * 2 * in_neurons, out_neurons)
    def forward(self, x):
        x = x.view(-1, in_neurons)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_neurons, 2 * in_neurons, k_size)
        self.conv2 = nn.Conv2d(2 * in_neurons, 2 * in_neurons, k_size)
        self.conv3 = nn.Conv2d(2 * in_neurons, out_neurons, k_size)
    def forward(self, x):
        #x = x.view(-1, in_neurons)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x.view(-1, out_neurons)

model = FCModel()
#model = CNNModel()

optimizer = optim.SGD(model.parameters(), lr = 0.01)
loss_func = nn.CrossEntropyLoss()

def train():
    sum_loss = 0
    print("Beginning training...")
    for epoch in range(1):
        for num, data in enumerate(trainloader):
            input, labels = data
            optimizer.zero_grad()
            output = model(input)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()
            sum_loss+=loss.item()
            if (num % 2000 == 1999):
                print("loss: " + str(sum_loss / 2000))
                sum_loss = 0
        print("Epoc " + str(epoch) + " complete")
    print("Training complete. Beginning testing...")

def test():
    num_correct = 0
    num_total = 0
    for input, target in testloader:
        predictions = model(input)
        prediction = torch.argmax(predictions, dim = 1)
        #print(prediction)
        #print(target)
        if (prediction == target):
            num_correct+=1
        num_total+=1
    print("Percent correct: " + str(num_correct/num_total*100) + "%")

train()
test()

#or input, target in testloader:
#    print(input.squeeze_(0).shape)
