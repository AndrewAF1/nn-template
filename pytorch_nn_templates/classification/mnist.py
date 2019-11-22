import pytorch_nn_templates.classification.model as m
import pytorch_nn_templates.classification.train_test as t

from torchvision import datasets as tv_sets
from torchvision import transforms as T
import torch.utils.data as D
import torch.nn as nn
import torch.optim as optim

MNIST_data = tv_sets.MNIST("data/MNIST/", train = True, download = True,
                                transform = T.ToTensor())

train_sampler = D.sampler.SubsetRandomSampler(range(0, int(0.7 * len(MNIST_data))))
test_sampler = D.sampler.SubsetRandomSampler(range(int(0.7*len(MNIST_data)), len(MNIST_data)-1))
trainloader = D.DataLoader(MNIST_data, sampler=train_sampler)
testloader = D.DataLoader(MNIST_data, sampler=test_sampler)

model = m.FCModel(
    n_in = 28**2,
    n_out = 10,
    n_hidden = 64
)
model = m.CNNModel(
    image_size = 28, 
    n_hidden = 64,
    n_out = 10,
    pool_size = 2,
    conv_size = 5
)


optimizer = optim.SGD(model.parameters(), lr = 1e-4)
loss_func = nn.CrossEntropyLoss()

t.train(
    model, 
    optimizer, 
    loss_func, 
    trainloader,
    device="cuda:0",
    epochs = 1,
    filename = "mnist.pt", 
    print_freq = 2000, 
    save_freq = 5000
)

t.test(model, testloader, "cuda:0")
