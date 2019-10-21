import torch
import torchvision as tv
import torchvision.transforms as T
import torch.utils.data as D
import sys
from os.path import isfile

import autoencoder
import train
import test

in_out_size = 28*28
encoded_size = 10
num_epochs = 500
print_freq = 400
save_freq = 8000
learning_rate = 0.005

filename = "mnist.pt"

MNIST_data = tv.datasets.MNIST("data/MNIST/", train = True, download = True,
                                transform = T.ToTensor())

train_sampler = D.sampler.SubsetRandomSampler(range(0, int(0.9995 * len(MNIST_data))))
test_sampler = D.sampler.SubsetRandomSampler(range(int(0.9995*len(MNIST_data)), len(MNIST_data)-1))
trainloader = D.DataLoader(MNIST_data, sampler=train_sampler)
testloader = D.DataLoader(MNIST_data, sampler=test_sampler)

autoenc = autoencoder.Autoencoder(in_out_size, encoded_size)

def load_saved():
    if isfile("saved_models/" + filename):
        autoenc.load_state_dict(torch.load("saved_models/" + filename))
        print("loading saved model")

if (sys.argv[1] == "train"):
    if "--load" in sys.argv:
        load_saved()
    train.train(autoenc, trainloader, filename, print_freq, save_freq, num_epochs, learning_rate)
elif (sys.argv[1] == "test"):
    load_saved()
    test.test(autoenc, testloader, filename)
else:
    train.train(autoenc, trainloader, filename, print_freq, save_freq, num_epochs, learning_rate)
    test.test(autoenc, testloader, filename)
