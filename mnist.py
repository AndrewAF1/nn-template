import model as m
import train_test as tt

from torchvision import datasets as tvsets
from torchvision import transforms as T
import torch.utils.data as D

MNIST_data = tvsets.MNIST("data/MNIST/", train = True, download = True,
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

hyperparams = in_neurons, out_neurons, k_size

#train settings
filename = "mnist.pt"
print_freq = 2000
save_freq = 5000
train_settings = filename, print_freq, save_freq


model = m.FCModel(hyperparams)
#model = CNNModel()

tt.train(model, trainloader, train_settings)
tt.test(model, testloader)
