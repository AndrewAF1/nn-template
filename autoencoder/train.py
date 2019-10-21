import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from autoencoder import Autoencoder

def train(autoenc, trainloader, filename, print_freq, save_freq, num_epochs, learning_rate):
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(autoenc.parameters(), learning_rate)

    sum_loss = 0
    print("Beginning training...")
    for epoch in range(num_epochs):
        for num, data in enumerate(trainloader):
            input, _ = data
            optimizer.zero_grad()
            decoded, encoded = autoenc(input)
            loss = loss_func(decoded, input.view(1, -1))
            loss.backward()
            optimizer.step()
            sum_loss+=loss.item()
            if (num % print_freq == print_freq-1):
                print("loss: " + str(sum_loss / print_freq))
                sum_loss = 0
            if (num % save_freq == save_freq-1):
                torch.save(autoenc.state_dict(), "saved_models/" + filename)
                print("saved model")
        print("Epoc " + str(epoch) + " complete")
        torch.save(autoenc.state_dict(), "saved_models/" + filename)
        print("saved model")
    print("Training complete. Beginning testing...")
