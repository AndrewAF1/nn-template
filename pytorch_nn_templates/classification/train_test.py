import torch
from torch import nn as nn
from torch import optim
from pytorch_nn_templates.classification.visualize import VisdomLinePlotter

def train(model, optimizer, loss_func, trainloader, device, epochs, filename, print_freq, save_freq):

    plotter = VisdomLinePlotter()

    model = model.to(device)

    sum_loss = 0
    print("Beginning training...")
    for epoch in range(epochs):
        for num, sample in enumerate(trainloader):
            data, labels = sample
            data = data.to(device); labels = labels.to(device)

            optimizer.zero_grad()
            
            output = model(data)
            
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()
            
            sum_loss+=loss.item()

            #print loss every n iterations
            if ((epoch*len(trainloader) + num) % print_freq == print_freq-1):
                print("loss: " + str(sum_loss / print_freq))
                plotter.plot('loss', 'train', 'Loss', epoch*len(trainloader) + num, (sum_loss / print_freq))
                sum_loss = 0

            #save model params every n iterations
            if (num % save_freq == save_freq-1):
                torch.save(model.state_dict(), "saved_models/" + filename)
                print("saved model")

        print("Epoc " + str(epoch) + " complete")
    print("Training complete.")

def test(model, testloader, device):
    model = model.to(device)

    num_correct = 0
    num_total = 0
    for data, labels in testloader:
        data = data.to(device); labels = labels.to(device)

        predictions = model(data)
        prediction = torch.argmax(predictions, dim = 1)
        #print(prediction)
        #print(target)
        if (prediction == labels):
            num_correct+=1
        num_total+=1
    print("Percent correct: " + str(num_correct/num_total*100) + "%")
    return num_correct/num_total*100
