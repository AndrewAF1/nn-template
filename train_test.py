from torch import nn as nn
from torch import optim

def train(model, trainloader, train_settings):
    filename, print_freq, save_freq = train_settings

    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    loss_func = nn.CrossEntropyLoss()

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

            #print loss every x iterations
            if (num % print_freq == print_freq-1):
                print("loss: " + str(sum_loss / print_freq))
                sum_loss = 0

            #save model params every x iterations
            if (num % save_freq == save_freq-1):
                torch.save(autoenc.state_dict(), "saved_models/" + filename)
                print("saved model")

        print("Epoc " + str(epoch) + " complete")
    print("Training complete.")

def test(model, testloader):
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
