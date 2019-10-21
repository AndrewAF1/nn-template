import torch
import matplotlib.pyplot as plt

from autoencoder import Autoencoder

def test(autoenc, testloader, filename):

    for num, data in enumerate(testloader):
        input, _ = data
        with torch.no_grad():
            decoded, encoded = autoenc(input)

        dec_image = decoded.view(28, 28)
        original_image = input.squeeze(0).squeeze(0)

        results = plt.figure()
        ax1 = results.add_subplot(2,2,1)
        ax1.imshow(original_image)
        ax2 = results.add_subplot(2,2,2)
        ax2.imshow(dec_image)

        print("showing")
        results.savefig("saved_results/img" + str(num) + ".png")
