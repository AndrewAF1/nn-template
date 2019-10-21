import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, in_out_size, encoded_size):
        super(Autoencoder, self).__init__()

        self.intermediate_size = int((in_out_size+encoded_size)*0.5)

        self.encoder = nn.Sequential(
                                    nn.Linear(in_out_size, self.intermediate_size),
                                    nn.ReLU(),
                                    nn.Linear(self.intermediate_size, encoded_size)
        )
        self.decoder = nn.Sequential(
                                    nn.ReLU(),
                                    nn.Linear(encoded_size, self.intermediate_size),
                                    nn.ReLU(),
                                    nn.Linear(self.intermediate_size, in_out_size)
        )

    def forward(self, input):
        input = input.view(1, -1)
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)

        return decoded, encoded
