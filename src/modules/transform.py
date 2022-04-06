import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(Encoder, self).__init__()
        layers = [
            nn.Linear(input_size, output_size),
            nn.ReLU()
        ]

        self.encoder = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.encoder(inputs)


class Decoder(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(Decoder, self).__init__()
        layers = [
            nn.Linear(input_size, output_size),
            nn.ReLU()
        ]

        self.decoder = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.decoder(inputs)


class AutoEncoder(nn.Module):
    def __init__(self, input_size, reduced_size) -> None:
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(input_size, reduced_size)
        self.decoder = Decoder(reduced_size, input_size)

    def forward(self, inputs):
        inputs = inputs.type(torch.float64)
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)

        return encoded, decoded