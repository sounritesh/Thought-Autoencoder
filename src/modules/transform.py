import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(Encoder, self).__init__()
        # input 768*2
        # output 2

        layers = [
            nn.Linear(input_size, int(input_size/2)),
            nn.Linear(int(input_size/2), int(input_size/4)),
            nn.Linear(int(input_size/4), int(input_size/8)),
            nn.Linear(int(input_size/8), output_size),
            # nn.ReLU()
        ]

        self.encoder = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.encoder(inputs)


class Decoder(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(Decoder, self).__init__()
        # input 2
        # output 768*2
        layers = [
            nn.Linear(input_size, int(output_size/8)),
            nn.Linear(int(output_size/8), int(output_size/4)),
            nn.Linear(int(output_size/4), int(output_size/2)),
            nn.Linear(int(output_size/2), output_size),
            # nn.ReLU()
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
        # inputs = inputs.type(torch.float)

        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)

        return encoded, decoded