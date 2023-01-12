import time
from torch import flatten
from torch.nn import (
        Module,
        Conv2d,
        Linear,
        MaxPool2d,
        ReLU,
        LogSoftmax,
        ModuleList
        )


class VanillaCNN(Module):
    def __init__(self, num_channels, name=None):
        super(VanillaCNN, self).__init__()

        if not name:
            name = f'vanillacnn-{time.strftime("%Y%m%d-%H%M%S")}'
        self.name = name

        # Setup architecture
        # Set 1
        self.conv1 = Conv2d(
                in_channels=num_channels,
                out_channels=20,
                kernel_size=(12,12),
                )
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(6,6), stride=(6,6))

        # Set 2
        self.conv2 = Conv2d(
                in_channels=20,
                out_channels=50,
                kernel_size=(5,5),
                )
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(3,3), stride=(3,3))

        # Get out to linear layer
        self.fc1 = Linear(in_features=3600, out_features=500)
        self.relu3 = ReLU()

        # Regression output
        self.fc2 = Linear(in_features=500, out_features=3)

        self.conv_base = ModuleList(
                [
                    self.conv1,
                    self.relu1,
                    self.maxpool1,
                    self.conv2,
                    self.relu2,
                    self.maxpool2,
                    ]
                )
        self.final_stage = ModuleList(
                [
                    self.fc1,
                    self.relu3,
                    self.fc2,
                    ]
                )


    def forward(self, x):
        # Run through convolutions
        for layer in self.conv_base:
            x = layer(x)
        # Flatten before final layers
        x = flatten(x, 1)
        # Run through final outputs
        for layer in self.final_stage:
            x = layer(x)
        return x
