import torch.nn as nn
import torch.nn.functional as F
import torch

# Set-up realIn_realConv Model Parameters
padd, num_InChannels = 'valid', 1
numFilts_l1 = 4
linFilts_l1, linFilts_l2, linFilts_l3 = 512, 1024, 2048

class maModel(nn.Module):
    def __init__(self):
        super(maModel, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channels=num_InChannels, out_channels=numFilts_l1, kernel_size=(3), padding=padd)
        self.mp1 = nn.MaxPool1d(kernel_size=2)

        self.conv2_1 = nn.Conv1d(in_channels=numFilts_l1, out_channels=numFilts_l1, kernel_size=(3), padding=padd)
        self.mp2 = nn.MaxPool1d(kernel_size=2)

        self.flat1 = nn.Flatten()
        self.fc1 = nn.Linear(in_features=linFilts_l3, out_features=linFilts_l2)
        self.fc2 = nn.Linear(in_features=linFilts_l2, out_features=linFilts_l1)
        self.fc3 = nn.Linear(in_features=linFilts_l1, out_features=1)

    def forward(self, input):  # defines structure
        out = F.relu(self.conv1_1(input))
        out = self.mp1(out)

        out = F.relu(self.conv2_1(out))
        out = self.mp2(out)

        out = self.flat1(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        outFinal = self.fc3(out)

        return outFinal


