import torch.nn as nn
import torch.nn.functional as F
import torch

# Set-up realIn_realConv Model Parameters
linFilts_l1, linFilts_l2, linFilts_l3 = 512, 1024, 2048

class tapperModel(nn.Module):
    def __init__(self):
        super(tapperModel, self).__init__()
        self.fc1 = nn.Linear(in_features=linFilts_l3, out_features=linFilts_l2)
        self.fc2 = nn.Linear(in_features=linFilts_l2, out_features=linFilts_l1)
        self.fc3 = nn.Linear(in_features=linFilts_l1, out_features=1)

    def forward(self, input):  # defines structure
        out = F.relu(self.fc1(input))
        out = F.relu(self.fc2(out))
        outFinal = self.fc3(out)

        return outFinal