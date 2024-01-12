import torch.nn as nn
import torch.nn.functional as F
import torch


class CompConv2d(torch.nn.Module):
    def __init__(self, in_filters=1, out_filters=1, kernel_size=(1,1), strides=(1,1), padding='same'):
        super(CompConv2d, self).__init__()
        self.convreal = nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=(kernel_size), stride=strides, padding=padding)
        self.convimag = nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=(kernel_size), stride=strides, padding=padding)

    def call(self, input_tensor):
        ureal, uimag = torch.split(input_tensor, 2, dim=0)
        oreal = self.convreal(ureal) - self.convimag(uimag)
        oimag = self.convimag(ureal) + self.convreal(uimag)
        return torch.concat((oreal, oimag), dim=0)

    # def get_config(self):
    #     config = {
    #         "convreal": self.convreal,
    #         "convimag": self.convimag
    #     }
    #     base_config = super(CompConv2d, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))

# Set-up realIn_realConv Model Parameters
padd, num_InChannels = 'same', 1
numFilts_l1, numFilts_l2 = 8, 16
linFilts_l1, linFilts_l2, linFilts_l3 = 1024, 2048, 4096

class realIn_realConv(nn.Module):
    def __init__(self):
        super(realIn_realConv, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channels=num_InChannels, out_channels=numFilts_l1, kernel_size=(5), padding=padd)
        self.conv1_2 = nn.Conv1d(in_channels=numFilts_l1, out_channels=numFilts_l1, kernel_size=(5), padding=padd)
        self.mp1 = nn.MaxPool1d(kernel_size=(2))

        self.conv2_1 = nn.Conv1d(in_channels=numFilts_l1, out_channels=numFilts_l2, kernel_size=(5), padding=padd)
        self.conv2_2 = nn.Conv1d(in_channels=numFilts_l2, out_channels=numFilts_l2, kernel_size=(5), padding=padd)
        self.mp2 = nn.MaxPool1d(kernel_size=(2))

        self.flat1 = nn.Flatten()
        self.fc1 = nn.Linear(in_features=linFilts_l3, out_features=linFilts_l2)
        self.fc2 = nn.Linear(in_features=linFilts_l2, out_features=linFilts_l1)
        self.fc3 = nn.Linear(in_features=linFilts_l1, out_features=1)

    def forward(self, input):  # defines structure
        out = F.relu(self.conv1_1(input))
        out = F.relu(self.conv1_2(out))
        out = self.mp1(out)

        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.mp2(out)

        out = self.flat1(out)
        out = F.relu(self.fc1(out))
        outFinal = F.relu(self.fc2(out))
        outFinal = self.fc3(outFinal)

        return outFinal


# Set-up compIn_realConv Model Parameters
num_CompInChannels = 2

class compIn_realConv(nn.Module):
    def __init__(self):
        super(compIn_realConv, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channels=num_CompInChannels, out_channels=numFilts_l1, kernel_size=(5), padding=padd)
        self.conv1_2 = nn.Conv1d(in_channels=numFilts_l1, out_channels=numFilts_l1, kernel_size=(5), padding=padd)
        self.mp1 = nn.MaxPool1d(kernel_size=(2))

        self.conv2_1 = nn.Conv1d(in_channels=numFilts_l1, out_channels=numFilts_l2, kernel_size=(5), padding=padd)
        self.conv2_2 = nn.Conv1d(in_channels=numFilts_l2, out_channels=numFilts_l2, kernel_size=(5), padding=padd)
        self.mp2 = nn.MaxPool1d(kernel_size=(2))

        self.flat1 = nn.Flatten()
        self.fc1 = nn.Linear(in_features=linFilts_l2, out_features=linFilts_l1)
        self.fc2 = nn.Linear(in_features=linFilts_l1, out_features=1)
        self.fc3 = nn.Linear(in_features=1, out_features=1)

    def forward(self, input):  # defines structure
        out = F.relu(self.conv1_1(input))
        out = F.relu(self.conv1_2(out))
        out = self.mp1(out)

        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.mp2(out)

        out = self.flat1(out)
        out = F.relu(self.fc1(out))
        outFinal = F.relu(self.fc2(out))
        # outFinal = F.linear(self.fc3(outFinal))

        return outFinal


class compIn_compConv(nn.Module):
    def __init__(self):
        super(compIn_compConv, self).__init__()
        self.conv1_1 = CompConv2d(in_filters=num_CompInChannels, out_filters=numFilts_l1, kernel_size=(5,1), strides=(1,1), padding=padd)
        self.conv1_2 = CompConv2d(in_filters=numFilts_l1, out_filters=numFilts_l1, kernel_size=(5,1), strides=(1,1), padding=padd)
        self.mp1 = nn.MaxPool2d(kernel_size=(2,1))

        self.conv2_1 = CompConv2d(in_filters=numFilts_l1, out_filters=numFilts_l2, kernel_size=(5,1), strides=(1,1), padding=padd)
        self.conv2_2 = CompConv2d(in_filters=numFilts_l2, out_filters=numFilts_l2, kernel_size=(5,1), strides=(1,1), padding=padd)
        self.mp2 = nn.MaxPool2d(kernel_size=(2,1))

        self.fc1 = nn.Linear(in_features=linFilts_l2, out_features=linFilts_l1)
        self.fc2 = nn.Linear(in_features=linFilts_l1, out_features=1)
        self.fc3 = nn.Linear(in_features=1, out_features=1)

    def forward(self, input):  # defines structure
        out = F.relu(self.conv1_1(input))
        out = F.relu(self.conv1_2(out))
        out = self.mp1(out)

        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.mp2(out)

        out = self.flat1(out)
        out = F.relu(self.fc1(out))
        outFinal = F.relu(self.fc2(out))
        # outFinal = F.linear(self.fc3(outFinal))

        return outFinal