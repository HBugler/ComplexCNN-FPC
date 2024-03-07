import torch.nn as nn
import torch.nn.functional as F
import torch


class Comp_Conv1d(torch.nn.Module):
    def __init__(self, in_filters, out_filters):
        super(Comp_Conv1d, self).__init__()
        self.convreal = nn.Conv1d(in_channels=in_filters, out_channels=out_filters, kernel_size=(5), padding='same')
        self.convimag = nn.Conv1d(in_channels=in_filters, out_channels=out_filters, kernel_size=(5), padding='same')

    def forward(self, input):  # defines structure
        # print(f'Comp_Conv2d input size is {input.shape}')
        ureal, uimag = torch.split(input, split_size_or_sections=[int(input.shape[1]/2), int(input.shape[1]/2)], dim=1)
        # print(f'Comp_Conv2d ureal size is {ureal.shape} and uimag {uimag.shape}')
        oreal = self.convreal(ureal) - self.convimag(uimag)
        # print(f'Comp_Conv2d oreal size is {oreal.shape}')
        oimag = self.convimag(ureal) + self.convreal(uimag)
        # print(f'Comp_Conv2d oimag size is {oimag.shape}')
        outFinal = torch.concat((oreal, oimag), dim=1)
        # print(f'Comp_Conv2d outFinal size is {outFinal.shape}')

        return outFinal

# Set-up realIn_realConv Model Parameters
padd, num_InChannels = 'same', 1
numFilts_l1, numFilts_l2 = 8, 16     #10 mp: 8, 16
linFilts_l1, linFilts_l2, linFilts_l3 = 1024, 2048, 4096    #10 mp: 1024, 2048, 4096


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

    def forward(self, input):
        out = F.relu(self.conv1_1(input))
        out = F.relu(self.conv1_2(out))
        out = self.mp1(out)

        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.mp2(out)

        out = self.flat1(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        outFinal = self.fc3(out)

        return outFinal


class compIn_realConv(nn.Module):
    def __init__(self):
        super(compIn_realConv, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channels=2, out_channels=numFilts_l1, kernel_size=(5), padding=padd)
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
        out = F.relu(self.fc2(out))
        outFinal = self.fc3(out)

        return outFinal

# halve the number of filters to reduce parameter count

class compIn_compConv(nn.Module):
    def __init__(self):
        super(compIn_compConv, self).__init__()
        self.conv1_1 = Comp_Conv1d(in_filters=1, out_filters=8)     #technically 1 channel because real and imaginary are going in seperately
        self.conv1_2 = Comp_Conv1d(in_filters=8, out_filters=8)
        self.mp1 = nn.MaxPool1d(kernel_size=(2))

        self.conv2_1 = Comp_Conv1d(in_filters=8, out_filters=16)
        self.conv2_2 = Comp_Conv1d(in_filters=16, out_filters=16)
        self.mp2 = nn.MaxPool2d(kernel_size=(2))

        self.flat1 = nn.Flatten()

        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=1)


    def forward(self, input):  # defines structure
        out = F.relu(self.conv1_1(input))
        out = F.relu(self.conv1_2(out))
        out = self.mp1(out)

        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.mp2(out)

        out = self.flat1(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        outFinal = self.fc3(out)

        return outFinal