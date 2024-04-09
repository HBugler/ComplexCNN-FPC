import torch.nn as nn
import torch
import torch.nn.functional as F

class Comp_Conv1d(torch.nn.Module):
    def __init__(self, in_filters, out_filters):
        super(Comp_Conv1d, self).__init__()
        self.convreal = nn.Conv1d(in_channels=in_filters, out_channels=out_filters, kernel_size=(5), padding='same')
        self.convimag = nn.Conv1d(in_channels=in_filters, out_channels=out_filters, kernel_size=(5), padding='same')

    def forward(self, input):  # defines structure
        ureal, uimag = torch.split(input, split_size_or_sections=[int(input.shape[1]/2), int(input.shape[1]/2)], dim=1)
        oreal = self.convreal(ureal) - self.convimag(uimag)
        oimag = self.convimag(ureal) + self.convreal(uimag)
        outFinal = torch.concat((oreal, oimag), dim=1)

        return outFinal

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

class compIn_compConv(nn.Module):
    def __init__(self):
        super(compIn_compConv, self).__init__()
        self.conv1_1 = Comp_Conv1d(in_filters=1, out_filters=8)
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


# Set-up realIn_realConv Model Parameters
paddCNN, num_InChannelsCNN = 'valid', 1
numFilts_l1CNN, numFilts_l2CNN = 8, 16
linFilts_l1CNN, linFilts_l2CNN, linFilts_l3CNN = 1012, 2024, 4048

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1_1CNN = nn.Conv1d(in_channels=num_InChannelsCNN, out_channels=numFilts_l1CNN, kernel_size=(3), padding=paddCNN)
        self.mp1CNN = nn.MaxPool1d(kernel_size=2)

        self.conv2_1CNN = nn.Conv1d(in_channels=numFilts_l1CNN, out_channels=numFilts_l2CNN, kernel_size=(3), padding=paddCNN)
        self.conv2_2CNN = nn.Conv1d(in_channels=numFilts_l2CNN, out_channels=numFilts_l2CNN, kernel_size=(3), padding=paddCNN)
        self.mp2CNN = nn.MaxPool1d(kernel_size=2)

        self.flat1CNN = nn.Flatten()
        self.fc1CNN = nn.Linear(in_features=linFilts_l3CNN, out_features=linFilts_l2CNN)
        self.fc2CNN = nn.Linear(in_features=linFilts_l2CNN, out_features=linFilts_l1CNN)
        self.fc3CNN = nn.Linear(in_features=linFilts_l1CNN, out_features=1)

    def forward(self, input):  # defines structure
        outCNN = F.relu(self.conv1_1CNN(input))
        outCNN = self.mp1CNN(outCNN)
        outCNN = F.relu(self.conv2_1CNN(outCNN))
        outCNN = F.relu(self.conv2_2CNN(outCNN))
        outCNN = self.mp2CNN(outCNN)
        outCNN = self.flat1CNN(outCNN)
        outCNN = F.relu(self.fc1CNN(outCNN))
        outCNN = F.relu(self.fc2CNN(outCNN))
        outCNNFinal = self.fc3CNN(outCNN)

        return outCNNFinal

# Set-up realIn_realConv Model Parameters
linFilts_l1MLP, linFilts_l2MLP, linFilts_l3MLP, linFilts_l4MLP, linFilts_l5MLP = 256, 512, 1024, 2048, 2496

class MLP_Model(nn.Module):
    def __init__(self):
        super(MLP_Model, self).__init__()
        self.fc_m2MLP = nn.Linear(in_features=linFilts_l3MLP, out_features=linFilts_l5MLP)
        self.fc_m1MLP = nn.Linear(in_features=linFilts_l5MLP, out_features=linFilts_l4MLP)
        self.fc0MLP = nn.Linear(in_features=linFilts_l4MLP, out_features=linFilts_l3MLP)
        self.fc1MLP = nn.Linear(in_features=linFilts_l3MLP, out_features=linFilts_l2MLP)
        self.fc2MLP = nn.Linear(in_features=linFilts_l2MLP, out_features=linFilts_l1MLP)
        self.fc3MLP = nn.Linear(in_features=linFilts_l1MLP, out_features=1)

    def forward(self, input):  # defines structure
        outMLP = F.relu(self.fc_m2MLP(input))
        outMLP = F.relu(self.fc_m1MLP(outMLP))
        outMLP = F.relu(self.fc0MLP(outMLP))
        outMLP = F.relu(self.fc1MLP(outMLP))
        outMLP = F.relu(self.fc2MLP(outMLP))
        outMLPFinal = self.fc3MLP(outMLP)

        return outMLPFinal