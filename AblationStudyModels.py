import torch.nn as nn
import torch.nn.functional as F
import torch

# Set-up realIn_realConv Model Parameters
padd, num_InChannels = 'same', 1
numFilts_l1, numFilts_l2 = 4, 8
linFilts_l1, linFilts_l2 = 1024, 2048

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
        self.fc1 = nn.Linear(in_features=linFilts_l2, out_features=linFilts_l1)
        self.fc2 = nn.Linear(in_features=linFilts_l1, out_features=1)

    def forward(self, input):  # defines structure
        # print(f'input shape is {input.shape}')
        out = F.relu(self.conv1_1(input))
        # print(f'first output shape is {out.shape}')
        out = F.relu(self.conv1_2(out))
        # print(f'2nd output shape is {out.shape}')
        out = self.mp1(out)
        # print(f'3rd output shape is {out.shape}')

        out = F.relu(self.conv2_1(out))
        # print(f'4th output shape is {out.shape}')
        out = F.relu(self.conv2_2(out))
        # print(f'5th output shape is {out.shape}')
        out = self.mp2(out)
        # print(f'6th output shape is {out.shape}')

        out = self.flat1(out)
        # print(f'7th output shape is {out.shape}')
        out = F.relu(self.fc1(out))
        # print(f'8th output shape is {out.shape}')
        out = F.relu(self.fc2(out))
        # print(f'9th output shape is {out.shape}')
        outFinal = F.linear(input=out, weight=out)
        # print(f'final output shape is {out.shape}')

        return outFinal

#
# # Set-up compIn_realConv Model Parameters
# num_CompInChannels = 1
#
# class compIn_realConv(nn.Module):
#     def __init__(self):
#         super(compIn_realConv, self).__init__()
#         self.conv1_1 = nn.Conv1d(in_channels=num_CompInChannels, out_channels=numFilts_l1, kernel_size=(5), padding=padd)
#         self.conv1_2 = nn.Conv1d(in_channels=numFilts_l1, out_channels=numFilts_l1, kernel_size=(5), padding=padd)
#         self.mp1 = nn.MaxPool1d(kernel_size=(2))
#
#         self.conv2_1 = nn.Conv1d(in_channels=numFilts_l1, out_channels=numFilts_l2, kernel_size=(5), padding=padd)
#         self.conv2_2 = nn.Conv1d(in_channels=numFilts_l2, out_channels=numFilts_l2, kernel_size=(5), padding=padd)
#         self.mp2 = nn.MaxPool1d(kernel_size=(2))
#
#         self.flat1 = nn.Flatten()
#         self.fc1 = nn.Linear(in_features=linFilts_l2, out_features=linFilts_l1)
#         self.fc2 = nn.Linear(in_features=linFilts_l1, out_features=linFilts_l2)
#
#     def forward(self, input):  # defines structure
#         out = F.relu(self.conv1_1(input))
#         out = F.relu(self.conv1_2(out))
#         out = self.mp1(out)
#
#         out = F.relu(self.conv2_1(out))
#         out = F.relu(self.conv2_2(out))
#         out = self.mp2(out)
#
#         out = self.flat1(out)
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))
#         outFinal = F.linear(out)
#
#         return outFinal
#
# class compIn_compConv(nn.Module):
#     def __init__(self):
#         super(compIn_compConv, self).__init__()
#
#
# class CompConv2D():
#     def __init__(self, filters=1, kernel_size=1, strides=1, padding='valid', dilation_rate=1, use_bias=True, **kwargs):
#         super(CompConv2D, self).__init__()
#         self.convreal = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(kernel_size), stride=strides, padding=padding,
#                                   dilation=dilation_rate, bias=use_bias)
#         self.convimag = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(kernel_size), stride=strides, padding=padding,
#                                   dilation=dilation_rate, bias=use_bias)
#
#     def call(self, input_tensor):
#         ureal, uimag = tf.split(input_tensor, num_or_size_splits=2, axis=3)
#         oreal = self.convreal(ureal) - self.convimag(uimag)
#         oimag = self.convimag(ureal) + self.convreal(uimag)
#         x = tf.concat([oreal, oimag], axis=3)
#         return x
#
#     def get_config(self):
#         config = {
#             "convreal": self.convreal,
#             "convimag": self.convimag
#         }
#         base_config = super(CompConv2D, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))