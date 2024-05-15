# Based on the publication
# "Frequency and phase correction of GABA-edited magnetic resonance spectroscopy using complex-valued convolutional neural networks"
# (doi: 10.1016/j.mri.2024.05.008) by Hanna Bugler, Rodrigo Berto, Roberto Souza and Ashley Harris (2024)

from torch.utils.data import Dataset

# Convert to [batch_size, channels, number of points (height), number of transients/samples (width)]
class FPC_Dataset_2C(Dataset):
    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels
        self.len = int(self.specs.shape[1])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        indv_specs = self.specs[:, index, :]
        indv_labels = self.labels[:, index]

        return indv_specs, indv_labels

class FPC_Dataset_1C(Dataset):
    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels
        self.len = int(self.specs.shape[1])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        indv_specs = self.specs[:, index, :]
        indv_labels = self.labels[:, index]

        return indv_specs, indv_labels

class FPC_Dataset_MLP(Dataset):
    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels
        self.len = int(self.specs.shape[1])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        indv_specs = self.specs[0, index, :]
        indv_labels = self.labels[:, index]

        return indv_specs, indv_labels