from torch.utils.data import Dataset

# Convert to [batch_size, channels, number of points (height), number of transients/samples (width)]
class FPC_Dataset_compReal(Dataset):
    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels
        self.len = int(self.specs.shape[1])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        indv_specs = self.specs[:, index, :]        #0 for tapper and : for Ma and Ours
        indv_labels = self.labels[:, index]

        return indv_specs, indv_labels

class FPC_Dataset_Ma(Dataset):
    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels
        self.len = int(self.specs.shape[1])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        indv_specs = self.specs[:, index, :]        #0 for tapper and : for Ma and Ours
        indv_labels = self.labels[:, index]

        return indv_specs, indv_labels

class FPC_Dataset_Tapper(Dataset):
    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels
        self.len = int(self.specs.shape[1])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        indv_specs = self.specs[0, index, :]  # 0 for tapper and : for Ma and Ours
        indv_labels = self.labels[:, index]

        return indv_specs, indv_labels