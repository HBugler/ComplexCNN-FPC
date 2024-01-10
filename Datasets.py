from torch.utils.data import Dataset

# Convert to [batch_size, channels, number of points (height), number of transients/samples (width)]
class FPC_Dataset(Dataset):
    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels
        self.len = int(self.specs.shape[1])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.specs[:, index, :], self.labels[:, index]