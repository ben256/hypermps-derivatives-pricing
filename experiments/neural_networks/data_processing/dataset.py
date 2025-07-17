from torch.utils.data import Dataset


class TTDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mu, cores = self.samples[idx]
        return mu, cores
