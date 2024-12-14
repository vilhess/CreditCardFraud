import torch
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    def __init__(self, df):
        # Reset index to ensure consistency
        self.features = df.drop(['Time', 'Class'], axis=1).values.astype('float32')  # Extract features
        self.targets = (1 - df['Class'].values).astype('float32')  # Invert targets if needed

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        # Access precomputed features and targets directly
        features = self.features[index]
        target = self.targets[index]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    

def get_transac_sec(df):
    new = df.copy()
    new['transactions_seconds']=1

    new['transactions_seconds'] = (new['Time'] != new['Time'].shift()).cumsum()
    new['transactions_seconds'] = new.groupby('transactions_seconds').cumcount() + 1

    return new