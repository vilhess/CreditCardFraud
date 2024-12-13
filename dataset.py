import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        subdf = self.df.iloc[index]
        features = subdf.drop(['Time', 'Class']).values
        tar = subdf['Class'].item()
        return torch.Tensor(features), 1-torch.tensor(tar)
