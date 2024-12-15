import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, df):
        self.features = df.drop(['Time', 'Class'], axis=1).values.astype('float32')
        self.targets = (1 - df['Class'].values).astype('float32')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        features = self.features[index]
        target = self.targets[index]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    

def get_transac_n_seconds(dataset, seconds=5):
    df = dataset.copy()
    
    df = df.sort_values('Time').reset_index(drop=True)
    
    df['n_transac_last_5s'] = 0
    
    start = 0
    for end in range(len(df)):
        while df.loc[end, 'Time'] - df.loc[start, 'Time'] > seconds:
            start += 1
        df.loc[end, 'n_transac_last_5s'] = end - start
    
    return df

def get_transac_sec(df):
    new = df.copy()
    new['transactions_seconds']=1

    new['transactions_seconds'] = (new['Time'] != new['Time'].shift()).cumsum()
    new['transactions_seconds'] = new.groupby('transactions_seconds').cumcount() + 1

    return new