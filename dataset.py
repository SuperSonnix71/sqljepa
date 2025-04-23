from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

class TabularDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.num_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

        num_data = self.scaler.fit_transform(df[self.num_cols])
        cat_data = self.encoder.fit_transform(df[self.cat_cols]) if self.cat_cols else np.empty((len(df), 0))
        self.data = np.hstack([num_data, cat_data]).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
 