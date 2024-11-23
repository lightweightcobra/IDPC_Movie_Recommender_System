import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class RatingsDataset(Dataset):
    def __init__(self, ratings_csv, movies_csv):
        # Load datasets
        ratings = pd.read_csv(ratings_csv)
        movies = pd.read_csv(movies_csv)
        
        # Merge datasets
        self.data = ratings.merge(movies, on="movieId")
        
        # Normalize userId and movieId for embedding layers
        self.data["userId"] = self.data["userId"].astype("category").cat.codes
        self.data["movieId"] = self.data["movieId"].astype("category").cat.codes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return torch.tensor(row["userId"]), torch.tensor(row["movieId"]), torch.tensor(row["rating"], dtype=torch.float)

def get_dataloader(batch_size, rank, world_size, ratings_csv, movies_csv):
    dataset = RatingsDataset(ratings_csv, movies_csv)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader
