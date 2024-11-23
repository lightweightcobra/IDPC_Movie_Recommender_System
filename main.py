import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.distributed import setup_distributed, cleanup_distributed
from data.dataset import get_dataloader, RatingsDataset
from models.recommender import RecommenderModel

def train(rank, world_size, ratings_csv, movies_csv):
    setup_distributed(rank, world_size)

    # Load datasets and initialize model
    dataset = RatingsDataset(ratings_csv, movies_csv)
    num_users = dataset.data["userId"].nunique()
    num_movies = dataset.data["movieId"].nunique()

    model = RecommenderModel(num_users, num_movies).to(rank)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    dataloader = get_dataloader(batch_size=64, rank=rank, world_size=world_size, 
                                ratings_csv=ratings_csv, movies_csv=movies_csv)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    model.train()
    for epoch in range(5):
        for user_id, movie_id, rating in dataloader:
            user_id, movie_id, rating = user_id.to(rank), movie_id.to(rank), rating.to(rank)

            optimizer.zero_grad()
            predictions = model(user_id, movie_id)
            loss = loss_fn(predictions, rating)
            loss.backward()
            optimizer.step()

        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup_distributed()

if __name__ == "__main__":
    import torch.multiprocessing as mp

    world_size = 2  # Number of nodes or GPUs
    ratings_csv = "./data/ratings.csv"
    movies_csv = "./data/movies.csv"
    mp.spawn(train, args=(world_size, ratings_csv, movies_csv), nprocs=world_size, join=True)
