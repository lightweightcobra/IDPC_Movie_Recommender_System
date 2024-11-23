import torch.nn as nn
import torch

class RecommenderModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        super(RecommenderModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, user_id, movie_id):
        user_vec = self.user_embedding(user_id)
        movie_vec = self.movie_embedding(movie_id)
        interaction = user_vec * movie_vec  # Element-wise product
        prediction = self.fc(interaction.sum(dim=1))
        return prediction.squeeze()
