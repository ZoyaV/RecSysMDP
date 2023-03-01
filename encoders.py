from torch import nn
import torch


class StateReprModule(nn.Module):
    """
    Compute state for RL environment. Based on `DRR paper
    <https://arxiv.org/pdf/1810.12027.pdf>`_

    Computes State is a concatenation of user embedding,
    weighted average pooling of `memory_size` latest relevant items
    and their pairwise product.
    """

    def __init__(
            self,
            user_num,
            item_num,
            embedding_dim,
            memory_size,
    ):
        super().__init__()
        self.user_embeddings = nn.Embedding(user_num, embedding_dim)
        self.item_embeddings = nn.Embedding(
            item_num + 1, embedding_dim, padding_idx=int(item_num)
        )
        self.drr_ave = torch.nn.Conv1d(
            in_channels=memory_size, out_channels=1, kernel_size=1
        )

        self.initialize()

    def initialize(self):
        """weight init"""
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        self.item_embeddings.weight.data[-1].zero_()

        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.uniform_(self.drr_ave.weight)

        self.drr_ave.bias.data.zero_()

    def forward(self, user, memory):
        """
        :param user: user batch
        :param memory: memory batch
        :return: vector of dimension 3 * embedding_dim
        """
        user_embedding = self.user_embeddings(user.long())

        item_embeddings = self.item_embeddings(memory.long())
        drr_ave = self.drr_ave(item_embeddings).squeeze(1)

        return torch.cat(
            (user_embedding, user_embedding * drr_ave, drr_ave), 1
        )


import torch
import torch.nn as nn
from d3rlpy.models.encoders import EncoderFactory

class ActorEncoder(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim, hidden_dim, memory_size, feature_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_size),
        )

        self.state_repr = StateReprModule(
            user_num, item_num, embedding_dim, memory_size
        )

        self.feature_size = feature_size

        self.initialize()

    def initialize(self):
        """weight init"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, x):
        """
        :param user: user batch
        :param memory: memory batch
        :return: output, vector of the size `feature_size`
        """
        user = x[:,-1]
        memory = x[:, :-1]
        state = self.state_repr(user, memory)
        return self.layers(state)

    def get_feature_size(self):
        return self.feature_size

class ActorEncoderFactory(EncoderFactory):
    TYPE = 'custom'

    def __init__(
            self,
            user_num,
            item_num,
            embedding_dim,
            hidden_dim,
            memory_size,
            feature_size,
    ):
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.feature_size = feature_size

    def create(self, observation_shape):
        return ActorEncoder(
            self.user_num,
            self.item_num,
            self.embedding_dim,
            self.hidden_dim,
            self.memory_size,
            self.feature_size,
        )

    def get_params(self, deep=False):
        return {
            'user_num': self.user_num,
            'item_num': self.item_num,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'memory_size': self.memory_size,
            'feature_size': self.feature_size,
        }


def load_embeddings(path_to_model="model_final.pt", model_params=[943, 1682, 8, 16, 5]):
    the_model = ActorEncoder(*model_params)
    the_model.load_state_dict(torch.load(path_to_model))
    user_emb = the_model.state_repr.user_embeddings.weight.data
    item_emb = the_model.state_repr.item_embeddings.weight.data
    return user_emb, item_emb