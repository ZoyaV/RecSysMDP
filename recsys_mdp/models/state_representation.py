import torch
import torch.nn as nn
from d3rlpy.models.encoders import EncoderFactory
import torch.nn.functional as F

class StateReprModuleWithAttention(nn.Module):
    def __init__(
            self,
            user_num,
            item_num,
            embedding_dim,
            memory_size,
            freeze_emb,
            attention_hidden_size,
            use_als,
            user_emb,
            item_emb
    ):
        super().__init__()
        self.use_als = use_als
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.freeze_emb = freeze_emb

        self.user_embeddings = nn.Embedding(user_num, embedding_dim)
        self.item_embeddings = nn.Embedding(
            item_num + 1, embedding_dim, padding_idx=int(item_num)
        )
        self.drr_ave = torch.nn.Conv1d(
            in_channels=memory_size, out_channels=1, kernel_size=1
        )
        self.attention_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, attention_hidden_size),
            nn.ReLU(),
            nn.Linear(attention_hidden_size, 1),
        )
        self.initialize()

    def initialize(self):
        """weight init"""
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        self.item_embeddings.weight.data[-1].zero_()
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.uniform_(self.drr_ave.weight)
        self.drr_ave.bias.data.zero_()

        if self.use_als:
            self.item_embeddings.weight.data.copy_(self.item_emb)
            self.user_embeddings.weight.data.copy_(self.user_emb)

        if self.freeze_emb:
            self.user_embeddings.weight.requires_grad = False
            self.item_embeddings.weight.requires_grad = False

    @property
    def out_embeddings(self):
        return 3
    def forward(self, user, memory):
        """
        :param user: user batch
        :param memory: memory batch
        :return: vector of dimension 3 * embedding_dim
        """


        user_embedding = self.user_embeddings(user.long())
        item_embeddings = self.item_embeddings(memory.long())

        # Compute attention weights
        attention_input = torch.cat(
            [user_embedding.unsqueeze(1).expand_as(item_embeddings),
             item_embeddings], dim=-1)
        attention_scores = self.attention_net(attention_input)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Compute weighted average of item embeddings
        weighted_item_embeddings = item_embeddings * attention_weights

        # Compute average DRR values for each item
        drr_ave = self.drr_ave(weighted_item_embeddings).squeeze(1)
       # print(drr_ave.shape)
        # Concatenate user embedding, weighted item embeddings, and DRR values
        state = torch.cat(
            [user_embedding, user_embedding * drr_ave, drr_ave], dim=-1)

        return state
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
            freeze_emb,
            use_als,
            user_emb,
            item_emb
    ):
        super().__init__()

        self.use_als = use_als
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.freeze_emb = freeze_emb

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

        if self.use_als:
            self.item_embeddings.weight.data.copy_(self.item_emb)
            self.user_embeddings.weight.data.copy_(self.user_emb)

        if self.freeze_emb:
            self.item_embeddings.weight.requires_grad = False
            self.user_embeddings.weight.requires_grad = False

    @property
    def out_embeddings(self):
        return 3
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


class FullHistory(nn.Module):
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
            freeze_emb,
            use_als,
            user_emb,
            item_emb,
            state_keys
    ):
        super().__init__()

        self.memory_size = memory_size
        self.use_als = use_als
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.freeze_emb = freeze_emb
        self.state_keys = state_keys

        print(self.state_keys)
        #exit()

        self.user_embeddings = nn.Embedding(user_num, embedding_dim)
        self.item_embeddings = nn.Embedding(
            item_num + 1, embedding_dim, padding_idx=int(item_num)
        )

        #TODO: co-style, make size of score emb flexible
        self.score_embeddings = nn.Embedding(
            6, 1
        )
        self.drr_ave = torch.nn.Conv1d(
            in_channels=memory_size, out_channels=1, kernel_size=1
        )

        self.initialize()

    @property
    def out_embeddings(self):
        if 'user' in self.state_keys:
            return self.memory_size + 1
        else:
            return self.memory_size
    def initialize(self):
        """weight init"""
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        self.item_embeddings.weight.data[-1].zero_()

        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.uniform_(self.drr_ave.weight)

        self.drr_ave.bias.data.zero_()

        if self.use_als:
            self.item_embeddings.weight.data.copy_(self.item_emb)
            self.user_embeddings.weight.data.copy_(self.user_emb)

        if self.freeze_emb:
            self.item_embeddings.weight.requires_grad = False
            self.user_embeddings.weight.requires_grad = False

    def forward(self, user, memory, scorers = None):
        """
        :param user: user batch
        :param memory: memory batch
        :return: vector of dimension 3 * embedding_dim
        """

        if 'item' not in self.state_keys:
            raise Exception("Memory is a required parameter!")
        item_embeddings = self.item_embeddings(memory.long())

        if 'score' in self.state_keys:
            scorers_embeddings = scorers/5
          #  item_embeddings = item_embeddings * scorers_embeddings

        if 'user' in self.state_keys:
            user_embedding = self.user_embeddings(user.long())[:,None,:]
            item_embeddings = torch.cat((item_embeddings, user_embedding), axis = 1)

        batch, i, j = item_embeddings.shape
        out =  item_embeddings.reshape(-1, i * j)
        if 'score' in self.state_keys:
            batch, i, j = scorers_embeddings.shape
           # print(batch, i, j)
            score = scorers_embeddings.reshape(batch, i * j)
            out = torch.cat((out, score), axis = 1)
        return out
