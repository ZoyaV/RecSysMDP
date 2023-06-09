from __future__ import annotations

import torch
import torch.nn as nn


class CategoricalEncoder(nn.Module):
    def __init__(
            self, n_elements: int, n_dims: int,
            learn: bool = True, initial_embeddings: torch.Tensor = None,
            use_gpu: int | None = None
    ):
        super().__init__()
        self.output_dim = n_dims

        assert initial_embeddings is not None or learn, 'Cannot freeze undefined embeddings'
        self.net = nn.Embedding(
            n_elements + 1, self.output_dim,
            # dummy element — we use an additional last element
            padding_idx=n_elements,
            _weight=initial_embeddings,
            _freeze=not learn,
        )
        self.to(device=use_gpu)

    def forward(self, ids: torch.Tensor):
        if ids.dtype != torch.long:
            ids = ids.long()
        return self.net(ids)


class StateReprModuleWithAttention(nn.Module):
    def __init__(
            self,
            embedding_dim,
            memory_size,
            attention_hidden_size,
    ):
        super().__init__()
        self.drr_ave = nn.Conv1d(
            in_channels=memory_size, out_channels=1, kernel_size=1
        )
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, attention_hidden_size),
            nn.ReLU(),
            nn.Linear(attention_hidden_size, 1),
            nn.Softmax(dim=-1)
        )
        self.output_dim = ...
        self.initialize()

    def initialize(self):
        nn.init.uniform_(self.drr_ave.weight)
        self.drr_ave.bias.data.zero_()

    @property
    def out_embeddings(self):
        return 3

    def forward(self, user_id, item_ids):
        user, items = self.observation_encoder(user_id=user_id, item_ids=item_ids)

        # Compute attention weights
        attention = self.attention(torch.cat([user, items], dim=-1))

        # Compute weighted average of item embeddings
        attended_items = items * attention

        # Compute average DRR values for each item
        drr_ave = self.drr_ave(attended_items).squeeze(1)

        # print(drr_ave.shape)
        # Concatenate user embedding, weighted item embeddings, and DRR values
        state = torch.cat(
            [user, user * drr_ave, drr_ave],
            dim=-1
        )
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
            initial_user_embeddings=None,
            initial_item_embeddings=None
    ):
        super().__init__()

        assert (initial_user_embeddings is None) == (initial_item_embeddings is None)
        self.initial_user_embeddings = initial_user_embeddings
        self.initial_item_embeddings = initial_item_embeddings
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
        nn.init.uniform_(self.drr_ave.weight)
        self.drr_ave.bias.data.zero_()

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


class ConcatState(nn.Module):
    """Computes state as a plain concatenation of user embedding and previous interactions."""

    def __init__(self, user_dim: int, interaction_dim: int, n_interactions: int):
        super().__init__()
        self.output_dim = user_dim + n_interactions * interaction_dim

    # noinspection PyMethodMayBeStatic
    def forward(self, pair: tuple[torch.Tensor, torch.Tensor]):
        user, interaction_history = pair
        interaction_history = interaction_history.flatten(start_dim=1)
        return torch.cat([user, interaction_history], dim=1)
