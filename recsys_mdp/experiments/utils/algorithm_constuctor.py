import torch

from recsys_mdp.mdp.base import USER_ID_COL, ITEM_ID_COL, RATING_COL
from recsys_mdp.models.als_model import ALSRecommender
from recsys_mdp.models.embedddings import als_embeddings


def init_als_embeddings(data, emb_dim):
    users, items = data[USER_ID_COL].values, data[ITEM_ID_COL].values
    rating = data[RATING_COL].values
    users_emb, items_emb = als_embeddings(users, items, rating, emb_dim)
    return torch.from_numpy(users_emb), torch.from_numpy(items_emb)


def init_hidden_state_encoder(**encoder):
    from recsys_mdp.models.models import ActorEncoderFactory
    return ActorEncoderFactory(**encoder)


def init_algo(actor_encoder_factory, use_gpu, batch_size, algo):

    if algo == 'CQL':
        from d3rlpy.algos import CQL
        algo = CQL(use_gpu=use_gpu, encoder_factory=actor_encoder_factory, batch_size=batch_size)

    if algo == 'DCQL':
        from d3rlpy.algos import DiscreteCQL
        algo = DiscreteCQL(
            use_gpu=use_gpu, encoder_factory=actor_encoder_factory, batch_size=batch_size
        )

    if algo == 'DBC':
        from d3rlpy.algos import DiscreteBC
        algo = DiscreteBC(use_gpu=use_gpu, batch_size=batch_size)

    if algo == 'SDAC':
        from d3rlpy.algos import SDAC
        algo = SDAC(use_gpu=use_gpu, critic_encoder_factory=actor_encoder_factory,
                    actor_encoder_factory = actor_encoder_factory,  batch_size=batch_size)
    if algo == 'DSAC':
        from d3rlpy.algos import DiscreteSAC
        algo = DiscreteSAC(use_gpu=use_gpu, critic_encoder_factory=actor_encoder_factory,
                    actor_encoder_factory = actor_encoder_factory,  batch_size=batch_size)

    return algo


def learn_als_model(data):
    model = ALSRecommender()
    model.fit(data)
    return model
