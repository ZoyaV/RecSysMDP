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
