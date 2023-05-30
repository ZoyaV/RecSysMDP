from __future__ import annotations

import d3rlpy.algos

from recsys_mdp.utils.run.config import LazyTypeResolver, TTypeOrFactory


class TypesResolver(LazyTypeResolver):
    def __init__(self):
        super().__init__()
        self.resolvers_by_prefix = [
            ('model.', _resolve_model),
            ('env', _resolve_env_related_types),
            ('', _resolve_the_rest)
        ]

    def resolve(self, type_tag: str) -> TTypeOrFactory:
        assert type_tag is not None, 'Type is not specified'
        resolved_type = None
        for prefix, resolver in self.resolvers_by_prefix:
            if type_tag.startswith(prefix):
                resolved_type = resolver(type_tag)
                break

        if not resolved_type:
            raise ValueError(f'Unknown type tag: {type_tag}')
        return resolved_type


def _resolve_the_rest(type_tag: str):
    pass


def _resolve_env_related_types(type_tag: str):
    if type_tag == 'env':
        from recsys_mdp.simulator.env import NextItemEnvironment
        return NextItemEnvironment
    if type_tag == 'env.user_state':
        from recsys_mdp.simulator.user_state import UserState
        return UserState
    if type_tag == 'env.embeddings.random':
        from recsys_mdp.simulator.embeddings import RandomEmbeddings
        return RandomEmbeddings
    if type_tag == 'env.embeddings.clusters':
        from recsys_mdp.simulator.embeddings import RandomClustersEmbeddings
        return RandomClustersEmbeddings


def _resolve_model(type_tag: str):
    prefix = 'model.'
    type_tag = type_tag[len(prefix):]

    discrete_prefix = 'discrete_'
    if type_tag.startswith(discrete_prefix):
        type_tag = type_tag[len(discrete_prefix):]
        return d3rlpy.algos.DISCRETE_ALGORITHMS[type_tag]

    # if not starting with discrete prefix, then the priority have continuous algos
    # e.g. our type tag 'sac' relates to SAC, while in d3rlpy discrete dict it is for DiscreteSAC
    if type_tag in d3rlpy.algos.CONTINUOUS_ALGORITHMS:
        return d3rlpy.algos.CONTINUOUS_ALGORITHMS[type_tag]

    if type_tag in d3rlpy.algos.DISCRETE_ALGORITHMS:
        return d3rlpy.algos.DISCRETE_ALGORITHMS[type_tag]

