from __future__ import annotations

from recsys_mdp.utils.run.config import LazyTypeResolver


class TypesResolver(LazyTypeResolver):
    def resolve(self, type_tag: str, **kwargs):
        if type_tag == 'embeddings.random':
            from recsys_mdp.simulator.embeddings import \
                RandomEmbeddingsGenerator
            return RandomEmbeddingsGenerator
        if type_tag == 'embeddings.clusters':
            from recsys_mdp.simulator.embeddings import \
                RandomClustersEmbeddingsGenerator
            return RandomClustersEmbeddingsGenerator
        if type_tag == 'model.random':
            from recsys_mdp.utils.random_recommender import RandomRecommender
            return RandomRecommender
        if type_tag == 'd3rlpy.cql':
            from d3rlpy.algos import CQL
            return CQL
        if type_tag == 'd3rlpy.sac':
            from d3rlpy.algos import SAC
            return SAC
        if type_tag == 'd3rlpy.ddpg':
            from d3rlpy.algos import DDPG
            return DDPG
        if type_tag == 'd3rlpy.discrete_cql':
            from d3rlpy.algos import DiscreteCQL
            return DiscreteCQL
        if type_tag == 'd3rlpy.sdac':
            from d3rlpy.algos import SDAC
            return SDAC
        if type_tag == 'd3rlpy.discrete_sac':
            from d3rlpy.algos import DiscreteSAC
            return DiscreteSAC
        if type_tag == 'd3rlpy.bc':
            from d3rlpy.algos.bc import BC
            return BC
        raise ValueError(f'Unknown type: {type_tag}')
