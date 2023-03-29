import torch
from models.embedddings import als_embeddings
def init_embedings(values, emb_dim):
    users, items = values['full_users'], values['full_items']
    rating = values['rating']
    users_emb, items_emb = als_embeddings(users, items, rating, emb_dim)
    return users_emb, items_emb
def init_algo(config, values):
    from models.models import ActorEncoderFactory
    user_num = config['experiment']['algo_settings']['model_parametrs']['user_num']
    item_num = config['experiment']['algo_settings']['model_parametrs']['item_num']
    emb_dim = config['experiment']['algo_settings']['model_parametrs']['emb_dim']
    hid_dim = config['experiment']['algo_settings']['model_parametrs']['hid_dim']
    memory_size = config['experiment']['algo_settings']['model_parametrs']['memory_size']
    feature_size = config['experiment']['algo_settings']['model_parametrs']['feature_size']
    use_attention = config['experiment']['algo_settings']['model_parametrs']['use_attention']
    freeze_emb = config['experiment']['algo_settings']['model_parametrs']['freeze_emb']
    attention_hidden_size = config['experiment']['algo_settings']['model_parametrs']['attention_hidden_size']

    use_als = config['experiment']['algo_settings']['use_als']
    model_params = [user_num, item_num, emb_dim, hid_dim, memory_size,
                    feature_size, use_attention,freeze_emb, attention_hidden_size]
    if use_als:
        users_emb, items_emb = init_embedings(values, emb_dim)
        users_emb = torch.from_numpy(users_emb)
        items_emb = torch.from_numpy(items_emb)
        model_params.append(True)
        model_params.append(users_emb)
        model_params.append(items_emb)

    actor_encoder_factory = ActorEncoderFactory(
        *model_params
    )

    use_gpu = config['experiment']['algo_settings']['use_gpu']
    batch_size = config['experiment']['algo_settings']['batch_size']
    algo = config['experiment']['algo_settings']['algo']
    if algo == 'CQL':
        from d3rlpy.algos import CQL
        algo = CQL(use_gpu=use_gpu, encoder_factory=actor_encoder_factory, batch_size=batch_size)

    if algo == 'DCQL':
        from d3rlpy.algos import DiscreteCQL
        algo = DiscreteCQL(use_gpu=use_gpu, encoder_factory=actor_encoder_factory, batch_size=batch_size)

    if algo == 'DBC':
        from d3rlpy.algos import DiscreteBC
        algo = DiscreteBC(use_gpu=use_gpu, batch_size=batch_size)

    return algo
