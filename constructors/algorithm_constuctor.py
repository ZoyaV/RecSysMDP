
def init_algo(config):
    from models.models import ActorEncoderFactory
    user_num = config['experiment']['algo_settings']['model_parametrs']['user_num']
    item_num = config['experiment']['algo_settings']['model_parametrs']['item_num']
    emb_dim = config['experiment']['algo_settings']['model_parametrs']['emb_dim']
    hid_dim = config['experiment']['algo_settings']['model_parametrs']['hid_dim']
    memory_size = config['experiment']['algo_settings']['model_parametrs']['memory_size']
    feature_size = config['experiment']['algo_settings']['model_parametrs']['feature_size']
    model_params = [user_num, item_num, emb_dim, hid_dim, memory_size, feature_size]
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
    return algo
