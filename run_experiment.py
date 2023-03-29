import argparse

import yaml

from recsys_mdp.utils import to_d3rlpy_form_ND
from constructors.algorithm_constuctor import init_algo, init_model
from constructors.mdp_constructor import load_data, make_mdp
from constructors.scorers_constructor import init_scorers

def main(config):
    prediction_type = True if config['experiment']['scorer']['prediction_type'] == "discrete" else False

    # Load train data
    data, data_mapping, train_values = load_data(data_path=config['experiment']['data_path'],
                                                 return_values=True,
                                                 col_mapping=config['experiment']['col_mapping'])
    mdp_preparator = make_mdp(data=data, data_mapping=data_mapping, **config['experiment']['mdp_settings'])
    states, rewards, actions, termations, state_tail = mdp_preparator.create_mdp()
    train_mdp = to_d3rlpy_form_ND(states, rewards, actions, termations, discrete=prediction_type)

    # Load test data
    test_data, _, test_values = load_data(data_path=config['experiment']['test_data_path'],
                                          return_values=True,
                                          col_mapping=config['experiment']['col_mapping'])

    test_mdp_preparator = make_mdp(data=test_data, data_mapping=data_mapping, **config['experiment']['mdp_settings'])
    states, rewards, actions, termations, _ = test_mdp_preparator.create_mdp()
    test_mdp = to_d3rlpy_form_ND(states, rewards, actions, termations, discrete=prediction_type)

    # Init RL algorithm
    model = init_model(train_values, **config['experiment']['algo_settings']['model_parametrs'])
    algo = init_algo(model, **config['experiment']['algo_settings']['general_parametrs'])

    # Init scorer
    top_k = config['experiment']['top_k']
    scorers = init_scorers(state_tail, test_values, top_k, **config['experiment']['scorer'])

    # Run experiment
    batch_size = config['experiment']['algo_settings']['n_epochs']
    algo.fit(train_mdp, n_epochs=batch_size, eval_episodes=test_mdp, scorers=scorers)
    algo.save_model(f'pretrained_models/{args.experiment_name}_{name}.pt')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config')
    parser.add_argument('--experiment_name', type = str, default="default_exp")
    parser.add_argument('--framestack', type = int, default = 0)
    parser.add_argument('--model_parametrs', nargs='+',default = [])
    parser.add_argument('--freeze_emb', type=int, default=0)
    parser.add_argument('--use_als', type=int, default=0)

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)

    if args.framestack!=0:
        config['experiment']['algo_settings']['model_parametrs']['memory_size'] = args.framestack
        config['experiment']['mdp_settings']['framestack_size'] = args.framestack
        print("Update framestack: ", config['experiment']['mdp_settings']['framestack_size'])
    name = f"top_k_{config['experiment']['top_k']}_framestack_size_\
                        {config['experiment']['mdp_settings']['framestack_size']}"

    config['experiment']['algo_settings']['model_parametrs']['freeze_emb'] = args.freeze_emb
    config['experiment']['algo_settings']['use_als'] = args.use_als

    prm = []
    if len(args.model_parametrs)>1:
        prm = [int(p) for p in args.model_parametrs]
        config['experiment']['algo_settings']['model_parametrs']['user_num'] = prm[0]
        config['experiment']['algo_settings']['model_parametrs']['item_num'] = prm[1]
        config['experiment']['algo_settings']['model_parametrs']['emb_dim'] = prm[2]
        config['experiment']['algo_settings']['model_parametrs']['hid_dim'] = prm[3]
        config['experiment']['algo_settings']['model_parametrs']['memory_size'] = prm[4]
        config['experiment']['algo_settings']['model_parametrs']['feature_size'] = prm[5]
        config['experiment']['algo_settings']['model_parametrs']['use_attention'] = prm[6]
        config['experiment']['algo_settings']['model_parametrs']['attention_hidden_size'] = prm[7]


    if config['use_wandb']:
        import wandb
        wandb.init(project=f"{config['name']}", \
                   group = args.experiment_name,
                   name=f"model_{prm}_top_k_{config['experiment']['top_k']}_framestack_size_\
                        {config['experiment']['mdp_settings']['framestack_size']}")

    main(config)

