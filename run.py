from recsys_mdp.utils.run.config import LazyTypeResolver
from recsys_mdp.utils.run.entrypoint import (
    run_experiment, default_run_arg_parser
)


class RunnerRegistry(LazyTypeResolver):
    def resolve(self, type_tag: str):
        if type_tag == 'test.next_item':
            from recsys_mdp.experiments.next_item import (
                NextItemExperiment
            )
            return NextItemExperiment
        if type_tag == 'test.next_item_on_data':
            from recsys_mdp.experiments.next_item_on_data import (
                NextItemOnDataExperiment
            )
            return NextItemOnDataExperiment
        raise ValueError(f'Unknown type: {type_tag}')


if __name__ == "__main__":
    run_experiment(
        arg_parser=default_run_arg_parser(),
        experiment_runner_registry=RunnerRegistry(),
    )
