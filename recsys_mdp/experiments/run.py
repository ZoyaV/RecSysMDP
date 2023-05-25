from recsys_mdp.utils.run.config import LazyTypeResolver
from recsys_mdp.utils.run.entrypoint import (
    run_experiment, default_run_arg_parser
)


class RunnerRegistry(LazyTypeResolver):
    def resolve(self, name: str):
        if name == 'test.mdp_next_item':
            from recsys_mdp.experiments.scenarios.mdp_next_item import (
                MdpNextItemExperiment
            )
            return MdpNextItemExperiment
        if name == 'test.mdp_nip_reset':
            from recsys_mdp.experiments.scenarios.mdp_nip_with_reset import (
                NIP_with_reset
            )
            return NIP_with_reset


if __name__ == "__main__":
    run_experiment(
        arg_parser=default_run_arg_parser(),
        experiment_runner_registry=RunnerRegistry(),
    )
