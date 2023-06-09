from __future__ import annotations

from argparse import ArgumentParser
from copy import deepcopy
from functools import partial
from multiprocessing import Process
from pathlib import Path

import wandb

from recsys_mdp.utils.run.argparse import parse_arg_list
from recsys_mdp.utils.run.config import extracted, read_config
from recsys_mdp.utils.run.entrypoint import RunParams, run_single_run_experiment
from recsys_mdp.utils.run.wandb import (
    set_wandb_sweep_threading,
    turn_off_gui_for_matplotlib
)


def run_sweep(
        sweep_id: str, n_agents: int, sweep_run_params: RunParams, run_arg_parser: ArgumentParser,
        individual_cpu_cores: tuple[int, int|None] = None
) -> None:
    """
    Manages a whole wandb sweep run.

    If provided with a sweep id, assumes that it is an existing sweep id and tries to continue it.
    Otherwise, it registers a new sweep id.

    Prepares sweep execution by setting the needed env/execution params.
    Finally, spawns the specified number of agents (=processes) and waits for their completion.
    """
    set_wandb_sweep_threading()
    n_agents = n_agents if n_agents is not None else 1

    # read sweep config, extract command for parsing and wandb project for sweep initialization
    sweep_config, run_command, wandb_project = extracted(
        sweep_run_params.config, 'command', 'project'
    )

    # if ID is not provided, create/register a new sweep
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep_config, project=wandb_project)

    # parse sweep run args, extract run config path, read run config
    run_config_path = _extract_config_filepath(run_arg_parser, run_command)
    run_config_path = Path.joinpath(sweep_run_params.config_path.parent, run_config_path)
    run_config = read_config(run_config_path)

    # construct run params shared between all agents (we will construct individual ones later)
    run_params = RunParams(
        config=run_config,
        config_path=run_config_path,
        config_overrides=sweep_run_params.config_overrides,
        type_resolver=sweep_run_params.type_resolver,
    )

    agent_processes = [
        Process(
            target=wandb.agent,
            kwargs={
                'sweep_id': sweep_id,
                'function': partial(
                    _wandb_agent_entry_point, run_params=run_params,
                    cpu_affinity=(i, ) + individual_cpu_cores
                ),
                'project': wandb_project,
            }
        )
        for i in range(n_agents)
    ]

    print(f'==> Sweep {sweep_id}')
    # TODO: [on any error,] should we terminate the whole sweep or only a single agent?
    for p in agent_processes:
        p.start()
    # then wait for their completion
    for p in agent_processes:
        p.join()
    print(f'<== Sweep {sweep_id}')


# noinspection PyBroadException
def _wandb_agent_entry_point(
        run_params: RunParams, cpu_affinity: tuple[int, int, int | None]
) -> None:
    """
    This method is used by the spawned agents as a starting point for each single run job.
    """
    # BE CAREFUL: this method is expected to run in parallel
    try:
        # we tell matplotlib to not touch GUI at all in each of the spawned sub-processes
        turn_off_gui_for_matplotlib()

        i_agent, start_core, n_cores_per_agent = cpu_affinity
        if n_cores_per_agent is not None:
            # If setting cpu affinity via Math libs env variables doesn't work, use this
            # It works only on linux, win, bsd, not on macos. But it won't raise any exception.
            import os
            start = start_core + i_agent * n_cores_per_agent
            end = start + n_cores_per_agent - 1
            cpu_list = f'{start}-{end}' if end > start else f'{start}'
            os.system(f"taskset -p --cpu-list {cpu_list} {os.getpid()}")

        # we know here that it's a sweep-induced run and can expect single sweep run config
        # to be passed via wandb.config, hence we have to take it and apply all overrides;
        run = wandb.init()
        single_run_overrides = parse_arg_list(run.config.items())

        # passed `run_params` is shared between all agents, now construct a specific one for a run
        run_params = RunParams(
            config=deepcopy(run_params.config),
            config_path=run_params.config_path,
            # while concatenating overrides, the order DOES matter: single run, then whole sweep
            config_overrides=single_run_overrides + run_params.config_overrides,
            type_resolver=run_params.type_resolver,
        )
        # remove `project` attribute from the config as it is set by the sweep
        run_params.config, _ = extracted(run_params.config, 'project')

        run_single_run_experiment(run_params)
    except Exception as _:
        import traceback
        import sys
        # we catch it only to print traces to the terminal as wandb doesn't do it in Agents!
        print(traceback.print_exc(), file=sys.stderr)
        # finish explicitly with error code (NB: I tend to think it's not necessary here)
        wandb.finish(1)
        # re-raise after printing so wandb catch it
        raise


def _extract_config_filepath(parser: ArgumentParser, run_command: list[str]) -> Path:
    # there are several ways to extract config filepath based on different conventions
    # we could introduce strict positional convention or parse it with hands, but..

    # here, we use the most simple, automated way by compatibility with an existing parser:
    # we pass a pair "--config" and "<config_filepath>", which can be parsed by the parser
    args, _ = parser.parse_known_args(run_command)
    return args.config_filepath
