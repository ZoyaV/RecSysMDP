from __future__ import annotations

from ast import literal_eval
from typing import Any

from recsys_mdp.experiments.run.config import TKeyPathValue


def parse_arg_list(args: list[str]) -> list[TKeyPathValue]:
    """Parse a list of command line arguments to the list of key-value pairs."""
    return list(map(parse_arg, args))


def parse_arg(arg: str | tuple[str, Any]) -> TKeyPathValue:
    """Parse a single command line argument to the key-value pair."""
    if isinstance(arg, str):
        # raw arg string: "key=value"

        # "--key=value" --> ["--key", "value"]
        key_path, value = arg.split('=', maxsplit=1)

        # "--key" --> "key"
        key_path = key_path.removeprefix('--')

        # parse value represented as str
        value = parse_str(value)
    else:
        # tuple ("key", value) from wandb config of the sweep single run
        # we assume that the passed value is already correctly parsed
        key_path, value = arg

    # parse key tokens as they can represent array indices
    # NB: skip empty key tokens (see [1] in the end of the file for an explanation)
    key_path = [
        parse_str(key_token)
        for key_token in key_path.split('.')
        if key_token
    ]

    return key_path, value


def parse_str(s: str) -> Any:
    """Parse string value to the most appropriate type."""
    # noinspection PyShadowingNames
    def boolify(s):
        if s in ['True', 'true']:
            return True
        if s in ['False', 'false']:
            return False
        raise ValueError('Not a boolean value!')

    # NB: try/except is widely accepted pythonic way to parse things
    assert isinstance(s, str)

    # NB: order of casters is important (from most specific to most general)
    for caster in (boolify, int, float, literal_eval):
        try:
            return caster(s)
        except ValueError:
            pass
    return s


# [1]: Using sweeps we have a problem with config logging. All parameters provided to
# a run from the sweep via run args are logged to wandb automatically. At the same time,
# when we also log our compiled config dictionary, its content is flattened such that
# each param key is represented as `path.to.nested.dict.key`. Note that we declare
# params in a sweep config the same way. Therefore, each sweep run will have such params
# visibly duplicated in wandb and there's no correct way to distinguish them
# (although, wandb itself does it)! Also, only sweep runs will have params duplicated.
# Simple runs don't have the duplicate entry because they don't have sweep param args.
#
# Problem: when you want to filter or group by a param in wandb interface,
# you cannot be sure which of the duplicated entries to select, while they're different
# — the only entry that is presented in all runs [either sweep or simple] is the entry
# from our config, not from a sweep.
#
# Solution: That's why we introduced a trick - you are allowed to specify sweep param
# with insignificant additional dots (e.g. `path..to...key.`) to de-duplicate entries.
# We ignore these dots [or empty path elements introduced by them after split-by-dots]
# while parsing the nested key path.
