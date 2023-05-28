from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Collection, Union, Type, Callable

from ruamel import yaml

# config-related types
TConfig = Union[
    dict[str, Any],
    list[Any]
]
TKeyPath = list
TKeyPathValue = tuple[TKeyPath, Any]


# ==================== resolve absolute or relative quantity ====================
# quantities can be specified as absolute or relative to some baseline value

def resolve_absolute_quantity(abs_or_relative: int | float, *, baseline: int) -> int:
    """
    Convert passed quantity to the absolute quantity regarding its type and the baseline value.
    Here we consider that ints relate to the absolute quantities and floats
    relate to the relative quantities (relative to the `baseline` value).

    Examples:
        ensure_absolute(10, 20) -> 10
        ensure_absolute(1.25, 20) -> 25


    Parameters
    ----------
    abs_or_relative: int or float
        The value to convert. If it's int then it's returned as is. Otherwise, it's
        converted to the absolute system relative to the `baseline` value
    baseline: int
        The baseline for the relative number system.

    Returns
    -------
        Integer value in the absolute quantities system
    """

    if isinstance(abs_or_relative, float):
        relative = abs_or_relative
        return int(baseline * relative)
    elif isinstance(abs_or_relative, int):
        absolute = abs_or_relative
        return absolute
    else:
        raise TypeError(f'Function does not support type {type(abs_or_relative)}')


def resolve_relative_quantity(abs_or_relative: int | float, *, baseline: int) -> float:
    """See `resolve_absolute_quantity` - this method is the opposite of it."""

    if isinstance(abs_or_relative, float):
        relative = abs_or_relative
        return relative
    elif isinstance(abs_or_relative, int):
        absolute = abs_or_relative
        return absolute / baseline
    else:
        raise TypeError(f'Function does not support type {type(abs_or_relative)}')


# ==================== config dict manipulation utils ====================
def override_config(
        config: TConfig,
        overrides: list[TKeyPathValue] | TKeyPathValue
) -> None:
    """Apply the number of overrides to the content of the config dictionary."""
    overrides = overrides if isinstance(overrides, list) else [overrides]
    for key_path, value in overrides:
        c = config
        for key_token in key_path[:-1]:
            c = c[key_token]
        c[key_path[-1]] = value


def filtered(d: TConfig, keys_to_remove: Collection[str], depth: int) -> TConfig:
    """
    Return a shallow copy of the provided dictionary without the items
    that match `keys_to_remove`.

    The `depth == 1` means filtering `d` itself,
        `depth == 2` — with its dict immediate descendants
        and so on.
    """
    if not isinstance(d, dict) or depth <= 0:
        return d

    return {
        k: filtered(v, keys_to_remove, depth - 1)
        for k, v in d.items()
        if k not in keys_to_remove
    }


def extracted(d: TConfig, *keys: str) -> tuple:
    """
    Return a copy of the dictionary without specified keys and each extracted value
    (or None if a specified key was absent).

    NOTE: Sadly, type checkers incorrectly understand the correct type hint here,
    which is tuple[TConfig, Optional[Any], ...], so a less strict type hint is provided

    Examples
    --------
    >>> extracted({'a': 1, 'b': 2, 'c': 3}, 'a', 'c')
    ({'b': 2}, 1, 3)
    """
    values = tuple([d.get(k, None) for k in keys])
    filtered_dict = filtered(d, keys, depth=1)
    return (filtered_dict, ) + values


# ==================== read config ====================

def read_config(filepath: str | Path) -> TConfig:
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    with filepath.open('r') as config_io:
        return yaml.load(config_io, Loader=yaml.Loader)


# ==> special value constants
# They are intended to be non-importable, i.e. to be used only here!


# value should be resolved as None — it is a more explicit way to indicate None than empty string
_TO_BE_NONE_VALUE = '...'

# value must be resolved later by induction on a building stage
#   e.g. layers dimension propagation, DRY principle, calculated dynamically
_TO_BE_INDUCED_VALUE = '???'


def resolve_value(
        value: Any, *,
        substitute_with: Any = _TO_BE_INDUCED_VALUE,
        key: str = None, induction_registry: dict[str, Any] = None
) -> Any:
    """
    Resolve value defined with the config. Some values have specific meaning, which is handled here.
    """
    if value == _TO_BE_NONE_VALUE:
        return None

    elif value == _TO_BE_INDUCED_VALUE:
        # substitute, but default substitution in both scenario — leave it to-be-induced

        if key is None:
            # direct substitution
            return substitute_with
        else:
            # try substitute using registry first then using direct substitution
            return induction_registry.get(key, substitute_with)

    # return already resolved value as is
    return value


# ==================== config dict value induction ====================
def get_unresolved_value() -> Any:
    return _TO_BE_INDUCED_VALUE


def is_resolved_value(value: Any) -> bool:
    return value != _TO_BE_NONE_VALUE and value != _TO_BE_INDUCED_VALUE


def resolve_init_params(config: TConfig, **induction_registry):
    """
    Resolve params defined with the config. Some values are intended to be resolved
    later at runtime - so, it tries to substitute special values with the
    values from the induction registry.
    """
    return {
        k: resolve_value(config[k], key=k, induction_registry=induction_registry)
        for k in config
    }


def check_all_resolved(*values) -> bool:
    """Check all provided values are resolved, i.e. there is no value equal to specific constant"""
    resolved = True
    for x in values:
        resolved &= is_resolved_value(x)
    return resolved


TTypeOrFactory = Union[Type, Callable]

TTypeResolver = Union[
    dict[str, TTypeOrFactory],
    'LazyTypeResolver'
]


class LazyTypeResolver(ABC):
    """
    TypeResolverBase is a convenient base class for all type resolvers.
    It is a dictionary-like object that lazily resolves type tags to types.
    """
    types: dict[str, TTypeOrFactory]

    def __init__(self):
        self.types = {}

    def __getitem__(self, type_tag: str) -> TTypeOrFactory:
        resolved_type = self.types.get(type_tag)

        if resolved_type is None:
            # lazy loading of types prevents unused imports
            resolved_type = self.resolve(type_tag)
            self.types[type_tag] = resolved_type

        return resolved_type

    @abstractmethod
    def resolve(self, type_tag: str) -> TTypeOrFactory:
        """Returns the type of object by its type tag."""
        raise NotImplementedError()


# ==> special keys
# They are intended to be non-importable, i.e. to be used only here!


# object type alias: str
#   alias should be registered to the real type in the type registry
_TYPE_KEY = '_type_'

# name or dot-separated full path to the config entry with the base config: str
#   name
#       - name of the entry in the same parent collection;
#   collection.name | path.to.the.name
#       - fully qualified path to the entry in the config
#   $another_config_filename.path.to.the.name
#       - fully qualified path to the entry in another config file;
#           filename should be w/o extension and the file should be in the same folder
_BASE_CONFIG_KEY = '_base_'


# ==================== config meta info extraction ====================
def extracted_type_tag(config: TConfig) -> tuple[TConfig, str | None]:
    """Extracts the type tagusing the type hinting convention for configs."""
    return extracted(config, _TYPE_KEY)


def extracted_base_config(config: TConfig) -> tuple[TConfig, str | None]:
    """Extracts the base config name using the meta key convention for configs."""
    return extracted(config, _BASE_CONFIG_KEY)


class ConfigResolver:
    """
    ConfigResolver is a class that resolves the config of an object or any sub-config in general.

    In most cases object constructors expect a dictionary config with named args. For dict configs
    we support the "_base_" meta key that allows to reference another dict config to take it as
    base config and override it with the current dict. Such referencing is recursive, i.e. the base
    config may also have a "_base_" key, and so on.

    We treat "_base_" as a dot-separated key path to the base config, i.e. path containing a
    sequence of keys that are used to recursively select subconfigs from the global config.
    For example "path.to.config" means following structure:
    path:
        to:
            config: <base config>

    If the key path starts with "$" it means that the first key of the path is a name (without
    extension) of the config file to load. The tail of the path is used to select the base config
    from the loaded config. For example "$another_config.path.to.config" means the same as above,
    but the base config is loaded from the "another_config.yaml" file in the same folder as the
    current global config file.

    To prevent repeated loading of the same external config, the whole loaded config is added to
    the global config under its root key (e.g. "$another_config"), so the key path resolution
    works with only "local" key paths as is.

    The `default_base_path` is used to resolve the key path when the key path is a single key.
    In this case provided key is treated as a key in the same parent collection as the last key
    of the `default_base_path` key path. For example, if the `default_base_path` is
    "path.to.obj1_config" and the key path is "obj2_config", then it is treated as:
    path:
        to:
            obj1_config:
            obj2_config: <base config>

    In several cases config is a list (e.g. if object is a list or tuple). As they don't have
    named args, we don't support referencing with "_base_" key.

    However, for both cases we support direct referencing with a single string key path.
    """
    global_config: TConfig
    global_config_path: Path

    def __init__(self, global_config: TConfig, global_config_path: Path):
        self.global_config = global_config
        self.global_config_path = global_config_path

    def resolve(
            self, config: TConfig | str | None,
            *,
            config_type: Type[dict | list],
            config_path: str = None,
    ) -> TConfig:
        """
        Resolves the config of an object or any sub-config in general.
        See the class docstring for details.
        """
        if config is None:
            return config_type()

        if isinstance(config, str):
            # it is a reference only
            reference_path = config
            config = config_type()
        else:
            # only dicts may contain reference key in their definition;
            if isinstance(config, dict):
                config, reference_path = extracted_base_config(config)
            else:
                # others' definition is either reference-only string or just a plain value
                reference_path = None

        if reference_path is not None:
            # resolve reference first, getting an unresolved base config
            base_config = self._resolve_reference(reference_path, default_base_path=config_path)

            # recursively resolve base config the same way as current
            resolved_base_config = self.resolve(
                base_config,
                config_type=config_type,
                config_path=reference_path,
            )
            if config_type == dict:
                # by the implementation we override the innermost base config with each outer one
                resolved_base_config.update(**config)
            else:
                resolved_base_config.extend(config)

            config = resolved_base_config
        return config

    def _resolve_reference(self, config_path: str, *, default_base_path: str) -> TConfig:
        """
        Resolves the reference to the base config and returns it raw.
        See the class docstring for details.
        """
        key_path: TKeyPath = config_path.split('.')

        if key_path[0].startswith('$') and key_path[0] not in self.global_config:
            self._load_external_config(key_path[0])

        elif len(key_path) == 1:
            # single key means a reference to file neighboring the `default_base_path` file
            key = key_path[0]
            # hence, replace the last key of the `default_base_path` key path
            key_path = default_base_path.split('.')
            key_path[-1] = key

        return self._select_by_key_path(key_path)

    def _load_external_config(self, reference_root_key: str):
        """
        Loads the external config and adds it to the global config.
        See the class docstring for details.
        """
        # key: $filename_without_extension
        file_stem = reference_root_key[1:]
        config_filepath = self.global_config_path.with_stem(file_stem)

        # add it to global config by the root key, such that key_path resolution just works as is
        self.global_config[reference_root_key] = read_config(config_filepath)

    def _select_by_key_path(self, key_path: TKeyPath) -> Any:
        """
        Traverses global config and selects a sub-config by the provided key path.
        See the class docstring for details.
        """
        sub_config = self.global_config
        for key_token in key_path:
            key_token = self._parse_key_token(key_token)
            sub_config = sub_config[key_token]
        return sub_config

    @staticmethod
    def _parse_key_token(key: str) -> str | int:
        # noinspection PyShadowingNames
        def boolify(s):
            if s in ['True', 'true']:
                return True
            if s in ['False', 'false']:
                return False
            raise ValueError('Not a boolean value!')

        assert isinstance(key, str)

        # NB: try/except is widely accepted pythonic way to parse things
        # NB: order of casters is important (from most specific to most general)
        for caster in (boolify, int):
            try:
                return caster(key)
            except ValueError:
                pass
        return key


class ObjectResolver:
    """
    ObjectResolver is a helper class for resolving and building objects from the configuration.
    """

    type_resolver: TTypeResolver
    config_resolver: ConfigResolver

    def __init__(
            self,
            type_resolver: TTypeResolver = None,
            config_resolver: ConfigResolver = None
    ):
        self.type_resolver = type_resolver
        self.config_resolver = config_resolver

    def resolve_requirements(
            self, config: TConfig, *,
            object_type_or_factory: TTypeOrFactory = None,
            config_type: Type[dict | list] = dict,
            **substitution_registry
    ) -> tuple[TConfig, TTypeOrFactory]:
        if not is_resolved_value(config) or config is None:
            raise ValueError(f'{config}')

        if self.config_resolver is not None:
            # we expect that referencing is enabled, so we need to resolve the config
            config = self.config_resolver.resolve(config, config_type=config_type)

        if config_type is dict:
            # substitute inducible args using substitution registry
            config = resolve_init_params(config, **substitution_registry)

        if object_type_or_factory is None:
            # have to resolve the type from the config as object type is not specified
            config, type_tag = extracted_type_tag(config)
            object_type_or_factory = self.type_resolver[type_tag]

        return config, object_type_or_factory

    def resolve(
            self, config: TConfig, *,
            object_type_or_factory: TTypeOrFactory = None,
            config_type: Type[dict | list] = dict,
            **substitution_registry
    ) -> Any:
        config, object_type_or_factory = self.resolve_requirements(
            config, object_type_or_factory=object_type_or_factory,
            config_type=config_type, **substitution_registry
        )

        try:
            if config_type is list:
                return object_type_or_factory(*config)
            return object_type_or_factory(**config)
        except TypeError:
            from pprint import pprint
            pprint(config)
            pprint(substitution_registry)
            print(f'object_type_or_factory: {object_type_or_factory} | config_type: {config_type}')
            raise


class GlobalConfig:
    config: TConfig
    config_path: Path

    config_resolver: ConfigResolver
    type_resolver: TTypeResolver
    object_resolver: ObjectResolver

    global_substitution_registry: dict

    def __init__(self, config: TConfig, config_path: Path, type_resolver: TTypeResolver):
        self.config = config
        self.config_path = config_path

        self.config_resolver = ConfigResolver(
            global_config=config, global_config_path=config_path
        )
        self.type_resolver = type_resolver
        self.object_resolver = ObjectResolver(
            type_resolver=type_resolver, config_resolver=self.config_resolver
        )
        self.global_substitution_registry = dict(
            global_config=self,
            seed=self.config['seed'],
        )

    def resolve_object(
            self, config: TConfig, *,
            object_type_or_factory: TTypeOrFactory = None,
            config_type: Type[dict | list] = dict,
            **substitution_registry
    ) -> Any:
        return self.object_resolver.resolve(
            config,
            object_type_or_factory=object_type_or_factory,
            config_type=config_type,
            **substitution_registry | self.global_substitution_registry
        )

    def resolve_object_requirements(
            self, config: TConfig, *,
            object_type_or_factory: TTypeOrFactory = None,
            config_type: Type[dict | list] = dict,
            **substitution_registry
    ) -> tuple[TConfig, TTypeOrFactory]:
        return self.object_resolver.resolve_requirements(
            config,
            object_type_or_factory=object_type_or_factory,
            config_type=config_type,
            **substitution_registry | self.global_substitution_registry
        )

