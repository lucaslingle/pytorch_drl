"""
Config util.
"""

from typing import Optional, Dict, Tuple, Any, Iterable

import yaml


class ConfigParser(dict):
    def __init__(self, defaults: Optional[Dict[str, Any]]):
        """
        Args:
            defaults (Optional[Dict[str, Any]]): Default argument values.
                If an argument is not in a config file, the value provided here
                is used. In addition, we support nested defaults by using '.' as
                a separator. For example, config['env']['id'] defaults to be
                defaults['env.id'].
        """
        super().__init__()
        self._defaults = defaults if defaults else dict()
        self._config = defaults

    def to_dict(self):
        return self._config

    def read(self, config_path: str, verbose: bool = False) -> None:
        """
        Reads a YAML configuration file, parses it into dictionary,
        and stores the result internally.

        Args:
            config_path (str): Path to the configuration file.
            verbose (bool): Print the config dictionary obtained?

        Returns:
            None.
        """
        config = {k: v for k, v in self._defaults.items() if '.' not in k}
        nested_defaults = {k: v for k, v in self._defaults.items() if '.' in k}

        # read in a yaml file.
        with open(config_path, 'rb') as f:
            config.update(yaml.safe_load(f))

        # support for nested defaults, makes syntax easier elsewhere.
        for k in nested_defaults:
            key_sequence = k.split('.')
            subconfig = config
            for key in key_sequence[0:-1]:
                subconfig = subconfig[key]
            if not hasattr(subconfig, key_sequence[-1]):
                subconfig[key_sequence[-1]] = nested_defaults[k]

        self._config = config
        if verbose:
            for k in self._config:
                print(f"{k}: {self._config[k]}")

    def __getitem__(self, item: str) -> Any:
        """
        Gets an item.

        Args:
            item (str): Item key.

        Returns:
            Any: item value.
        """
        return self._config[item]

    def get(self, item: str) -> Any:
        """
        Gets an item.

        Args:
            item (str): Item key.

        Returns:
            Any: item value.
        """
        return self._config[item]

    def items(self) -> Iterable[Tuple[str, Any]]:
        """
        Gets the items in the config.

        Returns:
            Iterable of key-value pairs.
        """
        return self._config.items()
