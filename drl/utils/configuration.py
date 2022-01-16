"""
Config util.
"""

from typing import Optional, Dict, Tuple, Any

import yaml


class ConfigParser(dict):
    def __init__(self, defaults: Optional[Dict[str, Any]]) -> None:
        super().__init__()
        self._defaults = defaults if defaults else dict()
        self._config = None

    def read(self, config_path: str, verbose: bool = False) -> None:
        config = {k: v for k,v in self._defaults.items() if '.' not in k}
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
        return self._config[item]

    def get(self, item: str) -> Any:
        return self._config[item]

    def items(self) -> Tuple[str, Any]:
        return self._config.items()