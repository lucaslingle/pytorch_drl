"""
Config util.
"""

from typing import Optional, Dict, Any
from collections.abc import ItemsView

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
        self._config = self.parse_defaults(defaults if defaults else dict())

    def make_nested(self, key: str, value: Any) -> Dict[str, Any]:
        """
        Transforms a key of the form 'a.b.c ... x.y.z' and a value into a
        dictionary of the form
            {'a': {'b': {'c': ... {'x': {'y': {'z': value}}} ... }}}.

        Args:
            key (str): Key.
            value (Any): Value.

        Returns:
            Dict[str, Any]
        """
        key_prefix, _, key_suffix = key.partition('.')
        if len(key_suffix) == 0:
            return {key_prefix: value}
        return {key_prefix: self.make_nested(key_suffix, value)}

    def parse_defaults(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a defaults dictionary, possibly having some defaults specified
        by keys with the format 'a.b.c ... x.y.z', indicating nesting.

        Args:
            defaults (Dict[str, Any]): Dictionary of defaults.

        Returns:
            Dict[str, Any]: Dictionary of parsed results.
        """
        nonnested_defaults = {k: v for k, v in defaults.items() if '.' not in k}
        nested_defaults = {k: v for k, v in defaults.items() if '.' in k}
        results = nonnested_defaults
        for k in nested_defaults:
            nested = self.make_nested(k, nested_defaults[k])
            results.update(nested)
        return results

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns:
           Dict[str, Any]: Dictionary of configuration keys and values.
        """
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
        with open(config_path, 'rb') as f:
            self._config.update(yaml.safe_load(f))
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
        return self.__getitem__(item)

    def items(self) -> ItemsView[str, Any]:
        """
        Gets the items in the config.

        Returns:
            Iterable of key-value pairs.
        """
        return self._config.items()
