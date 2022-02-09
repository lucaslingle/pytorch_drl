import tempfile
import os

import pytest

from drl.utils.configuration import ConfigParser

CONFIG_DIR = os.path.join(tempfile.gettempdir(), 'pytorch_drl_testing')
CONFIG_STR = """
foo:
    a: 1
    b: 2.0
    c: '3'
bar:
   d: '4.0'
   e: '5.0.0'
baz:
   f: 6
   g: 7
""".strip("\n")


def make_config_path():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    config_path = os.path.join(CONFIG_DIR, 'config.yaml')
    return config_path


def save_config(path):
    with open(path, 'w') as f:
        f.write(CONFIG_STR)


def make_config():
    return ConfigParser(defaults={'foo': 123, 'baz.spam': 42, 'baz.f': -1})


def test_config_make_nested():
    path = make_config_path()
    save_config(path)
    config = make_config()
    assert config.make_nested('a.b.c', 777) == {'a': {'b': {'c': 777}}}


def test_parse_defaults():
    path = make_config_path()
    save_config(path)
    config = make_config()

    # to just test the parse defaults function, here we use different defaults
    # than those actually provided to config in make_config.
    results = config.parse_defaults({'xyz': 1, 'abc.def': 2})
    assert results == {'xyz': 1, 'abc': {'def': 2}}


def test_config_read():
    path = make_config_path()
    save_config(path)
    config = make_config()

    config.read(path)
    tree = config.to_dict()
    assert tree == {
        'foo': {
            'a': 1,
            'b': 2.0,
            'c': '3',
        },
        'bar': {
            'd': '4.0',
            'e': '5.0.0',
        },
        'baz': {
            'f': 6,  # overwrote baz.f = -1
            'g': 7,  # provided but had no default value
            'spam': 42,  # default value unchanged
        }
    }


def test_config_get():
    path = make_config_path()
    save_config(path)
    config = make_config()
    assert config.get('foo') == 123
    assert config.get('baz').get('f') == -1
    with pytest.raises(KeyError):
        baz_g = config['baz']['g']

    config.read(path)
    assert config.get('foo') == {'a': 1, 'b': 2.0, 'c': '3'}
    assert config.get('bar') == {'d': '4.0', 'e': '5.0.0'}
    assert config.get('baz') == {'f': 6, 'g': 7, 'spam': 42}
