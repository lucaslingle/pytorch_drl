import tempfile
import os

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
""".strip("\n")


def make_config_path():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    config_path = os.path.join(CONFIG_DIR, 'config.yaml')
    return config_path


def make_config(path):
    with open(path, 'w') as f:
        f.write(CONFIG_STR)


def test_config_read():
    path = make_config_path()
    make_config(path)
    config = ConfigParser(defaults={})
    config.read(path)
    tree = config.to_dict()
    assert tree == {
        'foo': {
            'a': 1, 'b': 2.0, 'c': '3'
        }, 'bar': {
            'd': '4.0', 'e': '5.0.0'
        }
    }


def test_config_get():
    path = make_config_path()
    make_config(path)
    config = ConfigParser(defaults={'foo': 123})
    assert config.get('foo') == 123
    config.read(path)
    assert config.get('foo') == {'a': 1, 'b': 2.0, 'c': '3'}
    assert config.get('bar') == {'d': '4.0', 'e': '5.0.0'}
