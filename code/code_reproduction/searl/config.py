"""Minimal stand-in for searl.utils.handler.config.AttributeDict/ConfigHandler
from the original SEARL repo, trimmed of the file-saving machinery (YAML I/O,
experiment-folder bookkeeping) that this standalone reproduction does not need.
The attribute-style nested access (`cfg.rl.lr_actor`), `get_dict`, and
`set_attr` are kept identical so the copied algorithm files (mutation.py,
tournament_selection.py, td3_training.py) work unmodified.
"""


class AttributeDict:
    def __init__(self, dictionary, name="main"):
        self._name = name
        for key in dictionary:
            if isinstance(dictionary[key], dict):
                setattr(self, key, AttributeDict(dictionary[key], key))
            else:
                setattr(self, key, dictionary[key])

    def __repr__(self):
        return str({k: v for k, v in self.__dict__.items() if not k.startswith("_")})

    @property
    def get_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def set_attr(self, name, value):
        self.__setattr__(name, value)


def make_config(d):
    return AttributeDict(d)
