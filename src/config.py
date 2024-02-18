import yaml


# Global variable to store the configuration in memory
_config = None


def load_default_config(default_path="config.yaml"):
    """ Load default configuration file to memory
    """
    global _config
    with open(default_path, 'r') as file:
        _config = yaml.safe_load(file)


def update_config(updates):
    """ Update the in-memory configuration with a dictionary of updates
    """
    global _config
    if _config is None: raise RuntimeError("Configuration not defined yet")
    for key, value in updates.items():
        if key in _config:
            _config[key] = value


def get_config(default_path="config.yaml"):
    """ Get the current in-memory configuration (or default if not existent)
    """
    if _config is None: load_default_config(default_path)
    return _config
