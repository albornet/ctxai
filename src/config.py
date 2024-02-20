import yaml
import threading

configuration = None  # global configuration dictionary
config_lock = threading.Lock()  # lock for thread-safe configuration updates


def load_default_config(default_path: str = "config.yaml"):
    """ Load default configuration file to memory
    """
    global configuration
    with open(default_path, "r") as file:
        with config_lock:
            configuration = yaml.safe_load(file)


def update_config(request_data: dict):
    """ Update the in-memory configuration with a dictionary of updates
    """
    global configuration
    with config_lock:
        for key, value in request_data.items():
            if isinstance(configuration.get(key), dict) and isinstance(value, dict):
                configuration[key].update(value)
            else:
                configuration[key] = value


def get_config(default_path: str = "config.yaml"):
    """ Get the current in-memory configuration (or default one if non-existent)
    """
    global configuration
    if configuration is None:
        load_default_config(default_path)
    return configuration
