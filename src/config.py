import yaml
import logging
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


class CTxAILogger:
    """ This is copied from BERTopic -> https://maartengr.github.io/BERTopic/index.html
    """
    def __init__(self, level):
        self.logger = logging.getLogger('CTxAI')
        self.set_level(level)
        self._add_handler()
        self.logger.propagate = False

    def info(self, message):
        self.logger.info(f"{message}")

    def warning(self, message):
        self.logger.warning(f"WARNING: {message}")

    def set_level(self, level):
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level in levels:
            self.logger.setLevel(level)

    def _add_handler(self):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        self.logger.addHandler(sh)

        # Remove duplicate handlers
        if len(self.logger.handlers) > 1:
            self.logger.handlers = [self.logger.handlers[0]]