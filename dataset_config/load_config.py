import importlib

def load_config(config_name):
    """Dynamically import a configuration file as a module."""
    config_module = importlib.import_module(config_name)
    return {
        "base_path": config_module.base_path,
        "n_classes": config_module.n_classes,
        "n_channels": config_module.n_channels,
        "classes": config_module.classes,
        "num_tiles": config_module.num_tiles,
        "weights": config_module.weights
    }
