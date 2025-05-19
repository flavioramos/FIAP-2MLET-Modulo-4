"""Configuration loader module.

This module provides functionality to load parameters from a configuration file.
"""

import os
from config import PARAMS_DIR


def load_parameters():
    """Load parameters from the configuration file.
    
    Returns:
        dict: Dictionary containing the loaded parameters
    """
    params = {}
    file_path = f"{PARAMS_DIR}/params.txt"
    print(f"Loading parameters from {file_path}")
    
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                print(f"Line: {line}")
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        params[key] = int(value)
                    except ValueError:
                        try:
                            params[key] = float(value)
                        except ValueError:
                            params[key] = value
    else:
        print(f"File {file_path} not found. Using default parameters.")
    
    return params


def save_parameters(params):
    """Save parameters to the configuration file.
    
    Args:
        params (dict): Dictionary containing the parameters to save
    """
    file_path = f"{PARAMS_DIR}/params.txt"
    print(f"Saving parameters to {file_path}")
    
    with open(file_path, "w") as f:
        for key, value in sorted(params.items()):  # Sort keys for consistent ordering
            f.write(f"{key}={value}\n")
