"""Model versioning utility module.

This module provides functionality to manage model versions and handle versioned model files.
"""

import os
from config import MODEL_LOCAL_PATH
from utils.config_loader import load_parameters, save_parameters


class ModelVersioning:
    """Class to handle model versioning operations."""

    @staticmethod
    def get_version():
        """Get the current model version.
        
        Returns:
            int: Current model version, or 0 if not found
        """
        params = load_parameters()
        return params.get("MODEL_VERSION", 0)

    @staticmethod
    def increment_version():
        """Increment the model version number.
        
        Returns:
            int: New model version number
        """
        params = load_parameters()
        current_version = params.get("MODEL_VERSION", 0)
        new_version = current_version + 1
        params["MODEL_VERSION"] = new_version
        save_parameters(params)
        return new_version

    @staticmethod
    def get_versioned_path():
        """Get the path for the versioned model file.
        
        Returns:
            str: Path to the versioned model file
        """
        version = ModelVersioning.get_version()
        base_path = os.path.splitext(MODEL_LOCAL_PATH)[0]
        return f"{base_path}_v{version}.joblib"

    @staticmethod
    def update_latest_symlink(versioned_path):
        """Update the symlink to point to the latest model version.
        
        Args:
            versioned_path (str): Path to the versioned model file
        """
        if os.path.exists(MODEL_LOCAL_PATH):
            os.remove(MODEL_LOCAL_PATH)
        os.symlink(versioned_path, MODEL_LOCAL_PATH) 