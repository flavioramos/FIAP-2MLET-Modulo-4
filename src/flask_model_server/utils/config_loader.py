import os
from config import PARAMS_DIR

def load_parameters():
    params = {}
    file_path = f"{PARAMS_DIR}/params.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
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
        print(f"Arquivo {file_path} não encontrado. Usando parâmetros padrão.")
    return params
