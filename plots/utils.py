from typing import Dict, List
import json
import os

def read_metric_file(file: str) -> Dict[str, List[float]]:
    with open(file, "r") as f:
        obj = json.load(f)
        return obj

def get_file_name(file: str):
    return file.split("/")[-1].split(".")[0]

def get_folder_name(folder_path: str):
    return os.path.basename(os.path.normpath(folder_path))