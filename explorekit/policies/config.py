import yaml
from pathlib import Path

def load_policies_config(path="configs/policies.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)