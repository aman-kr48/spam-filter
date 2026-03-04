from pathlib import Path
import typing as T
import omegaconf as oc

ROOT = Path(__file__).resolve().parents[2]


MODEL_DIR = ROOT / "models"

Config = oc.ListConfig | oc.DictConfig


def parse_file(path: str) -> Config:
    """Parse a config file from a path."""
    return oc.OmegaConf.load(path)

def merge_configs(configs: T.Sequence[Config]) -> Config:
    """Merge a list of config into a single config."""
    return oc.OmegaConf.merge(*configs)

