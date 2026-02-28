from pathlib import Path

ROOT = Path("../")

DATA_PATH = ROOT / "data" / "SMSSpamCollection"
MODEL_DIR = ROOT / "models"

SEPARATOR = "\t"

COLUMN_NAMES = ["label", "message"]

TEXT_COLUMN = "message"
TARGET_COLUMN = "label"

TEST_SIZE = 0.2
RANDOM_STATE = 42
SHUFFLE = True

MAX_ITER = 1000