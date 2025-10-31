import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = os.path.join(BASE_DIR, "models", "saved_weights_finetuned")


TRIAGE_CONFIG = {
    "architecture": "large",
    "path": os.path.join(MODELS_DIR, "triage_model.pth"),
    "num_classes": 3
}

EXPERT_CONFIG = {
    "digits": {
        "architecture": "small",
        "path": os.path.join(MODELS_DIR, "digits_model.pth"),
        "num_classes": 10
    },
    "uppercase": {
        "architecture": "medium",
        "path": os.path.join(MODELS_DIR, "uppercase_model.pth"),
        "num_classes": 26
    },
    "lowercase": {
        "architecture": "medium",
        "path": os.path.join(MODELS_DIR, "lowercase_model.pth"),
        "num_classes": 26
    }
}

TRIAGE_OUTPUT_MAP = {
    0: 'digits',
    1: 'uppercase',
    2: 'lowercase'
}

EXPERT_LABEL_OFFSETS = {
    "digits": 0,
    "uppercase": 10,
    "lowercase": 36
}

EXPERT_CHARACTER_MAPS = {
    'digits': {i: str(i) for i in range(10)},
    'uppercase': {i: chr(ord('A') + i) for i in range(26)},
    'lowercase': {i: chr(ord('a') + i) for i in range(26)}
}

EMNIST_MAPPING = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
    46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't',
    56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'
}

SPACE_THRESHOLD_FACTOR = 0.4
NEWLINE_THRESHOLD_FACTOR = 0.7

MODEL_IMAGE_SIZE = 28

POPPLER_PATH = "/opt/homebrew/opt/poppler"

