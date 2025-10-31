import torch
from models import CNNModel_Small, CNNModel_Medium, CNNModel_Large
from config import settings
import os

ARCHITECTURE_MAP = {
    "small": CNNModel_Small,
    "medium": CNNModel_Medium,
    "large": CNNModel_Large
}


def _load_model(config: dict, model_type: str, model_name: str):
    """Generic helper function to load a finetuned model based on its config."""
    model_class = ARCHITECTURE_MAP[config["architecture"]]

    model_path = os.path.join(settings.MODELS_DIR, f"{model_name}_model_finetuned.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fine-tuned model not found at '{model_path}'. Please run the training script.")

    num_classes = config["num_classes"]
    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=settings.DEVICE))
    model.to(settings.DEVICE)
    model.eval()

    print(f"Successfully loaded fine-tuned {model_type} model: '{model_name}' ({config['architecture']})")
    return model


def load_all_models() -> dict:
    """
    Loads the finetuned triage model and all three expert models.
    """
    print("Loading all fine-tuned models for the OCR pipeline...")

    models = {"triage": _load_model(settings.TRIAGE_CONFIG, "triage", "triage")}

    for name, config in settings.EXPERT_CONFIG.items():
        models[name] = _load_model(config, f"expert", name)

    print("All fine-tuned models loaded and ready.")
    return models

