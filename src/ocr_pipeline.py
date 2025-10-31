import torch
import numpy as np
from typing import Dict, Tuple
import cv2
import utils
import model_loader
from config import settings


def predict_character(char_tensor: torch.Tensor, models: Dict[str, torch.nn.Module]) -> str:
    """Predicts a single character using the Triage and Expert system."""
    triage_model = models['triage']

    with torch.no_grad():
        triage_output = triage_model(char_tensor)
        _, predicted_class_idx = torch.max(triage_output, 1)
        expert_name = settings.TRIAGE_OUTPUT_MAP[predicted_class_idx.item()]

        expert_model = models[expert_name]
        expert_output = expert_model(char_tensor)
        _, predicted_char_idx = torch.max(expert_output, 1)

        offset = settings.EXPERT_LABEL_OFFSETS[expert_name]
        global_idx = predicted_char_idx.item() + offset

        return chr(global_idx)


def process_image_data(image_data: np.ndarray, models: Dict[str, torch.nn.Module]) -> str:
    """Performs end-to-end OCR, now with intelligent word spacing."""
    gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    bounding_boxes = utils.segment_characters(binary_image)
    if not bounding_boxes:
        return ""

    print(f"Found {len(bounding_boxes)} characters to recognize.")

    recognized_elements = []
    previous_box = bounding_boxes[0]

    for i, box in enumerate(bounding_boxes):
        if i > 0:
            previous_x, previous_y, previous_w, previous_h = previous_box
            current_x, current_y, _, _ = box

            if current_y > (previous_y + previous_h * settings.NEWLINE_THRESHOLD_FACTOR):
                recognized_elements.append('\n')
            elif current_x > (previous_x + previous_w + (previous_w * settings.SPACE_THRESHOLD_FACTOR)):
                recognized_elements.append(' ')

        x, y, w, h = box
        char_crop = binary_image[y:y + h, x:x + w]
        char_tensor = utils.prepare_char_for_model(char_crop)
        predicted_char = predict_character(char_tensor, models)
        recognized_elements.append(predicted_char)

        previous_box = box

    return "".join(recognized_elements)
