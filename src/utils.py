import cv2
import numpy as np
import torch
from config import settings

MODEL_IMAGE_SIZE = settings.MODEL_IMAGE_SIZE


def preprocess_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_image


def segment_characters(binary_image: np.ndarray, min_height: int = 10, min_width: int = 5):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > min_height and w > min_width:
            bounding_boxes.append((x, y, w, h))

    # --- THIS IS THE NEW, SMARTER SORTING LOGIC ---
    if not bounding_boxes:
        return []

    # Group boxes into lines based on vertical overlap
    lines = []
    sorted_boxes = sorted(bounding_boxes, key=lambda box: box[1])  # Sort by y-coordinate first

    current_line = [sorted_boxes[0]]
    for box in sorted_boxes[1:]:
        previous_box = current_line[-1]
        # If the current box's y-center is within the vertical span of the previous box, it's on the same line
        if (box[1] + box[3] / 2) < (previous_box[1] + previous_box[3]):
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]
    lines.append(current_line)  # Add the last line

    # Sort characters within each line by their x-coordinate
    final_sorted_boxes = []
    for line in lines:
        sorted_line = sorted(line, key=lambda box: box[0])
        final_sorted_boxes.extend(sorted_line)

    return final_sorted_boxes


def prepare_char_for_model(char_image: np.ndarray) -> torch.Tensor:
    h, w = char_image.shape
    if w > h:
        pad = (w - h) // 2
        padded_image = cv2.copyMakeBorder(char_image, pad, w - h - pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        pad = (h - w) // 2
        padded_image = cv2.copyMakeBorder(char_image, 0, 0, pad, h - w - pad, cv2.BORDER_CONSTANT, value=0)

    resized_image = cv2.resize(padded_image, (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))
    tensor = torch.from_numpy(resized_image).float().div(255)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor.to(settings.DEVICE)
