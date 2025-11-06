import torch
import numpy as np
import cv2
import os
import argparse
from pdf2image import convert_from_path

from config import settings
import utils
from model_loader import load_all_models


def predict_character(char_tensor, models):
    """Predicts a single character using the Triage and Expert system with the CORRECTED mapping."""
    with torch.no_grad():
        triage_output = models['triage'](char_tensor)
        _, triage_idx = torch.max(triage_output, 1)
        triage_decision = settings.TRIAGE_OUTPUT_MAP[triage_idx.item()]

        expert_model = models[triage_decision]
        expert_output = expert_model(char_tensor)
        _, expert_idx = torch.max(expert_output, 1)

        character_map = settings.EXPERT_CHARACTER_MAPS[triage_decision]
        final_prediction = character_map.get(expert_idx.item(), '?')

    return final_prediction


def run_ocr_pipeline(image_data, models):
    """Runs the full OCR pipeline with smarter sorting and word-gap detection."""
    gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    bounding_boxes = utils.segment_characters(binary_image)
    if not bounding_boxes:
        return ""

    print(f"Found {len(bounding_boxes)} characters to recognize.")

    recognized_elements = []
    previous_box = bounding_boxes[0]

    for box in bounding_boxes:
        prev_x, prev_y, prev_w, prev_h = previous_box
        curr_x, curr_y, _, _ = box

        if curr_y > (prev_y + prev_h * settings.NEWLINE_THRESHOLD_FACTOR):
            recognized_elements.append('\n')
        elif curr_x > (prev_x + prev_w + (prev_w * settings.SPACE_THRESHOLD_FACTOR)):
            recognized_elements.append(' ')

        x, y, w, h = box
        char_crop = binary_image[y:y + h, x:x + w]
        char_tensor = utils.prepare_char_for_model(char_crop)
        predicted_char = predict_character(char_tensor, models)
        recognized_elements.append(predicted_char)

        previous_box = box

    return "".join(recognized_elements)


def main():
    parser = argparse.ArgumentParser(description="Run the final, corrected OCR on an image or PDF.")
    parser.add_argument("file_path", type=str, help="The path to the input image or PDF file.")
    parser.add_argument("--page", type=int, default=12, help="Page number to process for a PDF.")
    args = parser.parse_args()

    try:
        models = load_all_models()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not os.path.exists(args.file_path):
        print(f"Error: Input file not found at '{args.file_path}'")
        return

    try:
        if args.file_path.lower().endswith('.pdf'):
            print(f"Processing PDF file, page {args.page}...")
            poppler_path = os.path.join(settings.POPPLER_PATH, "bin") if settings.POPPLER_PATH else None
            pil_image = \
            convert_from_path(args.file_path, first_page=args.page, last_page=args.page, poppler_path=poppler_path)[0]
            image_data = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            image_data = cv2.imread(args.file_path)

        final_text = run_ocr_pipeline(image_data, models)

        print("\n" + "=" * 50)
        print("           FINAL RECOGNIZED TEXT")
        print("=" * 50)
        print(final_text)
        print("=" * 50)

    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == '__main__':
    main()