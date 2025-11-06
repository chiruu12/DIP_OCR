import torch
import cv2
import numpy as np
import os
import shutil
from pdf2image import convert_from_path
from tqdm import tqdm

from src import utils
from config import settings
from src.model_loader import load_all_models

PDF_PATH = "sample_documents/books/Applied-Machine-Learning-and-AI-for-Engineers.pdf"
PAGE_TO_DEBUG = 2
DEBUG_OUTPUT_DIR = "debug_output/"


def main():
    print("--- Starting Pipeline Debugging Session ---")

    if os.path.exists(DEBUG_OUTPUT_DIR): shutil.rmtree(DEBUG_OUTPUT_DIR)
    os.makedirs(DEBUG_OUTPUT_DIR)
    print(f"Debug artifacts will be saved in: '{DEBUG_OUTPUT_DIR}'")

    try:
        models = load_all_models()
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        return

    try:
        print(f"\nConverting page {PAGE_TO_DEBUG} of the PDF to an image...")
        poppler_path = os.path.join(settings.POPPLER_PATH, "bin") if settings.POPPLER_PATH else None
        pil_image = \
        convert_from_path(PDF_PATH, first_page=PAGE_TO_DEBUG, last_page=PAGE_TO_DEBUG, poppler_path=poppler_path)[0]
        image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"PDF conversion failed: {e}")
        return

    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bounding_boxes = utils.segment_characters(binary_image)

    segmentation_viz_image = image_bgr.copy()
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(segmentation_viz_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    viz_path = os.path.join(DEBUG_OUTPUT_DIR, "_SEGMENTATION_RESULT.png")
    cv2.imwrite(viz_path, segmentation_viz_image)
    print(f"\nSAVED VISUAL EVIDENCE: Segmentation result saved to '{viz_path}'")

    print("\n--- Character-by-Character Recognition Log ---")

    for i, box in enumerate(tqdm(bounding_boxes, desc="Debugging Pipeline")):
        x, y, w, h = box

        char_crop = binary_image[y:y + h, x:x + w]
        crop_path = os.path.join(DEBUG_OUTPUT_DIR, f"contour_{i:04d}_input.png")
        cv2.imwrite(crop_path, char_crop)

        char_tensor = utils.prepare_char_for_model(char_crop)

        with torch.no_grad():
            triage_output = models['triage'](char_tensor)
            _, triage_idx = torch.max(triage_output, 1)
            triage_decision = settings.TRIAGE_OUTPUT_MAP[triage_idx.item()]

            expert_model = models[triage_decision]
            expert_output = expert_model(char_tensor)
            _, expert_idx = torch.max(expert_output, 1)

            character_map = settings.EXPERT_CHARACTER_MAPS[triage_decision]
            final_prediction = character_map.get(expert_idx.item(), '?')

        print(f"Contour #{i:03d} | Triage Decision: {triage_decision:<10} | Final Prediction: '{final_prediction}'")

    print("\n--- Debugging Session Complete ---")


if __name__ == "__main__":
    main()