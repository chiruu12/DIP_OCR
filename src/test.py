from pdf2image import convert_from_path
import os
import numpy as np
import cv2
from utils import segment_characters, prepare_char_for_model, MODEL_IMAGE_SIZE
from config import settings

def test_image_processing_utils():
    print("Testing image processing utilities with a PDF...")


    POPPLER_PATH = settings.POPPLER_PATH
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
    pdf_name = "Applied-Machine-Learning-and-AI-for-Engineers.pdf"
    sample_pdf_path = os.path.join(project_root, "sample_documents", pdf_name)

    try:
        print(f"Reading first page from '{sample_pdf_path}'...")
        page_image_pil = convert_from_path(
            sample_pdf_path,
            first_page=1,
            last_page=2,
            poppler_path=os.path.join(POPPLER_PATH, "bin")
        )[1]

        page_image_bgr = cv2.cvtColor(np.array(page_image_pil), cv2.COLOR_RGB2BGR)
        print("Successfully converted PDF page to image.")

        gray_image = cv2.cvtColor(page_image_bgr, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        print(f"Successfully preprocessed image. Shape: {binary_img.shape}")

        boxes = segment_characters(binary_img)
        print(f"Found {len(boxes)} potential character bounding boxes.")

        for x, y, w, h in boxes:
            cv2.rectangle(page_image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        output_path = os.path.join(project_root, "sample_documents", "pdf_segmentation_result.png")
        cv2.imwrite(output_path, page_image_bgr)
        print(f"Segmentation visualization saved to: {output_path}")

        if boxes:
            x, y, w, h = boxes[0]
            first_char_crop = binary_img[y:y + h, x:x + w]
            char_tensor = prepare_char_for_model(first_char_crop)
            print(f"Prepared first character for model. Tensor shape: {char_tensor.shape}")
            assert char_tensor.shape == (1, 1, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
            print("Tensor shape is correct.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"\nPlease ensure the PDF file exists at the absolute path: '{sample_pdf_path}'")
        print("Also check that your POPPLER_PATH is correct.")



if __name__ == "__main__":
    test_image_processing_utils()