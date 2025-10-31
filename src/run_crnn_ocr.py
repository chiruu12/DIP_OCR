import torch
import cv2
import numpy as np
import os
import argparse
from pdf2image import convert_from_path
import h5py
from tqdm import tqdm
import shutil

# Import our custom CRNN modules
from crnn_model import CRNN


# We include the decoder function here to make the script self-contained
def decode_ctc_output(preds, int_to_char):
    texts = []
    preds_idx = preds.argmax(2).cpu().numpy()
    for pred_sequence in preds_idx:
        decoded_sequence, last_char_idx = [], 0
        for char_idx in pred_sequence:
            if char_idx != last_char_idx:
                if char_idx != 0: decoded_sequence.append(char_idx)
                last_char_idx = char_idx
        texts.append("".join([int_to_char.get(c, '') for c in decoded_sequence]))
    return texts


# --- CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = os.path.join(PROJECT_ROOT, "src/models/crnn_final/crnn_real_data_model.pth")
DATA_FILE = os.path.join(PROJECT_ROOT, "src/data/real_line_dataset.h5")
IMAGE_HEIGHT = 32
POPPLER_PATH = None


def find_text_lines(image_data):
    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((1, 40), np.uint8)
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_images = []
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours = [c for _, c in sorted(zip(bounding_boxes, contours), key=lambda b: b[0][1])]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 15 and h > 8:
            pad = 2
            line_crop = binary[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
            line_images.append(line_crop)
    return line_images


def preprocess_line_for_model(line_image):
    inverted_image = cv2.bitwise_not(line_image)
    h, w = inverted_image.shape
    scale_factor = IMAGE_HEIGHT / h
    new_w = int(w * scale_factor)
    resized_image = cv2.resize(inverted_image, (new_w, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    normalized_image = (resized_image / 255.0).astype(np.float32)
    tensor = torch.from_numpy(normalized_image).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)


def main():
    parser = argparse.ArgumentParser(description="Run the final trained CRNN model on a full PDF page.")
    parser.add_argument("file_path", type=str, help="Path to PDF, relative to project root.")
    parser.add_argument("--page", type=int, default=12, help="Page number to process.")
    args = parser.parse_args()

    abs_file_path = os.path.join(PROJECT_ROOT, args.file_path)

    # --- THIS IS THE CRITICAL FIX ---
    # 1. Load the OFFICIAL character list FROM THE HDF5 FILE that the model was trained on.
    print("Loading trained CRNN model and OFFICIAL character set from HDF5 file...")
    try:
        with h5py.File(DATA_FILE, 'r') as hf:
            char_list = [c.decode('utf-8') for c in hf['char_list'][:]]
        int_to_char = {i + 1: char for i, char in enumerate(char_list)}
        print(f"Character map loaded successfully with {len(char_list)} characters.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Dataset file not found at '{DATA_FILE}'. Cannot determine character map.")
        return

    # 2. Initialize the model with the CORRECT number of characters from the dataset.
    model = CRNN(num_chars=len(char_list)).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found at '{MODEL_PATH}'. Please train the model first.")
        return
    except RuntimeError as e:
        print(
            f"FATAL ERROR: Model and saved weights have a size mismatch. This indicates the dataset has changed since training.")
        print(f"Error details: {e}")
        return

    model.eval()
    print("Model loaded successfully.")

    try:
        pil_image = \
        convert_from_path(abs_file_path, first_page=args.page, last_page=args.page, poppler_path=POPPLER_PATH)[0]
        image_data = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"PDF conversion failed: {e}");
        return

    line_crops = find_text_lines(image_data)
    print(f"Detected {len(line_crops)} lines of text. Recognizing...")

    full_text = []
    with torch.no_grad():
        for line_image in tqdm(line_crops, desc="Recognizing lines"):
            line_tensor = preprocess_line_for_model(line_image)
            preds = model(line_tensor)
            decoded_text = decode_ctc_output(preds, int_to_char)
            full_text.append(decoded_text[0])

    print("\n" + "=" * 50)
    print(f"        FINAL RECOGNIZED TEXT - PAGE {args.page}")
    print("=" * 50)
    print("\n".join(full_text))
    print("=" * 50)


if __name__ == "__main__":
    main()