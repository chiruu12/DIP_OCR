import torch
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import h5py

from src.crnn_model import CRNN
from train_crnn import decode_ctc_output

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/crnn_finetuned/crnn_book_model.pth")
DATA_FILE = os.path.join(PROJECT_ROOT, "data/line_dataset.h5")
FONT_PATH = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"

IMAGE_HEIGHT = 32
FONT_SIZE = 28

# --- Ground Truth Text from Page 12 of the Book ---
# We know this is what the model SHOULD be reading.
TEST_LINES = [
    "Praise for Applied Machine Learning and AI for Engineers",
    "This book is a fantastic guide to machine learning and AI",
    "the concrete examples with working code show how to take",
]


# --- HELPER FUNCTIONS ---

def render_perfect_line(text, font):
    """Re-creates a 'perfect' line image exactly like the training data."""
    bbox = font.getbbox(text)
    line_width = bbox[2] - bbox[0]
    image = Image.new("L", (line_width + 10, IMAGE_HEIGHT), 255)
    draw = ImageDraw.Draw(image)
    draw.text((5, (IMAGE_HEIGHT - FONT_SIZE) // 2), text, font=font, fill=0)
    return np.array(image)


def preprocess_for_model(line_image, is_from_scan=False):
    """Prepares an image for the model."""
    if is_from_scan:
        # For real scans, we must binarize and invert
        _, binary_image = cv2.threshold(line_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_to_process = binary_image
    else:
        # For our perfect synthetic data, it's already clean
        image_to_process = line_image

    h, w = image_to_process.shape
    scale_factor = IMAGE_HEIGHT / h
    new_w = int(w * scale_factor)
    resized_image = cv2.resize(image_to_process, (new_w, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)

    normalized_image = (resized_image / 255.0).astype(np.float32)
    tensor = torch.from_numpy(normalized_image).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)


def main():
    print("--- Running Final Diagnostic A/B Test ---")

    # 1. Load Model and Character Set
    print("Loading model...")
    with h5py.File(DATA_FILE, 'r') as hf:
        char_list = [c.decode('utf-8') for c in hf['char_list'][:]]
    int_to_char = {i + 1: char for i, char in enumerate(char_list)}
    model = CRNN(num_chars=len(char_list)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded.")
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # 2. Hardcode the coordinates for the test lines from a real scan
    # These are example coordinates from page 12.
    line_coords = [
        (118, 114, 1551, 44),  # "Praise for..."
        (118, 178, 1549, 36),  # "This book is..."
        (118, 298, 1551, 35)  # "the concrete examples..."
    ]

    # Load the real page image once
    # This requires `pdf2image` and `poppler` to be set up
    from pdf2image import convert_from_path
    pdf_path = os.path.join(PROJECT_ROOT, "sample_documents/books/Applied-Machine-Learning-and-AI-for-Engineers.pdf")
    pil_image = convert_from_path(pdf_path, first_page=12, last_page=12)[0]
    page_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)

    # 3. Run the A/B Test
    with torch.no_grad():
        for i, ground_truth_text in enumerate(TEST_LINES):
            print("\n" + "=" * 50)
            print(f"TESTING LINE {i + 1}: '{ground_truth_text}'")
            print("=" * 50)

            # --- Test A: The REAL Scanned Line ---
            x, y, w, h = line_coords[i]
            real_line_crop = page_image[y:y + h, x:x + w]
            real_tensor = preprocess_for_model(real_line_crop, is_from_scan=True)
            real_preds = model(real_tensor)
            real_decoded_text = decode_ctc_output(real_preds, int_to_char)[0]

            print(f"  -> Prediction from REAL SCAN: '{real_decoded_text}'")

            # --- Test B: The PERFECT Synthetic Line ---
            perfect_line_image = render_perfect_line(ground_truth_text, font)
            perfect_tensor = preprocess_for_model(perfect_line_image, is_from_scan=False)
            perfect_preds = model(perfect_tensor)
            perfect_decoded_text = decode_ctc_output(perfect_preds, int_to_char)[0]

            print(f"  -> Prediction from PERFECT RENDER: '{perfect_decoded_text}'")

            # --- Save Visual Evidence ---
            # Resize real crop to match height for easy comparison
            h_real, w_real = real_line_crop.shape
            scale = perfect_line_image.shape[0] / h_real
            resized_real_crop = cv2.resize(real_line_crop, (int(w_real * scale), perfect_line_image.shape[0]))

            # Pad the shorter image to match the width of the longer one
            width_diff = abs(resized_real_crop.shape[1] - perfect_line_image.shape[1])
            if resized_real_crop.shape[1] < perfect_line_image.shape[1]:
                resized_real_crop = cv2.copyMakeBorder(resized_real_crop, 0, 0, 0, width_diff, cv2.BORDER_CONSTANT,
                                                       value=255)
            else:
                perfect_line_image = cv2.copyMakeBorder(perfect_line_image, 0, 0, 0, width_diff, cv2.BORDER_CONSTANT,
                                                        value=255)

            comparison_image = np.vstack([resized_real_crop, perfect_line_image])
            cv2.imwrite(f"diagnostic_comparison_line_{i + 1}.png", comparison_image)
            print(f"  -> Saved visual evidence to 'diagnostic_comparison_line_{i + 1}.png'")


if __name__ == "__main__":
    main()