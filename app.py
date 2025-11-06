import torch
import cv2
import numpy as np
from pdf2image import convert_from_path
import gradio as gr

from src.crnn_model import CRNN

DEVICE = torch.device("cpu")
MODEL_PATH = "src/models/crnn_final/crnn_real_data_model.pth"
CHAR_LIST_PATH = "char_list.txt"
IMAGE_HEIGHT = 32



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
            line_images.append(binary[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2])
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


print("Loading CRNN model and character set...")


with open(CHAR_LIST_PATH, 'r', encoding='utf-8') as f:
    char_list = f.read().split('<SEP>')
int_to_char = {i + 1: char for i, char in enumerate(char_list)}

model = CRNN(num_chars=len(char_list)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully.")


def ocr_process(pdf_file, page_number):
    if pdf_file is None:
        return "Please upload a PDF file."

    page_number = int(page_number)
    print(f"Processing PDF '{pdf_file.name}', page {page_number}...")

    try:
        pil_images = convert_from_path(pdf_file.name, first_page=page_number, last_page=page_number)
        if not pil_images:
            return f"Error: Could not extract page {page_number}."
        image_data = cv2.cvtColor(np.array(pil_images[0]), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return f"PDF processing failed.\nError: {e}"

    line_crops = find_text_lines(image_data)
    if not line_crops:
        return "No text lines were detected on the page."

    full_text = []
    with torch.no_grad():
        for line_image in line_crops:
            line_tensor = preprocess_line_for_model(line_image)
            preds = model(line_tensor)
            decoded_text = decode_ctc_output(preds, int_to_char)
            full_text.append(decoded_text[0])

    print("Recognition complete.")
    return "\n".join(full_text)


iface = gr.Interface(
    fn=ocr_process,
    inputs=[gr.File(label="Upload PDF"), gr.Number(label="Page Number", value=1, precision=0)],
    outputs=gr.Textbox(label="Recognized Text", lines=15),
    title="Custom Book OCR Engine",
    description="A CRNN model built from scratch and trained on real book data to perform high-accuracy OCR.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
