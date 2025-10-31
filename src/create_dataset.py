import fitz
import cv2
import numpy as np
import os
import h5py
from tqdm import tqdm
import shutil
import argparse

PDF_SOURCE_DIR = "sample_documents/books/"
OUTPUT_DATA_DIR = "data/"
HDF5_FILE_PATH = os.path.join(OUTPUT_DATA_DIR, "real_line_dataset.h5")


def find_text_lines_from_image(image_data):
    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((1, 40), np.uint8)
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_boxes, line_crops = [], []
    if not contours: return line_boxes, line_crops

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    sorted_boxes = sorted(bounding_boxes, key=lambda b: b[1])

    for x, y, w, h in sorted_boxes:
        # --- THIS IS THE STRICTEST FIX ---
        # Only accept contours with a meaningful width AND height.
        if w > 15 and h > 8:
            pad = 2
            line_crop = binary[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]

            # Final safeguard: ensure the cropped image is not empty or malformed.
            if line_crop is not None and line_crop.shape[0] > 0 and line_crop.shape[1] > 0:
                line_boxes.append({'x': x, 'y': y, 'w': w, 'h': h, 'words': []})
                line_crops.append(line_crop)

    return line_boxes, line_crops


# ... (The rest of the script is identical to the previous correct version) ...
def align_text_with_lines(page_words, line_boxes):
    for x1, y1, x2, y2, word, _, _, _ in page_words:
        word_mid_y = (y1 + y2) / 2
        for line_box in line_boxes:
            if line_box['y'] <= word_mid_y <= (line_box['y'] + line_box['h']):
                line_box['words'].append((x1, word))
                break
    line_texts = []
    for box in line_boxes:
        if box['words']:
            sorted_words = sorted(box['words'], key=lambda w: w[0])
            line_texts.append(" ".join([word for _, word in sorted_words]))
        else:
            line_texts.append("")
    return line_texts


def main():
    parser = argparse.ArgumentParser(description="Build a robust, real-world CRNN dataset from PDFs.")
    parser.add_argument("--clean", action="store_true", help="Wipe the existing dataset.")
    args = parser.parse_args()
    if args.clean and os.path.exists(HDF5_FILE_PATH):
        os.remove(HDF5_FILE_PATH)
    if not os.path.exists(OUTPUT_DATA_DIR):
        os.makedirs(OUTPUT_DATA_DIR)
    pdf_files = [f for f in sorted(os.listdir(PDF_SOURCE_DIR)) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found. Aborting.");
        return
    all_chars = set()
    for pdf_filename in pdf_files:
        pdf_path = os.path.join(PDF_SOURCE_DIR, pdf_filename)
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = "".join(c for c in page.get_text() if c.isalnum() or c.isspace())
                all_chars.update(text)
    char_list = sorted(list(all_chars))
    with h5py.File(HDF5_FILE_PATH, 'w') as hf:
        hf.create_dataset('char_list', data=[s.encode('utf-8') for s in char_list])
        labels_ds = hf.create_dataset('labels', (0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'),
                                      chunks=True)
        images_ds = hf.create_dataset('image_data', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.uint8),
                                      chunks=True)
        total_lines_saved = 0
        for pdf_filename in pdf_files:
            pdf_path = os.path.join(PDF_SOURCE_DIR, pdf_filename)
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(tqdm(doc, desc=f"Processing {pdf_filename}")):
                    page_words = page.get_text("words")
                    pix = page.get_pixmap()
                    image_data = cv2.cvtColor(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n),
                                              cv2.COLOR_RGB2BGR)
                    line_boxes, line_crops = find_text_lines_from_image(image_data)
                    line_texts = align_text_with_lines(page_words, line_boxes)
                    image_chunk, label_chunk = [], []
                    for crop, text in zip(line_crops, line_texts):
                        filtered_text = "".join(c for c in text if c in all_chars)
                        if len(filtered_text) > 2:  # Check is already here
                            _, img_encoded = cv2.imencode('.png', crop)
                            image_chunk.append(img_encoded.flatten())
                            label_chunk.append(filtered_text)
                    if image_chunk:
                        start_idx = labels_ds.shape[0]
                        new_size = start_idx + len(image_chunk)
                        labels_ds.resize(new_size, axis=0)
                        images_ds.resize(new_size, axis=0)
                        labels_ds[start_idx:] = label_chunk
                        for i, img_data in enumerate(image_chunk):
                            images_ds[start_idx + i] = img_data
                        total_lines_saved += len(image_chunk)
    print(f"\n--- Real Dataset Creation Complete! ---")
    print(f"Total lines saved: {total_lines_saved}")


if __name__ == "__main__":
    main()