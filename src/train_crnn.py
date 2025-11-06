import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import os
import h5py
from tqdm import tqdm
import numpy as np
import cv2
from pdf2image import convert_from_path
from crnn_model import CRNN
from torchvision import transforms

DATA_FILE = "data/real_line_dataset.h5"
MODEL_OUTPUT_DIR = "models/crnn_final/"
MODEL_SAVE_NAME = "crnn_real_data_model.pth"
VALIDATION_PDF = "sample_documents/books/Applied-Machine-Learning-and-AI-for-Engineers.pdf"
VALIDATION_PAGE = 2
EPOCHS, BATCH_SIZE, LEARNING_RATE = 50, 16, 0.0001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMAGE_HEIGHT = 32
POPPLER_PATH = None


class RealLineDataset(Dataset):
    def __init__(self, h5_path, char_list, transform=None):
        self.h5_path, self.transform = h5_path, transform
        self.char_to_int = {char: i + 1 for i, char in enumerate(char_list)}
        self.int_to_char = {i + 1: char for i, char in enumerate(char_list)}
        with h5py.File(self.h5_path, 'r') as hf:
            self.num_samples = len(hf['labels'])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        try:
            with h5py.File(self.h5_path, 'r') as hf:
                img_encoded = hf['image_data'][idx]
                label_str = hf['labels'][idx].decode('utf-8')

            image = cv2.imdecode(np.frombuffer(img_encoded, np.uint8), cv2.IMREAD_GRAYSCALE)

            if image is None or image.shape[1] == 0 or image.shape[0] == 0:
                raise ValueError(f"Image at index {idx} is corrupted or has a zero dimension.")

            image = cv2.bitwise_not(image)

            h, w = image.shape
            scale_factor = IMAGE_HEIGHT / h
            new_w = int(w * scale_factor)

            if new_w <= 0:
                raise ValueError(f"Calculated new width is non-positive for image at index {idx}.")

            resized_image = cv2.resize(image, (new_w, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)

            label = torch.tensor([self.char_to_int.get(char, 0) for char in label_str])

            if self.transform:
                resized_image = self.transform(resized_image)

            return resized_image, label
        except Exception as e:
            print(f"\nWARNING: Corrupted data at index {idx}. Skipping. Error: {e}\n")
            return self.__getitem__(0)


def collate_fn(batch):
    images, labels = zip(*batch)
    image_widths = [img.shape[2] for img in images]
    max_width = max(image_widths)
    padded_images = []
    for img in images:
        padding = (0, max_width - img.shape[2], 0, 0)
        padded_images.append(torch.nn.functional.pad(img, padding, "constant", 0))
    images_tensor = torch.stack(padded_images, 0)
    labels_concat = torch.cat(labels, 0)
    label_lengths = torch.tensor([len(lab) for lab in labels])
    return images_tensor, labels_concat, label_lengths


def decode_ctc_output(preds, int_to_char):
    texts = []
    preds_idx = preds.argmax(2).cpu().numpy()
    for seq in preds_idx:
        decoded, last = [], 0
        for char_idx in seq:
            if char_idx != last:
                if char_idx != 0: decoded.append(char_idx)
                last = char_idx
        texts.append("".join([int_to_char.get(c, '') for c in decoded]))
    return texts


def validate_on_real_page(model, int_to_char, pdf_path, page_num):
    print("\n--- Running Real-World Validation ---")
    try:
        pil_image = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, poppler_path=POPPLER_PATH)[0]
        image_data = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Validation failed: Could not convert PDF page. Error: {e}");
        return
    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((1, 40), np.uint8)
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_crops = []
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours = [c for _, c in sorted(zip(bounding_boxes, contours), key=lambda b: b[0][1])]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 15 and h > 8 and h > 0:
            line_crops.append(binary[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2])
    full_text = []
    model.eval()
    with torch.no_grad():
        for line_image in line_crops:
            inverted_image = cv2.bitwise_not(line_image)
            h, w = inverted_image.shape
            scale = IMAGE_HEIGHT / h
            resized = cv2.resize(inverted_image, (int(w * scale), IMAGE_HEIGHT))
            tensor = transforms.ToTensor()(resized).unsqueeze(0).to(DEVICE)
            preds = model(tensor)
            decoded_text = decode_ctc_output(preds, int_to_char)
            full_text.append(decoded_text[0])
    print("--- Validation Page OCR Result ---")
    print("\n".join(full_text))
    print("----------------------------------\n")


def main():
    if not os.path.exists(MODEL_OUTPUT_DIR): os.makedirs(MODEL_OUTPUT_DIR)
    with h5py.File(DATA_FILE, 'r') as hf:
        char_list = [c.decode('utf-8') for c in hf['char_list'][:]]
    dataset = RealLineDataset(DATA_FILE, char_list, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, num_workers=2)
    model = CRNN(num_chars=len(char_list)).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    best_train_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        for images, labels, label_lengths in tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"):
            images, batch_size = images.to(DEVICE), images.size(0)
            preds_raw = model(images)
            preds_for_loss = preds_raw.permute(1, 0, 2)
            cnn_output_width = preds_for_loss.size(0)
            input_lengths = torch.full(size=(batch_size,), fill_value=cnn_output_width, dtype=torch.long)
            optimizer.zero_grad()
            loss = criterion(preds_for_loss.log_softmax(2).cpu(), labels.cpu(), input_lengths.cpu(),
                             label_lengths.cpu())
            if not (torch.isinf(loss) or torch.isnan(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(loader)
        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        validate_on_real_page(model, dataset.int_to_char, VALIDATION_PDF, VALIDATION_PAGE)
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), os.path.join(MODEL_OUTPUT_DIR, MODEL_SAVE_NAME))
            print(f"Train loss improved. Saved model to '{MODEL_SAVE_NAME}'")


if __name__ == "__main__":
    main()