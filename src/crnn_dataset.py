# In src/crnn_dataset.py
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import cv2


class CRNNDataset(Dataset):
    def __init__(self, h5_path, char_list):
        self.h5_path = h5_path
        self.char_list = char_list
        self.char_to_int = {char: i + 1 for i, char in enumerate(char_list)}
        self.int_to_char = {i + 1: char for i, char in enumerate(char_list)}

        with h5py.File(self.h5_path, 'r') as hf:
            self.num_samples = len(hf['labels'])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as hf:
            img_encoded = hf['image_data'][idx]
            image = cv2.imdecode(np.frombuffer(img_encoded, np.uint8), cv2.IMREAD_GRAYSCALE)
            label_str = hf['labels'][idx].decode('utf-8')

        image = (image / 255.0).astype(np.float32)
        label_int = [self.char_to_int[char] for char in label_str]

        return torch.from_numpy(image).unsqueeze(0), torch.tensor(label_int)


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

    # Return the widths as part of the batch
    return images_tensor, labels_concat, label_lengths, torch.tensor(image_widths)
