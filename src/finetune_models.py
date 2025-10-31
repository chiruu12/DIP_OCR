import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
import os
import numpy as np
from tqdm import tqdm
import h5py

# Import the ORIGINAL 28x28 architectures for loading Kaggle weights
from models import CNNModel_Small, CNNModel_Medium, CNNModel_Large

# --- CONFIGURATION ---
DATA_FILE = "data/book_dataset.h5"
KAGGLE_WEIGHTS_DIR = "models/saved_weights/"
OUTPUT_DIR = "models/saved_weights_finetuned/"

BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.0001  # Low learning rate for fine-tuning
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# --- DATASET AND MODEL CLASSES ---
class HDF5Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        with h5py.File(self.h5_path, 'r') as hf:
            self.labels = hf['labels'][:]
            self.images = hf['images'][:]  # Load into memory for faster access

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        if self.transform: image = self.transform(image)
        return image, label


class LabelRemapper:
    def __init__(self, remap_dict): self.remap_dict = remap_dict

    def __call__(self, label): return self.remap_dict.get(label, -1)


# --- TRAINING FUNCTION ---
def finetune_model(model_name, model, loader):
    print(f"\n{'=' * 60}\nFine-tuning: {model_name.upper()} on {len(loader.dataset)} balanced samples\n{'=' * 60}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} Avg Loss: {total_loss / len(loader):.6f}")

    save_path = os.path.join(OUTPUT_DIR, f"{model_name}_model_finetuned.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved fine-tuned model to {save_path}")


# --- MAIN EXECUTION ---
def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    print(f"Using device: {DEVICE}")

    print("\nLoading and analyzing dataset for balancing...")
    full_dataset = HDF5Dataset(DATA_FILE, transform=transforms.ToTensor())
    all_labels = np.array(full_dataset.labels)

    # --- 1. TRIAGE MODEL FINE-TUNING (Balanced by Undersampling) ---
    triage_model_path = os.path.join(OUTPUT_DIR, "triage_model_finetuned.pth")
    if os.path.exists(triage_model_path):
        print("Skipping Triage Model: Fine-tuned version already exists.")
    else:
        digit_indices = np.where((all_labels >= 48) & (all_labels <= 57))[0]
        upper_indices = np.where((all_labels >= 65) & (all_labels <= 90))[0]
        lower_indices = np.where((all_labels >= 97) & (all_labels <= 122))[0]

        min_class_size = min(len(digit_indices), len(upper_indices), len(lower_indices))
        print(f"Balancing Triage data: Using {min_class_size} samples per class.")

        balanced_indices = np.concatenate([
            np.random.choice(digit_indices, min_class_size, replace=False),
            np.random.choice(upper_indices, min_class_size, replace=False),
            np.random.choice(lower_indices, min_class_size, replace=False)
        ])

        triage_remap = {code: (0 if 48 <= code <= 57 else 1 if 65 <= code <= 90 else 2) for code in
                        np.unique(all_labels)}
        triage_target_transform = LabelRemapper(triage_remap)

        triage_subset = Subset(full_dataset, balanced_indices)
        # Apply target transform to the subset
        triage_dataset = [(img, triage_target_transform(label)) for img, label in
                          tqdm(triage_subset, desc="Remapping Triage labels")]

        triage_loader = DataLoader(triage_dataset, batch_size=BATCH_SIZE, shuffle=True)
        triage_model = CNNModel_Large(num_classes=3).to(DEVICE)

        kaggle_triage_path = os.path.join(KAGGLE_WEIGHTS_DIR, "triage_large_model.pth")
        if os.path.exists(kaggle_triage_path):
            print(f"Loading base weights from {kaggle_triage_path}")
            triage_model.load_state_dict(torch.load(kaggle_triage_path, map_location=DEVICE))

        finetune_model('triage', triage_model, triage_loader)

    # --- 2. EXPERT MODELS FINE-TUNING (Balanced by Weighted Sampling) ---
    expert_configs = {
        'digits': (CNNModel_Small, [chr(i) for i in range(48, 58)]),
        'uppercase': (CNNModel_Medium, [chr(i) for i in range(65, 91)]),
        'lowercase': (CNNModel_Medium, [chr(i) for i in range(97, 123)])
    }

    for name, (model_class, target_chars) in expert_configs.items():
        expert_model_path = os.path.join(OUTPUT_DIR, f"{name}_model_finetuned.pth")
        if os.path.exists(expert_model_path):
            print(f"Skipping {name.capitalize()} Model: Fine-tuned version already exists.")
            continue

        target_codes = {ord(c) for c in target_chars}
        indices = [i for i, label in enumerate(all_labels) if label in target_codes]

        if not indices: continue

        expert_remap = {code: i for i, code in enumerate(sorted(list(target_codes)))}
        expert_target_transform = LabelRemapper(expert_remap)

        expert_subset = Subset(full_dataset, indices)
        expert_dataset = [(img, expert_target_transform(label)) for img, label in
                          tqdm(expert_subset, desc=f"Remapping {name} labels")]

        # Calculate weights for sampling
        labels_in_subset = [item[1] for item in expert_dataset]
        class_counts = np.bincount(labels_in_subset)
        class_weights = 1. / np.where(class_counts > 0, class_counts, 1)
        sample_weights = [class_weights[label] for label in labels_in_subset]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        expert_loader = DataLoader(expert_dataset, batch_size=BATCH_SIZE, sampler=sampler)
        expert_model = model_class(num_classes=len(target_chars)).to(DEVICE)

        kaggle_expert_path = os.path.join(KAGGLE_WEIGHTS_DIR, f"{name}_model.pth")
        if os.path.exists(kaggle_expert_path):
            print(f"Loading base weights from {kaggle_expert_path}")
            expert_model.load_state_dict(torch.load(kaggle_expert_path, map_location=DEVICE))

        finetune_model(name, expert_model, expert_loader)


if __name__ == '__main__':
    main()