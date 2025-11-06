import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
import os
import numpy as np
import h5py
from tqdm import tqdm

from models import CNNModel_Small, CNNModel_Medium, CNNModel_Large

DATA_FILE = "data/book_dataset.h5"
MODEL_OUTPUT_DIR = "models/saved_weights_finetuned/"

EPOCHS = 3
BATCH_SIZE = 256
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 2

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience=2, path='checkpoint.pth'):
        self.patience, self.path = patience, path
        self.counter, self.val_loss_min = 0, np.inf
        self.best_score, self.early_stop = None, False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(f'Val loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class LabelRemapper:
    def __init__(self, remap_dict):
        self.remap_dict = remap_dict

    def __call__(self, label):
        return self.remap_dict.get(label, -1)


class HDF5Dataset(Dataset):
    def __init__(self, h5_path, transform=None, target_transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self.target_transform = target_transform
        with h5py.File(self.h5_path, 'r') as hf:
            self.length = len(hf['labels'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as hf:
            image = hf['images'][idx]
            label = hf['labels'][idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def prepare_dataloaders():
    print("Preparing dataloaders...")
    image_transform = transforms.Compose([transforms.ToTensor()])

    with h5py.File(DATA_FILE, 'r') as hf:
        all_labels = hf['labels'][:]
    unique_labels = np.unique(all_labels)

    triage_remap = {code: (0 if 48 <= code <= 57 else 1 if 65 <= code <= 90 else 2) for code in unique_labels}
    triage_target_transform = LabelRemapper(triage_remap)

    triage_dataset = HDF5Dataset(DATA_FILE, transform=image_transform, target_transform=triage_target_transform)
    val_size = int(len(triage_dataset) * VALIDATION_SPLIT)
    train_size = len(triage_dataset) - val_size
    triage_train, triage_val = random_split(triage_dataset, [train_size, val_size])

    datasets = {'triage': (triage_train, triage_val)}
    expert_filters = {
        'digits': (lambda c: 48 <= c <= 57), 'uppercase': (lambda c: 65 <= c <= 90),
        'lowercase': (lambda c: 97 <= c <= 122)
    }

    for name, condition in expert_filters.items():
        class_codes = sorted([c for c in unique_labels if condition(c)])
        expert_remap = {code: i for i, code in enumerate(class_codes)}
        expert_target_transform = LabelRemapper(expert_remap)

        indices = [i for i, code in enumerate(all_labels) if condition(code)]

        expert_full_dataset = HDF5Dataset(DATA_FILE, transform=image_transform,
                                          target_transform=expert_target_transform)
        expert_subset = Subset(expert_full_dataset, indices)

        val_expert_size = int(len(expert_subset) * VALIDATION_SPLIT)
        train_expert_size = len(expert_subset) - val_expert_size
        expert_train, expert_val = random_split(expert_subset, [train_expert_size, val_expert_size])
        datasets[name] = (expert_train, expert_val)

    print("Dataloaders are ready.")
    return datasets


def train_model(model_name, model, train_dataset, val_dataset):
    print(f"\n{'=' * 60}\nTraining: {model_name.upper()} MODEL on {len(train_dataset)} samples\n{'=' * 60}")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model_save_path = os.path.join(MODEL_OUTPUT_DIR, f"{model_name}_model_finetuned.pth")
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, path=model_save_path)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        avg_train_loss = train_loss / len(train_dataset)
        avg_val_loss = val_loss / len(val_dataset)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Validation Loss: {avg_val_loss:.6f}")

        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Model saved to {model_save_path}")


def main():
    if not os.path.exists(MODEL_OUTPUT_DIR): os.makedirs(MODEL_OUTPUT_DIR)
    print(f"Using device: {DEVICE}")

    datasets = prepare_dataloaders()

    models_to_train = {
        'triage': CNNModel_Large(num_classes=3),
        'digits': CNNModel_Small(num_classes=10),
        'uppercase': CNNModel_Medium(num_classes=26),
        'lowercase': CNNModel_Medium(num_classes=26)
    }
    for name, model in models_to_train.items():
        model_path = os.path.join(MODEL_OUTPUT_DIR, f"{name}_model_finetuned.pth")
        if os.path.exists(model_path):
            print(f"Skipping training for '{name}': Model already exists at {model_path}")
            continue

        train_d, val_d = datasets[name]
        model.to(DEVICE)
        train_model(name, model, train_d, val_d)



if __name__ == "__main__":
    main()