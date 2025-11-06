import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import os
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

from models import CNNModel_Small, CNNModel_Medium, CNNModel_Large

DATA_FILE = "data/book_dataset.h5"
MODEL_DIR = "models/saved_weights_finetuned/"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 256
NUM_SAMPLES_PER_CLASS = 1000

class HDF5Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        with h5py.File(self.h5_path, 'r') as hf:
            self.images = hf['images'][:]
            self.labels = hf['labels'][:]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        if self.transform: image = self.transform(image)
        return image, label


def load_finetuned_model(name, model_class, num_classes):
    model_path = os.path.join(MODEL_DIR, f"{name}_model_finetuned.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Please run the fine-tuning script.")

    model = model_class(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def create_balanced_test_set():
    print(f"Creating a balanced test set with up to {NUM_SAMPLES_PER_CLASS} samples per class...")
    with h5py.File(DATA_FILE, 'r') as hf:
        all_labels = hf['labels'][:]

    digit_indices = np.where((all_labels >= 48) & (all_labels <= 57))[0]
    upper_indices = np.where((all_labels >= 65) & (all_labels <= 90))[0]
    lower_indices = np.where((all_labels >= 97) & (all_labels <= 122))[0]

    digit_samples = np.random.choice(digit_indices, min(NUM_SAMPLES_PER_CLASS, len(digit_indices)), replace=False)
    upper_samples = np.random.choice(upper_indices, min(NUM_SAMPLES_PER_CLASS, len(upper_indices)), replace=False)
    lower_samples = np.random.choice(lower_indices, min(NUM_SAMPLES_PER_CLASS, len(lower_indices)), replace=False)

    test_indices = np.concatenate([digit_samples, upper_samples, lower_samples])
    np.random.shuffle(test_indices)

    full_dataset = HDF5Dataset(DATA_FILE, transform=transforms.ToTensor())
    test_subset = Subset(full_dataset, test_indices)
    print(f"Test set created with {len(test_subset)} total samples.")
    return test_subset


def main():
    print("--- Starting Full Model Evaluation ---")
    print(f"Using device: {DEVICE}")

    print("\nLoading all fine-tuned models...")
    models = {
        'triage': load_finetuned_model('triage', CNNModel_Large, 3),
        'digits': load_finetuned_model('digits', CNNModel_Small, 10),
        'uppercase': load_finetuned_model('uppercase', CNNModel_Medium, 26),
        'lowercase': load_finetuned_model('lowercase', CNNModel_Medium, 26)
    }
    print("All models loaded.")

    test_dataset = create_balanced_test_set()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_true_labels, all_pred_labels = [], []
    all_true_triage, all_pred_triage = [], []
    expert_true = {'digits': [], 'uppercase': [], 'lowercase': []}
    expert_preds = {'digits': [], 'uppercase': [], 'lowercase': []}

    triage_map = {0: 'digits', 1: 'uppercase', 2: 'lowercase'}
    expert_remaps = {
        'digits': {code: i for i, code in enumerate(range(48, 58))},
        'uppercase': {code: i for i, code in enumerate(range(65, 91))},
        'lowercase': {code: i for i, code in enumerate(range(97, 123))}
    }

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating Full System"):
            images = images.to(DEVICE)

            triage_out = models['triage'](images)
            _, pred_triage_indices = torch.max(triage_out, 1)

            for i in range(len(images)):
                true_label_code = labels[i].item()

                if 48 <= true_label_code <= 57:
                    true_triage_class_idx, true_expert_name = 0, 'digits'
                elif 65 <= true_label_code <= 90:
                    true_triage_class_idx, true_expert_name = 1, 'uppercase'
                else:
                    true_triage_class_idx, true_expert_name = 2, 'lowercase'

                pred_triage_class_idx = pred_triage_indices[i].item()
                expert_to_use = triage_map[pred_triage_class_idx]

                expert_out = models[expert_to_use](images[i].unsqueeze(0))
                _, pred_expert_idx = torch.max(expert_out, 1)

                remap = expert_remaps[expert_to_use]
                inv_remap = {v: k for k, v in remap.items()}
                pred_label_code = inv_remap.get(pred_expert_idx.item(), -1)

                all_true_labels.append(true_label_code)
                all_pred_labels.append(pred_label_code)
                all_true_triage.append(true_triage_class_idx)
                all_pred_triage.append(pred_triage_class_idx)

                correct_expert_model = models[true_expert_name]
                correct_expert_out = correct_expert_model(images[i].unsqueeze(0))
                _, pred_correct_expert_idx = torch.max(correct_expert_out, 1)

                true_expert_label = expert_remaps[true_expert_name][true_label_code]
                expert_true[true_expert_name].append(true_expert_label)
                expert_preds[true_expert_name].append(pred_correct_expert_idx.item())

    print("\n\n" + "=" * 50)
    print("         OCR EVALUATION REPORT")
    print("=" * 50)

    overall_accuracy = accuracy_score(all_true_labels, all_pred_labels)
    print("\n--- 1. Overall OCR System Accuracy ---")
    print(f"   End-to-End Accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")

    print("\n--- 2. Triage Model Performance (Class-wise) ---")
    triage_class_names = ['digits', 'uppercase', 'lowercase']
    print(classification_report(all_true_triage, all_pred_triage, target_names=triage_class_names))

    print("\n--- 3. Individual Expert Accuracy ---")
    print("(This measures accuracy assuming the Triage model was correct)")
    for name in ['digits', 'uppercase', 'lowercase']:
        expert_acc = accuracy_score(expert_true[name], expert_preds[name])
        print(f"   - {name.capitalize()} Expert Accuracy: {expert_acc:.4f} ({expert_acc * 100:.2f}%)")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
