import os
import h5py
import numpy as np

DATA_FILE = "data/book_dataset.h5"


def get_file_size(path):
    """Calculates the size of a single file."""
    size_bytes = os.path.getsize(path)
    if size_bytes > 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 ** 3):.2f} GB"
    elif size_bytes > 1024 * 1024:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    return f"{size_bytes / 1024:.2f} KB"


def analyze_hdf5_dataset(file_path):
    """
    Analyzes the HDF5 character dataset, printing a summary of its contents.
    """
    if not os.path.exists(file_path):
        print(f"Error: Dataset file not found at '{file_path}'")
        return

    print("=" * 50)
    print("        HDF5 Dataset Inspection Report")
    print("=" * 50)
    print(f"Analyzing file: '{file_path}'\n")

    with h5py.File(file_path, 'r') as hf:
        if 'labels' not in hf or 'images' not in hf:
            print("Error: HDF5 file is missing 'images' or 'labels' datasets.")
            return

        labels = hf['labels'][:]
        total_images = hf['images'].shape[0]

    unique_labels, counts = np.unique(labels, return_counts=True)
    char_counts = {chr(int(label)): count for label, count in zip(unique_labels, counts)}

    print("--- Character Frequency ---")
    sorted_chars = sorted(char_counts.items(), key=lambda item: item[1], reverse=True)

    for char, count in sorted_chars:
        print(f"Character: '{char}' | Samples: {count}")

    print("\n" + "-" * 27)

    print("\n--- Summary ---")
    print(f"Total number of unique characters: {len(char_counts)}")
    print(f"Total number of image samples: {total_images}")
    print(f"Total dataset size on disk: {get_file_size(file_path)}")
    print("=" * 50)


if __name__ == "__main__":
    analyze_hdf5_dataset(DATA_FILE)