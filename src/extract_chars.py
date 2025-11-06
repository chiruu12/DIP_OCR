import h5py

DATA_FILE = "data/real_line_dataset.h5"
OUTPUT_FILE = "char_list.txt"

print(f"Reading character list from '{DATA_FILE}'...")
with h5py.File(DATA_FILE, 'r') as hf:
    char_list = [c.decode('utf-8') for c in hf['char_list'][:]]

separator = "<SEP>"
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(separator.join(char_list))

print(f"Successfully extracted {len(char_list)} characters to '{OUTPUT_FILE}'.")
