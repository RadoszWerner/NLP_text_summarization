# split_datasets.py
import os
import json
from datasets import load_dataset
import random

# Parametry podziału
thresholds = {
    'small': 500,
    'medium': 800,
    'large': 1200
}

sample_size = 10000  # liczba próbek na grupę
random_seed = 42

print("Ładowanie zbioru abisee/cnn_dailymail (train)...")
data = load_dataset('abisee/cnn_dailymail', '3.0.0')['train']

# Podział na grupy wg długości
groups = {'small': [], 'medium': [], 'large': []}
for idx, item in enumerate(data):
    length = len(item['article'].split())
    if length <= thresholds['small']:
        groups['small'].append(idx)
    elif length <= thresholds['medium']:
        groups['medium'].append(idx)
    else:
        groups['large'].append(idx)

# Balansowanie: dokładnie `sample_size` wpisów na grupę
random.seed(random_seed)
balanced = {}
for name, idxs in groups.items():
    if len(idxs) >= sample_size:
        # unikalne próbki
        samp = random.sample(idxs, sample_size)
    else:
        # jeśli za mało, losujemy z powtórzeniami, aby uzyskać sample_size
        samp = random.choices(idxs, k=sample_size)
    balanced[name] = samp
    print(f"Grupa {name}: przed={len(idxs)}, po={len(samp)}")

os.makedirs('datasets', exist_ok=True)
for name, idxs in balanced.items():
    with open(f'datasets/{name}_indices.json', 'w', encoding='utf-8') as f:
        json.dump(idxs, f)
    print(f"Grupa {name}: przed={len(groups[name])}, po={len(idxs)}")