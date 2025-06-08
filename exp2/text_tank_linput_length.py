# run_textrank_length.py
import os
import json
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from datasets import load_dataset
import spacy
from tqdm import tqdm

# Ładowanie spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Wczytanze indeksów
groups = {}
for name in ['small', 'medium', 'large']:
    with open(f'datasets/{name}_indices.json', encoding='utf-8') as f:
        groups[name] = json.load(f)

# Funkcje
def preprocess_text(text):
    text = text.replace("(CNN)", "").replace("--", " ").replace("''", '"')
    original = sent_tokenize(text)
    processed = [" ".join(tok.lemma_ for tok in nlp(sent.lower())
                  if tok.is_alpha and not tok.is_stop and tok.pos_ in {"NOUN","VERB","ADJ","ADV"})
                 for sent in original]
    return processed, original


def textrank_summary(text):
    proc, orig = preprocess_text(text)
    if not proc:
        return ""
    mat = TfidfVectorizer().fit_transform(proc).toarray()
    sim = cosine_similarity(mat)
    np.fill_diagonal(sim, 0)
    sim[sim < 0.1] = 0
    graph = nx.from_numpy_array(sim)
    scores = nx.pagerank(graph)
    top_idxs = sorted(scores, key=scores.get, reverse=True)[:5]
    return " ".join(orig[i] for i in sorted(top_idxs))

# Ewaluacja z paskiem postępu
data = load_dataset('abisee/cnn_dailymail', '3.0.0')['train']
g = Rouge()
results = []
for name, idxs in groups.items():
    print(f"Przetwarzanie grupy: {name} ({len(idxs)} artykułów)")
    scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
    for i in tqdm(idxs, desc=f"{name}"):
        art = data[i]['article']
        ref = data[i]['highlights']
        pred = textrank_summary(art)
        if not pred:
            continue
        sc = g.get_scores(pred, ref)[0]
        for k in scores:
            scores[k].append(sc[k]['f'])
    results.append({'group': name,
                    'count': len(idxs),
                    **{k: np.mean(v) for k, v in scores.items()}})

# Zapis wyników
df = __import__('pandas').DataFrame(results)
os.makedirs('results', exist_ok=True)
df.to_csv('results/length_experiment_results.csv', index=False)
print("Zapisano wyniki do results/length_experiment_results.csv")
