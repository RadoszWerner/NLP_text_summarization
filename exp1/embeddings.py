# embeddings.py

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import KeyedVectors
import gensim.downloader as api

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel

# 1) Tu jakis loading modeli do pamieci cache zamiast miec to lokalnie w repo
print("Word2Vec")
w2v = api.load("word2vec-google-news-300")
print("  → Word2Vec gotowy!")

# GloVe (Wikipedia + Gigaword, 300d)
print("GloVe")
glove = api.load("glove-wiki-gigaword-300")
print("  → GloVe gotowy!")

# FastText (Wikipedia 300d + subwords)
print("FastText")
ft = api.load("fasttext-wiki-news-subwords-300")
print("  → FastText gotowy!")

# BERT (klasyczny, "bert-base-uncased" z poolingiem)
print("Tokenizer i model BERT")
bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained(
    "bert-base-uncased", output_hidden_states=True
)
bert_model.eval()
print("BERT gotowy")

# Sentence‐BERT (SBERT, np. "all-MiniLM-L6-v2")
print("Sentence-BERT")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
print("SBERT gotowy")



# 2) Funkcje pomocnicze do wyliczania embeddingu zdania
def sentence_embedding_mean_gensim(tokens: list[str], model: KeyedVectors) -> np.ndarray:
    vecs = []
    for w in tokens:
        if w in model.key_to_index:
            vecs.append(model.get_vector(w))
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)


def sentence_embedding_bert(sent_text: str, device: str = "cpu") -> np.ndarray:
    encoding = bert_tok(
        sent_text,
        is_split_into_words=False,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**encoding, return_dict=True)
        hidden_states = outputs.hidden_states
        last_hidden = hidden_states[-1].squeeze(0)
        mean_vec = last_hidden.mean(dim=0).cpu().numpy()
    return mean_vec


def build_sentence_matrix(
    preprocessed_sentences: list[str],
    embedding_type: str = "tfidf"
) -> np.ndarray:
    # —— TF-IDF (sklearn) ——
    if embedding_type == "tfidf":
        vect = TfidfVectorizer()
        X = vect.fit_transform(preprocessed_sentences)  # csr_matrix (N, V)
        return X.toarray()                              # (N, V)
    sentence_vecs = []
    # —— Word2Vec / GloVe / FastText — średnia wektorów ——
    if embedding_type in {"word2vec", "glove", "fasttext"}:
        model = {
            "word2vec": w2v,
            "glove": glove,
            "fasttext": ft
        }[embedding_type]

        for sent in preprocessed_sentences:
            toks = sent.split()
            vec = sentence_embedding_mean_gensim(toks, model)
            sentence_vecs.append(vec)
        return np.vstack(sentence_vecs)

    # —— BERT (mean pooling) ——
    if embedding_type == "bert":
        for sent in preprocessed_sentences:
            vec = sentence_embedding_bert(sent)
            sentence_vecs.append(vec)
        return np.vstack(sentence_vecs)

    # —— SBERT ——
    if embedding_type == "sbert":
        vecs = sbert.encode(preprocessed_sentences, show_progress_bar=False)
        return np.array(vecs)

    raise ValueError(f"Nieznany typ embeddingu: {embedding_type}")
