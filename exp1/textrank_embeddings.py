# textrank_embeddings.py

import os
import numpy as np
import networkx as nx

from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from rouge import Rouge

import spacy
import pandas as pd

# Importujemy funkcje z embeddings.py
from embeddings import build_sentence_matrix

# ------------------------------------------------------------
# 1) Inicjalizacja spaCy (do preprocesu tekstu: lematyzacja, POS, stopwords)
# ------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    # Usuwamy "(CNN)", "--", "''"
    text = text.replace("(CNN)", "").replace("--", "").replace("''", '"')
    # Dzielimy tekst na zdania
    sentences = sent_tokenize(text)

    preprocessed = []
    for sent in sentences:
        doc = nlp(sent.lower())
        cleaned_words = []
        for token in doc:
            if token.is_alpha and not token.is_stop and token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}:
                cleaned_words.append(token.lemma_)
        preprocessed.append(" ".join(cleaned_words))

    return preprocessed, sentences


def textrank_with_embedding(article_text, embedding_type="tfidf", num_sentences=5,
                            damping_factor=0.85, similarity_threshold=0.1,
                            max_iter=100, tol=1e-6, max_input_sentences=None):
    # 1. Preprocess
    preprocessed, original_sentences = preprocess_text(article_text)

    # 2. Przycięcie
    if max_input_sentences is not None:
        preprocessed = preprocessed[:max_input_sentences]
        original_sentences = original_sentences[:max_input_sentences]

    # 3. Budujemy macierz embeddingów / TF-IDF
    sentence_matrix = build_sentence_matrix(preprocessed, embedding_type=embedding_type)

    # 4. Kosinusowa macierz podobieństw
    from sklearn.metrics.pairwise import cosine_similarity

    sim_matrix = cosine_similarity(sentence_matrix, sentence_matrix)
    np.fill_diagonal(sim_matrix, 0.0)
    sim_matrix[sim_matrix < similarity_threshold] = 0.0

    # 6. Zbuduj graf i PageRank
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph, alpha=damping_factor, max_iter=max_iter, tol=tol)

    # 7. Sortowanie zdań po PageRank i wybór top num_sentences
    ranked = sorted(((score, idx) for idx, score in scores.items()), reverse=True)
    top_count = min(num_sentences, len(original_sentences))
    top_idxs = [idx for (_score, idx) in ranked[:top_count]]
    top_idxs.sort()

    # 8. Połącz oryginalne zdania i zwróć
    summary = " ".join([original_sentences[i] for i in top_idxs])
    return summary


def evaluate_embeddings(dataset, embedding_types, num_articles=200, num_sentences=5,
                        damping_factor=0.85, similarity_threshold=0.1,
                        max_iter=100, tol=1e-6):
    rouge = Rouge()
    results = []

    for emb in embedding_types:
        print(f"\n=== Eksperymentujemy na embeddingu: {emb} ===")
        for i in range(num_articles):
            article = dataset["train"][i]["article"]
            reference = dataset["train"][i]["highlights"]

            summary = textrank_with_embedding(
                article, embedding_type=emb, num_sentences=num_sentences,
                damping_factor=damping_factor, similarity_threshold=similarity_threshold,
                max_iter=max_iter, tol=tol, max_input_sentences=None
            )

            scores = rouge.get_scores(summary, reference)[0]
            results.append({
                "article_id": i + 1,
                "embedding": emb,
                "rouge-1": scores["rouge-1"]["f"],
                "rouge-2": scores["rouge-2"]["f"],
                "rouge-l": scores["rouge-l"]["f"],
                "summary": summary,
            })

            if (i + 1) % 10 == 0:
                print(f"  • {i+1}/{num_articles} artykułów przetworzono")

    df = pd.DataFrame(results)

    # 1) Zapis do CSV w folderze results
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "embeddings_experiments123.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Wyniki zapisano do: {csv_path}")

    # 3) Dla każdego embeddingu: znajdź rekord z najwyższym rouge-l
    print("\n=== Najlepsze podsumowanie wg rouge-l dla każdego embeddingu ===\n")
    for emb in embedding_types:
        subset = df[df["embedding"] == emb]
        best_idx = subset["rouge-l"].idxmax()
        best_row = subset.loc[best_idx]
        art_id = int(best_row["article_id"])
        best_summary = best_row["summary"]
        best_score = best_row["rouge-l"]

        original_article = dataset["train"][art_id - 1]["article"]
        original_highlight = dataset["train"][art_id - 1]["highlights"]

        num_chars = len(original_article)
        num_words = len(original_article.split())

        print(f"> Embedding: {emb}")
        print(f"  • Article ID: {art_id} (ROUGE-L = {best_score:.3f})")
        print(f"  • Długość oryginalnego artykułu: {num_words} słów, {num_chars} znaków\n")
        print("  • Najlepsze podsumowanie:")
        print(f"    {best_summary}\n")
        print("  • Oryginalny highlight:")
        print(f"    {original_highlight}\n")
        print("-" * 80)

    return df


if __name__ == "__main__":
    print("Ładuję zbiór CNN/DailyMail...")
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    embedding_list = ["tfidf", "word2vec", "glove", "fasttext", "bert", "sbert"]

    print("Start eksperymentu")
    results_df = evaluate_embeddings(
        dataset, embedding_list, num_articles=200, num_sentences=5,
        damping_factor=0.85, similarity_threshold=0.1, max_iter=500, tol=1e-6
    )

