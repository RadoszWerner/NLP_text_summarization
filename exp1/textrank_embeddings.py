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


def preprocess_text(text: str) -> tuple[list[str], list[str]]:
    text = text.replace("(CNN)", "").replace("--", "").replace("''", '"')
    sentences = sent_tokenize(text)

    preprocessed: list[str] = []
    for sent in sentences:
        doc = nlp(sent.lower())
        cleaned_words = []
        for token in doc:
            if (
                token.is_alpha
                and not token.is_stop
                and token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
            ):
                cleaned_words.append(token.lemma_)
        preprocessed.append(" ".join(cleaned_words))

    return preprocessed, sentences


def textrank_with_embedding(
    article_text: str,
    embedding_type: str = "tfidf",
    num_sentences: int = 5,
    damping_factor: float = 0.85,
    similarity_threshold: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-6,
    max_input_sentences: int | None = None,
) -> str:
    """
    1) Preprocess → (preprocessed_sentences, original_sentences)
    2) (opcjonalnie) Przycięcie do max_input_sentences (pierwsze N zdań).
    3) build_sentence_matrix(...) wg `embedding_type` (np. "tfidf","word2vec","sbert" itd.)
    4) cosine_similarity → macierz podobieństw
    5) odetnij wagi < similarity_threshold
    6) zbuduj graf i nalicz PageRank (networkx)
    7) wybierz top num_sentences zdań (wg rankingów), posortuj wg oryginalnej kolejności
    8) zwróć je jako jeden string (połączone spacją)
    """
    preprocessed, original_sentences = preprocess_text(article_text)

    if max_input_sentences is not None:
        preprocessed = preprocessed[:max_input_sentences]
        original_sentences = original_sentences[:max_input_sentences]

    sentence_matrix = build_sentence_matrix(
        preprocessed, embedding_type=embedding_type
    )  # np. (N_zd, dim_emb)

    from sklearn.metrics.pairwise import cosine_similarity

    sim_matrix = cosine_similarity(sentence_matrix, sentence_matrix)  # (N, N)
    np.fill_diagonal(sim_matrix, 0.0)

    sim_matrix[sim_matrix < similarity_threshold] = 0.0

    # 6. Zbuduj graf i PageRank
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(
        graph, alpha=damping_factor, max_iter=max_iter, tol=tol
    )  # słownik {i: score_i}

    # 7. Sortowanie zdań po PageRank (malejąco), potem weź `num_sentences`
    ranked = sorted(
        ((score, idx) for idx, score in scores.items()), reverse=True
    )
    top_idxs = [idx for (_score, idx) in ranked[: min(num_sentences, len(original_sentences))]]
    top_idxs.sort()

    # 8. Połącz wybrane oryginalne zdania w logiczną kolejność i zwróć
    summary = " ".join([original_sentences[i] for i in top_idxs])
    return summary


def evaluate_embeddings(
    dataset,
    embedding_types: list[str],
    num_articles: int = 200,
    num_sentences: int = 5,
    damping_factor: float = 0.85,
    similarity_threshold: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Dla każdego embeddingu w `embedding_types`:
      - Dla i ∈ [0..num_articles-1]:
          • bierz dataset["train"][i]["article"] oraz ["highlights"]
          • generuj summary = textrank_with_embedding(...)
          • licz Rouge(get_scores) i zapisz (rouge-1, rouge-2, rouge-l) oraz summary
    Zwraca DataFrame z kolumnami:
      ["article_id", "embedding", "rouge-1", "rouge-2", "rouge-l", "summary"]
    """
    rouge = Rouge()
    results = []

    for emb in embedding_types:
        print(f"\n=== Eksperymentujemy na embeddingu: {emb} ===")
        for i in range(num_articles):
            article = dataset["train"][i]["article"]
            reference = dataset["train"][i]["highlights"]

            summary = textrank_with_embedding(
                article_text=article,
                embedding_type=emb,
                num_sentences=num_sentences,
                damping_factor=damping_factor,
                similarity_threshold=similarity_threshold,
                max_iter=max_iter,
                tol=tol,
                max_input_sentences=None,
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
        best_row = subset.loc[subset["rouge-l"].idxmax()]
        art_id = int(best_row["article_id"])
        best_summary = best_row["summary"]
        best_score = best_row["rouge-l"]

        original_article = dataset["train"][art_id - 1]["article"]
        original_highlight = dataset["train"][art_id - 1]["highlights"]

        # Liczba słów i znaków w oryginalnym artykule
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
    # ---------------------------------------
    # Przykładowe użycie / główna pętla „main”
    # ---------------------------------------
    print("Ładuję zbiór CNN/DailyMail...")
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    # Spis embeddingów, które chcemy przetestować
    embedding_list = ["tfidf", "word2vec", "glove", "fasttext", "bert", "sbert"]

    print("Start eksperymentu z różnymi embeddingami (200 artykułów)...")
    results_df = evaluate_embeddings(
        dataset=dataset,
        embedding_types=embedding_list,
        num_articles=1,
        num_sentences=5,
        damping_factor=0.85,
        similarity_threshold=0.1,
        max_iter=500,
        tol=1e-6,
    )

    # 4) Porównanie referatu z wygenerowanymi podsumowaniami dla 5 pierwszych artykułów
    print("\n=== Porównanie referatu z wygenerowanymi podsumowaniami dla 5 artykułów ===\n")
    for i in range(5):
        article = dataset["train"][i]["article"]
        reference = dataset["train"][i]["highlights"]
        print(f"> Artykuł {i+1}:")
        print("  • Referat (highlight):")
        print(f"    {reference}\n")
        for emb in embedding_list:
            summary = textrank_with_embedding(
                article_text=article,
                embedding_type=emb,
                num_sentences=5,
                damping_factor=0.85,
                similarity_threshold=0.1,
                max_iter=500,
                tol=1e-6,
                max_input_sentences=None,
            )
            print(f"  - {emb} summary:")
            print(f"    {summary}\n")
        print("-" * 80)
