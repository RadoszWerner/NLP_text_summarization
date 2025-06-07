# summarize_results.py

import pandas as pd
from datasets import load_dataset

def main():
    df = pd.read_csv("results/embeddings_experiments.csv")

    stats = df.groupby("embedding")[["rouge-1", "rouge-2", "rouge-l"]].agg(["mean", "std"])
    
    stats.columns = [
        f"{metric}_{stat}" 
        for metric, stat in stats.columns
    ]
    stats = stats.reset_index()

    print("%%% Średnie i odchylenia metryk (LaTeX) %%%\n")
    print(stats.to_latex(index=False, float_format="%.3f"))

    # 3) Wczytaj zbiór, aby uzyskać dostęp do artykułów
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")

    # 4) Tabela z najlepszymi wynikami (najwyższe rouge-l) + długość oryginalnego artykułu
    best_rows = []
    for emb in df["embedding"].unique():
        subset = df[df["embedding"] == emb]
        best = subset.loc[subset["rouge-l"].idxmax()]
        art_id = int(best["article_id"]) - 1
        article_text = dataset[art_id]["article"]
        num_chars = len(article_text)
        best_rows.append({
            "embedding": emb,
            "article_id": art_id + 1,
            "rouge-1": best["rouge-1"],
            "rouge-2": best["rouge-2"],
            "rouge-l": best["rouge-l"],
            "num_chars": num_chars
        })
    best_df = pd.DataFrame(best_rows)
    print("\n%%% Najlepsze wyniki (LaTeX) %%%\n")
    print(best_df.to_latex(index=False, float_format="%.3f"))

    # 5) Tabela z najgorszymi wynikami (najniższe rouge-l) + długość oryginalnego artykułu
    worst_rows = []
    for emb in df["embedding"].unique():
        subset = df[df["embedding"] == emb]
        worst = subset.loc[subset["rouge-l"].idxmin()]
        art_id = int(worst["article_id"]) - 1
        article_text = dataset[art_id]["article"]
        num_chars = len(article_text)
        worst_rows.append({
            "embedding": emb,
            "article_id": art_id + 1,
            "rouge-1": worst["rouge-1"],
            "rouge-2": worst["rouge-2"],
            "rouge-l": worst["rouge-l"],
            "num_chars": num_chars
        })
    worst_df = pd.DataFrame(worst_rows)
    print("\n%%% Najgorsze wyniki (LaTeX) %%%\n")
    print(worst_df.to_latex(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
