import numpy as np
import pandas as pd
from pathlib import Path


def extract_embedding_features(texts: pd.Series, cfg: dict) -> pd.DataFrame:
    ec = cfg["embedding"]
    cache_path = Path(ec["cache_path"])
    model_name = ec["model_name"]
    batch_size = ec.get("batch_size", 32)
    normalize = ec.get("normalize_embeddings", True)

    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        embs = np.load(str(cache_path))
    else:
        print(f"Computing embeddings with {model_name} ...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        text_list = texts.tolist()
        embs = model.encode(
            text_list,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), embs)
        print(f"Embeddings saved to {cache_path}")

    n_dim = embs.shape[1]
    col_names = [f"emb_{i:03d}" for i in range(n_dim)]
    return pd.DataFrame(embs, columns=col_names)
