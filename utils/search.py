import json
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/faiss.index"
META_PATH = "data/meta.json"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
Q_PREFIX = "query: "


def main():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    index = faiss.read_index(INDEX_PATH)
    model = SentenceTransformer(MODEL_NAME)

    while True:
        q = input("\nQuery (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break

        q_emb = model.encode([Q_PREFIX + q], normalize_embeddings=True).astype("float32")
        k = 5
        scores, idx = index.search(q_emb, k)

        print("\nTop sources:")
        for rank, (i, s) in enumerate(zip(idx[0], scores[0]), start=1):
            m = meta[i]
            snippet = m["chunk_text"][:350].replace("\n", " ")
            print(f"\n#{rank} score={float(s):.3f} | {m['title']} ({m.get('year')})")
            print(f"  doc_id={m['doc_id']} chunk_id={m['chunk_id']}")
            print(f"  url={m.get('url')}")
            print(f"  snippet: {snippet}...")


if __name__ == "__main__":
    main()