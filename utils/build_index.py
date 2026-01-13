import json
import faiss
from sentence_transformers import SentenceTransformer 

CHUNKS_PATH = 'data/chunks.jsonl'
INDEX_PATH = 'data/faiss.index'
META_PATH = 'data/meta.json'

MODEL_NAME = "BAAI/bge-base-en-v1.5"  
DOC_PREFIX = "passage: "
Q_PREFIX = "query: "


def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out
        

def main():

    chunks = read_jsonl(CHUNKS_PATH)
    model = SentenceTransformer(MODEL_NAME)

    texts = []
    meta = []

    for c in chunks:
        t = c.get("chunk_text_for_embedding") or c["chunk_text"]
        texts.append(DOC_PREFIX + t)
        meta.append({
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "title": c.get("title"),
            "year": c.get("year"),
            "url": c.get("url"),
            "chunk_text": c["chunk_text"]
        })

    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved index: {INDEX_PATH} | vectors: {index.ntotal}")
    print(f"Saved meta:  {META_PATH}")


if __name__ == "__main__":
    main()
