import re
import json
from typing import Dict
from tqdm import tqdm


SECTION_MARKERS = [
    "background", "introduction", "results", "result", "discussion", "conclusion", "conclusions", "summary"
]

def clean_text(text: str):
    if not text:
        return ""

    text = text.replace("\u00a0", " ")  
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    for m in SECTION_MARKERS:
        text = re.sub(rf"(?i)([a-z0-9\.\)])\s+({re.escape(m)})\b", r"\1\n\n\2", text)

    text = text.strip()
    return text


def chunk_by_words(
        text: str,
        chunk_size: int = 220,
        overlap_words: int = 40
):
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    
    chunks = []
    step = max(1, chunk_size - overlap_words)
    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end]).strip()

        if chunk: 
            chunks.append(chunk)

        if end == len(words):
            break
    
    return chunks


def read_jsonl(path):
    with open(path, 'r', encoding = 'utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            yield json.loads(line)


def write_jsonl(path, records):
    with open(path, 'w', encoding = 'utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_chunks(doc: Dict, chunk_size: int, overlap_words: int):
    doc_id = doc.get('id')
    if not doc_id: 
        doc_id = re.sub(r"\W+", "_", (doc.get("title") or "unknown").lower()).strip("_")[:80]

    title = (doc.get('title')).strip()
    url = doc.get('url')
    year = doc.get('year')
    authors = doc.get('authors')

    raw_text = doc.get('content')
    text = clean_text(raw_text)

    chunks = chunk_by_words(text, chunk_size=chunk_size, overlap_words=overlap_words)

    out = []
    for i, ch in enumerate(chunks):
        out.append({
            "chunk_id": f"{doc_id}__{i:04d}",
            "doc_id": doc_id,
            "chunk_index": i,
            "title": title,
            "year": year,
            "authors": authors,
            "url": url,
            "chunk_text": ch,
            "chunk_text_for_embedding": (
                f"Title: {title}\nYear: {year}\nDocument ID: {doc_id}\n\n{ch}"
            ).strip()
        })
    return out


def main(
    input_path: str = "data/articles.jsonl",
    output_path: str = "data/chunks.jsonl",
    chunk_size: int = 220,
    overlap_words: int = 40,
    min_chunk_chars: int = 200
):
    all_chunks = []
    docs = list(read_jsonl(input_path))

    for doc in tqdm(docs, desc="Chunking"):
        doc_chunks = make_chunks(doc, chunk_size, overlap_words)
        doc_chunks = [c for c in doc_chunks if len(c["chunk_text"]) >= min_chunk_chars]
        all_chunks.extend(doc_chunks)

    write_jsonl(output_path, all_chunks)
    print(f"Done. Docs: {len(docs)} | Chunks: {len(all_chunks)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()