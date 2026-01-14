import json
import re
import faiss
from sentence_transformers import SentenceTransformer
from ollama import chat

# ============ Конфигурация ============
INDEX_PATH = "data/faiss.index"
META_PATH = "data/meta.json"

EMB_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "qwen2.5:7b-instruct"

DOC_PREFIX = "passage: "
Q_PREFIX = "query: "

SYSTEM_PROMPT = """
You are a biomedical research assistant who helps identify potential therapeutic targets for Alzheimer's disease.

Rules:
- Answer ONLY in English.
- Use ONLY the provided context. Do not use external knowledge.
- Every non-trivial statement (targets, evidence, methods, proposed research) SHOULD be supported by references in square brackets, for example [1] or [1][3].
- If the context does not contain enough information, specify it explicitly.
- Do not invent targets, mechanisms, or claims about the possibility of using drugs.

Answer in detail, but do not write any unnecessary information.
Keep citations in-text only. DO NOT list your sources at the end of your answer.
"""


def load_meta(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_resources():

    meta = load_meta(META_PATH)
    index = faiss.read_index(INDEX_PATH)
    emb_model = SentenceTransformer(EMB_MODEL)
    return {
        "index": index,
        "meta": meta,
        "emb_model": emb_model
    }


def get_context(top_chunks: list) -> str:
    blocks = []
    for i, c in enumerate(top_chunks, start=1):
        header = (
            f"[{i}] {c.get('title', '')} ({c.get('year', '')}) | "
            f"doc_id={c.get('doc_id')} | chunk_id={c.get('chunk_id')}\n"
            f"URL: {c.get('url', '')} | authors={c.get('authors', '')}"
        )
        body = c["chunk_text"].strip()
        blocks.append(header + "\n" + body)
    return "\n\n---\n\n".join(blocks)


def extract_used_citations(text: str) -> set:
    found = re.findall(r"\[(\d+)\]", text)
    return set(int(x) for x in found)


def build_sources_list(top_chunks: list, used_citations: set) -> str:
    lines = []
    for idx in sorted(used_citations):
        if 1 <= idx <= len(top_chunks):
            c = top_chunks[idx - 1]
            lines.append(
                f"[{idx}] {c.get('title', '')} ({c.get('year', '')}) — {c.get('url', '')}"
            )
    return "\n".join(lines) if lines else "No sources cited."


def rag_answer(query: str, top_k: int, resources: dict = None):

    if resources is None:
        resources = load_resources()
    
    index = resources["index"]
    meta = resources["meta"]
    emb_model = resources["emb_model"]
    
    query_emb = emb_model.encode(
        [Q_PREFIX + query], 
        normalize_embeddings=True
    ).astype("float32")
    
    scores, idx = index.search(query_emb, top_k)
    top_chunks = [meta[i] for i in idx[0]]
    
    context = get_context(top_chunks)
    
    user_prompt = f"""Question: {query}

Context:
{context}"""
    
    response = chat(
        model=LLM_MODEL,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt}
        ],
        options={'temperature': 0.2}
    )
    
    answer = response['message']['content'].strip()
    
    citations = extract_used_citations(answer)
    sources = build_sources_list(top_chunks, citations)
    
    return answer, sources, context, top_chunks


def main():
    print("Assistant is ready! Type your question")
    resources = load_resources()
    
    while True:
        q = input("\nQuery (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        
        answer, sources, _, _ = rag_answer(q, top_k=6, resources=resources)
        
        print("\n=== Answer ===\n")
        print(answer)
        
        print("\n=== Sources ===\n")
        print(sources)


if __name__ == "__main__":
    main()