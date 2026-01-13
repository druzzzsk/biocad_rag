import json 
import faiss
from sentence_transformers import SentenceTransformer
from ollama import chat 

INDEX_PATH = "data/faiss.index"
META_PATH = "data/meta.json"

EMB_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "qwen2.5:7b-instruct"

TOP_K = 6

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


def load_meta(path):
    with open(path, 'r', encoding = 'utf-8') as f:
        return json.load(f)
    

def get_context(top_chunks):

    blocks = []
    for i,c in enumerate(top_chunks, start = 1):
        header = f"[{i}] {c.get('title','')} ({c.get('year','')}) | doc_id={c.get('doc_id')} | chunk_id={c.get('chunk_id')}\nURL: {c.get('url','')} | authors={c.get('authors', '')}"
        body = c["chunk_text"].strip()
        blocks.append(header + "\n" + body)
    return "\n\n---\n\n".join(blocks)


def build_sources_list(top_chunks, used_citations):
    lines = []
    for idx in sorted(used_citations):
        if 1 <= idx <= len(top_chunks):
            c = top_chunks[idx-1]
            lines.append(f"[{idx}] {c.get('title','')} ({c.get('year','')}) — {c.get('url','')}")
    
    return "\n".join(lines) if lines else "No sources cited."


def extract_used_citations(text: str):
    import re
    found = re.findall(r"\[(\d+)\]", text)
    return set(int(x) for x in found)


def rag_answer(query, top_k):
    
    print('Waiting for answer ...')
    meta = load_meta(META_PATH)
    index = faiss.read_index(INDEX_PATH)
    emb_model = SentenceTransformer(EMB_MODEL)

    query_emb = emb_model.encode([Q_PREFIX + query], normalize_embeddings=True).astype("float32")
    scores, idx = index.search(query_emb, top_k)

    top_chunks = [meta[i] for i in idx[0]]

    context = get_context(top_chunks)

    user_prompt = f"""
    Question: {query}
    Context: {context}"""

    response = chat(
        model = LLM_MODEL,
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt}
        ], options= {'temperature': 0.2}
    )

    answer = response['message']['content'].strip()

    citations = extract_used_citations(answer)
    sourses = build_sources_list(top_chunks, citations)

    return answer, sourses, context


def main():
    print("Assistant is ready! Type your question")
    while True:
        q = input("\nQuery (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break

        answer, sources, _ = rag_answer(q, top_k=TOP_K)

        print("\n=== Answer ===\n")
        print(answer)

        print("\n=== Sources ===\n")
        print(sources)
    

if __name__ == "__main__":
    main()

    