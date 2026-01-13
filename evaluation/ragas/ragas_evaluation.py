import sys
import os
current_file = os.path.abspath(__file__)
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.run_config import RunConfig 
from rag_answer import rag_answer, LLM_MODEL, EMB_MODEL

eval_questions = [
    "What are potential targets for Alzheimer's disease treatment?",
    "Are the targets druggable with small molecules, biologics, or other modalities?",
    "What additional studies are needed to advance these targets?"
]

def run_evaluation():

    data_samples = {
        "question": [],
        "answer": [],
        "contexts": []
    }

    print("Generating answers for evaluation...")
    for q in eval_questions:
        ans, _, ctx_str, _ = rag_answer(q, top_k=3)
        data_samples["question"].append(q)
        data_samples["answer"].append(ans)
        data_samples["contexts"].append(ctx_str.split("\n\n---\n\n"))

    dataset = Dataset.from_dict(data_samples)

    eval_llm = ChatOllama(model=LLM_MODEL, temperature=0)
    eval_embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    my_run_config = RunConfig(timeout=600, max_workers=1)

    print("\nStarting Ragas evaluation (sequential mode)...")
    
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=my_run_config
    )

    print("\n=== Evaluation Results ===")
    print(results)
    
    df = results.to_pandas()
    pd.set_option('display.max_colwidth', 50)
    df.to_csv('ragas_results.csv')

if __name__ == '__main__': 
    run_evaluation()