import sys
import os
current_file = os.path.abspath(__file__)
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

from langchain_community.llms import Ollama
import pandas as pd
import prompts
from rag_core import rag_answer
from prompts import (prompt_factual, 
                     prompt_citation, 
                     prompt_coverage, 
                     prompt_paraphrase, 
                     prompt_relevance)

PROMPTS_AND_METRICS = {
    'factual_consistency': prompt_factual,
    'coverage': prompt_coverage,
    'citation': prompt_citation,
    'paraphrase': prompt_paraphrase,
    'relevance': prompt_relevance 
}

QUESTIONS_LIST = [
    'What are potential targets for Alzheimers disease treatment?',
]


def build_judges(model):
    judges = {}
    
    for metric_name, prompt_template in PROMPTS_AND_METRICS.items():
        judges[metric_name] = prompt_template | model
    
    return judges
    

def evaluate(judges, data):

    print('Evaluation started!')
    results = []
    required_columns = ['query', 'answer', 'context']

    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f'Column {col} is missed!')
        
    for idx, row in data.iterrows():

        query = row.get('query', '')
        answer = row.get('answer', '')
        context = row.get('context', '')
        
        evaluation_row = {
            'query' : query,
            'answer' : answer
        }

        for name, judge in judges.items():

            payload = {
                'query' : query,
                'answer' : answer,
                'context' : context
            }
            
            try:
                response = judge.invoke(payload)
                evaluation_row[name] = response
            except Exception as e:
                evaluation_row[name] = f"Error: {e}"
        results.append(evaluation_row)

    return pd.DataFrame(results)


def main(): 
    model = Ollama(model="qwen2.5:7b-instruct")

    results = []
    for query in QUESTIONS_LIST:
        print(f'Query #{QUESTIONS_LIST.index(query) + 1}')
        answer, sources, context, _ = rag_answer(query, 6)
        
        row = {
            'query': query,
            'answer': answer,
            'sources': sources,
            'context': context
        }
        results.append(row)
    
    data = pd.DataFrame(results)
    judges = build_judges(model)
    evaluation_result = evaluate(judges, data)

    evaluation_result.to_csv('evaluation/llm-as-a-judge/judge_results.csv', index=False)
    
    print("Evaluation complete!")


if __name__ == '__main__':
    main()



