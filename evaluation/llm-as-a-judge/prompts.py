from langchain_core.prompts import ChatPromptTemplate

prompt_factual = ChatPromptTemplate.from_messages([
    ("system", 
     """
     You are an expert in evaluating factual consistency in RAG systems. 
     Assess to what extent all statements in the answer are supported by the provided context. 
     If there is information not present in the chunks, this lowers the score. Use a 5-point scale (1 — very poor, 5 — excellent). 
     Answer ONLY with a number and do not write anything extra"""),
    ("human", 
     """
     Question: {query}
     Context (chunks): {context}
     Generated answer: {answer} 
     Please rate the factual consistency on a scale from 1 to 5 and explain your choice.""")
])

prompt_citation = ChatPromptTemplate.from_messages([
    ("system", 
     """
     You are an expert in evaluating citation accuracy in RAG systems. 
     Assess whether all statements that require support are provided with correct citations to the relevant chunks. 
     Are there any citations to irrelevant parts? Use a 5-point scale (1 — very poor, 5 — excellent). 
     Answer ONLY with a number and do not write anything extra"""),
    ("human", 
     """
     Question: {query}
     Context (chunks): {context}
     Generated answer: {answer} 
     Please rate the citation accuracy on a scale from 1 to 5 and explain your choice.""")
])


prompt_coverage = ChatPromptTemplate.from_messages([
    ("system", 
     """
     You are an expert in evaluating coverage in RAG systems. 
     Assess how fully and accurately the relevant information from the provided chunks is used in the answer. 
     Is any important information missing? Use a 5-point scale (1 — very poor, 5 — excellent). 
     Answer ONLY with a number and do not write anything extra"""),
    ("human", 
     """
     Question: {query}
     Context (chunks): {context} 
     Generated answer: {answer} 
     Please rate the coverage on a scale from 1 to 5 and explain your choice.""")
])


prompt_relevance = ChatPromptTemplate.from_messages([
    ("system", 
     """
     You are an expert in evaluating relevance and coherence in RAG systems. 
     Assess how well the answer addresses the original question and how logically it is structured. 
     Are there any deviations from the topic or logical gaps? Use a 5-point scale (1 — very poor, 5 — excellent). 
     Answer ONLY with a number and do not write anything extra"""),
    ("human", 
     """
     Question: {query}
     Context (chunks): {context}
     Generated answer: {answer} 
     Please rate the relevance and coherence on a scale from 1 to 5 and explain your choice.""")
])


prompt_paraphrase = ChatPromptTemplate.from_messages([
    ("system", 
     """
     You are an expert in evaluating paraphrasing in RAG systems. 
     Assess how well the system paraphrases information from the chunks instead of simply copying them verbatim. 
     Use a 5-point scale (1 — very poor, 5 — excellent). Answer ONLY with a number and do not write anything extra"""),
    ("human", 
     """
     Question: {query}
     Context (chunks): {context} 
     Generated answer:\n{answer} 
     Please rate the paraphrasing and lack of copy-paste on a scale from 1 to 5 and explain your choice.""")
])
