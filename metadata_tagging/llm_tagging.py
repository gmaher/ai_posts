import openai
import json
import os
from rank_bm25 import BM25Okapi
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import numpy as np

MAX_DOCS = 20

ds = load_dataset("okite97/news-data")
titles = ds['train']['Title']
texts = ds['train']['Excerpt']
documents = [t for t in zip(titles, texts) if t[0] is not None and t[1] is not None]
documents = [t[0] + ' ' + t[1] for t in documents][:MAX_DOCS]

openai.api_key = os.environ['OPENAI_KEY']

def llm_tagging(documents):
    chatgpt_tags = []
    for doc in documents:
        prompt = (
            f"Please provide the entities, keywords, topics, and categories for the following document:\n\n"
            f"{doc}\n\n"
            "Provide the answer as a JSON object with keys 'entities', 'keywords', 'topics', 'categories'."
            "Only output the JSON object, do not use ```json tags."
        )
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0
        )
        content = response.choices[0].message.content
        try:
            tags = json.loads(content)
        except json.JSONDecodeError:
            tags = {'entities': [], 'keywords': [], 'topics': [], 'categories': []}
        
        chatgpt_tags.append({
            'document': doc,
            'entities': tags.get('entities', []),
            'keywords': tags.get('keywords', []),
            'topics': tags.get('topics', []),
            'categories': tags.get('categories', [])
        })

    return chatgpt_tags

def bm25_search(documents, query, k=5):
    tokenized_documents = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_documents)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = scores.argsort()[-k:][::-1]
    return [(index, documents[index], scores[index]) for index in top_indices]

def vector_search(model, document_embeddings, query_embedding, k=5):
    scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0].cpu()
    top_indices = np.argsort(-scores)[:k]
    return [(index, scores[index]) for index in top_indices]

chatgpt_tags = llm_tagging(documents)
stringified_tags = [str(s) for s in chatgpt_tags]
remove_chars = ["{", "}", ",", "'", ":", "[", "]"]
stringified_tags = ["".join([c for c in s if c not in remove_chars]) for s in stringified_tags]

query = "Football news"

bm25_results_docs = bm25_search(documents, query)
bm25_results_tags = bm25_search(stringified_tags, query)

model = SentenceTransformer('all-MiniLM-L6-v2')

doc_embeddings = model.encode(documents, convert_to_tensor=True)
tag_embeddings = model.encode(stringified_tags, convert_to_tensor=True)
query_embedding = model.encode(query, convert_to_tensor=True)

vector_results_docs = vector_search(model, doc_embeddings, query_embedding)
vector_results_tags = vector_search(model, tag_embeddings, query_embedding)

print("BM25 NO TAGS")
for index, doc, score in bm25_results_docs:
    print(f"Document index: {index}, Score: {score}\n{doc}\n---\n")

print("BM25 TAGS")
for index, doc, score in bm25_results_tags:
    print(f"Document index: {index}, Score: {score}\n{doc}\n---\n")

print("VECTOR NO TAGS")
for index, score in vector_results_docs:
    print(f"Document index: {index}, Score: {score}\n{documents[index]}\n---\n")

print("VECTOR TAGS")
for index, score in vector_results_tags:
    print(f"Document index: {index}, Score: {score}\n{stringified_tags[index]}\n---\n")