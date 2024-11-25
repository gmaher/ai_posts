# Import necessary libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Sample documents (simulating Web of Science abstracts)
documents = [
    "This paper explores the use of machine learning algorithms in predicting stock market trends.",
    "The study investigates the impact of climate change on global agricultural production.",
    "An analysis of quantum computing and its potential applications in cryptography.",
    "A review of recent advancements in renewable energy technologies.",
    "This research focuses on the social and economic effects of the COVID-19 pandemic.",
    # Add more documents as needed
]

# Example query
query = "machine learning applications in finance"

# Function to get embeddings using SentenceTransformer
def get_local_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings

# Get embeddings for documents and query using SentenceTransformer
local_doc_embeddings = get_local_embeddings(documents)
local_query_embedding = get_local_embeddings([query])[0]

# Function to compute cosine similarities
def compute_similarities(query_embedding, doc_embeddings):
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    return similarities

# Compute similarities using local embeddings
local_similarities = compute_similarities(local_query_embedding, local_doc_embeddings)
local_rankings = np.argsort(-local_similarities)

# Print top documents retrieved using SentenceTransformer embeddings
print("Top documents retrieved using SentenceTransformer embeddings:")
for idx in local_rankings[:3]:
    print(f"Similarity Score: {local_similarities[idx]:.4f}")
    print(f"Document: {documents[idx]}")
    print("-----")