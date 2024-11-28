# Metadata Tagging with LLMs to Improve RAG Document Search and Retrieval

Being able to retrieve the right documents for a given query is important when building any kind of RAG application. If retrieval performance is poor, then all downstream generation steps will not produce good results. Typically, documents are retrieved by running a search algorithm such as BM25 or using vector search based on the input query. However, this does not work well if the query is missing keywords, or if the embedding computed from the query is not similar to the embeddings of relevant documents.

One way to improve the retrieval of relevant documents is by adding metadata at the document ingestion phase. This metadata increases the likelihood that a query will match relevant documents when using search algorithms. In this blog post, we'll explore this approach using language models for metadata tagging.

## Approach to Metadata Tagging

To do metadata tagging we need to decide what metadata to extract and how to extract it. The best metadata to extract will depend on the type of queries you expect to receive. For example, for a financial application users may often submit queries that require key financial numbers to answer. Consider e.g. revenue and cost numbers from an earnings report, queried based on company ticker or sector. Having this information in the metadata for each document would make it easier to surface relevant documents.

Next we need to determine how to extract the metadata for an arbitrary document. There are many classical NLP algorithms for extracting various types of data, but these tend to be focused on only extracting specific categories such as entities or locations. A more flexible approach is just to ask an LLM to extract the metadata, this way we can just specify the metadata we want without having to worry about the algorithm (although we would need to test the accuracy).

In this article we create an example application of that uses LLMs to extract various kinds of metadata from news articles, and compares search performance with and without tags.

## Code Example

Here is the code we will use. At a high level we load the dataset, tag the documents using the LLM and then, given a query, search for relevant documents.

```python
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
```

## Code Explanation

Here is a step-by-step explanation of the code.

1. **Load Dataset**: It loads a dataset using `load_dataset("okite97/news-data")`, and extracts titles and texts from it.

2. **Prepare Documents**: Loads a news data set using the Huggingface `datasets` library

3. **Set API Key**: The OpenAI API key is set using an environment variable.

4. **Define LLM Tagging Function**: The `llm_tagging` function sends each document to the OpenAI API to retrieve entities, keywords, topics, and categories. It handles JSON decoding errors by providing default empty lists.

5. **Define BM25 Search Function**: The `bm25_search` function uses the BM25 algorithm to search for documents most relevant to a given query.

6. **Define Vector Search Function**: The `vector_search` function uses cosine similarity on embeddings from the SentenceTransformer model to search for relevant documents.

7. **Tag Documents**: Calls the `llm_tagging` function to obtain metadata (entities, keywords, topics, categories) for each document.

8. **Stringify Tags**: Converts the metadata tags to strings by concatenating lists into single strings. Special characters need to be removed as they interfere with the keyword search.

9. **Define Query**: Sets the search query as "Football news".

10. **Perform BM25 Searches**:
    - **Without Tags**: Executes a BM25 search on the original documents using the query.
    - **With Tags**: Executes a BM25 search on the stringified tags using the query.

11. **Generate Embeddings**:
    - Computes embeddings for the original documents.
    - Computes embeddings for the stringified tags.
    - Computes an embedding for the query.

12. **Perform Vector Searches**:
    - **Without Tags**: Conducts a vector search using document embeddings.
    - **With Tags**: Conducts a vector search using the embeddings of stringified tags.

13. **Print Results**:
    - Outputs BM25 search results for both the documents and the tags.
    - Outputs vector search results for both the documents and the tags, displaying document indices, scores, and content.

## Improvement through Metadata Tagging

The results from running the code with the query `Football news` are:

```plaintext
BM25 NO TAGS
Document index: 19, Score: 0.0
200,000 Undetected Tuberculosis Infections Occur Annually in Nigeria, Says FG Nigeria's federal government on Thursday said despite significant progress made in the fight against tuberculosis (TB) epidemic in the country,
---

Document index: 18, Score: 0.0
Nigeria’s Senate Intent on Passing Oil Reform Bill The Nigerian Senate has fixed October 20 to commence debate on the Petroleum Industry Bill (PIB), a move the Senate
---

Document index: 1, Score: 0.0
Amazon Blames Inflation as It Increases Cost of Prime in Europe The increases are steeper than the 17 percent jump in the Prime membership price that came into effect for new
---

Document index: 2, Score: 0.0
Nigeria’s Parliament Passes Amended Electoral Bill Nigeria's Senate on Tuesday passed the harmonised Clause 84 of the 2010 Electoral Act (Amendment) Bill 2022, which allows political
---

Document index: 3, Score: 0.0
Nigeria: Lagos Governor Tests Positive for Covid-19, Kaduna Governor Self-Isolates The Lagos State Governor, Mr. Babajide Sanwo-Olu, has tested positive for Covid-19, Commissioner for Health, Professor Akin Abayomi, has said.
---

BM25 TAGS
Document index: 9, Score: 1.5775930589568545
document Premier League Clubs Reject ‘Project Big Picture’ Premier League clubs have "unanimously agreed" that \Project Big Picture\ will not be endorsed or pursued. The controversial plans proposed entities Premier League Clubs Project Big Picture keywords Reject unanimously agreed endorsed pursued controversial plans proposed topics Premier League Project Big Picture Football Sports Management categories Sports Football Club Management
---

Document index: 5, Score: 1.5775930589568545
document Guardiola To Leave Man City When Contract Expires in 2023 Pep Guardiola has said that he will leave Manchester City when his contract runs out in 2023 - and hopes entities Guardiola Man City Manchester City Contract Expires 2023 keywords leave contract expires 2023 Manchester City Guardiola topics Football Contract Expiration Managerial Changes categories Sports Football Contracts
---

Document index: 11, Score: 1.5553293057727469
document Old Trafford Modified for 23500 Socially Distanced Fans Manchester United have modified Old Trafford to accommodate 23500 socially distanced spectators and insists they are "bemused" by the ongoing entities Old Trafford 23500 Socially Distanced Fans Manchester United keywords Old Trafford modified 23500 socially distanced spectators Manchester United bemused ongoing topics Football COVID-19 precautions Stadium modifications categories Sports Football Public Health
---

Document index: 0, Score: 1.4989200220654944
document Uefa Opens Proceedings against Barcelona Juventus and Real Madrid Over European Super League Plan Uefa has opened disciplinary proceedings against Barcelona Juventus and Real Madrid over their involvement in the proposed European Super League. entities Uefa Barcelona Juventus Real Madrid European Super League keywords Uefa proceedings Barcelona Juventus Real Madrid European Super League disciplinary involvement proposed topics Football Disciplinary Proceedings European Super League categories Sports Football Legal Proceedings
---

Document index: 13, Score: 1.421582279669606
document Brilliant Brentford Beat Arsenal After 75-year Exile From England’s Top Flight Brentford got the better of Arsenal in the first match of the 2021-22 Premier League season at the Community Stadium entities Brentford Arsenal 75-year Exile England’s Top Flight first match 2021-22 Premier League season Community Stadium keywords Brentford Arsenal 75-year Exile England’s Top Flight first match 2021-22 Premier League season Community Stadium beat topics Football Premier League Brentford vs Arsenal 2021-22 Premier League season categories Sports Football Premier League
---

VECTOR NO TAGS
Document index: 13, Score: 0.29005610942840576
Brilliant Brentford Beat Arsenal After 75-year Exile From England’s Top Flight Brentford got the better of Arsenal in the first match of the 2021-22 Premier League season at the Community Stadium
---

Document index: 11, Score: 0.2587330639362335
Old Trafford Modified for 23,500 Socially Distanced Fans Manchester United have modified Old Trafford to accommodate 23,500 socially distanced spectators, and insists they are "bemused" by the ongoing
---

Document index: 0, Score: 0.2265966683626175
Uefa Opens Proceedings against Barcelona, Juventus and Real Madrid Over European Super League Plan Uefa has opened disciplinary proceedings against Barcelona, Juventus and Real Madrid over their involvement in the proposed European Super League.
---

Document index: 9, Score: 0.17919565737247467
Premier League Clubs Reject ‘Project Big Picture’ Premier League clubs have "unanimously agreed" that 'Project Big Picture' will not be endorsed or pursued. The controversial plans, proposed
---

Document index: 5, Score: 0.16245952248573303
Guardiola To Leave Man City When Contract Expires in 2023 Pep Guardiola has said that he will leave Manchester City when his contract runs out in 2023 - and hopes
---

VECTOR TAGS
Document index: 11, Score: 0.3497103452682495
document Old Trafford Modified for 23500 Socially Distanced Fans Manchester United have modified Old Trafford to accommodate 23500 socially distanced spectators and insists they are "bemused" by the ongoing entities Old Trafford 23500 Socially Distanced Fans Manchester United keywords Old Trafford modified 23500 socially distanced spectators Manchester United bemused ongoing topics Football COVID-19 precautions Stadium modifications categories Sports Football Public Health
---

Document index: 13, Score: 0.3287414610385895
document Brilliant Brentford Beat Arsenal After 75-year Exile From England’s Top Flight Brentford got the better of Arsenal in the first match of the 2021-22 Premier League season at the Community Stadium entities Brentford Arsenal 75-year Exile England’s Top Flight first match 2021-22 Premier League season Community Stadium keywords Brentford Arsenal 75-year Exile England’s Top Flight first match 2021-22 Premier League season Community Stadium beat topics Football Premier League Brentford vs Arsenal 2021-22 Premier League season categories Sports Football Premier League
---

Document index: 0, Score: 0.2898913323879242
document Uefa Opens Proceedings against Barcelona Juventus and Real Madrid Over European Super League Plan Uefa has opened disciplinary proceedings against Barcelona Juventus and Real Madrid over their involvement in the proposed European Super League. entities Uefa Barcelona Juventus Real Madrid European Super League keywords Uefa proceedings Barcelona Juventus Real Madrid European Super League disciplinary involvement proposed topics Football Disciplinary Proceedings European Super League categories Sports Football Legal Proceedings
---

Document index: 9, Score: 0.28569144010543823
document Premier League Clubs Reject ‘Project Big Picture’ Premier League clubs have "unanimously agreed" that \Project Big Picture\ will not be endorsed or pursued. The controversial plans proposed entities Premier League Clubs Project Big Picture keywords Reject unanimously agreed endorsed pursued controversial plans proposed topics Premier League Project Big Picture Football Sports Management categories Sports Football Club Management
---

Document index: 5, Score: 0.2489248812198639
document Guardiola To Leave Man City When Contract Expires in 2023 Pep Guardiola has said that he will leave Manchester City when his contract runs out in 2023 - and hopes entities Guardiola Man City Manchester City Contract Expires 2023 keywords leave contract expires 2023 Manchester City Guardiola topics Football Contract Expiration Managerial Changes categories Sports Football Contracts
---
```

* The BM25 search without tags performed very poorly, no relevant documents were surfaced. We see that keyword search can fail if a query requires more broad information to answer such as topics or categories.

* Adding the metadata tags to the documents drastically improved the BM25 search, we now correctly get articles related to football, even though the football keyword does not appear in the documents.

* A vector search without tags surface some relevant documents, however adding the tags improves the similarity scores and reorders the documents. This can become increasingly important the larger the document corpus.


## Applications and Conclusion

Metadata tagging is a simple but powerful way to improve document search results in RAG applications. Particularly using an LLM for metadata tagging creates a flexible approach that can be customized to the application at hand.

The method is broadly applicable but some applications could be academia, legal research, and enterprise document management, customer support and knowledge bases. Here metadata-driven search can improve response times and the accuracy of RAG.