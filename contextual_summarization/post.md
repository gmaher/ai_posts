**Title:** Handling Long Context RAG with Contextual Summarization in LLMs

*By [Your Name]*

---

Handling user queries over large contexts or many documents can be challenging. A common approach is to slice the documents into smaller chunks and use a search algorithm to retrieve relevant chunks when answering the user query. However using chunks may leave out relevant context, and it is not always clear which chunks to retrieve based on the user's query. Alternatively whole documents or many chunks can be added to the prompt, but LLMs have token limits, making it difficult or impossible to process long contexts in a single prompt. Additionally the LLM may ignore important context when the prompt becomes long. 

So what can we do to improve our ability to handle user queries that involve large amounts of context? One solution is to use contextual summarization. With contextual summarization we use an LLM to summarize or extract relevant information from the documents, and then use the summaries as context when answering the user query. Importantly the summarization LLM is also given the user query which ensures it produces summaries that with relevant information. This is why the technique is called contextual summarization vs just creating a regular summary.

In this blog post I show how contextual summarization can enhance query answering performance when dealing with long contexts or extensive document collections. As an example I implement contextual summarization in python using OpenAI and use it to answer queries on one of Alphabet's earnings reports.

#### Code Example

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ['OPENAI_KEY'])

system_prompt = """
You are a summarization AI.
You will be given a user query and a sequence of text chunks.
Your task is to extract relevant information and summarize each text chunk contextually based on the user query.
The extracted information and summaries will be used by a downstream LLM to answer the user query, so it is important you extract relevant information.

For your output only output a string with the relevant information/summary. Do not output anything else.
"""

instruction_template = """
The user query is:
{query}

The log of previous summaries is:
{summaries}

Now extract information from and summarize passage {page_number}/{total}:
{passage}
"""

# Read the full text of Shakespeare's works
full_text = open("./data/google.txt").read()

# Define the chunk size and split the text
chunk_size = 2000  # Adjusted for demonstration purposes
chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

query = "How did the performance compare to last quarter?"

summaries = []

def call_llm(system_prompt, instruction):
    response = client.chat.completions.create(
        model="gpt-4o",  # Use "gpt-4" if you have access
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ],
        max_tokens=2000,  # Adjust as needed
        temperature=0.2,
    )
    return response.choices[0].message.content

for i, chunk in enumerate(chunks):
    print(i,len(chunks))
    summary_string = "\n".join(summaries)
    instruction = instruction_template.format(
        query=query,
        summaries=summary_string,
        page_number=(i + 1),
        total=len(chunks),
        passage=chunk
    )
    # Call the LLM to get the summary
    summary = call_llm(system_prompt, instruction)
    summaries.append(summary)
    print(f"Processed chunk {i + 1}/{len(chunks)}")

# Combine all summaries
final_summary = "\n".join(summaries)

f = open("summary.txt",'w')
f.write(final_summary)
f.close()

# Now, use the final summary to answer the user's query
answer_prompt = f"""
Based on the following summaries, answer the user's query in detail:

User Query:
{query}

Summaries:
{final_summary}
"""

# Get the final answer from the LLM
final_answer = call_llm(system_prompt="", instruction=answer_prompt)
print("Final Answer:")
print(final_answer)

f = open("answer.txt",'w')
f.write(final_answer)
f.close()
```

### Step-by-Step Explanation

Let's break down the code step by step to understand how contextual summarization is implemented:

1. **Environment Setup**: The script starts by importing required modules and setting up an API client for OpenAI:
    ```python
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.environ['OPENAI_KEY'])
    ```
    Here, the OpenAI API key is fetched from the environment, allowing secure access to the LLM services.

2. **System Prompt Definition**: The `system_prompt` is defined to instruct the LLM on its role:
    ```python
    system_prompt = """
    You are a summarization AI.
    You will be given a user query and a sequence of text chunks.
    Your task is to extract relevant information and summarize each text chunk contextually based on the user query.
    For your output only output a string with the relevant information/summary. Do not output anything else.
    """
    ```
    This prompt ensures the LLM understands the specific task—extracting information contextually for each query.

3. **Instruction Template**: The instruction template defines the instructions given to the LLM for each chunk of text:
    ```python
    instruction_template = """
    The user query is:
    {query}

    The log of previous summaries is:
    {summaries}

    Now extract information from and summarize passage {page_number}/{total}:
    {passage}
    """
    ```
    By including a log of previous summaries, the LLM can maintain continuity across multiple chunks.

4. **Splitting the Text into Chunks**: The document is split into chunks, each containing approximately `2000` characters:
    ```python
    full_text = open("./data/google.txt").read()
    chunk_size = 2000
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    ```
    This approach helps overcome token limitations, as the entire document cannot be processed in a single request.

5. **Looping through Chunks and Summarizing**: The code processes each chunk one at a time, contextualizing the summarization:
    ```python
    for i, chunk in enumerate(chunks):
        summary_string = "\n".join(summaries)
        instruction = instruction_template.format(
            query=query,
            summaries=summary_string,
            page_number=(i + 1),
            total=len(chunks),
            passage=chunk
        )
        summary = call_llm(system_prompt, instruction)
        summaries.append(summary)
        print(f"Processed chunk {i + 1}/{len(chunks)}")
    ```
    Each chunk is processed in relation to the user query and prior summaries, allowing the LLM to keep track of important context.

6. **Final Summarization and Query Answering**: After summarizing all chunks, the summaries are combined to create a final summary:
    ```python
    final_summary = "\n".join(summaries)
    answer_prompt = f"""
    Based on the following summaries, answer the user's query in detail:
    User Query:
    {query}
    Summaries:
    {final_summary}
    """
    ```
    The final answer is then generated using the complete summary to address the user's query effectively.

### Contextual Summarization vs Traditional Retrieval-Augmented Generation (RAG)

- **Contextual Summarization**: The key advantage of contextual summarization is that it extracts information in direct relation to the user's query. This ensures that the generated summaries are directly relevant, reducing the chances of missing crucial information hidden across different chunks. However, summarizing all chunks requires processing more data, which can be resource-intensive and time-consuming.

- **Traditional RAG**: In traditional RAG, the document is split into smaller chunks, and only the chunks most relevant to the user's query are retrieved. While this reduces processing costs and allows more efficient handling of large corpora, it can potentially miss important context when the query is ambiguous or when relevance isn't captured well by the retrieval model. RAG is generally faster, but contextual gaps might occur.

Overall, contextual summarization tends to provide more comprehensive and query-specific responses, while traditional RAG offers speed and simplicity.

### Applications of Contextual Summarization

1. **Customer Support**: Summarizing long support tickets or chat histories to quickly get the context relevant to a specific customer query.

2. **Medical Records Analysis**: When a doctor has to make a decision based on a patient's extensive medical history, contextual summarization could highlight the most pertinent medical notes based on a particular health concern.

3. **Legal Documents**: Extracting relevant clauses or details from lengthy contracts based on specific user questions, helping lawyers quickly understand potential issues or opportunities.

4. **Research Papers**: Summarizing relevant information from multiple research papers for scientists working on a specific hypothesis, reducing the time spent reading unrelated content.

5. **Financial Reports**: Extracting key insights related to performance, trends, or growth across quarterly or annual reports, particularly when comparing current performance to prior periods.

