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

query = "What is the main driver of growth?"

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