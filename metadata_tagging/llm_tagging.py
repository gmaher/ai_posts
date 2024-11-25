# Import necessary libraries
import openai
import json
import os

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
            model="gpt-4o-mini",  # Use "gpt-4" if available
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0
        )
        # Parse the response
        content = response.choices[0].message.content
        print(content)
        try:
            tags = json.loads(content)
        except json.JSONDecodeError:
            # Handle cases where ChatGPT does not return valid JSON
            tags = {'entities': [], 'keywords': [], 'topics': [], 'categories': []}
        
        chatgpt_tags.append({
            'document': doc,
            'entities': tags.get('entities', []),
            'keywords': tags.get('keywords', []),
            'topics': tags.get('topics', []),
            'categories': tags.get('categories', [])
        })

    return chatgpt_tags

# Example usage
documents = [
    "This paper explores the use of machine learning algorithms in predicting stock market trends.",
    "The study investigates the impact of climate change on global agricultural production.",
    "An analysis of quantum computing and its potential applications in cryptography.",
    "A review of recent advancements in renewable energy technologies.",
    "This research focuses on the social and economic effects of the COVID-19 pandemic.",
]
chatgpt_tags = llm_tagging(documents)