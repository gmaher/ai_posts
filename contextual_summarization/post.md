# Handling Long Context RAG for LLMs with Contextual Summarization

---

Responding to user queries when they require analyzing large contexts or many documents can be challenging. A common approach is to slice the documents into smaller chunks and use a search algorithm to retrieve relevant chunks when answering the user query. However using chunks may leave out relevant context, and it is not always clear which chunks to retrieve based on the user's query. Alternatively whole documents or many chunks can be added to the prompt, but LLMs have token limits, making it difficult or impossible to process long contexts in a single prompt. Additionally the LLM may ignore important context when the prompt becomes long. 

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
    The extracted information and summaries will be used by a downstream LLM to answer the user query, so it is important you extract relevant information.

    For your output only output a string with the relevant information/summary. Do not output anything else.
    """
    ```
    This prompt ensures the LLM understands the specific task—extracting information contextually from each passage based on the user query.

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
    By including a log of previous summaries, the LLM can maintain continuity when summarizing multiple chunks.

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
    The final answer is then generated using the complete summary to address the user's query.

### Testing Contextual Summarization on Earnings Report
To test the code I downloaded Alphabest's Q3 2024 earnings report.
If we now run the code with different queries, we will see that we get detailed answers of our queries about the earnings report (for the generated summaries see the appendix).

For example, using the query `How did the performance compare to last quarter?`, the answer we get is:

```markdown
In comparing Alphabet's performance in the third quarter of 2024 to the previous quarter (second quarter of 2024), several key financial metrics indicate positive growth and improvements:

1. **Revenue Growth**: Alphabet's total constant currency revenues increased by $3.5 billion from the previous quarter, rising from $84.6 billion in Q2 2024 to $88.1 billion in Q3 2024. This represents a 4% increase quarter-over-quarter. Including hedging effects, total revenues were $88.3 billion in Q3 2024.

2. **Geographical Revenue Growth**: The revenue growth was geographically diverse, with the United States and APAC regions each experiencing a 5% increase, EMEA seeing a 2% increase, and Other Americas showing a 7% increase from Q2 2024 to Q3 2024.

3. **Operating Income and Margin**: Operating income rose from $21.3 billion in Q3 2023 to $28.5 billion in Q3 2024, with the operating margin expanding from 28% to 32%. Although specific quarter-over-quarter figures for operating income are not provided, the year-over-year increase suggests continued momentum.

4. **Net Income and Earnings Per Share**: Net income increased significantly to $26.3 billion in Q3 2024 from $19.7 billion in Q3 2023. Diluted earnings per share grew to $2.12 from $1.55 year-over-year, indicating strong profitability. The quarter-over-quarter comparison is not explicitly detailed, but the year-over-year growth reflects a positive trend.

5. **Segment Performance**: Both Google Services and Google Cloud showed substantial growth. Google Services' operating income increased from $23.9 billion to $30.9 billion year-over-year, while Google Cloud's operating income rose from $266 million to $1.9 billion. These segments likely contributed to the quarter-over-quarter revenue increase.

6. **Non-GAAP Measures**: Alphabet's non-GAAP free cash flow for Q3 2024 was $17.637 billion, calculated by subtracting capital expenditures from net cash provided by operating activities. This reflects strong cash generation capabilities, although specific quarter-over-quarter changes are not detailed.

7. **Investment and Financial Activities**: Alphabet experienced a $1.8 billion gain on equity securities, which positively impacted net income and diluted EPS. This gain, along with a reversal of performance fees, increased the provision for income tax, net income, and diluted EPS by $389 million, $1.5 billion, and $0.12, respectively.

Overall, Alphabet's performance in Q3 2024 showed robust growth compared to the previous quarter, driven by increased revenues across regions, improved operating income and margins, and strong contributions from key business segments like Google Services and Google Cloud. The company's strategic focus on AI innovations and efficiency improvements appears to be yielding positive results.
```

But if we use a query like `What is the current main driver of growth?`, we get the following answer:
```markdown
The main driver of growth for Alphabet in Q3 2024 was the substantial increase in revenues, particularly from two key segments: Google Cloud and Google Services. Here's a detailed breakdown of these growth drivers:

1. **Google Cloud**: 
   - Google Cloud experienced a remarkable 35% increase in revenue, rising from $8,411 million to $11,353 million. This growth was primarily fueled by accelerated advancements in AI Infrastructure, Generative AI Solutions, and core Google Cloud Platform (GCP) products. The focus on AI innovation and infrastructure enhancements has positioned Google Cloud as a significant contributor to Alphabet's overall revenue growth.

2. **Google Services**: 
   - Google Services revenue grew significantly, from $67,986 million to $76,510 million, marking a 13% increase. This growth was driven by several factors:
     - **Google Search**: Continued to be a major revenue generator, benefiting from increased user engagement and advertising demand.
     - **YouTube Ads**: Contributed to the revenue increase, reflecting strong performance in video advertising.
     - **Subscriptions, Platforms, and Devices**: Growth in consumer subscription-based products like YouTube TV and Music, as well as sales from platforms and devices, added to the revenue surge.

3. **Operating Income and Margin**:
   - The increase in revenues from both Google Cloud and Google Services translated into higher operating income. Google Services' operating income increased from $23,937 million to $30,856 million, while Google Cloud's operating income rose from $266 million to $1,947 million. This improvement in operating income, along with efficiency enhancements, led to an expanded operating margin, further contributing to Alphabet's growth.

4. **Constant Currency Revenues**:
   - The growth in constant currency revenues across various geographic regions also played a role. Total constant currency revenues increased by $12.6 billion compared to the same period in 2023, reflecting a 16% growth. This metric helps in assessing true revenue growth by excluding the effects of foreign exchange rate movements.

Overall, the main driver of growth for Alphabet in Q3 2024 was the robust increase in revenues, particularly from Google Cloud and Google Services, supported by strategic investments in AI and operational efficiencies.
```

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

## Appendix: Contextual Summaries
Summary when using the query `How did the performance compare to last quarter?`:
```markdown
In the third quarter of 2024, Alphabet's revenues increased by 15% year-over-year to $88.3 billion, with Google Services and Google Cloud showing significant growth. Operating income and net income both rose by 34%, and earnings per share increased by 37% to $2.12. This performance reflects strong momentum across the business, driven by innovations in AI and efficiency improvements.

In the third quarter of 2024, Alphabet's total revenues increased to $88.3 billion from $76.7 billion in the same quarter of 2023. Operating income rose to $28.5 billion from $21.3 billion, and the operating margin expanded from 28% to 32%. Net income increased to $26.3 billion from $19.7 billion, and diluted earnings per share grew to $2.12 from $1.55. Google Services and Google Cloud both showed significant growth, with Google Services' operating income increasing to $30.9 billion from $23.9 billion, and Google Cloud's operating income rising to $1.9 billion from $266 million. The number of employees decreased slightly from 182,381 to 181,269.

In the third quarter of 2024, Alphabet incurred $607 million in office space charges related to optimization efforts. Dividend payments for the quarter totaled $2.5 billion across Class A, B, and C shares. A cash dividend of $0.20 per share was announced for December 2024.

The passage discusses the use of non-GAAP financial measures by Alphabet to supplement their GAAP financial statements. These measures include free cash flow, constant currency revenues, and percentage change in constant currency revenues. Alphabet uses these non-GAAP measures for financial and operational decision-making and to evaluate period-to-period comparisons, as they provide meaningful supplemental information by excluding certain non-recurring items. The passage also notes that forward-looking statements should not be overly relied upon, as they are based on information available as of October 29, 2024.

The passage discusses Alphabet's use of non-GAAP financial measures to provide transparency and facilitate comparisons to historical performance and competitors. These measures help in planning, forecasting, and analyzing future periods, despite having limitations. The passage also includes a snippet of Alphabet's consolidated balance sheets, showing a decrease in cash and cash equivalents from $24,048 million as of December 31, 2023, to $19,959 million as of September 30, 2024, and a decrease in marketable securities from $86,868 million to $73,271 million over the same period.

In the third quarter of 2024, Alphabet's total assets increased to $430,266 million from $402,392 million in the previous year. Total liabilities decreased slightly to $116,147 million from $119,013 million. Stockholders' equity rose to $314,119 million from $283,379 million. Revenues for the quarter increased to $88,268 million from $76,693 million in the same quarter of 2023. Costs and expenses also rose, with the cost of revenues increasing to $36,474 million from $33,229 million, and research and development expenses rising to $12,447 million from $11,258 million.

In the third quarter of 2024, Alphabet's net income increased to $26.3 billion from $19.7 billion in the same quarter of 2023. Basic earnings per share rose to $2.14 from $1.56, and diluted earnings per share increased to $2.12 from $1.55. Operating income grew to $28.5 billion from $21.3 billion. Net cash provided by operating activities remained stable at approximately $30.7 billion, compared to $30.7 billion in the previous year. Investing activities saw increased expenditures, with purchases of property and equipment rising to $13.1 billion from $8.1 billion, and purchases of marketable securities increasing to $22.0 billion from $13.8 billion.

In the third quarter of 2024, Alphabet's total revenues increased to $88.3 billion from $76.7 billion in the same quarter of 2023. Google Services revenues rose from $67.99 billion to $76.51 billion, and Google Cloud revenues increased from $8.41 billion to $11.35 billion. Operating income for Google Services grew from $23.94 billion to $30.86 billion, while Google Cloud's operating income rose significantly from $266 million to $1.95 billion. Other Bets' operating loss decreased slightly from $1.19 billion to $1.12 billion. Overall, total income from operations increased from $21.34 billion to $28.52 billion.

Google Services generates revenue primarily from advertising, consumer subscriptions (like YouTube TV and Google One), and sales of apps and devices. Google Cloud's revenue comes from consumption-based fees and subscriptions for its platform services and collaboration tools. Other Bets' revenue is mainly from healthcare and internet services. Certain costs, such as AI R&D and corporate initiatives, are not allocated to segments. In terms of other income and expenses, interest income increased from $1,066 million in 2023 to $1,243 million in 2024, while interest expense decreased. There was a foreign currency exchange gain of $23 million in 2024 compared to a loss in 2023. Gains on equity securities significantly increased from a loss of $366 million to a gain of $1,821 million.

In the third quarter of 2024, Alphabet reported a net effect of a $1.8 billion gain on equity securities, which, along with a $29 million reversal of previously accrued performance fees, increased the provision for income tax, net income, and diluted EPS by $389 million, $1.5 billion, and $0.12, respectively. The fluctuations in investment values could contribute to future volatility in other income and expenses. Alphabet's non-GAAP free cash flow for the quarter was $17.637 billion, calculated by subtracting $13.061 billion in capital expenditures from $30.698 billion in net cash provided by operating activities. Additionally, Alphabet uses non-GAAP constant currency revenues to exclude the effects of foreign exchange rate movements and hedging activities, facilitating comparisons to historical performance.
The passage provides a comparison of Alphabet's revenues by geography for the quarter ended September 30, 2024, against both the same quarter in 2023 and the previous quarter ended June 30, 2024. 

From Q3 2023 to Q3 2024, total constant currency revenues increased by 16%, with notable growth in the United States (19%), EMEA (13%), APAC (13%), and Other Americas (25%). 

Comparing Q2 2024 to Q3 2024, revenues showed a 5% increase in the United States and APAC, a 2% increase in EMEA, and a 7% increase in Other Americas. Overall, the total constant currency revenues for Q3 2024 were $89.3 billion, reflecting a $12.6 billion increase from the same quarter in the previous year.
In the third quarter of 2024, Alphabet's total constant currency revenues were $88.1 billion, reflecting a $3.5 billion increase from the previous quarter ended June 30, 2024, which had revenues of $84.6 billion excluding hedging effects. This represents a 4% increase quarter-over-quarter. Comparing the third quarter of 2023 to the third quarter of 2024, revenues excluding hedging effects increased from $76.7 billion to $88.1 billion, marking a 15% increase year-over-year. Total revenues, including hedging effects, rose from $76.7 billion in Q3 2023 to $88.3 billion in Q3 2024, an increase of 15%.
```

Summary when using the query `How did the performance compare to last quarter?`:

```markdown
The main driver of growth for Alphabet in Q3 2024 was the significant increase in revenues, particularly from Google Cloud, which saw a 35% rise due to accelerated growth in AI Infrastructure, Generative AI Solutions, and core GCP products. Additionally, Google Services revenues grew by 13%, driven by Google Search, subscriptions, platforms, devices, and YouTube ads. The company's commitment to AI innovation and efficiency improvements also contributed to the growth, as reflected in the increased operating income and expanded operating margin.

The main driver of growth in this passage is the increase in revenues across various segments, particularly Google Services and Google Cloud. Google Services revenue grew from $67,986 million to $76,510 million, driven by Google Search, YouTube ads, and subscriptions, platforms, and devices. Google Cloud revenue increased from $8,411 million to $11,353 million. The operating income for Google Services and Google Cloud also saw significant increases, contributing to the overall growth in operating income and operating margin.

The passage primarily discusses Alphabet's financial activities for the quarter ended September 30, 2024, focusing on charges related to employee severance and office space optimization, which amounted to $607 million. Additionally, it highlights the dividend payments totaling $2.5 billion to stockholders of Class A, B, and C shares. The passage does not directly address the main driver of growth but provides context on financial management and shareholder returns.

The passage discusses the use of non-GAAP financial measures by Alphabet to supplement their GAAP financial statements. These measures, including free cash flow and constant currency revenues, are used for financial and operational decision-making and to evaluate period-to-period comparisons. They provide additional insights into performance and liquidity by excluding certain items that may not reflect the core business operating results. The passage does not directly address the main driver of growth.

The passage discusses the use of non-GAAP financial measures by Alphabet to provide greater transparency and facilitate internal and external performance assessments. These measures help management and investors analyze the company's health by excluding certain items that may not reflect core business operations. The passage does not directly address the main driver of growth.

The passage provides a detailed breakdown of Alphabet Inc.'s financial position, including assets, liabilities, and stockholders' equity, as well as a summary of revenues and expenses. The main driver of growth indicated in this passage is the increase in revenues, which rose from $76,693 million to $88,268 million for the quarter ended September 30, and from $221,084 million to $253,549 million year-to-date. This growth in revenue suggests a positive financial performance, although the passage does not specify which segments contributed most to this increase.

The main driver of growth in this passage is the increase in net income, which rose from $19,689 million to $26,301 million for the quarter ended September 30, and from $53,108 million to $73,582 million year-to-date. This growth in net income is supported by an increase in income from operations, which grew from $21,343 million to $28,521 million for the quarter, and from $60,596 million to $81,418 million year-to-date. Additionally, the net cash provided by operating activities increased slightly from $82,831 million to $86,186 million year-to-date, indicating strong cash flow management.

The main driver of growth in this passage is the increase in revenues, particularly from Google Services and Google Cloud. Google Services revenue rose from $67,986 million to $76,510 million, and Google Cloud revenue increased from $8,411 million to $11,353 million for the quarter ended September 30. This growth in revenue is also reflected in the operating income, with Google Services' operating income increasing from $23,937 million to $30,856 million, and Google Cloud's operating income rising from $266 million to $1,947 million. Overall, total revenues increased from $76,693 million to $88,268 million, and total income from operations grew from $21,343 million to $28,521 million.

The main driver of growth in this passage is the revenue generated by Google Services and Google Cloud. Google Services primarily earns from advertising, consumer subscription-based products like YouTube TV and Music, app sales, and devices. Google Cloud generates revenue from consumption-based fees and subscriptions for its platform services and collaboration tools. Additionally, the passage mentions that certain costs, such as AI-focused R&D and corporate initiatives, are not allocated to specific segments, which may impact the overall financial assessment. The passage also provides a breakdown of other income and expenses, highlighting gains and losses from interest, foreign currency exchange, and securities.

The passage discusses financial metrics related to Alphabet's performance, focusing on other income and expenses, free cash flow, and constant currency revenues. It highlights a net effect from gains on equity securities and a reversal of performance fees, which increased income tax provisions, net income, and diluted EPS. The passage also explains the reconciliation from GAAP net cash provided by operating activities to non-GAAP free cash flow, which amounted to $17,637 million for the quarter ended September 30, 2024. Additionally, it describes the use of non-GAAP constant currency revenues to exclude foreign exchange effects for better comparison with historical performance. The passage does not directly address the main driver of growth.

The main driver of growth in this passage is the increase in constant currency revenues across various geographic regions. For the quarter ended September 30, 2024, total constant currency revenues increased by $12.6 billion compared to the same period in 2023, reflecting a 16% growth. This growth was observed across all regions, with the United States showing a 19% increase, EMEA a 13% increase, APAC a 13% increase, and Other Americas a 25% increase. The passage highlights the impact of foreign exchange rate movements and hedging activities on reported revenues, emphasizing the importance of constant currency measures for assessing true revenue growth.

The main driver of growth in this passage is the increase in constant currency revenues. For the quarter ended September 30, 2024, total constant currency revenues reached $88.1 billion, marking a $3.5 billion increase compared to the previous quarter. This growth highlights the impact of constant currency measures in assessing revenue performance, excluding the effects of foreign exchange and hedging. Additionally, when comparing the quarter ended September 30, 2023, to the same period in 2022, revenues excluding hedging effects increased by 12%, further emphasizing the role of constant currency revenue growth as a key driver.
```
