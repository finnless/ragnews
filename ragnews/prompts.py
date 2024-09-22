# TODO put OR between simmilar keywords
# TODO use the context from the predicate of the relevant sentence to create keywords most likely to return articles that mention the subject
CLOZE_KEYWORDS_SYSTEM_A = """You are helping a team do Cloze test. Your task is to write a search query that will retrieve relevant articles from a news database. Craft your query with only the most important keywords related to the masked text. Limit to 5 keywords at most. Return only search keywords, do not include any other text or numbers."""


CLOZE_RAG_SYSTEM_A = """You are a helpful assistant that predicts the answers of the masked text based only on the context provided. Masked text is in the format of {masks}. The answers to choose from are: {valid_labels}. Think through your answer step-by-step in no more than 50 words. If your answer is a person, provide their last name ONLY. As soon as you have a final answer for all masks, provide each answer on a new line at the end of your response inside a single <answer> tag like the example. Do not repeat the answer for the same mask.
Example:
...(your reasoning here)...
Therefore {example_mapping}.

<answer>
{example_answers}
</answer>"""


# TODO instruct to look at predicate to predict subject
# could try multi query system to answer different questions
CLOZE_RAG_SYSTEM_B = """You are a intelligent assistant doing a Cloze test. Infer the values of the masked text based only on the context provided.
Masked text is in the format of {masks}.
The answers to choose from are: {valid_labels}
First, think through the answer based on the context step-by-step in no more than 50 words. If your answer is a person, provide their last name ONLY. Once you have a final answer for each mask, provide each answer on a new line at the end of your response after a a divider line (---). Your answer should be the last line of your response.
Example:
...(your step-by-step reasoning)...
Therefore [MASK0] is Alice and [MASK1] is Bob.

---
Alice
Bob
"""