# %%
import pandas as pd
import ollama
import evaluate
from tqdm import tqdm

# %%
exact_match = evaluate.load("exact_match")

# %%
modelfile = """
FROM llama3
PARAMETER temperature 0.5
"""

ollama.create(model="llama3-temp0.5", modelfile=modelfile)

# %%
df = pd.read_json("dataset/data/dev.json").head(1200)


# %%
def make_prompt(question, contexts):
    return "\n\n".join(
        (
            "Please answer the given question based on the given contexts below.",
            *[
                f"Context {i}: {' '.join(texts)}"
                for i, (_, texts) in enumerate(contexts, start=1)
            ],
            f"Question: {question}",
            "Constraint: Don't give any explanations and use MAX 5 tokens in your response. No yapping.",
        )
    )


# %%
results = []
for row in tqdm(df.head(5).itertuples()):
    prompt = make_prompt(row.question, row.context)
    prediction = ollama.generate(model="llama3-temp0.5", prompt=prompt)
    results.append((row.question, row.answer, prediction["response"]))

pd.DataFrame(results, columns=("question", "answer", "prediction"))

# %%
[_, references, predictions] = list(zip(*results))
exact_match.compute(
    predictions=predictions,
    references=references,
    ignore_case=True,
    ignore_punctuation=True,
)

# %%
