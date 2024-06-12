# %% Imports
import re
from collections import Counter
from statistics import mean
import evaluate
import ollama
import pandas as pd
from tqdm import tqdm

# %% Metrics
exact_match = evaluate.load("exact_match")


def calculate_metrics(prediction, ground_truth):
    prediction_tokens = re.findall(r"\w+", prediction.lower())
    ground_truth_tokens = re.findall(r"\w+", ground_truth.lower())

    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)

    num_common_tokens = sum(common_tokens.values())

    if len(prediction_tokens) == 0:
        precision = 0.0
    else:
        precision = num_common_tokens / len(prediction_tokens)

    if len(ground_truth_tokens) == 0:
        recall = 0.0
    else:
        recall = num_common_tokens / len(ground_truth_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


# %% Prepare llama3 with temperature=0.5
modelfile = """
FROM llama3
PARAMETER temperature 0.5
"""

ollama.create(model="llama3-temp0.5", modelfile=modelfile)

# %% Load test data
df = pd.read_json("dataset/data/dev.json").head(1200)


# %% Make prompt for a given question/contexts pair
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


# %% Make predictions
results = []
for row in tqdm(df.itertuples()):
    prompt = make_prompt(row.question, row.context)
    prediction = ollama.generate(model="llama3-temp0.5", prompt=prompt)
    results.append((row.question, row.answer, prediction["response"]))

res_df = pd.DataFrame(results, columns=("question", "answer", "prediction"))
res_df.to_csv("oracle_prediction_results.csv")
res_df

# %% Compute metrics
[_, references, predictions] = list(zip(*results))

calculated_metrics = list(
    map(lambda args: calculate_metrics(*args), zip(predictions, references))
)
precisions, recalls, f1s = zip(*calculated_metrics)

avg_exact_match = exact_match.compute(
    predictions=predictions,
    references=references,
    ignore_case=True,
    ignore_punctuation=True,
)

avg_precision = mean(precisions)
avg_recall = mean(recalls)
avg_f1 = mean(f1s)

# %%
metrics = {
    "avg_exact_match": avg_exact_match["exact_match"],
    "avg_precision": avg_precision,
    "avg_recall": avg_recall,
    "avg_f1": avg_f1,
}

with open("oracle_prediction_metrics.json", "w") as fp:
    import json

    json.dump(metrics, fp)

metrics
