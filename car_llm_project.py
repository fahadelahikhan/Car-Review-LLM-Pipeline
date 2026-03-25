# Import necessary packages
import warnings
import pandas as pd
import torch
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import evaluate

from transformers import pipeline, logging
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")
logging.set_verbosity(logging.WARNING)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# DATA LOADING

df = pd.read_csv("data/car_reviews.csv", sep=";", encoding="utf-8-sig")

print("Dataset shape:", df.shape)
print(df.head())
print("\nColumns:", df.columns.tolist())


# Classify all reviews as POSITIVE or NEGATIVE, then score it

sentiment_classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1,
)

predicted_labels = sentiment_classifier(df["Review"].tolist())

label_map = {"POSITIVE": 1, "NEGATIVE": 0}
predictions = [label_map[item["label"]] for item in predicted_labels]
true_labels = df["Class"].map(label_map).tolist()

accuracy_result = accuracy_score(true_labels, predictions)
f1_result = f1_score(true_labels, predictions, average="binary")

print(f"\n[Task 1] Accuracy : {accuracy_result:.4f}")
print(f"[Task 1] F1 Score : {f1_result:.4f}")

# Confusion matrix
cm = confusion_matrix(true_labels, predictions)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["NEGATIVE", "POSITIVE"],
    yticklabels=["NEGATIVE", "POSITIVE"],
    ax=ax,
)
ax.set_title("Sentiment Classification — Confusion Matrix")
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("[Task 1] Confusion matrix saved → confusion_matrix.png")


# Translate the first two sentences of review[0], score with BLEU

first_review = df["Review"].iloc[0]
sentences = nltk.sent_tokenize(first_review)
text_to_translate = " ".join(sentences[:2])

print(f"\n[Task 2] Source text: {text_to_translate}")

translator = pipeline(
    "translation_en_to_es",
    model="Helsinki-NLP/opus-mt-en-es",
    device=0 if torch.cuda.is_available() else -1,
)

translated_output = translator(text_to_translate, max_length=512)
translated_review = translated_output[0]["translation_text"]

print(f"[Task 2] Translation : {translated_review}")

# Load reference translations
with open("data/reference_translations.txt", "r", encoding="utf-8") as f:
    raw_refs = [line.strip() for line in f.readlines() if line.strip()]

# List of per sentence
references = [[ref.split() for ref in raw_refs]]
hypothesis = [translated_review.split()]

bleu_metric = evaluate.load("bleu")
bleu_score = bleu_metric.compute(
    predictions=[translated_review],
    references=[raw_refs]
)
print(f"[Task 2] BLEU Score : {bleu_score}")


# Pull a direct answer from review[1] using MiniLM QA

second_review = df["Review"].iloc[1]

question = "What did he like about the brand?"
context = second_review

qa_pipeline = pipeline(
    "question-answering",
    model="deepset/minilm-uncased-squad2",
    device=0 if torch.cuda.is_available() else -1,
)

qa_output = qa_pipeline(question=question, context=context)
answer = qa_output["answer"]

print(f"\n[Task 3] Question   : {question}")
print(f"[Task 3] Answer     : {answer}")
print(f"[Task 3] Confidence : {qa_output['score']:.4f}")


# Summarize the last review, then check the output for bias signals

last_review = df["Review"].iloc[-1]

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0 if torch.cuda.is_available() else -1,
)

summary_output = summarizer(
    last_review,
    min_length=50,
    max_length=55,
    do_sample=False,
    truncation=True,
)
summarized_text = summary_output[0]["summary_text"]

print(f"\n[Task 4] Original : {last_review[:200]}...")
print(f"[Task 4] Summary  : {summarized_text}")

# Toxicity — any harmful signals
toxicity_metric = evaluate.load("toxicity", module_type="measurement")
tox_result = toxicity_metric.compute(predictions=[summarized_text])
max_toxicity = max(tox_result["toxicity"])

# Regard — sentiment polarity
regard_metric = evaluate.load("regard", module_type="measurement")
regard_result = regard_metric.compute(data=[summarized_text])

print(f"[Task 4] Max Toxicity : {max_toxicity:.4f}")
print(f"[Task 4] Regard       : {regard_result['regard']}")


# RESULTS DASHBOARD

metrics = {
    "Sentiment\nAccuracy": accuracy_result,
    "Sentiment\nF1 Score": f1_result,
    "Translation\nBLEU Score": bleu_score["bleu"],
    "QA\nConfidence": qa_output["score"],
    "Max\nToxicity": max_toxicity,
}

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(
    metrics.keys(),
    metrics.values(),
    color=["#4C72B0", "#4C72B0", "#55A868", "#C44E52", "#8172B2"],
    edgecolor="white",
    width=0.55,
)

for bar, val in zip(bars, metrics.values()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax.set_ylim(0, 1.15)
ax.set_title("Car-ing Is Sharing — LLM Pipeline Results", fontsize=14, fontweight="bold")
ax.set_ylabel("Score")
ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(4.6, 0.81, "0.8 threshold", fontsize=8, color="gray")
sns.despine()
plt.tight_layout()
plt.savefig("results_dashboard.png", dpi=150)
plt.show()
print("\n[Dashboard] Saved → results_dashboard.png")


# FINAL PRINT

print("\n" + "=" * 58)
print("  CAR-ING IS SHARING — LLM PIPELINE | FINAL RESULTS")
print("=" * 58)
print(f"  Task 1 | Sentiment Accuracy   : {accuracy_result:.4f}")
print(f"  Task 1 | Sentiment F1 Score   : {f1_result:.4f}")
print(f"  Task 2 | BLEU Score           : {bleu_score['bleu']:.4f}")
print(f"  Task 2 | Translated Review    : {translated_review[:60]}...")
print(f"  Task 3 | QA Answer            : {answer}")
print(f"  Task 3 | QA Confidence        : {qa_output['score']:.4f}")
print(f"  Task 4 | Summarized Text      : {summarized_text[:60]}...")
print(f"  Task 4 | Max Toxicity         : {max_toxicity:.4f}")
print("=" * 58)
