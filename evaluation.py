import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import classification_report, confusion_matrix

print("Starting evaluation script...")

# Load samples.csv
df = pd.read_csv("samples.csv")
print(f"Loaded {len(df)} samples")

texts = df["text"].tolist()
true_labels = df["true_label"].tolist()

# Load CardiffNLP twitter-roberta-base-sentiment model (3-class: NEG, NEU, POS)
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Helper to map model's labels to your format
def map_label(label_str):
    # The model labels look like: 'LABEL_0', 'LABEL_1', 'LABEL_2'
    label_id = int(label_str.split("_")[-1])
    mapping = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    return mapping.get(label_id, "NEUTRAL")

# Run predictions in batches (for efficiency)
batch_size = 16
predicted_labels = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    results = sentiment_pipeline(batch)
    batch_preds = [map_label(res['label']) for res in results]
    predicted_labels.extend(batch_preds)

# Evaluation
print("\nClassification Report:")
report = classification_report(true_labels, predicted_labels, digits=4)
print(report)

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=["POSITIVE", "NEGATIVE", "NEUTRAL"])

plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=["POSITIVE", "NEGATIVE", "NEUTRAL"], yticklabels=["POSITIVE", "NEGATIVE", "NEUTRAL"], cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion matrix plot as confusion_matrix.png")
plt.show()
