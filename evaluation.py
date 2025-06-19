import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import classification_report, confusion_matrix
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import io

print("🔍 Starting evaluation script...")

try:
    # --- Load Data ---
    df = pd.read_csv("samples.csv")
    print(f"✅ Loaded {len(df)} samples")

    texts = df["text"].tolist()
    true_labels = df["true_label"].str.upper().tolist()

    # --- Load 3-class model with NEUTRAL support ---
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    print(f"📦 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("✅ Model loaded successfully.")

    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print("✅ Pipeline initialized.")

    # --- Label Mapping ---
    def map_label(label_str):
        label_id = int(label_str.split("_")[-1])
        mapping = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        return mapping.get(label_id, "NEUTRAL")

    # --- Run Predictions ---
    batch_size = 8
    predicted_labels = []
    confidence_scores = []

    print(f"▶️ Starting predictions on {len(texts)} samples in batches of {batch_size}...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        results = sentiment_pipeline(batch)
        for res in results:
            label = map_label(res['label'])
            score = round(res['score'], 4)
            predicted_labels.append(label)
            confidence_scores.append(score)
        print(f"  Processed batch {i // batch_size + 1} / {(len(texts) + batch_size - 1) // batch_size}")

    # --- Add predictions to DataFrame ---
    df["predicted_label"] = predicted_labels
    df["confidence_score"] = confidence_scores

    # --- Classification Report ---
    print("\n📊 Classification Report:")
    report = classification_report(true_labels, predicted_labels, digits=4)
    print(report)

    # --- Confusion Matrix ---
    labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("✅ Saved confusion matrix as 'confusion_matrix.png'")
    plt.show()

    # --- Save CSV ---
    df.to_csv("evaluation_results.csv", index=False)
    print("✅ Saved results to 'evaluation_results.csv'")

    # --- Save PDF ---
    def export_to_pdf(df, filename="evaluation_results.pdf"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = [Paragraph("Sentiment Evaluation Results", styles['Title']), Spacer(1, 12)]

        data = [list(df.columns)] + df.values.tolist()
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN',(0,0),(-1,-1),'LEFT'),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
        ]))
        elements.append(table)
        doc.build(elements)
        print(f"✅ Saved PDF: {filename}")

    export_to_pdf(df)

except Exception as e:
    print("❌ Error during evaluation:", e)
