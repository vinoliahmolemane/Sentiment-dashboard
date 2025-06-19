# app.py

# --- README Section ---
"""
README - Sentiment Analysis Dashboard (CAPACITI TCA Project)

Instructions:
-------------
1. Install dependencies:
   pip install streamlit transformers scikit-learn matplotlib seaborn reportlab pandas openpyxl

2. Place this app.py and any CSV/XLSX/TXT/JSON (with a 'text' column) in one folder.

3. Run the app:
   streamlit run app.py

Features:
---------
✓ Upload CSV, XLSX, TXT, or JSON (with 'text' column)
✓ Manual text input
✓ Multi-class sentiment classification: POSITIVE, NEGATIVE, NEUTRAL
✓ Confidence scoring
✓ Keyword extraction
✓ Sentiment explanation
✓ Batch processing (efficient)
✓ Visualizations: bar chart, pie chart
✓ Compare two datasets side-by-side
✓ Export results: CSV, JSON, PDF (with charts)
✓ Evaluation metrics (confusion matrix, classification report)

Author: Your Name | For CAPACITI Tech Career Accelerator
"""

# --- Imports ---
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# --- Config ---
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("😊 Sentiment Analysis Dashboard")
st.write("Analyze the sentiment of uploaded text data or manual entries with detailed insights and visualizations.")

# --- Load Sentiment Model ---
sentiment_pipeline = pipeline("sentiment-analysis")

# --- Helper Functions ---
def extract_keywords(text, num_keywords=3):
    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = set(['the', 'is', 'and', 'in', 'it', 'this', 'i', 'with', 'a', 'of', 'to'])
    keywords = [word for word in words if word not in stop_words]
    return ", ".join([kw for kw, _ in Counter(keywords).most_common(num_keywords)])

def extract_sentiment_drivers(text):
    pos_words = {"love", "great", "excellent", "good", "amazing", "happy", "fantastic", "wonderful", "like"}
    neg_words = {"hate", "bad", "terrible", "awful", "worst", "poor", "disappointed", "angry", "dislike"}
    words = re.findall(r'\b\w+\b', text.lower())
    pos_matches = [w for w in words if w in pos_words]
    neg_matches = [w for w in words if w in neg_words]
    if pos_matches and not neg_matches:
        return f"Positive words: {', '.join(pos_matches)}"
    elif neg_matches and not pos_matches:
        return f"Negative words: {', '.join(neg_matches)}"
    elif pos_matches and neg_matches:
        return f"Mixed — Positive: {', '.join(pos_matches)} | Negative: {', '.join(neg_matches)}"
    return "No strong sentiment words detected."

def classify_neutral(label, score):
    return "NEUTRAL" if 0.4 < score < 0.6 else label

def analyze_texts(texts):
    results = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        results.extend(sentiment_pipeline(texts[i:i+batch_size]))
    return results

def create_pdf(df, bar_chart_path, pie_chart_path):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = [Paragraph("Sentiment Analysis Results", getSampleStyleSheet()['Title']), Spacer(1, 12)]

    elements.append(Image(bar_chart_path, width=400, height=200))
    elements.append(Spacer(1, 12))
    elements.append(Image(pie_chart_path, width=400, height=200))
    elements.append(Spacer(1, 12))

    data = [list(df.columns)] + df.astype(str).values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f2f2f2")),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
    ]))
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()

def read_uploaded_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file, encoding_errors='ignore')
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    elif file.name.endswith('.json'):
        return pd.read_json(file)
    elif file.name.endswith('.txt'):
        lines = file.read()
        try:
            text = lines.decode('utf-8')
        except:
            text = lines.decode('ISO-8859-1')
        return pd.DataFrame({'text': text.splitlines()})
    return None

def process_dataframe(df):
    texts = df['text'].astype(str).tolist()
    predictions = analyze_texts(texts)
    df['label'] = [classify_neutral(p['label'], p['score']) for p in predictions]
    df['score'] = [round(p['score'], 4) for p in predictions]
    df['keywords'] = df['text'].apply(extract_keywords)
    df['explanation'] = df['text'].apply(extract_sentiment_drivers)
    return df

# --- Upload Section ---
st.subheader("📂 Upload File")
uploaded_file = st.file_uploader("Upload CSV, Excel, Text or JSON", type=["csv", "xlsx", "txt", "json"])

if uploaded_file:
    df = read_uploaded_file(uploaded_file)
    if df is not None and 'text' in df.columns:
        df = process_dataframe(df)

        st.write("### 📊 Results Preview")
        st.dataframe(df)

        sentiment_counts = df['label'].value_counts().reindex(['POSITIVE', 'NEGATIVE', 'NEUTRAL'], fill_value=0)

        st.write("### 📈 Sentiment Distribution")
        fig_bar, ax_bar = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax_bar, color=['green', 'red', 'gray'])
        plt.tight_layout()
        bar_chart_path = "bar_chart.png"
        plt.savefig(bar_chart_path)
        st.pyplot(fig_bar)

        st.write("### 🥧 Pie Chart")
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        ax_pie.axis('equal')
        pie_chart_path = "pie_chart.png"
        plt.savefig(pie_chart_path)
        st.pyplot(fig_pie)

        # Downloads
        st.download_button("💾 Download CSV", df.to_csv(index=False).encode(), file_name="results.csv")
        st.download_button("💾 Download JSON", df.to_json(orient='records', indent=2).encode(), file_name="results.json")
        st.download_button("💾 Download PDF", create_pdf(df, bar_chart_path, pie_chart_path), file_name="results.pdf")

        if 'true_label' in df.columns:
            st.write("### 🧪 Evaluation Metrics")
            report = classification_report(df['true_label'], df['label'], output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            cm = confusion_matrix(df['true_label'], df['label'], labels=['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['POS', 'NEG', 'NEU'], yticklabels=['POS', 'NEG', 'NEU'])
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)
    else:
        st.warning("❌ Make sure the uploaded file contains a 'text' column.")

# --- Comparison Upload ---
st.subheader("📂 Upload Second File for Comparison (Optional)")
comparison_file = st.file_uploader("Upload comparison file", type=["csv", "xlsx", "txt", "json"], key="comparison")

if comparison_file:
    comp_df = read_uploaded_file(comparison_file)
    if comp_df is not None and 'text' in comp_df.columns:
        comp_df = process_dataframe(comp_df)
        comp_counts = comp_df['label'].value_counts().reindex(['POSITIVE', 'NEGATIVE', 'NEUTRAL'], fill_value=0)

        st.write("### 🔍 Comparison Charts")
        col1, col2 = st.columns(2)
        with col1:
            st.write("File 1 Sentiment Distribution")
            st.bar_chart(sentiment_counts)
        with col2:
            st.write("File 2 Sentiment Distribution")
            st.bar_chart(comp_counts)
    else:
        st.warning("❌ Comparison file must also contain a 'text' column.")

# --- Manual Input ---
st.subheader("📝 Manual Text Input")
user_input = st.text_area("Enter text to analyze")
if st.button("Analyze Text"):
    if user_input.strip():
        result = sentiment_pipeline(user_input)[0]
        label = classify_neutral(result['label'], result['score'])
        score = round(result['score'], 4)
        keywords = extract_keywords(user_input)
        explanation = extract_sentiment_drivers(user_input)

        st.markdown("### 🧠 Prediction")
        st.write(f"**Sentiment:** {label}")
        st.write(f"**Confidence Score:** {score}")
        st.write(f"**Keywords:** {keywords}")
        st.write(f"**Explanation:** {explanation}")
    else:
        st.warning("⚠️ Please enter some text.")