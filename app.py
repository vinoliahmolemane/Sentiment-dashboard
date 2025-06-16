"""
README - Sentiment Analysis Dashboard (CAPACITI TCA Project)

Instructions:
-------------
1. Install dependencies:
   pip install streamlit transformers scikit-learn matplotlib seaborn reportlab pandas

2. Place this app.py and any CSV (with a 'text' column) in one folder.

3. Run the app:
   streamlit run app.py

Features:
---------
✓ Upload CSV or type text manually
✓ Multi-class sentiment classification: POSITIVE, NEGATIVE, NEUTRAL
✓ Confidence scoring
✓ Keyword extraction
✓ Sentiment explanation (driver words)
✓ Batch processing (efficient)
✓ Visualizations: bar chart, pie chart
✓ Compare two datasets side by side
✓ Export results: CSV, JSON, PDF
✓ Evaluation metrics (confusion matrix, classification report)
✓ NEUTRAL visible in all visualizations

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
import warnings
from collections import Counter
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# --- Streamlit config ---
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("😊 Sentiment Analysis Dashboard")
st.write("Upload a CSV with a 'text' column, or enter text below.")

# --- Load sentiment model ---
sentiment_pipeline = pipeline("sentiment-analysis")

# --- Helper functions ---
def extract_keywords(text, num_keywords=3):
    words = re.findall(r'\b\w+\b', text.lower())
    common_words = set(['the', 'is', 'and', 'in', 'it', 'this', 'i', 'with', 'a', 'of', 'to'])
    keywords = [word for word in words if word not in common_words]
    most_common = Counter(keywords).most_common(num_keywords)
    return ", ".join([kw[0] for kw in most_common])

def extract_sentiment_drivers(text):
    positive_words = {"love", "great", "excellent", "good", "amazing", "happy", "fantastic", "wonderful", "like"}
    negative_words = {"hate", "bad", "terrible", "awful", "worst", "poor", "disappointed", "angry", "dislike"}
    
    words = re.findall(r'\b\w+\b', text.lower())
    pos_matches = [word for word in words if word in positive_words]
    neg_matches = [word for word in words if word in negative_words]

    if pos_matches and not neg_matches:
        explanation = f"Positive words: {', '.join(pos_matches)}"
    elif neg_matches and not pos_matches:
        explanation = f"Negative words: {', '.join(neg_matches)}"
    elif pos_matches and neg_matches:
        explanation = f"Mixed words — Positive: {', '.join(pos_matches)} | Negative: {', '.join(neg_matches)}"
    else:
        explanation = "No strong sentiment words detected."
    return explanation

def create_pdf(dataframe):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    title = Paragraph("Sentiment Analysis Results", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    data = [list(dataframe.columns)] + dataframe.values.tolist()
    table = Table(data)
    table_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f2f2f2")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ])
    table.setStyle(table_style)
    elements.append(table)
    doc.build(elements)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

def analyze_texts(texts):
    results = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = sentiment_pipeline(texts[i:i+batch_size])
        results.extend(batch)
    return results

def classify_neutral(label, score):
    return "NEUTRAL" if 0.4 < score < 0.6 else label

# --- Upload Section ---
st.subheader("📂 Upload CSV")
uploaded_file = st.file_uploader("Upload your input.csv", type=["csv"])
df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, quotechar='"', on_bad_lines='skip')
        if 'text' not in df.columns:
            st.error("❌ The CSV must contain a 'text' column.")
            df = None
        else:
            st.success("✅ File uploaded successfully!")
            st.write("### 📥 Input Preview")
            st.dataframe(df.head())

            predictions = analyze_texts(df['text'].tolist())
            df['label'] = [classify_neutral(pred['label'], pred['score']) for pred in predictions]
            df['score'] = [round(pred['score'], 4) for pred in predictions]
            df['keywords'] = df['text'].apply(lambda x: extract_keywords(str(x)))
            df['explanation'] = df['text'].apply(lambda x: extract_sentiment_drivers(str(x)))

            st.write("### 📊 Sentiment Results")
            st.dataframe(df)

            sentiment_counts = df['label'].value_counts().reindex(['POSITIVE', 'NEGATIVE', 'NEUTRAL'], fill_value=0)
            st.write("### 📈 Sentiment Distribution (Bar Chart)")
            st.bar_chart(sentiment_counts)

            st.write("### 🥧 Sentiment Distribution (Pie Chart)")
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            csv_output = df.to_csv(index=False).encode('utf-8')
            json_output = df.to_json(orient='records', indent=2).encode('utf-8')
            pdf_output = create_pdf(df)

            st.download_button("💾 Download as CSV", data=csv_output, file_name='sentiment_results.csv', mime='text/csv')
            st.download_button("💾 Download as JSON", data=json_output, file_name='sentiment_results.json', mime='application/json')
            st.download_button("💾 Download as PDF", data=pdf_output, file_name='sentiment_results.pdf', mime='application/pdf')

            if 'true_label' in df.columns:
                st.write("### 📊 Evaluation Metrics")
                report = classification_report(df['true_label'], df['label'], output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                cm = confusion_matrix(df['true_label'], df['label'], labels=['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', xticklabels=['POS', 'NEG', 'NEU'], yticklabels=['POS', 'NEG', 'NEU'], ax=ax_cm, cmap="Blues")
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)

    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")

# --- Comparison Section ---
st.subheader("📂 Upload Second CSV (Optional for Comparison)")
comparison_file = st.file_uploader("Upload another CSV for side-by-side comparison", type=["csv"], key="comp")
if comparison_file:
    try:
        comp_df = pd.read_csv(comparison_file, quotechar='"', on_bad_lines='skip')
        if 'text' in comp_df.columns:
            comp_predictions = analyze_texts(comp_df['text'].tolist())
            comp_df['label'] = [classify_neutral(pred['label'], pred['score']) for pred in comp_predictions]
            comp_counts = comp_df['label'].value_counts().reindex(['POSITIVE', 'NEGATIVE', 'NEUTRAL'], fill_value=0)

            st.write("### 📊 Comparison Chart")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Uploaded File 1")
                st.bar_chart(sentiment_counts)
            with col2:
                st.write("Comparison File")
                st.bar_chart(comp_counts)
        else:
            st.error("❌ Second CSV must also contain a 'text' column.")
    except Exception as e:
        st.error(f"❌ Error in comparison file: {e}")

# --- Manual Text Input ---
st.subheader("💬 Or Enter Text Manually")
user_input = st.text_area("Type or paste your text here")
if st.button("Analyze Text"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
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
