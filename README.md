# Sentiment Analysis Dashboard

---

## Overview

This **Sentiment Analysis Dashboard** is a Streamlit web app designed to analyze text data sentiment using a Hugging Face transformer model. It supports multi-class classification with three labels: **POSITIVE**, **NEGATIVE**, and **NEUTRAL**, and provides confidence scores and keyword extraction.

The app allows users to upload  CSV, XLSX, TXT, JSON files containing text data or input text manually, perform batch sentiment analysis efficiently, visualize sentiment distributions, compare two datasets side-by-side, and export results in CSV, JSON, or PDF formats. It also includes evaluation metrics such as a confusion matrix and classification report when true labels are provided.

---

## Features

- Upload CSV,XLSX, TXT, JSON file with a **`text`** column for batch sentiment analysis  
- Manual text input for quick single-text analysis  
- Multi-class sentiment classification: **POSITIVE**, **NEGATIVE**, **NEUTRAL**  
- Confidence scoring for each prediction  
- Keyword extraction from the text to highlight important words  
- Sentiment explanation using driver words (simple positive/negative word matching)  
- Batch processing for efficient analysis of large datasets  
- Visualizations including bar charts and pie charts of sentiment distribution  
- Compare two uploaded datasets side-by-side with bar charts  
- Export results in multiple formats: CSV, JSON, PDF  
- Evaluation metrics (confusion matrix and classification report) if ground-truth labels (`true_label` column) are present  
- NEUTRAL sentiment handling is integrated in predictions and visualizations

---

## How It Works

1. **Upload  CSV, XLSX, TXT, JSON or enter text manually:**  
   The CSV must contain a `text` column. You can upload one or two CSV files (for comparison) or just enter text manually in the input box.

2. **Sentiment classification:**  
   Uses Hugging Face's pre-trained `sentiment-analysis` pipeline to classify text in batches. Predictions include a label and confidence score.

3. **Post-processing:**  
   Scores between 0.4 and 0.6 are classified as **NEUTRAL** to improve classification nuance.

4. **Keyword extraction & explanations:**  
   Extracts the top keywords by frequency excluding common stopwords and provides simple explanations based on presence of positive/negative words.

5. **Visualizations:**  
   Displays sentiment distributions as bar and pie charts, and side-by-side comparisons if a second dataset is uploaded.

6. **Exports:**  
   You can download the results as CSV, JSON, or PDF (the PDF includes a formatted table).

7. **Evaluation:**  
   If your CSV has a `true_label` column, the app calculates and displays classification metrics and confusion matrix.

---

## Installation and Running Locally

1. Clone this repository or download the source files.

2. Make sure you have **Python 3.7+** installed.

3. Install the required dependencies using:

   ```bash
   pip install -r requirements.txt

To run the Streamlit app (app.py):
streamlit run app.py

How to Run the Evaluation Script
python evaluation.py


GitHub Repository
Access the full source code and latest updates here:
https://github.com/vinoliahmolemane/Sentiment-dashboard

Deployment
Try the live app deployed on Streamlit Cloud here:
https://sentiment-dashboard-g4sttofyca3gmtpysn9dvy.streamlit.app/
