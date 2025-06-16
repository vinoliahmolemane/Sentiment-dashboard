# 📊 Sentiment Analysis Dashboard

This is a multi-class Sentiment Analysis Dashboard built using **Streamlit** and **Hugging Face Transformers**. It classifies input text as **POSITIVE**, **NEUTRAL**, or **NEGATIVE**, and allows for both real-time and batch processing.

Model used: [`cardiffnlp/twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)

---

## 🚀 Features

- 📂 Upload a CSV file (`input.csv`) or enter a single sentence manually
- 🤖 Predicts sentiment with **confidence scores**
- 🔍 **Keyword highlighting** with sentiment-based emojis
- 📈 **Visual summaries** using Pie and Bar charts
- 🧪 **Evaluation script** using 50 manually labeled samples
- 📁 Download results in **CSV**, **JSON**, and **PDF**
- 🧠 Built-in support for 3 classes: **POSITIVE**, **NEGATIVE**, **NEUTRAL**

---

## 🛠️ Setup Instructions

1. **Clone or download** this repository:
   ```bash
   git clone https://github.com/your-username/sentiment-dashboard.git
   cd sentiment-dashboard
