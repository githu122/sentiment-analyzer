import gradio as gr
from transformers import pipeline

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_ID)

LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
    "NEGATIVE": "Negative",
    "NEUTRAL": "Neutral",
    "POSITIVE": "Positive",
}

def analyze_sentiment(text):
    text = (text or "").strip()
    if not text:
        return "⚠️ Please enter some text."
    result = sentiment_pipeline(text, truncation=True)[0]
    label = LABEL_MAP.get(result["label"], result["label"].title())
    score = round(float(result["score"]), 3)
    return f"{label} (confidence: {score})"

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Type a sentence..."),
    outputs="text",
    title="Sentiment Analyzer",
    description="Classifies text as Negative, Neutral, or Positive using a Hugging Face transformer."
)

if __name__ == "__main__":
    demo.launch()
