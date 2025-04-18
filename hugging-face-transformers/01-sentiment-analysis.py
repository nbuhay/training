# python 01-sentiment-analysis.py
from transformers import pipeline

# Initialize a sentiment-analysis pipeline to
#  classify the sentiment of input text.
sentiment_analyzer = pipeline("sentiment-analysis")

# List of texts to analyze
texts = [
    "I absolutely love this product! It has exceeded my expectations.",
    "The service was okay, nothing special.",
    "I'm not satisfied with the quality of the item.",
    "Best purchase I've made this year!",
    "It's decent, but I've seen better."
]

# Perform sentiment analysis
results = sentiment_analyzer(texts)

# Display the results
for text, result in zip(texts, results):
    print(f"Text: {text}\nSentiment: {result['label']} (Confidence: {result['score']:.2f})\n")