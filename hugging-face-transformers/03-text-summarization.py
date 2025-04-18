# python 03-text-summarization.py
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Sample text to summarize
text = (
    "The Hugging Face Transformers library provides an extensive collection of pre-trained models "
    "for natural language processing tasks. It simplifies the process of integrating state-of-the-art "
    "models into applications, allowing developers to perform tasks like text classification, question "
    "answering, and text generation with minimal code."
)

# Generate summary
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)

# Display the summary
print("Summary:", summary[0]['summary_text'])
