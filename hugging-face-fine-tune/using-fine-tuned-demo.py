# =================================================================
# Text Classification Demo: Comparing Base vs Fine-tuned BERT Model
# This script shows how a fine-tuned model can perform better on 
# specific tasks than the original pre-trained model
# =================================================================

from transformers import pipeline

# Path to the model we previously fine-tuned on our custom data
# This directory contains the model weights and configuration
fine_tuned_model_path = "test_trainer"

# ---------------------------------
# Setting up the models for comparison
# ---------------------------------

# Create a text classification pipeline with the original BERT model
# A pipeline is a ready-to-use tool that handles all the ML complexity
base_classifier = pipeline("text-classification", model="bert-base-cased")

# Create another pipeline using our fine-tuned version of BERT
# This model has been adapted to better understand our specific task
fine_tuned_classifier = pipeline("text-classification", model=fine_tuned_model_path)

# ---------------------------------
# Test data covering different sentiment levels
# ---------------------------------

# Sample hotel reviews ranging from very negative to very positive
# We'll use these to demonstrate how each model classifies different sentiments
texts = [
    "The hotel had roaches in the bathroom.",           # 0 - Very Negative - 1 star
    "The service was slow and the food was cold.",      # 1 - Negative - 2 stars
    "The room was okay, nothing special.",              # 2 - Neutral - 3 stars
    "The staff was friendly and the location was convenient.",  # 3 - Pos. - 4 stars
    "Absolutely loved the experience, will come back again!"    # 4 - Very Pos. - 5 stars
]

# ---------------------------------
# Running and comparing model predictions
# ---------------------------------

# Test each review with both models and compare the results
for idx, text in enumerate(texts):
    # The [0] accesses the first (and only) prediction in the returned list
    base_result = base_classifier(text)[0]
    fine_tuned_result = fine_tuned_classifier(text)[0]
    
    # Print the text and the predictions from both models
    # The score represents the model's confidence in its prediction
    print(f"Text {idx} (Rating {idx}): {text}")
    print(f"  Base Model Prediction: {base_result['label']} "
          f"(Score: {base_result['score']:.4f})")
    print(f"  Fine-Tuned Model Prediction: {fine_tuned_result['label']} "
          f"(Score: {fine_tuned_result['score']:.4f})\n")