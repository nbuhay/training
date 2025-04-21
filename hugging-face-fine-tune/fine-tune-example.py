# ========== FINE-TUNING EXAMPLE ==========
# Fine-tuning takes a pre-trained AI model and adapts it for a specific task.
# Think of it like taking a chef who knows general cooking principles 
# and training them to specialize in Italian cuisine.
# This approach needs much less data and computing power than training from scratch.

# ========== STEP 1: LOAD THE DATASET ==========
# We're using Yelp reviews - a collection of customer reviews with star ratings (1-5)
from datasets import load_dataset

# Load the Yelp review dataset (restaurant reviews with 1-5 star ratings)
dataset = load_dataset("yelp_review_full")
print('===example training data===')
print(dataset["train"][100])
print('===example test data===')
print(dataset["test"][100])

# ======= SHOW LABEL EXAMPLES =======
print("\n===Labels Information===")
unique_labels = set(dataset["train"]["label"])
print(f"Unique labels: {unique_labels}")
print(f"Number of unique labels: {len(unique_labels)}")

# Get distribution of labels
from collections import Counter
label_counts = Counter(dataset["train"]["label"])
print(f"Label distribution: {label_counts}")

# Show examples of each label
print("\n===One Example Per Label===")
label_examples = {}

# Find one example for each label
for example in dataset["train"]:
    label = example["label"]
    if label not in label_examples:
        label_examples[label] = example["text"]
        # If we found examples for all 5 labels, we can stop
        if len(label_examples) == 5:
            break

# Print one example for each label (star rating)
for label, text in sorted(label_examples.keys(), reverse=True):
    print(f"\nLabel {label} (Rating: {label+1} stars):")
    # Print a shorter preview if the text is very long
    preview = text[:150] + "..." if len(text) > 150 else text
    print(preview)

# ========== STEP 2: PREPARE THE TEXT FOR THE AI ==========
# The AI can't understand raw text - we need to convert it to numbers (tokens)
# This is like translating English to a language the AI understands
from transformers import AutoTokenizer

# BERT is a popular pre-trained AI model for understanding text
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    # Convert each review to tokens, with consistent length and padding
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True
    )

# Process all examples at once for speed
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ========== STEP 3: PREPARE FOR TRAINING ==========
# Set up our data format for PyTorch (the deep learning framework)
# 'labels' are what we want the model to predict (star ratings 0-4, representing 1-5 stars)
tokenized_datasets = tokenized_datasets.map(
    lambda examples: {'labels': examples['label']}, 
    batched=True
)
tokenized_datasets.set_format(
    type="torch", 
    columns=["input_ids", "attention_mask", "labels"]
)

# ========== STEP 4: LOAD THE PRE-TRAINED MODEL ==========
# We're using BERT, configuring it to predict 5 different star ratings
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", 
    num_labels=5
)

# ========== STEP 5: SET UP EVALUATION ==========
# We need a way to measure how well our model is learning
import numpy as np
import evaluate

# Accuracy measures what percentage of predictions are correct
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert model's raw outputs (logits) to predicted star ratings
    predictions = np.argmax(logits, axis=-1)
    # Calculate what percentage of predictions match the true ratings
    return metric.compute(predictions=predictions, references=labels)

# ========== STEP 6: CONFIGURE AND RUN TRAINING ==========
from transformers import TrainingArguments, Trainer

output_dir = "test_trainer"  # Folder to save the model and logs

# Training settings control how the learning happens
training_args = TrainingArguments(
    output_dir=output_dir,           # Where to save results
    eval_steps=500,                  # How often to check progress
    per_device_train_batch_size=16,  # How many examples to process at once
    per_device_eval_batch_size=16,   # Batch size for evaluation
    num_train_epochs=1,              # How many times to go through the data
    logging_dir="logs",              # Where to save training logs
    logging_steps=10,                # How often to log progress
    save_strategy="no",              # Don't save intermediate models
    fp16=True,                       # Use mixed precision for faster training
)

# Create the trainer - this handles all the complex training logic
trainer = Trainer(
    model=model,
    args=training_args,
    # Use 10,000 random examples for training (for speed)
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(10000)),
    # Use 10,000 random examples for evaluation
    eval_dataset=tokenized_datasets["test"].shuffle(seed=42).select(range(10000)),
    compute_metrics=compute_metrics,
)

# Start the training process!
trainer.train()

# ========== STEP 7: SAVE THE RESULTS ==========
# Save the fine-tuned model for future use
trainer.save_model(output_dir)

# Save the tokenizer so we can process new text the same way
tokenizer.save_pretrained(output_dir)