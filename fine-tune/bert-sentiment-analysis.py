# ​Fine-tuning a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model for sentiment analysis allows the model to adapt to specific tasks, such as determining whether movie reviews are positive or negative. Below is a simplified Python example with detailed, step-by-step comments to help you understand the process. This example uses the Hugging Face transformers library and the IMDB dataset.

# Import necessary libraries
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Check if a GPU is available and use it; otherwise, fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the IMDB dataset from the 'datasets' library
# The dataset contains movie reviews labeled as 'pos' (positive) or 'neg' (negative)
dataset = load_dataset("imdb")

# Initialize the BERT tokenizer
# The tokenizer converts text into tokens that correspond to BERT's vocabulary
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a function to preprocess the text data
def preprocess_function(examples):
    # Tokenize the text (reviews) with padding and truncation
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Apply the preprocessing function to the dataset
# The 'map' function applies the preprocessing to each example in the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Load the pre-trained BERT model with a classification head
# The 'num_labels' parameter is set to 2 because we have two classes: positive and negative
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Move the model to the appropriate device (GPU or CPU)
model.to(device)

# Define training arguments using the TrainingArguments class
# These arguments specify the training configuration
training_args = TrainingArguments(
    output_dir="./results",          # Directory to save the model checkpoints
    evaluation_strategy="epoch",     # Evaluate the model at the end of each epoch
    learning_rate=2e-5,              # Learning rate for the optimizer
    per_device_train_batch_size=8,   # Batch size for training
    per_device_eval_batch_size=8,    # Batch size for evaluation
    num_train_epochs=3,              # Number of training epochs
    weight_decay=0.01,               # Weight decay for regularization
)

# Initialize the Trainer class for training and evaluation
# The Trainer class provides an easy-to-use interface for training models
trainer = Trainer(
    model=model,                         # The pre-trained BERT model with a classification head
    args=training_args,                  # Training arguments defined above
    train_dataset=tokenized_datasets["train"],  # Training dataset
    eval_dataset=tokenized_datasets["test"],    # Evaluation dataset
)

# Train the model
trainer.train()

# Evaluate the model on the test dataset
trainer.evaluate()
