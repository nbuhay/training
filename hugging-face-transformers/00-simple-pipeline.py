# This script demonstrates how to use the Hugging Face Transformers library 
#   to create a simple text generation pipeline.
#
# It uses a pre-trained model (GPT-2) to generate text based on a given prompt.
#
# python 00-simple-pipeline.py
from transformers import pipeline

# Initialize a text generation pipeline with a pre-trained model
#
# By specifying "text-generation" and the model "gpt2", 
#   we set up a pipeline that can generate text continuations.​
generator = pipeline("text-generation", model="gpt2")

# Generate text based on a prompt
prompt = "The future of artificial intelligence is"
# max 50 character response, generate 3 different outputs
results = generator(prompt, max_length=50, num_return_sequences=3)

# Display the generated text
print('===pipeline output===')
print('prompt:', prompt)
print('===generated text===')
for i, result in enumerate(results):
    print(f'generated output {i+1}:', result['generated_text'], '\n')