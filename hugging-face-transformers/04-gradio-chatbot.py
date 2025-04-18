import gradio as gr  # Import Gradio library for creating web interfaces for ML models
from transformers import AutoModelForCausalLM, AutoTokenizer  # Import HuggingFace transformers components

# Initialize the language model
# AutoTokenizer handles tokenization (converting text to tokens the model can process)
# AutoModelForCausalLM loads a pre-trained generative language model for text generation
model_name = "facebook/blenderbot-400M-distill"  # BlenderBot: Meta's conversational AI model (400M parameters, distilled version)
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load tokenizer with model's vocabulary and encoding rules
model = AutoModelForCausalLM.from_pretrained(model_name)  # Load pre-trained weights for the transformer architecture

def chat_response(message, history):
    """
    Process user input and generate AI response using the transformer model.
    
    Args:
        message (str): The user's input text
        history (list): Conversation history (not used in current implementation)
    
    Returns:
        str: Generated response from the model
    """
    # Tokenization: Convert input text to token IDs and format for the model
    # The eos_token (End of Sequence) is added to mark the end of the input
    # return_tensors="pt" returns PyTorch tensors rather than plain lists
    input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors="pt")
    
    # Model inference with specific generation parameters
    output = model.generate(
        input_ids,  # Input sequence to continue from
        max_length=1000,  # Maximum length of generated sequence
        do_sample=True,  # Use sampling instead of greedy decoding
        top_p=0.95,  # Nucleus sampling: Consider tokens comprising the top 95% probability mass
        top_k=50,  # Limit vocabulary to top 50 tokens at each step
        temperature=0.7,  # Controls randomness: lower=more deterministic, higher=more random
        pad_token_id=tokenizer.eos_token_id  # Use EOS token ID for padding
    )
    
    # Extract only the generated response (not the input) and decode back to text
    # The slicing operation output[:, input_ids.shape[-1]:] removes the input portion
    # skip_special_tokens=True removes special tokens like [PAD], [EOS], etc.
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Create Gradio interface with the ChatInterface template
# ChatInterface automatically handles chat history display and user input
demo = gr.ChatInterface(
    fn=chat_response,  # Callback function that processes input and returns output
    title="Simple HuggingFace Transformer Chatbot",  # Header displayed on the web interface
    description="Ask a question and get a response from a small, efficient model.",  # Subheader with usage instructions
)

if __name__ == "__main__":
    demo.launch()  # Start the Gradio web server to host the interface