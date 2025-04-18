# Named Entity Recognition (NER) is a powerful tool in natural language processing
#  that identifies and classifies key elements in text into predefined categories
#  such as names of people, organizations, locations, dates, and more. This capability
#  is instrumental in transforming unstructured text into structured data, 
#  enabling various real-world applications across multiple industries.
# 
# python 02-named-entity-recognition.py
from transformers import pipeline

# By specifying "ner" and setting grouped_entities=True, 
#   we set up a pipeline that can identify and group named entities 
#   in the input text.
ner_pipeline = pipeline("ner", grouped_entities=True)

# We provide a list of customer support messages, and the model returns 
#   the named entities along with their labels and confidence scores for each.
customer_messages = [
    "Hi, my internet has been down since yesterday in New York. My account number is 123456.",
    "I'm experiencing issues with my 5G service in San Francisco.",
    "Can you check if there's an outage in the 94107 area code?",
    "My name is John Doe, and I'm unable to access my voicemail.",
    "There's a problem with my billing for account 789012."
]

# We iterate through each text and its corresponding entities, 
#   printing out the entity, its label, and confidence score.
#
# Entity Labels: Common labels include PER (person), ORG (organization), 
#   LOC (location), and MISC (miscellaneous).
#
# showcases how NER can extract info from teleco customer support msgs
#   to identify key entities like names, locations, etc
for text in customer_messages:
    print(f"\nText: {text}")
    entities = ner_pipeline(text)
    for entity in entities:
        print(f" - Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']:.2f}")
