import json
import pandas as pd

# Specify the path to your JSON file
json_file_path = "../api_QA/json_data/parsed_qa_pairs_output_mixed_6k.json"  # Update the path as needed

# Load the JSON data
with open(json_file_path, "r", encoding="utf-8") as file:
    qa_pairs = json.load(file)

# Verify the data
print(f"Loaded {len(qa_pairs)} QA pairs.")
print("Sample QA pair:", qa_pairs[0])

import re

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(qa_pairs)

# Check if 'answer' column exists
if 'answer' not in df.columns:
    raise KeyError("The JSON data does not contain an 'answer' key.")

# Function to count sentences using regex
def count_sentences(text):
    # This regex splits text into sentences based on ., !, or ?
    # It may not be perfect for all Vietnamese texts but serves as a basic splitter
    sentences = re.split(r'[.!?]+', text)
    # Remove any empty strings resulting from the split
    sentences = [s for s in sentences if s.strip()]
    return len(sentences)

# Compute lengths in words
df['answer_length_word'] = df['answer'].apply(lambda x: len(x.split()))

# Compute lengths in sentences
df['answer_length_sentence'] = df['answer'].apply(count_sentences)

# Display some statistics
print("\nDescriptive Statistics for Answer Lengths (Words):")
print(df['answer_length_word'].describe())

print("\nDescriptive Statistics for Answer Lengths (Sentences):")
print(df['answer_length_sentence'].describe())
