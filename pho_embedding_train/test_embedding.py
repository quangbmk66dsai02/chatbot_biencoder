from transformers import AutoTokenizer, AutoModel
import torch

def list_mean(arr):
    ls_sum = sum(arr)
    return ls_sum/len(arr)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load fine-tuned model and tokenizer
tokenizer_ft = AutoTokenizer.from_pretrained('fine-tuned-phobert-embedding-model', use_fast=False)
model_ft = AutoModel.from_pretrained('fine-tuned-phobert-embedding-model').to(device)

# Load base model and tokenizer
tokenizer_base = AutoTokenizer.from_pretrained('vinai/phobert-base-v2', use_fast=False)
model_base = AutoModel.from_pretrained('vinai/phobert-base-v2').to(device)

import json
# Specify the path to your test data JSON file
test_file_path = "../qa_pairs_with_query.json"

# Load the test QA pairs from the JSON file
with open(test_file_path, "r", encoding="utf-8") as f:
    test_qa_pairs = json.load(f)

# Check the number of test samples and the structure
print(f"Loaded {len(test_qa_pairs)} test QA pairs")
print(test_qa_pairs[0])  # Print the first pair as an example

# Initialize lists to hold questions, answers, and labels
questions = []
answers = []
labels = []
queries = []
# Iterate over the loaded test data
for item in test_qa_pairs:
    question = item['question']
    answer = item['answer']
    query = item.get('query', "")
    label = item.get('label', 1)  # Default label is 1 if missing

    # Append to the respective lists
    questions.append(question)
    answers.append(answer)
    labels.append(label)
    queries.append(query)



def get_embeddings(model, tokenizer, texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        embeddings = sum_embeddings / sum_mask
    return embeddings.cpu()

# Generate embeddings with fine-tuned model
embeddings_q_ft = get_embeddings(model_ft, tokenizer_ft, queries)
embeddings_a_ft = get_embeddings(model_ft, tokenizer_ft, answers)

# Generate embeddings with base model
embeddings_q_base = get_embeddings(model_base, tokenizer_base, queries)
embeddings_a_base = get_embeddings(model_base, tokenizer_base, answers)

# Compute cosine similarities
import torch.nn.functional as F

def compute_similarities(embeddings_q, embeddings_a):
    similarities = F.cosine_similarity(embeddings_q, embeddings_a)
    return similarities.numpy()

# Fine-tuned model similarities
similarities_ft = compute_similarities(embeddings_q_ft, embeddings_a_ft)

# Base model similarities
similarities_base = compute_similarities(embeddings_q_base, embeddings_a_base)

print("Base model similarities:", similarities_base)
print("Fine-tuned model similarities:", similarities_ft)
print("FT_MEAN", list_mean(similarities_ft), "BASE_MEAN", list_mean(similarities_base))