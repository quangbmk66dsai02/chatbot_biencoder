import json
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import random

# Load the QA pairs from the JSON file =============================================
file_path = "api_QA/json_data/parsed_qa_pairs_output_20.json"

with open(file_path, "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

# Check the structure of the data
print(f"Loaded {len(qa_pairs)} QA pairs")
print(qa_pairs[0])  # Print the first pair as an example

# Load the pre-trained PhoBERT tokenizer and model =============================================
bpe_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)
phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2")

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
phobert_model.to(device)

# Function to tokenize question-answer pairs separately ============================
def tokenize_qa_pairs(qa_pairs):
    question_input_ids = []
    answer_input_ids = []
    for pair in qa_pairs:
        # Tokenize question
        question_tokens = bpe_tokenizer.encode(pair["question"], add_special_tokens=True, truncation=True, max_length=256)
        question_input_ids.append(torch.tensor(question_tokens))

        # Tokenize answer
        answer_tokens = bpe_tokenizer.encode(pair["answer"], add_special_tokens=True, truncation=True, max_length=256)
        answer_input_ids.append(torch.tensor(answer_tokens))
        
    return question_input_ids, answer_input_ids

# Tokenize the loaded QA pairs
question_input_ids, answer_input_ids = tokenize_qa_pairs(qa_pairs)
# Create negative samples ==========================================================
def create_negative_samples(question_input_ids, answer_input_ids, num_negatives=3):
    negative_question_input_ids = []
    negative_answer_input_ids = []
    labels = []

    num_samples = len(question_input_ids)
    indices = list(range(num_samples))

    for i in range(num_samples):
        q = question_input_ids[i]

        # Add positive sample
        positive_a = answer_input_ids[i]
        negative_question_input_ids.append(q)
        negative_answer_input_ids.append(positive_a)
        labels.append(1)  # Label 1 for positive pairs

        # Generate multiple negative samples
        for _ in range(num_negatives):
            j = random.randint(0, num_samples - 1)
            while j == i:
                j = random.randint(0, num_samples - 1)
            a = answer_input_ids[j]

            negative_question_input_ids.append(q)
            negative_answer_input_ids.append(a)
            labels.append(0)  # Label 0 for negative pairs

    return negative_question_input_ids, negative_answer_input_ids, labels


# Generate negative samples
negative_question_input_ids, negative_answer_input_ids, negative_labels = create_negative_samples(question_input_ids, answer_input_ids, num_negatives=100)

# Positive samples
positive_samples = list(zip(question_input_ids, answer_input_ids))
positive_labels = [1]*len(positive_samples)

# Negative samples
negative_samples = list(zip(negative_question_input_ids, negative_answer_input_ids))
negative_labels = negative_labels  # Already set to 0

# Combine positive and negative samples
all_samples = positive_samples + negative_samples
all_labels = positive_labels + negative_labels
print("THIS IS ALL LABEL", len(all_labels))

# Create dataset class =============================================================
class QADataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        question_input_ids = self.samples[idx][0]
        answer_input_ids = self.samples[idx][1]
        label = self.labels[idx]
        return {
            'question_input_ids': question_input_ids,
            'answer_input_ids': answer_input_ids,
            'label': label
        }

# Create a dataset from the tokenized inputs
qa_dataset = QADataset(all_samples, all_labels)
print("THIS IS QA0", qa_dataset[0])
# Collate function to handle variable-length sequences =============================
def collate_fn(batch):
    question_input_ids = [item['question_input_ids'] for item in batch]
    answer_input_ids = [item['answer_input_ids'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float)

    question_input_ids = pad_sequence(question_input_ids, batch_first=True, padding_value=bpe_tokenizer.pad_token_id)
    answer_input_ids = pad_sequence(answer_input_ids, batch_first=True, padding_value=bpe_tokenizer.pad_token_id)

    attention_mask_q = (question_input_ids != bpe_tokenizer.pad_token_id).long()
    attention_mask_a = (answer_input_ids != bpe_tokenizer.pad_token_id).long()

    return {
        'question_input_ids': question_input_ids,
        'question_attention_mask': attention_mask_q,
        'answer_input_ids': answer_input_ids,
        'answer_attention_mask': attention_mask_a,
        'labels': labels
    }

# Create a DataLoader for training
qa_dataloader = DataLoader(qa_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
print("FINISHED LOADER")
# Function to get embeddings from PhoBERT ==========================================
def get_embeddings(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)
    # Mean pooling
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask
    return embeddings

# Set up the optimizer =============================================================
optimizer = torch.optim.AdamW(phobert_model.parameters(), lr=2e-5)

# Training loop ====================================================================
epochs = 10  # You can adjust the number of epochs
print("ENTER TRAINING", device)

from tqdm import tqdm
for epoch in tqdm(range(epochs)):
    phobert_model.train()
    total_loss = 0
    for batch in qa_dataloader:
        # Move inputs to device
        question_input_ids = batch['question_input_ids'].to(device)
        question_attention_mask = batch['question_attention_mask'].to(device)
        answer_input_ids = batch['answer_input_ids'].to(device)
        answer_attention_mask = batch['answer_attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Get embeddings
        question_embeddings = get_embeddings(phobert_model, question_input_ids, question_attention_mask)
        answer_embeddings = get_embeddings(phobert_model, answer_input_ids, answer_attention_mask)

        # Compute cosine similarity loss
        loss_fn = torch.nn.CosineEmbeddingLoss()
        # Convert labels 1 (similar) and 0 (dissimilar) to 1 and -1
        targets = labels * 2 - 1  # 1 becomes 1, 0 becomes -1
        loss = loss_fn(question_embeddings, answer_embeddings, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(qa_dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

phobert_model.save_pretrained('fine-tuned-phobert-embedding-model')
bpe_tokenizer.save_pretrained('fine-tuned-phobert-embedding-model')