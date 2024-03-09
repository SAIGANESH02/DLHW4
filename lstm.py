import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import numpy as np
from tqdm import tqdm

# Assuming `corpus` is a list of sentences from the Wikitext-2 dataset
def preprocess_data(corpus, vocab_size=10000, unk_token='<unk>'):
    flat_list = corpus.split(' ')

    word_counts = Counter(flat_list)

    # Create vocabulary
    vocab = [word for word, count in word_counts.most_common(vocab_size)]
    vocab.append(unk_token)

    # Create word to index and index to word dictionaries
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    print(len(word_to_idx))

    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    # Replace rare words with <unk>
    processed_corpus = [word if word in word_to_idx else unk_token for word in flat_list]
    encoded_corpus = [word_to_idx[word] for word in processed_corpus]

    return encoded_corpus, word_to_idx, idx_to_word

class TextDataset(Dataset):
    def __init__(self, corpus, seq_length):
        self.corpus = corpus
        self.seq_length = seq_length

    def __len__(self):
        return len(self.corpus) - self.seq_length

    def __getitem__(self, idx):
        return (torch.tensor(self.corpus[idx:idx+self.seq_length]), torch.tensor(self.corpus[idx+self.seq_length:idx+self.seq_length+1]))

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=100, dropout_rate=0.2):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_dim), torch.zeros(1, batch_size, self.hidden_dim))


# Read corpus from the file in data folder
with open('wiki2.train.txt', 'r', encoding='utf-8') as f:
    corpus = f.read()
    encoded_corpus, word_to_idx, idx_to_word = preprocess_data(corpus)

# Hyperparameters
vocab_size = len(word_to_idx) + 1
embed_size = 32
hidden_size = 32
dropout = 0.2
seq_length = 10
batch_size = 256
learning_rate = 0.001
num_epochs = 10

# print all the hyperparameters
print(f'Vocab size: {vocab_size}')
print(f'Embedding size: {embed_size}')
print(f'Hidden size: {hidden_size}')
print(f'Dropout: {dropout}')
print(f'Sequence length: {seq_length}')
print(f'Batch size: {batch_size}')
print(f'Learning rate: {learning_rate}')
print(f'Number of epochs: {num_epochs}')


# Create dataset and dataloader
dataset = TextDataset(encoded_corpus, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Create model, loss function and optimizer
model = LSTMLanguageModel(vocab_size, embed_size, hidden_size, dropout)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


torch.autograd.set_detect_anomaly(True)
for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader):
        hidden = model.init_hidden(batch_size)
        model.zero_grad()
        output, hidden = model(inputs, hidden)
        predictions = output[:, -1, :]
        loss = criterion(predictions, targets.view(-1))
        loss.backward(retain_graph=True)
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch}, Loss: {total_loss / len(dataloader)}')
