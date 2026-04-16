"""Reference sentiment models with non-functional readability improvements.

This file intentionally preserves behavior from pu_model.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import html
import string
import re
from tqdm import tqdm
import math


SEED = 42
TRAIN_SPLIT = 0.9


def preprocess_text(text):
    """
    Clean and tokenize text
    """
    if isinstance(text, str):
        text = html.unescape(text)
        text = re.sub(r"<br\s*/?>", " ", text)
        text = text.lower()
        text = re.sub(r"\d+", " ", text)
        tokens = re.findall(r"[a-z']+", text)
        return [token.strip(string.punctuation) for token in tokens if token.strip(string.punctuation)]
    return []

class Vocabulary:
    """
    Build a vocabulary from the word count
    """
    def __init__(self, max_size):
        self.max_size = max_size
        # Add <cls> token for transformer classification
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<cls>"}
        self.word_count = {}
        self.size = 3  # Start with pad, unk, and cls tokens
        
    def add_word(self, word):
        if not isinstance(word, str):
            return
        word = word.strip()
        if not word or word in {"<pad>", "<unk>", "<cls>"}:
            return
        if word not in self.word_count:
            self.word_count[word] = 0
        self.word_count[word] += 1
            
    def build_vocab(self):
        sorted_words = sorted(
            self.word_count.items(),
            key=lambda item: (-item[1], item[0]),
        )
        for word, _ in sorted_words:
            if self.size >= self.max_size:
                break
            if word not in self.word2idx:
                self.word2idx[word] = self.size
                self.idx2word[self.size] = word
                self.size += 1
                
    def text_to_indices(self, tokens, max_len, model_type="lstm"):
        if model_type == "transformer":
            indices = [self.word2idx["<cls>"]]
            indices.extend(self.word2idx[token] for token in tokens if token in self.word2idx)
        else:
            indices = [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]
        
        if len(indices) < max_len:
            indices += [self.word2idx["<pad>"]] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
            
        return indices

class IMDBDataset(Dataset):
    """
    A dataset for the IMDB dataset
    """
    def __init__(self, dataframe, vocabulary, max_len, is_training=True, model_type="lstm"):
        self.dataframe = dataframe
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.model_type = model_type

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # Support both Kaggle parquet ('text'/'label') and autograder data ('review'/'sentiment')
        text_col = "review" if "review" in self.dataframe.columns else "text"
        label_col = "sentiment" if "sentiment" in self.dataframe.columns else "label"
        text, label = row[text_col], row[label_col]
        
        tokens = preprocess_text(text)
        indices = self.vocabulary.text_to_indices(tokens, self.max_len, self.model_type)
        
        text_tensor = torch.LongTensor(indices)
        if isinstance(label, str):
            label_tensor = torch.FloatTensor([1 if label == "positive" else 0])
        else:
            label_tensor = torch.FloatTensor([float(label)])
        
        if self.model_type == "transformer":
            attention_mask = (text_tensor != self.vocabulary.word2idx["<pad>"]).float()
            return text_tensor, attention_mask, label_tensor
        else:
            return text_tensor, label_tensor


# LSTM model
class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size=20000,
        embedding_dim=300,
        hidden_dim=512,
        output_dim=1,
        n_layers=2,
        bidirectional=True,
        dropout=0.3,
        pad_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.bidirectional = bidirectional
        self.pad_idx = pad_idx

    def forward(self, text=None, input_ids=None, attention_mask=None):
        if text is None:
            text = input_ids
        if text is None:
            raise ValueError("LSTM.forward expected text or input_ids")
        if text.dim() == 0:
            text = text.unsqueeze(0)
        if text.dim() == 1:
            text = text.unsqueeze(0)

        embedded = self.dropout(self.embedding(text))
        lengths = text.ne(self.pad_idx).sum(dim=1).cpu()
        lengths = torch.clamp(lengths, min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        return self.fc(self.dropout(hidden))


# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) — batch_first compatible
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size=20000,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        max_len=400,
        dropout=0.2,
        pad_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)
        self.pad_idx = pad_idx

    def forward(self, text=None, attention_mask=None, input_ids=None):
        if text is None:
            text = input_ids
        if text is None:
            raise ValueError("TransformerEncoder.forward expected text or input_ids")
        if text.dim() == 0:
            text = text.unsqueeze(0)
        if text.dim() == 1:
            text = text.unsqueeze(0)

        if attention_mask is None:
            src_key_padding_mask = text.eq(self.pad_idx)
            attention_mask = text.ne(self.pad_idx).long()
        else:
            src_key_padding_mask = attention_mask.eq(0)

        embedded = self.embedding(text) * math.sqrt(self.embedding.embedding_dim)
        encoded = self.encoder(
            self.positional_encoding(self.dropout(embedded)),
            src_key_padding_mask=src_key_padding_mask,
        )
        encoded = self.norm(encoded)

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.fc(self.dropout(pooled))

def _normalize_labels(series):
    if series.dtype.kind in {"i", "u", "f", "b"}:
        return series.astype(int)

    label_map = {
        "negative": 0,
        "neg": 0,
        "0": 0,
        "false": 0,
        "positive": 1,
        "pos": 1,
        "1": 1,
        "true": 1,
    }

    normalized = series.astype(str).str.strip().str.lower().map(label_map)
    if normalized.isnull().any():
        unknown_values = series[normalized.isnull()].dropna().unique().tolist()
        raise ValueError(f"Unsupported label values found: {unknown_values[:5]}")

    return normalized.astype(int)


def load_and_preprocess_data(data_path, data_type="train", model_type="lstm", shared_vocab=None, batch_size=64, max_len=400, max_vocab_size=20000):
    """
    Load and preprocess the IMDB dataset
    
    Args:
        data_path: Path to the data files
        data_type: Type of data to load ('train' or 'test')
        model_type: Type of model ('lstm' or 'transformer')
        shared_vocab: Optional vocabulary to use (for test data)
    
    Returns:
        data_loader: DataLoader for the specified data type
        vocab: Vocabulary object (only returned for train data)
    """
    normalized_data_type = {
        "validation": "valid",
        "val": "valid",
    }.get(str(data_type).strip().lower(), str(data_type).strip().lower())

    df = pd.read_parquet(data_path).copy()

    expected_columns = {"text", "label"}
    if not expected_columns.issubset(df.columns):
        rename_map = {}
        text_candidates = ["review", "content", "sentence", "document"]
        label_candidates = ["sentiment", "target", "y", "class"]

        for candidate in text_candidates:
            if candidate in df.columns:
                rename_map[candidate] = "text"
                break

        for candidate in label_candidates:
            if candidate in df.columns:
                rename_map[candidate] = "label"
                break

        df = df.rename(columns=rename_map)

    if "text" not in df.columns:
        raise ValueError("Expected a text column such as 'text' or 'review'.")

    df["text"] = df["text"].fillna("").astype(str)
    if "label" in df.columns:
        df["label"] = _normalize_labels(df["label"])

    split_df = df
    if normalized_data_type in {"train", "valid"} and "label" in df.columns:
        try:
            train_df, valid_df = train_test_split(
                df,
                train_size=TRAIN_SPLIT,
                random_state=SEED,
                stratify=df["label"],
            )
        except ValueError:
            train_df, valid_df = train_test_split(
                df,
                train_size=TRAIN_SPLIT,
                random_state=SEED,
                stratify=None,
            )

        split_df = train_df if normalized_data_type == "train" else valid_df
    else:
        train_df = df

    if normalized_data_type == "train":
        if shared_vocab is None:
            vocab = Vocabulary(max_vocab_size)
            for text in tqdm(train_df["text"], desc="Building vocabulary"):
                for token in preprocess_text(text):
                    vocab.add_word(token)
            vocab.build_vocab()
        else:
            vocab = shared_vocab

        train_dataset = IMDBDataset(split_df.reset_index(drop=True), vocab, max_len, model_type=model_type)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, vocab

    if normalized_data_type == "valid":
        val_dataset = IMDBDataset(split_df.reset_index(drop=True), shared_vocab, max_len, model_type=model_type)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return val_loader

    # test mode (loads full file)
    test_dataset = IMDBDataset(df.reset_index(drop=True), shared_vocab, max_len, model_type=model_type)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def train(model, iterator, optimizer, criterion, device, model_type="lstm", label_smoothing=0.0):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        
        if model_type == "transformer":
            text, mask, labels = batch
            text, labels, mask = text.to(device), labels.to(device), mask.to(device)
            predictions = model(input_ids=text, attention_mask=mask).squeeze(1)
        else:
            text, labels = batch
            text, labels = text.to(device), labels.to(device)
            predictions = model(text).squeeze(1)
        
        targets = labels.squeeze(1)
        if label_smoothing > 0.0:
            # Shift targets: 1 → (1 - smooth/2), 0 → smooth/2
            targets = targets * (1.0 - label_smoothing) + label_smoothing * 0.5
        loss = criterion(predictions, targets)
        
        rounded_preds = torch.round(torch.sigmoid(predictions))
        correct = (rounded_preds == labels.squeeze(1)).float()
        acc = correct.sum() / len(correct)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device, model_type='lstm'):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            if model_type == "transformer":
                text, mask, labels = batch
                text, labels, mask = text.to(device), labels.to(device), mask.to(device)
                predictions = model(input_ids=text, attention_mask=mask).squeeze(1)
            else:
                text, labels = batch
                text, labels = text.to(device), labels.to(device)
                predictions = model(text).squeeze(1)
            
            loss = criterion(predictions, labels.squeeze(1))
            
            rounded_preds = torch.round(torch.sigmoid(predictions))
            correct = (rounded_preds == labels.squeeze(1)).float()
            acc = correct.sum() / len(correct)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def main():
    # Hyperparameters
    MAX_VOCAB_SIZE = 20000
    MAX_LEN = 256
    BATCH_SIZE = 32
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.3
    N_EPOCHS = 20
    
    # Transformer specific
    D_MODEL = 256
    NHEAD = 4
    NUM_ENCODER_LAYERS = 3
    DIM_FEEDFORWARD = 1024
    TRANSFORMER_DROPOUT = 0.25
    TRANSFORMER_LR = 1.0e-4
    WARMUP_EPOCHS = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_path = "/kaggle/input/datasets/pranavuttarkar/hw5train/hw5_data_train.parquet"

    # --- LSTM ---
    # print("--- Training LSTM ---")
    # train_loader, vocab = load_and_preprocess_data(data_path, data_type='train', model_type='lstm', batch_size=BATCH_SIZE, max_len=MAX_LEN, max_vocab_size=MAX_VOCAB_SIZE)
    # val_loader = load_and_preprocess_data(data_path, data_type='val', model_type='lstm', shared_vocab=vocab, batch_size=BATCH_SIZE, max_len=MAX_LEN)
    
    # lstm_model = LSTM().to(device)
    # optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    # criterion = nn.BCEWithLogitsLoss()

    # best_valid_loss = float('inf')

    # LSTM_EPOCHS = 8
    # for epoch in range(LSTM_EPOCHS):
    #     train_loss, train_acc = train(lstm_model, train_loader, optimizer, criterion, device, model_type='lstm')
    #     valid_loss, valid_acc = evaluate(lstm_model, val_loader, criterion, device, model_type='lstm')
    #     scheduler.step(valid_loss)
        
    #     if valid_loss < best_valid_loss:
    #         best_valid_loss = valid_loss
    #         torch.save(lstm_model.state_dict(), 'lstm.pt')
        
    #     print(f'Epoch: {epoch+1:02}')
    #     print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    #     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    # --- Transformer ---
    print("\n--- Training Transformer ---")
    train_loader, vocab = load_and_preprocess_data(data_path, data_type="train", model_type="transformer", batch_size=BATCH_SIZE, max_len=MAX_LEN, max_vocab_size=MAX_VOCAB_SIZE)
    val_loader = load_and_preprocess_data(data_path, data_type="val", model_type="transformer", shared_vocab=vocab, batch_size=BATCH_SIZE, max_len=MAX_LEN)

    transformer_model = TransformerEncoder().to(device)
    optimizer = optim.AdamW(transformer_model.parameters(), lr=TRANSFORMER_LR, weight_decay=1e-2)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(WARMUP_EPOCHS)
        progress = (epoch - WARMUP_EPOCHS) / max(1, N_EPOCHS - WARMUP_EPOCHS)
        return max(1e-6 / TRANSFORMER_LR, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.BCEWithLogitsLoss()

    best_valid_acc = 0.0

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(transformer_model, train_loader, optimizer, criterion, device, model_type="transformer", label_smoothing=0.05)
        valid_loss, valid_acc = evaluate(transformer_model, val_loader, criterion, device, model_type="transformer")
        scheduler.step()

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(transformer_model.state_dict(), "transformer.pt")
        
        print(f"Epoch: {epoch+1:02}")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

if __name__ == "__main__":
    main()