import html
import math
import re
import string
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


SEED = 42
MAX_VOCAB_SIZE = 20000
MAX_LEN = 400
BATCH_SIZE = 32
EMBEDDING_DIM = 300
HIDDEN_DIM = 512
NUM_LAYERS = 2
TRANSFORMER_HEADS = 8
DROPOUT = 0.3
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 12
TRAIN_SPLIT = 0.9
GRAD_CLIP = 1.0
TRANSFORMER_BATCH_SIZE = 24
TRANSFORMER_EMBEDDING_DIM = 256
TRANSFORMER_HIDDEN_DIM = 512
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_DROPOUT = 0.2
TRANSFORMER_LEARNING_RATE = 2e-4
TRANSFORMER_WEIGHT_DECAY = 1e-4
TRANSFORMER_EPOCHS = 14


def get_runtime_device():
    """
    Pick the best available execution device.

    Uses CUDA when it is available and can successfully execute a small probe;
    otherwise falls back to CPU so the code remains portable for autograders.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        device_arch = "sm_{}{}".format(capability[0], capability[1])
        arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []

        if arch_list and device_arch not in arch_list:
            print(
                "Warning: CUDA device '{}' uses architecture {}, while this PyTorch build reports support for {}. "
                "Proceeding with a runtime CUDA/LSTM probe."
                .format(device_name, device_arch, ", ".join(arch_list))
            )

        # Validate the actual runtime instead of relying only on the reported
        # architecture list, which can be conservative on some cluster builds.
        test_tensor = torch.randn(2, 2, device="cuda")
        _ = test_tensor @ test_tensor.t()

        lstm_probe = nn.LSTM(input_size=4, hidden_size=4, batch_first=True).to("cuda")
        probe_input = torch.randn(1, 3, 4, device="cuda")
        _ = lstm_probe(probe_input)
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception as exc:
        print("Warning: CUDA probe failed ({}). Falling back to CPU.".format(exc))
        return torch.device("cpu")


def preprocess_text(text):
    """
    Clean and tokenize text.
    """
    if not isinstance(text, str):
        return []

    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = text.lower()
    text = re.sub(r"\d+", " ", text)

    # Keep alphabetic tokens and simple contractions without relying on NLTK.
    tokens = re.findall(r"[a-z']+", text)
    return [token.strip(string.punctuation) for token in tokens if token.strip(string.punctuation)]


class Vocabulary:
    """
    Build a vocabulary from word counts.
    """

    def __init__(self, max_size=MAX_VOCAB_SIZE):
        self.max_size = max_size
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<cls>"}
        self.word_count = {}
        self.size = 3

    def add_word(self, word):
        self.word_count[word] = self.word_count.get(word, 0) + 1

    def build_vocab(self):
        sorted_words = sorted(
            self.word_count.items(),
            key=lambda item: (-item[1], item[0]),
        )

        for word, _ in sorted_words[: max(0, self.max_size - self.size)]:
            self.word2idx[word] = self.size
            self.idx2word[self.size] = word
            self.size += 1

    def text_to_indices(self, tokens, max_len, model_type="lstm"):
        if model_type == "transformer":
            # Match the skeleton note by skipping OOV tokens for the transformer path.
            indices = [self.word2idx["<cls>"]]
            indices.extend(self.word2idx[token] for token in tokens if token in self.word2idx)
        else:
            indices = [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]

        indices = indices[:max_len]
        pad_length = max_len - len(indices)

        if pad_length > 0:
            indices += [self.word2idx["<pad>"]] * pad_length

        return indices


class IMDBDataset(Dataset):
    """
    A dataset for IMDB-style sentiment classification.
    """

    def __init__(self, dataframe, vocabulary, max_len, is_training=True, model_type="lstm"):
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.is_training = is_training
        self.model_type = model_type
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist() if "label" in dataframe.columns else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = preprocess_text(self.texts[idx])
        indices = self.vocabulary.text_to_indices(tokens, self.max_len, self.model_type)
        text_tensor = torch.tensor(indices, dtype=torch.long)
        attention_mask = (text_tensor != self.vocabulary.word2idx["<pad>"]).long()

        if not self.is_training or self.labels is None:
            if self.model_type == "transformer":
                return text_tensor, attention_mask
            return text_tensor

        label_tensor = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        if self.model_type == "transformer":
            return text_tensor, label_tensor, attention_mask
        return text_tensor, label_tensor


class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size=MAX_VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=1,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=0,
        bidirectional=True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.bidirectional = bidirectional
        self.pad_idx = pad_idx

    def forward(self, text):
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

        logits = self.fc(self.dropout(hidden))
        return logits.reshape(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class PositionalEmbedding(PositionalEncoding):
    pass


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size=MAX_VOCAB_SIZE,
        embedding_dim=TRANSFORMER_EMBEDDING_DIM,
        n_heads=TRANSFORMER_HEADS,
        hidden_dim=TRANSFORMER_HIDDEN_DIM,
        num_layers=TRANSFORMER_NUM_LAYERS,
        dropout=TRANSFORMER_DROPOUT,
        output_dim=1,
        pad_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=MAX_LEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.pad_idx = pad_idx

    def forward(self, text, attention_mask=None):
        if text.dim() == 1:
            text = text.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

        if attention_mask is None:
            src_key_padding_mask = text.eq(self.pad_idx)
        else:
            src_key_padding_mask = attention_mask.eq(0)

        embedded = self.embedding(text) * math.sqrt(self.embedding.embedding_dim)
        encoded = self.encoder(
            self.positional_encoding(self.dropout(embedded)),
            src_key_padding_mask=src_key_padding_mask,
        )
        encoded = self.norm(encoded)

        if attention_mask is None:
            attention_mask = text.ne(self.pad_idx).long()

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        logits = self.fc(self.dropout(pooled))
        return logits.reshape(-1)


def _resolve_data_path(data_path, data_type):
    if data_path.endswith(".parquet"):
        return data_path

    candidate_paths = [
        f"{data_path.rstrip('/')}/hw5_data_{data_type}.parquet",
        f"{data_path.rstrip('/')}/{data_type}.parquet",
        f"{data_path.rstrip('/')}/data_{data_type}.parquet",
    ]

    for candidate in candidate_paths:
        try:
            with open(candidate, "rb"):
                return candidate
        except FileNotFoundError:
            continue

    raise FileNotFoundError(f"Could not resolve a parquet file for data_type='{data_type}' from '{data_path}'.")


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


def load_and_preprocess_data(data_path, data_type="train", model_type="lstm", shared_vocab=None):
    """
    Load and preprocess the IMDB dataset.

    Args:
        data_path: Path to the data files
        data_type: Type of data to load ('train' or 'test')
        model_type: Type of model ('lstm' or 'transformer')
        shared_vocab: Optional vocabulary to use (for evaluation data)

    Returns:
        data_loader: DataLoader for the specified data type
        vocab: Vocabulary object (only returned for train data)
    """

    parquet_path = _resolve_data_path(data_path, data_type)
    dataframe = pd.read_parquet(parquet_path).copy()

    expected_columns = {"text", "label"}
    if not expected_columns.issubset(dataframe.columns):
        rename_map = {}
        text_candidates = ["review", "content", "sentence", "document"]
        label_candidates = ["sentiment", "target", "y", "class"]

        for candidate in text_candidates:
            if candidate in dataframe.columns:
                rename_map[candidate] = "text"
                break

        for candidate in label_candidates:
            if candidate in dataframe.columns:
                rename_map[candidate] = "label"
                break

        dataframe = dataframe.rename(columns=rename_map)

    if "text" not in dataframe.columns:
        raise ValueError("Expected a text column such as 'text' or 'review'.")

    dataframe["text"] = dataframe["text"].fillna("").astype(str)

    if "label" in dataframe.columns:
        dataframe["label"] = _normalize_labels(dataframe["label"])

    split_df = dataframe
    if data_type in {"train", "valid", "val", "validation"} and "label" in dataframe.columns:
        train_df, valid_df = train_test_split(
            dataframe,
            train_size=TRAIN_SPLIT,
            random_state=SEED,
            stratify=dataframe["label"],
        )
        split_df = train_df if data_type == "train" else valid_df
    else:
        train_df = dataframe

    if shared_vocab is None:
        vocab = Vocabulary(max_size=MAX_VOCAB_SIZE)
        for text in tqdm(train_df["text"], desc="Building vocabulary"):
            for token in preprocess_text(text):
                vocab.add_word(token)
        vocab.build_vocab()
    else:
        vocab = shared_vocab

    dataset = IMDBDataset(
        dataframe=split_df.reset_index(drop=True),
        vocabulary=vocab,
        max_len=MAX_LEN,
        is_training="label" in split_df.columns,
        model_type=model_type,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=TRANSFORMER_BATCH_SIZE if model_type == "transformer" else BATCH_SIZE,
        shuffle=data_type == "train",
        num_workers=0,
    )

    if data_type == "train":
        return data_loader, vocab
    return data_loader


def train(model, iterator, optimizer, criterion, device, model_type="lstm"):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(iterator, desc=f"Training {model_type}", leave=False):
        if model_type == "transformer":
            text, labels, attention_mask = batch
            text = text.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
        else:
            text, labels = batch
            text = text.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()
        if model_type == "transformer":
            predictions = model(text, attention_mask)
        else:
            predictions = model(text)
        loss = criterion(predictions, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        epoch_loss += loss.item()
        predicted_labels = (torch.sigmoid(predictions) >= 0.5).float()
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

    average_loss = epoch_loss / max(len(iterator), 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


def evaluate(model, iterator, criterion, device, model_type="lstm"):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(iterator, desc=f"Evaluating {model_type}", leave=False):
            if model_type == "transformer":
                text, labels, attention_mask = batch
                text = text.to(device)
                labels = labels.to(device)
                attention_mask = attention_mask.to(device)
            else:
                text, labels = batch
                text = text.to(device)
                labels = labels.to(device)

            if model_type == "transformer":
                predictions = model(text, attention_mask)
            else:
                predictions = model(text)
            loss = criterion(predictions, labels)

            epoch_loss += loss.item()
            predicted_labels = (torch.sigmoid(predictions) >= 0.5).float()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

    average_loss = epoch_loss / max(len(iterator), 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


def build_model(model_type, vocab, device):
    if model_type == "transformer":
        model = TransformerEncoder(
            vocab_size=vocab.size,
            embedding_dim=TRANSFORMER_EMBEDDING_DIM,
            n_heads=TRANSFORMER_HEADS,
            hidden_dim=TRANSFORMER_HIDDEN_DIM,
            num_layers=TRANSFORMER_NUM_LAYERS,
            dropout=TRANSFORMER_DROPOUT,
            pad_idx=vocab.word2idx["<pad>"],
        )
    else:
        model = LSTM(
            vocab_size=vocab.size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            pad_idx=vocab.word2idx["<pad>"],
        )

    return model.to(device)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data_path = "hw5_data_train.parquet"
    device = get_runtime_device()
    model_type = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "lstm"
    if model_type not in {"lstm", "transformer"}:
        raise ValueError("model_type must be 'lstm' or 'transformer'. Example: python3 hw5_ske.py lstm")

    checkpoint_path = "transformer.pt" if model_type == "transformer" else "lstm.pt"

    print(f"Using device: {device}")
    train_loader, vocab = load_and_preprocess_data(data_path, data_type="train", model_type=model_type)
    valid_loader = load_and_preprocess_data(
        data_path,
        data_type="valid",
        model_type=model_type,
        shared_vocab=vocab,
    )

    model = build_model(model_type, vocab, device)

    if model_type == "transformer":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=TRANSFORMER_LEARNING_RATE,
            weight_decay=TRANSFORMER_WEIGHT_DECAY,
        )
        num_epochs = TRANSFORMER_EPOCHS
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        num_epochs = NUM_EPOCHS
    criterion = nn.BCEWithLogitsLoss()

    best_valid_accuracy = 0.0
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device, model_type=model_type)
        valid_loss, valid_accuracy = evaluate(model, valid_loader, criterion, device, model_type=model_type)

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            torch.save(model.state_dict(), checkpoint_path)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_accuracy:.4f}"
        )

    print(f"Best valid accuracy: {best_valid_accuracy:.4f}")
    print(f"Saved best {model_type} checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
