# data_loader.py
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import amrlib
import os
import re

class AMRDataset(Dataset):
    def __init__(self, articles, highlights, vocab):
        self.articles = articles
        self.highlights = [self.text_to_ids(s, vocab, add_special=True) for s in highlights]

    def text_to_ids(self, text, vocab, add_special=False):
        ids = [vocab.word2idx.get(word, vocab.word2idx["<unk>"]) for word in text.split()]
        if add_special:
            ids = [vocab.word2idx["<sos>"]] + ids + [vocab.word2idx["<eos>"]]
        return ids

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        return (
            self.articles[idx],
            torch.tensor(self.highlights[idx])
        )

class Vocabulary:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def build_vocab(self, texts):
        print("Building vocabulary...")
        words = [word for text in texts for word in text.split()]
        word_counts = Counter(words)
        common_words = word_counts.most_common(self.vocab_size - len(self.word2idx))
        for word, _ in common_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        print(f"Vocabulary size: {len(self.word2idx)}")

def _transform_graph(graph, transformation):
    if transformation == 'OAMR':
        return graph
    elif transformation == 'OAMRWS':
        # Remove sense numbers like -01
        return re.sub(r'-\d{2}', '', graph)
    elif transformation == 'SAMR':
        # Simplified AMR
        graph = re.sub(r'\w+ / ', '', graph) # remove variables like 'w / '
        graph = graph.replace('(', '').replace(')', '') # remove brackets
        return graph
    elif transformation == 'SAMRWS':
        # Simplified AMR without sense numbers
        graph = re.sub(r'-\d{2}', '', graph)
        graph = re.sub(r'\w+ / ', '', graph)
        graph = graph.replace('(', '').replace(')', '')
        return graph
    else:
        raise ValueError(f"Unknown graph transformation: {transformation}")

def get_dataloader(csv_path, batch_size, vocab, graph_construction, graph_transformation, is_train=True):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Load AMR parser
    # We load it here to avoid pickling issues with multiprocessing in DataLoader
    stog = amrlib.load_stog_model(device='cpu')

    print("Parsing sentences into AMR graphs...")
    # In a real scenario, you'd parse per article, especially for 'combination'
    graphs = stog.parse_sents(df['article'].tolist())
    
    print("Applying graph transformations...")
    transformed_graphs = [_transform_graph(g, graph_transformation) for g in graphs]

    highlights = df['highlights'].tolist()

    if is_train:
        vocab.build_vocab(transformed_graphs + highlights)

    dataset = AMRDataset(transformed_graphs, highlights, vocab)

    def collate_fn(batch):
        srcs, trgs = zip(*batch)
        # Here we just pass the graph strings directly for model-specific tokenization (like BERT)
        # For non-BERT models, we'd convert to IDs here.
        # This is a simplification for this refactoring. A full implementation would handle this better.
        
        # For now, let's assume non-BERT and pad highlight IDs
        trgs_padded = torch.nn.utils.rnn.pad_sequence(trgs, padding_value=vocab.word2idx["<pad>"], batch_first=True)
        return srcs, trgs_padded

    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, collate_fn=collate_fn)
