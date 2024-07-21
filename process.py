import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset

train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)
    
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

class IMDBDataset(Dataset):
    def __init__(self, data_iter, vocab):
        self.data = list(data_iter)
        self.vocab = vocab
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, text = self.data[idx]
        label = 1 if label == 'pos' else 0
        text = torch.tensor([self.vocab[token] for token in self.tokenizer(text)], dtype=torch.long)
        return label, text

def collate_batch(batch):
    labels = torch.tensor([entry[1] for entry in batch], dtype=torch.float)
    texts = [entry[0] for entry in batch]
    lengths = [len(text) for text in texts]
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)
    return texts, labels, lengths

train_dataset = IMDBDataset(train_iter, vocab)
test_dataset = IMDBDataset(test_iter, vocab)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)
