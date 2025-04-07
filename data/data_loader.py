import os
import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader

def load_20newsgroups(subset='all', categories=None, remove=('headers', 'footers', 'quotes')):
    """Load 20 Newsgroups dataset
    
    Args:
        subset: 'train', 'test', or 'all'
        categories: List of categories to load (None for all)
        remove: Tuple of components to remove
        
    Returns:
        documents: List of documents
        labels: List of numeric labels
        label_names: List of category names
    """
    print(f"Loading 20 Newsgroups dataset (subset={subset})...")
    dataset = fetch_20newsgroups(subset=subset, categories=categories, remove=remove)
    documents = dataset.data
    labels = dataset.target
    label_names = dataset.target_names
    
    print(f"Loaded {len(documents)} documents with {len(label_names)} categories")
    return documents, labels, label_names

class NewsGroupsDataset(Dataset):
    def __init__(self, documents, labels, bow_matrix):
        self.documents = documents
        self.labels = labels
        self.bow_matrix = bow_matrix
        
    def __len__(self):
        return len(self.documents)
        
    def __getitem__(self, idx):
        # Convert sparse matrix to dense if needed
        if hasattr(self.bow_matrix, 'toarray'):
            bow_vector = self.bow_matrix[idx].toarray().squeeze()
        else:
            bow_vector = self.bow_matrix[idx]
            
        return {
            'document': self.documents[idx],
            'bow': torch.FloatTensor(bow_vector),
            'label': self.labels[idx]
        }

def yield_tokens(data_iter, tokenizer):
    """Yield tokens from WikiText iterator"""
    for text in data_iter:
        yield tokenizer(text)

def load_wikitext(tokenizer=None):
    """Load WikiText dataset with vocabulary
    
    Args:
        tokenizer: Text tokenizer (None for default)
        
    Returns:
        train_data: Training data iterator
        valid_data: Validation data iterator
        test_data: Test data iterator
        vocab: Vocabulary object
    """
    print("Loading WikiText-2 dataset...")
    
    try:
        # Try to get dataset iterators
        train_iter = WikiText2(split='train')
        valid_iter = WikiText2(split='valid')
        test_iter = WikiText2(split='test')
        
        # Save train data for reuse
        train_data = list(train_iter)
        train_iter = iter(train_data)
        
        if tokenizer is None:
            tokenizer = get_tokenizer('basic_english')
        
        # Build vocabulary
        vocab = build_vocab_from_iterator(
            yield_tokens(train_iter, tokenizer),
            specials=['<unk>', '<pad>', '<bos>', '<eos>']
        )
        vocab.set_default_index(vocab['<unk>'])
        
        print(f"Loaded WikiText-2 with vocabulary size: {len(vocab)}")
        
        return train_data, list(valid_iter), list(test_iter), vocab
    
    except Exception as e:
        print(f"Error loading WikiText dataset: {e}")
        print("Creating mock WikiText dataset instead...")
        
        # Create a mock vocabulary and dataset
        mock_vocab_words = ["the", "of", "and", "in", "to", "a", "is", "was", "for", "on", 
                          "that", "by", "with", "as", "at", "from", "be", "this", "an", "which",
                          "or", "have", "one", "had", "not", "but", "what", "all", "were", "when",
                          "we", "there", "can", "who", "been", "more", "if", "will", "would", "about",
                          "their", "other", "new", "some", "could", "time", "my", "than", "first", "only"]
        
        # Add special tokens
        special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
        vocab_list = special_tokens + mock_vocab_words
        
        # Create a custom Vocab class instead of using a dict
        class CustomVocab:
            def __init__(self, tokens):
                self.tokens = tokens
                self.stoi = {word: i for i, word in enumerate(tokens)}
                self.default_index = 0  # <unk> index
                
            def __getitem__(self, token):
                return self.stoi.get(token, self.default_index)
                
            def __len__(self):
                return len(self.tokens)
                
            def get_itos(self):
                return self.tokens
                
            def get_stoi(self):
                return self.stoi
                
            def set_default_index(self, index):
                self.default_index = index
        
        # Create vocab instance
        vocab = CustomVocab(vocab_list)
        
        # Create mock dataset - simple sentences
        train_data = [
            "the quick brown fox jumps over the lazy dog",
            "a journey of a thousand miles begins with a single step",
            "to be or not to be that is the question",
            "all that glitters is not gold",
            "the early bird catches the worm",
            "actions speak louder than words",
            "practice makes perfect",
            "knowledge is power",
            "time and tide wait for no man",
            "fortune favors the bold"
        ] * 10  # Repeat to get a decent sized dataset
        
        valid_data = train_data[:5]
        test_data = train_data[5:10]
        
        if tokenizer is None:
            tokenizer = lambda x: x.split()
        
        print(f"Created mock WikiText dataset with vocabulary size: {len(vocab)}")
        
        return train_data, valid_data, test_data, vocab

class WikiTextDataset(Dataset):
    def __init__(self, data, vocab, tokenizer=None, seq_length=128, bow_mode=False):
        self.data = data
        self.vocab = vocab
        self.seq_length = seq_length
        self.bow_mode = bow_mode  # If True, return BoW tensors for topic modeling
        
        # Ensure we have a valid tokenizer
        if tokenizer is None:
            # Simple tokenizer that splits on whitespace
            self.tokenizer = lambda x: x.split()
        else:
            self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer(text)
        
        # For Bag-of-Words mode (topic modeling)
        if self.bow_mode:
            # Count word occurrences
            token_counts = {}
            for token in tokens:
                if token in self.vocab.get_stoi():
                    token_idx = self.vocab[token]
                    if token_idx in token_counts:
                        token_counts[token_idx] += 1
                    else:
                        token_counts[token_idx] = 1
            
            # Create BoW tensor
            bow = torch.zeros(len(self.vocab), dtype=torch.float32)
            for idx, count in token_counts.items():
                if isinstance(idx, int) and idx < len(self.vocab):
                    bow[idx] = count
            return bow
        
        # For sequence mode (language modeling)
        else:
            # Truncate or pad to sequence length
            if len(tokens) > self.seq_length:
                tokens = tokens[:self.seq_length]
            else:
                tokens = tokens + ['<pad>'] * (self.seq_length - len(tokens))
            
            # Convert to indices
            indices = [self.vocab[token] for token in tokens]
            return torch.LongTensor(indices)

def get_dataloaders(dataset, batch_size=32, train_ratio=0.8, val_ratio=0.1, shuffle=True):
    """Split dataset and create DataLoaders"""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    if shuffle:
        np.random.shuffle(indices)
        
    train_end = int(train_ratio * dataset_size)
    val_end = train_end + int(val_ratio * dataset_size)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create samplers
    from torch.utils.data.sampler import SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, val_loader, test_loader 