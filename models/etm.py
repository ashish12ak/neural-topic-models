import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors

class ETM(nn.Module):
    """Embeddings-based Topic Model
    
    This model incorporates pre-trained word embeddings to enhance
    topic coherence and interpretability. Topics are represented as
    embeddings in the same space as words.
    """
    
    def __init__(self, vocab_size, num_topics, embedding_dim=300, 
                 hidden_size=500, pretrained_embeddings=None, trainable_embeddings=False):
        """Initialize ETM model
        
        Args:
            vocab_size: Size of vocabulary
            num_topics: Number of topics to learn
            embedding_dim: Dimension of word embeddings
            hidden_size: Size of hidden layer
            pretrained_embeddings: Optional pretrained word embeddings
            trainable_embeddings: Whether to train word embeddings
        """
        super(ETM, self).__init__()
        
        # Word embeddings
        if pretrained_embeddings is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=not trainable_embeddings
            )
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Topic embeddings
        self.topic_embeddings = nn.Parameter(torch.randn(num_topics, embedding_dim))
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        # Topic inference layer
        self.topic_layer = nn.Linear(hidden_size, num_topics)
        
        # Save dimensions
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.embedding_dim = embedding_dim
        
        # Initialize topic embeddings
        nn.init.xavier_uniform_(self.topic_embeddings)
    
    def get_topic_word_dist(self):
        """Get topic-word distribution through embeddings inner product
        
        Returns:
            Topic-word distribution [num_topics, vocab_size]
        """
        # Normalize topic embeddings
        topic_embeddings_norm = F.normalize(self.topic_embeddings, dim=1)
        
        # Normalize word embeddings
        word_embeddings_norm = F.normalize(self.word_embeddings.weight, dim=1)
        
        # Topic-word distribution = inner product of embeddings
        topic_word_dist = torch.mm(topic_embeddings_norm, word_embeddings_norm.t())
        
        # Apply softmax to get distribution
        topic_word_dist = F.softmax(topic_word_dist, dim=1)
        
        return topic_word_dist
    
    def get_document_topic_dist(self, x):
        """Get document-topic distribution
        
        Args:
            x: Input BoW tensor [batch_size, vocab_size]
            
        Returns:
            Document-topic distribution [batch_size, num_topics]
        """
        h = self.encoder(x)
        logits = self.topic_layer(h)
        return F.softmax(logits, dim=1)
    
    def forward(self, x):
        """Forward pass through the model
        
        Args:
            x: Input BoW tensor [batch_size, vocab_size]
            
        Returns:
            doc_word_dist: Reconstructed document-word distribution
            doc_topic_dist: Document-topic distribution
        """
        # Get document-topic distribution
        doc_topic_dist = self.get_document_topic_dist(x)
        
        # Get topic-word distribution
        topic_word_dist = self.get_topic_word_dist()
        
        # Reconstruct document distribution
        doc_word_dist = torch.mm(doc_topic_dist, topic_word_dist)
        
        return doc_word_dist, doc_topic_dist

def load_pretrained_embeddings(vocab, embedding_path, embedding_dim=300):
    """Load pretrained word embeddings for vocabulary
    
    Args:
        vocab: Vocabulary object or list of words
        embedding_path: Path to pretrained word embeddings
        embedding_dim: Embedding dimension
        
    Returns:
        Embedding tensor for vocabulary
    """
    # Get vocabulary list
    if hasattr(vocab, 'get_itos'):
        vocab_list = vocab.get_itos()
    elif hasattr(vocab, 'get_feature_names_out'):
        vocab_list = vocab.get_feature_names_out()
    else:
        vocab_list = vocab
    
    # Load word2vec model
    print(f"Loading pretrained embeddings from {embedding_path}...")
    word_vectors = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
    
    # Initialize embeddings
    embeddings = torch.randn(len(vocab_list), embedding_dim)
    
    # Fill with pretrained embeddings
    found = 0
    for i, word in enumerate(vocab_list):
        if word in word_vectors:
            embeddings[i] = torch.tensor(word_vectors[word])
            found += 1
    
    print(f"Found pretrained embeddings for {found}/{len(vocab_list)} words")
    
    return embeddings

def etm_loss(recon_x, x, beta=1.0):
    """Compute ETM loss
    
    Args:
        recon_x: Reconstructed document-word distribution
        x: Original BoW tensor
        beta: Regularization weight
        
    Returns:
        total_loss: Total loss
    """
    # Reconstruction loss (negative log likelihood)
    recon_loss = -torch.sum(x * torch.log(recon_x + 1e-10), dim=1).mean()
    
    return recon_loss

def train_etm(model, train_loader, val_loader=None, epochs=100, lr=0.002, device="cuda"):
    """Train ETM model
    
    Args:
        model: ETM model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate
        device: Device to use
        
    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            if isinstance(batch, dict):
                x = batch['bow'].to(device)
            else:
                x = batch.to(device)
            
            # Forward pass
            recon_x, _ = model(x)
            
            # Compute loss
            loss = etm_loss(recon_x, x)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            train_loss += loss.item()
        
        # Average loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, dict):
                        x = batch['bow'].to(device)
                    else:
                        x = batch.to(device)
                    
                    # Forward pass
                    recon_x, _ = model(x)
                    
                    # Compute loss
                    loss = etm_loss(recon_x, x)
                    
                    # Track loss
                    val_loss += loss.item()
                
                # Average loss
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                # Print progress
                if (epoch + 1) % 10 == 0:
                    print(f"  Val Loss: {val_loss:.4f}")
    
    return model, train_losses, val_losses

def get_topic_embeddings_tsne(model, perplexity=30):
    """Get t-SNE projection of topic embeddings
    
    Args:
        model: Trained ETM model
        perplexity: t-SNE perplexity
        
    Returns:
        t-SNE projection of topic embeddings
    """
    from sklearn.manifold import TSNE
    
    # Get topic embeddings
    topic_embeddings = model.topic_embeddings.detach().cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    topic_embeddings_tsne = tsne.fit_transform(topic_embeddings)
    
    return topic_embeddings_tsne

def get_nearest_words_to_topics(model, vocab, k=10):
    """Get nearest words to each topic embedding
    
    Args:
        model: Trained ETM model
        vocab: Vocabulary object or list of words
        k: Number of nearest words
        
    Returns:
        List of lists of nearest words to each topic
    """
    # Get vocabulary list
    if hasattr(vocab, 'get_itos'):
        vocab_list = vocab.get_itos()
    elif hasattr(vocab, 'get_feature_names_out'):
        vocab_list = vocab.get_feature_names_out()
    else:
        vocab_list = vocab
    
    # Get normalized embeddings
    topic_embeddings = F.normalize(model.topic_embeddings, dim=1)
    word_embeddings = F.normalize(model.word_embeddings.weight, dim=1)
    
    # Compute similarity
    similarity = torch.mm(topic_embeddings, word_embeddings.t())
    
    # Get top words
    top_indices = torch.topk(similarity, k=k, dim=1).indices.cpu().numpy()
    
    # Get words
    topic_words = []
    for indices in top_indices:
        topic_words.append([vocab_list[idx] for idx in indices])
    
    return topic_words 