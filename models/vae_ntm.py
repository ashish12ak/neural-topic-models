import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

class VAE_NTM(nn.Module):
    """VAE-based Neural Topic Model
    
    This model implements a Variational Autoencoder for topic modeling.
    The encoder maps documents to a latent topic space, and the decoder
    reconstructs documents from the latent representation.
    """
    
    def __init__(self, vocab_size, num_topics, hidden_size=500, dropout=0.2):
        """Initialize VAE-NTM model
        
        Args:
            vocab_size: Size of vocabulary
            num_topics: Number of topics to learn
            hidden_size: Size of hidden layer
            dropout: Dropout rate
        """
        super(VAE_NTM, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Mean and log variance layers
        self.mu_layer = nn.Linear(hidden_size, num_topics)
        self.logvar_layer = nn.Linear(hidden_size, num_topics)
        
        # Decoder (topic-word distribution)
        self.decoder = nn.Linear(num_topics, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)
        
        # Initialize decoder weight
        nn.init.xavier_uniform_(self.decoder.weight)
        
        # Save dimensions
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        
    def encode(self, x):
        """Encode input to latent parameters
        
        Args:
            x: Input BoW tensor [batch_size, vocab_size]
            
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Sample from latent distribution using reparameterization trick
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, just return mean
            return mu
    
    def decode(self, z):
        """Decode latent vector to document distribution
        
        Args:
            z: Latent vector [batch_size, num_topics]
            
        Returns:
            Document-word distribution [batch_size, vocab_size]
        """
        logits = self.decoder(z)
        return F.softmax(self.bn(logits), dim=1)
    
    def forward(self, x):
        """Forward pass through the model
        
        Args:
            x: Input BoW tensor [batch_size, vocab_size]
            
        Returns:
            recon_x: Reconstructed document distribution
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def get_topic_word_dist(self, k=10):
        """Get topic-word distribution
        
        Args:
            k: Number of top words per topic
            
        Returns:
            topic_words: List of lists of top words per topic
            topic_word_probs: Topic-word distribution matrix
        """
        # Get decoder weights
        topic_word_dist = F.softmax(self.decoder.weight.detach(), dim=1).T
        return topic_word_dist

def get_topic_words(topic_word_dist, vocab, k=10):
    """Get top words for each topic
    
    Args:
        topic_word_dist: Topic-word distribution [num_topics, vocab_size]
        vocab: Vocabulary object or list of words
        k: Number of top words per topic
        
    Returns:
        List of lists of top words per topic
    """
    # Get top word indices
    top_indices = torch.topk(topic_word_dist, k=k, dim=1).indices.cpu().numpy()
    
    # Get vocabulary list
    if hasattr(vocab, 'get_itos'):
        # For torchtext Vocab
        vocab_list = vocab.get_itos()
    elif hasattr(vocab, 'get_feature_names_out'):
        # For sklearn vectorizers
        vocab_list = vocab.get_feature_names_out()
    else:
        # Assume it's already a list
        vocab_list = vocab
    
    # Get top words
    topic_words = []
    for indices in top_indices:
        topic_words.append([vocab_list[idx] for idx in indices])
    
    return topic_words

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """Compute VAE loss
    
    Args:
        recon_x: Reconstructed document-word distribution
        x: Original BoW tensor
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight of KL divergence term
        
    Returns:
        total_loss: Total loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence
    """
    # Reconstruction loss (negative log likelihood)
    recon_loss = -torch.sum(x * torch.log(recon_x + 1e-10), dim=1).mean()
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

def train_vae_ntm(model, train_loader, val_loader=None, epochs=100, lr=0.002, beta=1.0, device="cuda"):
    """Train VAE-NTM model
    
    Args:
        model: VAE-NTM model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate
        beta: Weight of KL divergence term
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
        train_recon_loss = 0
        train_kl_loss = 0
        
        for batch in train_loader:
            if isinstance(batch, dict):
                x = batch['bow'].to(device)
            else:
                x = batch.to(device)
            
            # Forward pass
            recon_x, mu, logvar = model(x)
            
            # Compute loss
            loss, recon, kl = vae_loss(recon_x, x, mu, logvar, beta)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track losses
            train_loss += loss.item()
            train_recon_loss += recon.item()
            train_kl_loss += kl.item()
        
        # Average losses
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)
        train_losses.append((train_loss, train_recon_loss, train_kl_loss))
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"    Recon Loss: {train_recon_loss:.4f}")
            print(f"    KL Loss: {train_kl_loss:.4f}")
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            val_recon_loss = 0
            val_kl_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, dict):
                        x = batch['bow'].to(device)
                    else:
                        x = batch.to(device)
                    
                    # Forward pass
                    recon_x, mu, logvar = model(x)
                    
                    # Compute loss
                    loss, recon, kl = vae_loss(recon_x, x, mu, logvar, beta)
                    
                    # Track losses
                    val_loss += loss.item()
                    val_recon_loss += recon.item()
                    val_kl_loss += kl.item()
                
                # Average losses
                val_loss /= len(val_loader)
                val_recon_loss /= len(val_loader)
                val_kl_loss /= len(val_loader)
                val_losses.append((val_loss, val_recon_loss, val_kl_loss))
                
                # Print progress
                if (epoch + 1) % 10 == 0:
                    print(f"  Val Loss: {val_loss:.4f}")
                    print(f"    Recon Loss: {val_recon_loss:.4f}")
                    print(f"    KL Loss: {val_kl_loss:.4f}")
    
    return model, train_losses, val_losses

def get_document_topic_dist(model, dataloader, device="cuda"):
    """Get document-topic distribution for all documents
    
    Args:
        model: Trained VAE-NTM model
        dataloader: Data loader
        device: Device to use
        
    Returns:
        Document-topic distribution matrix
    """
    model.to(device)
    model.eval()
    
    doc_topic_dist = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch['bow'].to(device)
            else:
                x = batch.to(device)
            
            # Encode documents to get topic distribution
            mu, _ = model.encode(x)
            theta = F.softmax(mu, dim=1)
            
            # Add to list
            doc_topic_dist.append(theta.cpu().numpy())
    
    # Concatenate batches
    doc_topic_dist = np.vstack(doc_topic_dist)
    
    return doc_topic_dist 