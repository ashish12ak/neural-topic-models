import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

# Update the GRAPH_MODULES_AVAILABLE flag to False to use simplified version
GRAPH_MODULES_AVAILABLE = False

# Define a simplified GraphNTM that doesn't require torch_geometric
class GraphNTM(nn.Module):
    """Simplified Graph-based Neural Topic Model
    
    This is a simplified version that doesn't require torch_geometric.
    It still uses the concept of modeling relationships between words and topics
    but with standard neural network components.
    """
    
    def __init__(self, vocab_size, num_topics, hidden_size=200, embedding_dim=128, dropout=0.2):
        """Initialize simplified GraphNTM model
        
        Args:
            vocab_size: Size of vocabulary
            num_topics: Number of topics to learn
            hidden_size: Size of hidden layers
            embedding_dim: Dimension of embeddings
            dropout: Dropout rate
        """
        super(GraphNTM, self).__init__()
        
        # Node embeddings (words, topics)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.topic_embeddings = nn.Parameter(torch.randn(num_topics, embedding_dim))
        
        # Document encoder for topic inference
        self.document_encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Topic inference layer
        self.topic_layer = nn.Linear(hidden_size, num_topics)
        
        # Word prediction layer
        self.word_decoder = nn.Linear(num_topics, vocab_size)
        
        # Save dimensions
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.word_embeddings.weight)
        nn.init.xavier_uniform_(self.topic_embeddings)
        
    def get_topic_word_dist(self):
        """Get topic-word distribution
        
        Returns:
            Topic-word distribution [num_topics, vocab_size]
        """
        # Compute similarity between topic and word embeddings
        topic_word_sim = torch.mm(self.topic_embeddings, self.word_embeddings.weight.t())
        
        # Apply softmax to get distribution
        topic_word_dist = F.softmax(topic_word_sim, dim=1)
        
        return topic_word_dist
    
    def get_document_topic_dist(self, x):
        """Get document-topic distribution
        
        Args:
            x: Input BoW tensor [batch_size, vocab_size]
            
        Returns:
            Document-topic distribution [batch_size, num_topics]
        """
        h = self.document_encoder(x)
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
        
        # Reconstruct document distribution using standard topic model approach
        doc_word_dist = torch.mm(doc_topic_dist, topic_word_dist)
        
        return doc_word_dist, doc_topic_dist

# Define training function that doesn't depend on graph structure
def train_graph_ntm(model, train_loader, val_loader, epochs=100, lr=0.001, 
                  device='cuda', beta=1.0, with_graph_structure=False):
    """Train GraphNTM model
    
    Args:
        model: GraphNTM model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate
        device: Device to use
        beta: KL divergence weight
        with_graph_structure: Flag to use graph structure (ignored in simplified version)
        
    Returns:
        model: Trained model
        train_losses: Training losses
        val_losses: Validation losses
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            if isinstance(batch, dict):
                x = batch['bow'].to(device)
            else:
                x = batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, doc_topic_dist = model(x)
            
            # Reconstruction loss
            recon_loss = F.kl_div(reconstructed.log(), x.float(), reduction='batchmean')
            
            # Topic diversity regularization
            topic_word_dist = model.get_topic_word_dist()
            diversity_loss = topic_diversity_loss(topic_word_dist)
            
            # Total loss
            loss = recon_loss + beta * diversity_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    x = batch['bow'].to(device)
                else:
                    x = batch.to(device)
                
                reconstructed, doc_topic_dist = model(x)
                recon_loss = F.kl_div(reconstructed.log(), x.float(), reduction='batchmean')
                
                topic_word_dist = model.get_topic_word_dist()
                diversity_loss = topic_diversity_loss(topic_word_dist)
                
                loss = recon_loss + beta * diversity_loss
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress every 10 epochs or at the end
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
    
    return model, train_losses, val_losses

def topic_diversity_loss(topic_word_dist):
    """Calculate topic diversity loss
    
    Args:
        topic_word_dist: Topic-word distribution [num_topics, vocab_size]
        
    Returns:
        loss: Topic diversity loss
    """
    num_topics = topic_word_dist.size(0)
    
    # Calculate pairwise cosine similarity between topics
    topic_sim = torch.mm(topic_word_dist, topic_word_dist.t())
    
    # Normalize by vector norms
    norms = torch.norm(topic_word_dist, dim=1, keepdim=True)
    norm_matrix = torch.mm(norms, norms.t())
    topic_sim = topic_sim / norm_matrix
    
    # We want topics to be different, so we minimize similarity
    # Excluding self-similarity by masking diagonal
    eye = torch.eye(num_topics, device=topic_word_dist.device)
    masked_sim = topic_sim * (1 - eye)
    
    # Calculate mean similarity
    diversity_loss = masked_sim.sum() / (num_topics * (num_topics - 1))
    
    return diversity_loss

def build_word_cooccurrence_graph(docs, vocab_size, window_size=5):
    """Build word co-occurrence graph from documents
    
    Args:
        docs: List of documents (each document is a list of token IDs)
        vocab_size: Size of vocabulary
        window_size: Co-occurrence window size
        
    Returns:
        nx_graph: NetworkX graph
    """
    # Create a graph
    graph = nx.Graph()
    
    # Add word nodes
    for word_id in range(vocab_size):
        graph.add_node(word_id, type='word')
    
    # Count co-occurrences
    for doc in docs:
        for i, word_i in enumerate(doc):
            if word_i >= vocab_size:
                continue
            
            # Consider words within window
            window_start = max(0, i - window_size)
            window_end = min(len(doc), i + window_size + 1)
            
            for j in range(window_start, window_end):
                if i != j and j < len(doc):
                    word_j = doc[j]
                    
                    if word_j >= vocab_size:
                        continue
                    
                    # Add edge or update weight
                    if graph.has_edge(word_i, word_j):
                        graph[word_i][word_j]['weight'] += 1
                    else:
                        graph.add_edge(word_i, word_j, weight=1)
    
    return graph 