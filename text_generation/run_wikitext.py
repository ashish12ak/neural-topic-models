import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Add project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from data.data_loader import load_wikitext, WikiTextDataset, get_dataloaders
from data.preprocessing import preprocess_documents
from models.vae_ntm import VAE_NTM, train_vae_ntm, get_topic_words
from models.etm import ETM, train_etm
from models.graph_ntm import GraphNTM, train_graph_ntm
from models.clustering_ntm import ClusteringNTM
from evaluation.perplexity import compute_perplexity, compute_topic_guided_perplexity, evaluate_generation_diversity
from visualization.text_vis import (
    plot_generated_text_length_distribution, plot_topic_influenced_texts,
    plot_topic_word_usage, plot_generation_wordcloud, plot_perplexity_comparison,
    create_text_examples_table
)

class TopicGuidedGenerator(nn.Module):
    """Topic-guided text generation model
    
    This model uses a pre-trained topic model to guide text generation.
    It consists of an LSTM language model that takes topic vectors as
    additional input to guide the generation process.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, 
                 num_topics, topic_model=None, dropout=0.5):
        """Initialize the model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            num_topics: Number of topics
            topic_model: Pre-trained topic model (optional)
            dropout: Dropout probability
        """
        super(TopicGuidedGenerator, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_topics = num_topics
        self.topic_model = topic_model
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim + num_topics,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights with small random values"""
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.fc.bias)
        nn.init.uniform_(self.fc.weight, -initrange, initrange)
    
    def forward(self, x, topics=None, hidden=None):
        """Forward pass
        
        Args:
            x: Input token indices [batch_size, seq_len]
            topics: Topic distribution [batch_size, num_topics]
            hidden: Initial hidden state
            
        Returns:
            output: Output logits [batch_size, seq_len, vocab_size]
            hidden: Final hidden state
        """
        batch_size, seq_len = x.size()
        
        # Get word embeddings
        emb = self.dropout(self.embedding(x))  # [batch_size, seq_len, emb_dim]
        
        # If topics not provided, use uniform distribution
        if topics is None:
            topics = torch.ones(batch_size, self.num_topics, device=x.device)
            topics = topics / self.num_topics
        
        # Replicate topic vector for each token in sequence
        topic_vectors = topics.unsqueeze(1).expand(batch_size, seq_len, self.num_topics)
        
        # Concatenate embeddings with topic vectors
        lstm_input = torch.cat([emb, topic_vectors], dim=2)
        
        # Pass through LSTM
        output, hidden = self.lstm(lstm_input, hidden)
        
        # Apply dropout and final linear layer
        output = self.dropout(output)
        output = self.fc(output)
        
        return output, hidden
    
    def generate(self, seed_text, max_length, tokenizer, topic_vector=None, 
                temperature=1.0, device='cuda'):
        """Generate text with a specific topic influence
        
        Args:
            seed_text: Seed text to start generation
            max_length: Maximum length to generate
            tokenizer: Tokenizer object
            topic_vector: Topic vector to guide generation
            temperature: Sampling temperature (higher = more random)
            device: Device to use
            
        Returns:
            generated_text: Generated text
        """
        self.eval()
        
        # Tokenize seed text
        if hasattr(tokenizer, '__call__'):
            # For callable tokenizers (like transformers)
            tokens = tokenizer(seed_text)
        else:
            # For non-callable tokenizers (like CustomVocab)
            tokens = seed_text.split()
        
        # Convert to indices
        if hasattr(tokenizer, 'encode'):
            indices = tokenizer.encode(tokens)
        else:
            # For simple tokenizers
            if hasattr(tokenizer, 'get_stoi'):
                vocab = tokenizer.get_stoi() 
            elif hasattr(tokenizer, 'stoi'):
                vocab = tokenizer.stoi
            else:
                vocab = getattr(tokenizer, 'vocab', {})
            
            # Handle tokens differently based on what we have
            if isinstance(tokens, list):
                indices = [vocab.get(token, vocab.get('<unk>', 0)) for token in tokens]
            else:
                # Tokens is a string, we need to split it
                indices = [vocab.get(token, vocab.get('<unk>', 0)) for token in tokens.split()]
        
        # Convert to tensor
        input_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)
        
        # Set topic vector
        if topic_vector is None:
            # Use uniform distribution if not provided
            topic_vector = torch.ones(1, self.num_topics, device=device) / self.num_topics
        else:
            topic_vector = torch.tensor(topic_vector, dtype=torch.float).unsqueeze(0).to(device)
        
        # Generate text
        generated_indices = indices.copy()
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                output, hidden = self.forward(input_tensor, topic_vector, hidden)
                
                # Get last token prediction
                logits = output[0, -1, :] / temperature
                probs = F.softmax(logits, dim=0)
                
                # Sample from distribution
                next_token = torch.multinomial(probs, 1).item()
                
                # Add to generated indices
                generated_indices.append(next_token)
                
                # Update input tensor for next iteration
                input_tensor = torch.LongTensor([[next_token]]).to(device)
                
                # Stop if end token
                if next_token == vocab.get('<eos>', -1):
                    break
        
        # Convert indices back to text
        if hasattr(tokenizer, 'decode'):
            generated_text = tokenizer.decode(generated_indices)
        else:
            # For simple tokenizers
            if hasattr(tokenizer, 'get_itos'):
                idx_to_token = tokenizer.get_itos()
                # Handle idx_to_token as a list
                generated_text = ' '.join([idx_to_token[idx] if idx < len(idx_to_token) else '<unk>' for idx in generated_indices])
            else:
                idx_to_token = {v: k for k, v in vocab.items()}
                # Handle idx_to_token as a dictionary
                generated_text = ' '.join([idx_to_token.get(idx, '<unk>') for idx in generated_indices])
        
        return generated_text

def parse_args():
    parser = argparse.ArgumentParser(description='Run topic-guided text generation on WikiText dataset')
    parser.add_argument('--topic_model', type=str, default='vae', 
                      choices=['vae', 'etm', 'graph', 'clustering'],
                      help='Topic model to use (default: vae)')
    parser.add_argument('--num_topics', type=int, default=20,
                      help='Number of topics (default: 20)')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size (default: 64)')
    parser.add_argument('--topic_epochs', type=int, default=30,
                      help='Number of epochs for topic model (default: 30)')
    parser.add_argument('--gen_epochs', type=int, default=10,
                      help='Number of epochs for generator (default: 10)')
    parser.add_argument('--vocab_size', type=int, default=10000,
                      help='Vocabulary size (default: 10000)')
    parser.add_argument('--embedding_dim', type=int, default=300,
                      help='Embedding dimension (default: 300)')
    parser.add_argument('--hidden_size', type=int, default=500,
                      help='Hidden layer size (default: 500)')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of LSTM layers (default: 2)')
    parser.add_argument('--seq_length', type=int, default=64,
                      help='Sequence length (default: 64)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (default: cuda if available, else cpu)')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Output directory (default: results)')
    return parser.parse_args()

def train_topic_model(args, train_loader, val_loader, vocab):
    """Train the topic model
    
    Args:
        args: Command line arguments
        train_loader: Training data loader
        val_loader: Validation data loader
        vocab: Vocabulary
        
    Returns:
        model: Trained topic model
        topic_word_dist: Topic-word distribution
        topic_words: List of top words for each topic
    """
    print(f"\n===== Training {args.topic_model.upper()} Topic Model =====")
    
    # Get actual vocabulary size
    vocab_size = len(vocab)
    print(f"Using vocabulary size: {vocab_size} (instead of {args.vocab_size})")
    
    if args.topic_model == 'vae':
        # Create VAE-NTM model
        model = VAE_NTM(
            vocab_size=vocab_size,
            num_topics=args.num_topics,
            hidden_size=args.hidden_size,
            dropout=0.2
        )
        
        # Train model
        model, train_losses, val_losses = train_vae_ntm(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.topic_epochs,
            lr=0.002,
            beta=1.0,
            device=args.device
        )
        
        # Get topic-word distribution
        topic_word_dist = model.get_topic_word_dist()
        
        # Get top words for each topic
        topic_words = get_topic_words(topic_word_dist, vocab, k=10)
        
    elif args.topic_model == 'etm':
        # Create ETM model
        model = ETM(
            vocab_size=vocab_size,
            num_topics=args.num_topics,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            pretrained_embeddings=None,
            trainable_embeddings=True
        )
        
        # Train model
        model, train_losses, val_losses = train_etm(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.topic_epochs,
            lr=0.002,
            device=args.device
        )
        
        # Get topic-word distribution
        topic_word_dist = model.get_topic_word_dist()
        
        # Get top words for each topic
        from models.etm import get_nearest_words_to_topics
        topic_words = get_nearest_words_to_topics(model, vocab, k=10)
        
    elif args.topic_model == 'graph':
        # Create GraphNTM model
        model = GraphNTM(
            vocab_size=vocab_size,
            num_topics=args.num_topics,
            hidden_size=args.hidden_size,
            embedding_dim=args.embedding_dim,
            dropout=0.2
        )
        
        # Train model
        model, train_losses, val_losses = train_graph_ntm(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.topic_epochs,
            lr=0.002,
            device=args.device,
            with_graph_structure=False
        )
        
        # Get topic-word distribution
        topic_word_dist = model.get_topic_word_dist()
        
        # Get top words for each topic
        if isinstance(topic_word_dist, torch.Tensor):
            topic_word_dist_np = topic_word_dist.detach().cpu().numpy()
        else:
            topic_word_dist_np = topic_word_dist
        
        # Get top word indices for each topic
        top_indices = np.argsort(-topic_word_dist_np, axis=1)[:, :10]
        
        # Get vocabulary list
        if hasattr(vocab, 'get_itos'):
            vocab_list = vocab.get_itos()
        else:
            vocab_list = vocab
        
        # Get top words
        topic_words = []
        for indices in top_indices:
            topic_words.append([vocab_list[idx] for idx in indices])
            
    elif args.topic_model == 'clustering':
        # For clustering model, we need raw documents
        if hasattr(train_loader.dataset, 'documents'):
            documents = train_loader.dataset.documents
        else:
            # Convert indices back to texts
            if hasattr(vocab, 'get_itos'):
                vocab_list = vocab.get_itos()
            else:
                vocab_list = vocab
                
            documents = []
            for batch in train_loader:
                indices = batch.numpy() if isinstance(batch, torch.Tensor) else batch
                for idx_seq in indices:
                    # Convert indices to integers and filter invalid ones
                    valid_indices = [int(idx) for idx in idx_seq if isinstance(idx, (int, float, np.float32, np.int64)) and int(idx) < len(vocab_list)]
                    doc = ' '.join([vocab_list[idx] for idx in valid_indices])
                    documents.append(doc)
        
        # Create ClusteringNTM model
        model = ClusteringNTM(
            embedding_model='all-MiniLM-L6-v2',
            num_topics=args.num_topics,
            umap_dim=5,
            min_cluster_size=5,
            random_state=42
        )
        
        # Train model
        model.fit(documents, method='kmeans')
        
        # Get topic-word distribution
        topic_word_dist = model.get_topic_word_dist()
        
        # Get top words for each topic
        topic_words = model.get_topics()
    
    # Print top words for each topic
    print("\nTop 10 words for each topic:")
    for i, words in enumerate(topic_words):
        print(f"Topic {i+1}: {', '.join(words)}")
    
    return model, topic_word_dist, topic_words

def train_generator(args, topic_model, train_loader, val_loader, vocab):
    """Train the topic-guided generator
    
    Args:
        args: Command line arguments
        topic_model: Pre-trained topic model
        train_loader: Training data loader
        val_loader: Validation data loader
        vocab: Vocabulary
        
    Returns:
        model: Trained generator model
    """
    print("\n===== Training Topic-Guided Generator =====")
    
    # Get actual vocabulary size
    vocab_size = len(vocab)
    print(f"Using vocabulary size: {vocab_size} (instead of {args.vocab_size})")
    
    # Create generator model
    model = TopicGuidedGenerator(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_topics=args.num_topics,
        topic_model=topic_model,
        dropout=0.5
    )
    
    # Move model to device
    model.to(args.device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
    
    # Training loop
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.gen_epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            # Get batch data
            inputs = batch[:, :-1].to(args.device)  # all tokens except last
            targets = batch[:, 1:].to(args.device)  # all tokens except first
            
            # Extract topics using topic model
            topics = None
            if topic_model is not None:
                with torch.no_grad():
                    topics = get_document_topics(topic_model, inputs, args.device)
            
            # Forward pass
            output, _ = model(inputs, topics)
            
            # Reshape for loss computation
            output_flat = output.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)
            
            # Compute loss
            loss = criterion(output_flat, targets_flat)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
        
        # Average loss
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Get batch data
                inputs = batch[:, :-1].to(args.device)
                targets = batch[:, 1:].to(args.device)
                
                # Extract topics using topic model
                topics = None
                if topic_model is not None:
                    topics = get_document_topics(topic_model, inputs, args.device)
                
                # Forward pass
                output, _ = model(inputs, topics)
                
                # Reshape for loss computation
                output_flat = output.reshape(-1, vocab_size)
                targets_flat = targets.reshape(-1)
                
                # Compute loss
                loss = criterion(output_flat, targets_flat)
                
                # Track loss
                val_loss += loss.item()
        
        # Average loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.gen_epochs}")
        print(f"  Train Loss: {epoch_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'wikitext', 'best_generator.pt'))
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'wikitext', 'best_generator.pt')))
    
    return model

def generate_topic_texts(args, generator, topic_model, topic_words, tokenizer, seed_texts):
    """Generate texts guided by different topics
    
    Args:
        args: Command line arguments
        generator: Trained generator model
        topic_model: Pre-trained topic model
        topic_words: List of top words for each topic
        tokenizer: Tokenizer
        seed_texts: List of seed texts
        
    Returns:
        generated_texts: Dictionary of generated texts for each topic
    """
    print("\n===== Generating Topic-Guided Texts =====")
    
    # Create output directory
    os.makedirs(os.path.join(args.output_dir, 'wikitext', 'generated_texts'), exist_ok=True)
    
    # Create one-hot topic vectors
    topic_vectors = np.eye(args.num_topics)
    
    # Generate texts for each topic
    generated_texts = {}
    
    for i, topic_vec in enumerate(topic_vectors):
        print(f"Generating texts for Topic {i+1}: {', '.join(topic_words[i])}")
        
        topic_texts = []
        
        for seed_text in seed_texts:
            # Generate text
            generated = generator.generate(
                seed_text=seed_text,
                max_length=100,
                tokenizer=tokenizer,
                topic_vector=topic_vec,
                temperature=0.7,
                device=args.device
            )
            
            topic_texts.append(generated)
        
        generated_texts[i] = topic_texts
        
        # Save to file
        with open(os.path.join(args.output_dir, 'wikitext', 'generated_texts', f'topic_{i+1}.txt'), 'w') as f:
            for j, text in enumerate(topic_texts):
                f.write(f"Seed: {seed_texts[j]}\n\n")
                f.write(f"Generated: {text}\n\n")
                f.write("-" * 80 + "\n\n")
    
    # Create a mixed topic for comparison
    mixed_topic = np.ones(args.num_topics) / args.num_topics
    mixed_texts = []
    
    for seed_text in seed_texts:
        # Generate text
        generated = generator.generate(
            seed_text=seed_text,
            max_length=100,
            tokenizer=tokenizer,
            topic_vector=mixed_topic,
            temperature=0.7,
            device=args.device
        )
        
        mixed_texts.append(generated)
    
    generated_texts['mixed'] = mixed_texts
    
    # Save to file
    with open(os.path.join(args.output_dir, 'wikitext', 'generated_texts', 'mixed_topic.txt'), 'w') as f:
        for j, text in enumerate(mixed_texts):
            f.write(f"Seed: {seed_texts[j]}\n\n")
            f.write(f"Generated: {text}\n\n")
            f.write("-" * 80 + "\n\n")
    
    return generated_texts

def evaluate_generation(args, generator, topic_model, topic_words, tokenizer, test_loader, seed_texts, generated_texts):
    """Evaluate the quality of generated texts
    
    Args:
        args: Command line arguments
        generator: Trained generator model
        topic_model: Pre-trained topic model
        topic_words: List of top words for each topic
        tokenizer: Tokenizer
        test_loader: Test data loader
        seed_texts: List of seed texts
        generated_texts: Dictionary of generated texts for each topic
        
    Returns:
        evaluation_results: Dictionary of evaluation results
    """
    print("\n===== Evaluating Generated Texts =====")
    
    # Create output directory
    os.makedirs(os.path.join(args.output_dir, 'wikitext', 'evaluation'), exist_ok=True)
    
    # 1. Perplexity
    print("Computing perplexity...")
    perplexity = compute_perplexity(generator, test_loader, device=args.device)
    print(f"Overall perplexity: {perplexity:.4f}")
    
    # 2. Topic-specific perplexity
    # Create one-hot topic vectors
    topic_vectors = np.eye(args.num_topics)
    topic_vectors = np.vstack([topic_vectors, np.ones(args.num_topics) / args.num_topics])  # Add mixed topic
    
    topic_perplexity = compute_topic_guided_perplexity(
        generator, test_loader, topic_vectors, device=args.device
    )
    
    # Print topic perplexities
    print("Topic-specific perplexities:")
    for i in range(args.num_topics):
        print(f"  Topic {i+1}: {topic_perplexity[f'topic_{i}']:.4f}")
    print(f"  Mixed: {topic_perplexity['average']:.4f}")
    
    # 3. Diversity metrics
    print("Computing diversity metrics...")
    all_texts = []
    for texts in generated_texts.values():
        all_texts.extend(texts)
    
    diversity_metrics = evaluate_generation_diversity(all_texts, tokenizer)
    
    # Print diversity metrics
    print("Diversity metrics:")
    for metric, value in diversity_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 4. Topic word usage
    print("Analyzing topic word usage...")
    
    # Compute topic word usage for each topic's generated texts
    topic_word_usage = {}
    
    for topic_idx, texts in generated_texts.items():
        if topic_idx == 'mixed':
            topic_words_idx = None
        else:
            topic_words_idx = topic_words[topic_idx]
        
        # Count occurrences of each topic word
        word_counts = {}
        
        if topic_words_idx:
            for word in topic_words_idx:
                count = sum(text.lower().count(word.lower()) for text in texts)
                word_counts[word] = count / len(texts)  # Normalize by number of texts
        
        topic_word_usage[topic_idx] = word_counts
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 1. Text length distribution
    fig = plot_generated_text_length_distribution(all_texts)
    fig.savefig(os.path.join(args.output_dir, 'wikitext', 'evaluation', 'text_length_distribution.png'))
    plt.close(fig)
    
    # 2. Topic-influenced texts
    fig = plot_topic_influenced_texts(generated_texts)
    fig.savefig(os.path.join(args.output_dir, 'wikitext', 'evaluation', 'topic_influenced_texts.png'))
    plt.close(fig)
    
    # 3. Topic word usage
    topic_labels = [f"Topic {i+1}" for i in range(args.num_topics)] + ["Mixed"]
    topic_indices = list(range(args.num_topics)) + ['mixed']
    
    fig = plot_topic_word_usage(
        {i: generated_texts[idx] for i, idx in enumerate(topic_indices)},
        [topic_words[i] for i in range(args.num_topics)] + [[]],
        topic_labels
    )
    fig.savefig(os.path.join(args.output_dir, 'wikitext', 'evaluation', 'topic_word_usage.png'))
    plt.close(fig)
    
    # 4. Word clouds
    for topic_idx, texts in generated_texts.items():
        if topic_idx == 'mixed':
            title = "Mixed Topics"
        else:
            title = f"Topic {topic_idx+1}"
        
        fig = plot_generation_wordcloud(texts, title=title)
        fig.savefig(os.path.join(args.output_dir, 'wikitext', 'evaluation', f'wordcloud_{title.lower().replace(" ", "_")}.png'))
        plt.close(fig)
    
    # 5. Perplexity comparison
    fig = plot_perplexity_comparison(topic_perplexity)
    fig.savefig(os.path.join(args.output_dir, 'wikitext', 'evaluation', 'perplexity_comparison.png'))
    plt.close(fig)
    
    # 6. Text examples table
    examples_df = create_text_examples_table(generated_texts, topic_labels)
    examples_df.to_csv(os.path.join(args.output_dir, 'wikitext', 'evaluation', 'text_examples.csv'), index=False)
    
    # Compile results
    evaluation_results = {
        'perplexity': perplexity,
        'topic_perplexity': topic_perplexity,
        'diversity_metrics': diversity_metrics,
        'topic_word_usage': topic_word_usage
    }
    
    return evaluation_results

def get_document_topics(topic_model, tokens, device='cuda'):
    """Extract document topics from the token sequence using the topic model
    
    Args:
        topic_model: Pre-trained topic model
        tokens: Token indices tensor [batch_size, seq_len]
        device: Device to use
        
    Returns:
        topic_dist: Topic distribution tensor [batch_size, num_topics]
    """
    # Use the topic model to get document topic distribution
    if topic_model is None:
        return None
    
    batch_size = tokens.size(0)
    
    # If topic model has a method to extract topics from tokens directly, use it
    if hasattr(topic_model, 'get_topics_from_tokens'):
        return topic_model.get_topics_from_tokens(tokens)
    
    # Otherwise, extract topic distribution using encode and softmax
    with torch.no_grad():
        # For VAE-based models
        if hasattr(topic_model, 'encode'):
            # Create a pseudo bag-of-words representation from tokens
            vocab_size = topic_model.vocab_size
            bow = torch.zeros(batch_size, vocab_size, device=device)
            for i in range(batch_size):
                # Count token occurrences, ignoring padding
                for token in tokens[i]:
                    if token > 0 and token < vocab_size:  # Ignore padding token (0)
                        bow[i, token] += 1
            
            # Normalize
            row_sums = bow.sum(dim=1, keepdim=True)
            bow = bow / (row_sums + 1e-10)
            
            # Extract topic distribution
            mu, _ = topic_model.encode(bow)
            topic_dist = F.softmax(mu, dim=1)
            return topic_dist
        
        # For other models, use a dummy uniform distribution
        num_topics = topic_model.num_topics if hasattr(topic_model, 'num_topics') else 20
        return torch.ones(batch_size, num_topics, device=device) / num_topics

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'wikitext'), exist_ok=True)
    
    # Load data
    print("Loading WikiText dataset...")
    train_data, valid_data, test_data, vocab = load_wikitext()
    
    # Create datasets and dataloaders
    # For topic modeling, use BoW mode
    train_topic_dataset = WikiTextDataset(train_data, vocab, tokenizer=None, seq_length=args.seq_length, bow_mode=True)
    val_topic_dataset = WikiTextDataset(valid_data, vocab, tokenizer=None, seq_length=args.seq_length, bow_mode=True)
    
    # For text generation, use sequence mode
    train_gen_dataset = WikiTextDataset(train_data, vocab, tokenizer=None, seq_length=args.seq_length, bow_mode=False)
    val_gen_dataset = WikiTextDataset(valid_data, vocab, tokenizer=None, seq_length=args.seq_length, bow_mode=False)
    test_dataset = WikiTextDataset(test_data, vocab, tokenizer=None, seq_length=args.seq_length, bow_mode=False)
    
    # Create dataloaders for topic modeling
    train_topic_loader = DataLoader(train_topic_dataset, batch_size=args.batch_size, shuffle=True)
    val_topic_loader = DataLoader(val_topic_dataset, batch_size=args.batch_size)
    
    # Create dataloaders for text generation
    train_gen_loader = DataLoader(train_gen_dataset, batch_size=args.batch_size, shuffle=True)
    val_gen_loader = DataLoader(val_gen_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Train topic model
    topic_model, topic_word_dist, topic_words = train_topic_model(
        args, train_topic_loader, val_topic_loader, vocab
    )
    
    # Train generator
    generator = train_generator(
        args, topic_model, train_gen_loader, val_gen_loader, vocab
    )
    
    # Create seed texts for generation
    seed_texts = [
        "The history of",
        "In recent studies",
        "Scientists have discovered",
        "According to the latest",
        "The development of"
    ]
    
    # Generate texts
    generated_texts = generate_topic_texts(
        args, generator, topic_model, topic_words, vocab, seed_texts
    )
    
    # Evaluate generation
    evaluation_results = evaluate_generation(
        args, generator, topic_model, topic_words, vocab, test_loader, seed_texts, generated_texts
    )
    
    print("\nGeneration and evaluation complete. Results saved to:", os.path.join(args.output_dir, 'wikitext'))

if __name__ == "__main__":
    main() 