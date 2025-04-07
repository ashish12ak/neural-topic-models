import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from matplotlib.colors import ListedColormap
from wordcloud import WordCloud
import pandas as pd

def plot_topic_word_heatmap(topic_word_dist, vocab, top_n=10, figsize=(16, 12)):
    """Plot heatmap of topic-word distribution
    
    Args:
        topic_word_dist: Topic-word distribution matrix [n_topics, n_words]
        vocab: Vocabulary (CountVectorizer, list, or vocabulary object)
        top_n: Number of top words to display
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    # Convert to numpy array if needed
    if not isinstance(topic_word_dist, np.ndarray):
        try:
            import torch
            if isinstance(topic_word_dist, torch.Tensor):
                topic_word_dist = topic_word_dist.detach().cpu().numpy()
        except:
            pass
    
    # Get vocabulary list
    if hasattr(vocab, 'get_itos'):
        vocab_list = vocab.get_itos()
    elif hasattr(vocab, 'get_feature_names_out'):
        vocab_list = vocab.get_feature_names_out()
    else:
        vocab_list = vocab
    
    # Get top words for each topic - handle potentially mismatched dimensions
    n_topics = topic_word_dist.shape[0]
    
    # Create a modified topic_word_dist that only considers valid vocab indices
    valid_cols = min(topic_word_dist.shape[1], len(vocab_list))
    safe_topic_word_dist = topic_word_dist[:, :valid_cols].copy()
    
    # Get top word indices within the valid range
    top_words_indices = np.argsort(-safe_topic_word_dist, axis=1)[:, :top_n]
    
    # Create a matrix with top words only
    plot_matrix = np.zeros((n_topics, top_n))
    top_words = []
    
    for t in range(n_topics):
        topic_top_words = []
        for i, word_idx in enumerate(top_words_indices[t]):
            # Double-check that word_idx is valid
            if word_idx < len(vocab_list):
                plot_matrix[t, i] = safe_topic_word_dist[t, word_idx]
                topic_top_words.append(vocab_list[word_idx])
            else:
                # Fallback for any indices that are still out of range
                plot_matrix[t, i] = 0.0
                topic_top_words.append(f"word_{word_idx}")
        top_words.append(topic_top_words)
    
    # Create a dataframe for heatmap
    topic_names = [f"Topic {i+1}" for i in range(n_topics)]
    df = pd.DataFrame(plot_matrix, index=topic_names)
    
    # Create word labels for columns
    word_labels = []
    for i in range(top_n):
        words = [top_words[t][i] for t in range(n_topics)]
        # Find most common word for this position
        from collections import Counter
        counter = Counter(words)
        most_common = counter.most_common(1)[0][0]
        count = counter[most_common]
        
        if count > 1:
            # If there are duplicates, use position-specific labels
            col_labels = [f"{i+1}. {top_words[t][i]}" for t in range(n_topics)]
            word_labels.append(col_labels)
        else:
            # Otherwise just use the words
            word_labels.append([top_words[t][i] for t in range(n_topics)])
    
    # Flatten the labels for the dataframe
    flat_labels = []
    for i in range(top_n):
        if isinstance(word_labels[i], list):
            flat_labels.append(word_labels[i][0])
        else:
            flat_labels.append(word_labels[i])
    
    df.columns = [f"{i+1}. {w}" for i, w in enumerate(flat_labels)]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    plt.title("Top Words for Each Topic", fontsize=16)
    plt.tight_layout()
    
    return fig, top_words

def plot_wordclouds(topic_words, topic_word_dist=None, n_topics=None, n_cols=3, figsize=(16, 10)):
    """Plot wordclouds for each topic
    
    Args:
        topic_words: List of lists of top words for each topic
        topic_word_dist: Optional topic-word distribution for word sizes
        n_topics: Number of topics to plot (None for all)
        n_cols: Number of columns in the grid
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    # Limit number of topics if needed
    if n_topics is None:
        n_topics = len(topic_words)
    else:
        n_topics = min(n_topics, len(topic_words))
    
    # Calculate grid dimensions
    n_rows = (n_topics + n_cols - 1) // n_cols
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    for i in range(n_topics):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        
        words = topic_words[i]
        
        # Create word frequencies dictionary
        if topic_word_dist is not None:
            # Use actual distribution values if available
            frequencies = {word: topic_word_dist[i, j] for j, word in enumerate(words)}
        else:
            # Otherwise, use decreasing weights
            frequencies = {word: 1.0 / (j + 1) for j, word in enumerate(words)}
        
        # Generate word cloud
        wordcloud = WordCloud(
            background_color='white',
            width=800,
            height=400,
            colormap='viridis',
            relative_scaling=0.5,
            normalize_plurals=False
        ).generate_from_frequencies(frequencies)
        
        # Plot word cloud
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f'Topic {i+1}', fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def plot_document_topic_heatmap(doc_topic_dist, document_labels=None, n_docs=20, figsize=(14, 10)):
    """Plot heatmap of document-topic distribution
    
    Args:
        doc_topic_dist: Document-topic distribution matrix [n_docs, n_topics]
        document_labels: Optional document labels
        n_docs: Number of documents to plot
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    # Convert to numpy array if needed
    if not isinstance(doc_topic_dist, np.ndarray):
        try:
            import torch
            if isinstance(doc_topic_dist, torch.Tensor):
                doc_topic_dist = doc_topic_dist.detach().cpu().numpy()
        except:
            pass
    
    # Limit number of documents
    n_docs = min(n_docs, doc_topic_dist.shape[0])
    plot_matrix = doc_topic_dist[:n_docs]
    
    # Create document labels if not provided
    if document_labels is None:
        document_labels = [f"Doc {i+1}" for i in range(n_docs)]
    else:
        document_labels = document_labels[:n_docs]
    
    # Create topic labels
    topic_labels = [f"Topic {i+1}" for i in range(doc_topic_dist.shape[1])]
    
    # Create dataframe
    df = pd.DataFrame(plot_matrix, index=document_labels, columns=topic_labels)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    plt.title("Document-Topic Distribution", fontsize=16)
    plt.tight_layout()
    
    return fig

def plot_topic_embeddings(topic_embeddings, topic_labels=None, method='tsne', figsize=(12, 10)):
    """Plot topic embeddings in 2D space
    
    Args:
        topic_embeddings: Topic embedding matrix [n_topics, embedding_dim]
        topic_labels: Optional topic labels
        method: Dimensionality reduction method ('tsne' or 'umap')
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    # Convert to numpy array if needed
    if not isinstance(topic_embeddings, np.ndarray):
        try:
            import torch
            if isinstance(topic_embeddings, torch.Tensor):
                topic_embeddings = topic_embeddings.detach().cpu().numpy()
        except:
            pass
    
    # Create topic labels if not provided
    n_topics = topic_embeddings.shape[0]
    if topic_labels is None:
        topic_labels = [f"Topic {i+1}" for i in range(n_topics)]
    
    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=min(30, max(5, n_topics - 1)), 
                       random_state=42, n_iter=1000)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    embeddings_2d = reducer.fit_transform(topic_embeddings)
    
    # Plot embeddings
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create colormap
    cmap = plt.cm.get_cmap('hsv', n_topics)
    
    # Plot points
    for i in range(n_topics):
        ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], s=100, 
                   color=cmap(i), label=topic_labels[i])
        ax.text(embeddings_2d[i, 0], embeddings_2d[i, 1], str(i+1), 
                fontsize=12, ha='center', va='center')
    
    ax.set_title(f"Topic Embeddings ({method.upper()})", fontsize=16)
    
    # Add legend if not too many topics
    if n_topics <= 15:
        ax.legend(loc='best')
    
    plt.tight_layout()
    return fig

def plot_document_embeddings(doc_embeddings, doc_topic_assignments, 
                          topic_labels=None, n_docs=1000, method='umap', figsize=(14, 10)):
    """Plot document embeddings in 2D space, colored by dominant topic
    
    Args:
        doc_embeddings: Document embedding matrix [n_docs, embedding_dim]
        doc_topic_assignments: Document-topic assignment matrix [n_docs, n_topics]
                               or array of topic indices
        topic_labels: Optional topic labels
        n_docs: Number of documents to plot (None for all)
        method: Dimensionality reduction method ('tsne' or 'umap')
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    # Convert to numpy array if needed
    if not isinstance(doc_embeddings, np.ndarray):
        try:
            import torch
            if isinstance(doc_embeddings, torch.Tensor):
                doc_embeddings = doc_embeddings.detach().cpu().numpy()
        except:
            pass
    
    if not isinstance(doc_topic_assignments, np.ndarray):
        try:
            import torch
            if isinstance(doc_topic_assignments, torch.Tensor):
                doc_topic_assignments = doc_topic_assignments.detach().cpu().numpy()
        except:
            pass
    
    # Limit number of documents if needed
    if n_docs is not None:
        n_docs = min(n_docs, doc_embeddings.shape[0])
        doc_embeddings = doc_embeddings[:n_docs]
        doc_topic_assignments = doc_topic_assignments[:n_docs]
    
    # Get dominant topic for each document
    if len(doc_topic_assignments.shape) == 2:
        # If distribution, get argmax
        dominant_topics = np.argmax(doc_topic_assignments, axis=1)
    else:
        # If already indices, use directly
        dominant_topics = doc_topic_assignments
    
    # Create topic labels if not provided
    n_topics = len(np.unique(dominant_topics))
    if topic_labels is None:
        topic_labels = [f"Topic {i+1}" for i in range(n_topics)]
    
    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    embeddings_2d = reducer.fit_transform(doc_embeddings)
    
    # Plot embeddings
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create colormap
    cmap = plt.cm.get_cmap('hsv', n_topics)
    
    # Plot points
    for i in range(n_topics):
        mask = (dominant_topics == i)
        if np.any(mask):  # Only plot if there are documents
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                      s=20, alpha=0.6, color=cmap(i), label=topic_labels[i])
    
    ax.set_title(f"Document Embeddings by Topic ({method.upper()})", fontsize=16)
    
    # Add legend if not too many topics
    if n_topics <= 15:
        ax.legend(loc='best')
    
    plt.tight_layout()
    return fig 