import numpy as np
from scipy.spatial.distance import pdist, squareform

def compute_topic_diversity(topic_words, topk=10):
    """Compute topic diversity
    
    Topic diversity measures the proportion of unique words across all topics.
    A higher score indicates less repetition of words between topics.
    
    Args:
        topic_words: List of lists of top words for each topic
        topk: Number of top words to consider
        
    Returns:
        diversity_score: Topic diversity score (0-1)
    """
    # Limit to top-k words per topic
    if topk > 0:
        topic_words = [words[:topk] for words in topic_words]
    
    # Get unique words
    unique_words = set()
    for words in topic_words:
        unique_words.update(words)
    
    # Compute diversity
    total_words = len(topic_words) * len(topic_words[0])
    diversity_score = len(unique_words) / total_words
    
    return diversity_score

def compute_pairwise_jaccard_distance(topic_words, topk=10):
    """Compute pairwise Jaccard distance between topics
    
    Jaccard distance measures the dissimilarity between topics based on shared words.
    A higher mean distance indicates more distinct topics.
    
    Args:
        topic_words: List of lists of top words for each topic
        topk: Number of top words to consider
        
    Returns:
        mean_distance: Mean Jaccard distance between topics
        distance_matrix: Pairwise distance matrix
    """
    # Limit to top-k words per topic
    if topk > 0:
        topic_words = [words[:topk] for words in topic_words]
    
    # Convert to sets
    topic_sets = [set(words) for words in topic_words]
    
    # Compute Jaccard distances
    distances = []
    n_topics = len(topic_sets)
    distance_matrix = np.zeros((n_topics, n_topics))
    
    for i in range(n_topics):
        for j in range(i + 1, n_topics):
            set_i = topic_sets[i]
            set_j = topic_sets[j]
            
            # Jaccard distance: 1 - |intersection| / |union|
            intersection = len(set_i.intersection(set_j))
            union = len(set_i.union(set_j))
            distance = 1.0 - (intersection / union)
            
            distances.append(distance)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    mean_distance = np.mean(distances)
    
    return mean_distance, distance_matrix

def compute_topic_embedding_distance(topic_embeddings):
    """Compute pairwise distances between topic embeddings
    
    Args:
        topic_embeddings: Topic embedding matrix [num_topics, embedding_dim]
        
    Returns:
        mean_distance: Mean distance between topics
        distance_matrix: Pairwise distance matrix
    """
    if isinstance(topic_embeddings, list):
        topic_embeddings = np.array(topic_embeddings)
    
    # Handle PyTorch tensors
    try:
        import torch
        if isinstance(topic_embeddings, torch.Tensor):
            # Detach tensor to avoid gradient computation
            topic_embeddings = topic_embeddings.detach().cpu().numpy()
    except:
        pass
    
    # Compute cosine distances
    distances = pdist(topic_embeddings, metric='cosine')
    distance_matrix = squareform(distances)
    mean_distance = np.mean(distances)
    
    return mean_distance, distance_matrix

def compute_topic_word_distribution_distance(topic_word_dist):
    """Compute distance between topic-word distributions
    
    Args:
        topic_word_dist: Topic-word distribution matrix [num_topics, vocab_size]
        
    Returns:
        mean_distance: Mean distance between topics
        distance_matrix: Pairwise distance matrix
    """
    if not isinstance(topic_word_dist, np.ndarray):
        # Convert PyTorch tensor to numpy
        try:
            import torch
            if isinstance(topic_word_dist, torch.Tensor):
                topic_word_dist = topic_word_dist.detach().cpu().numpy()
        except:
            topic_word_dist = np.array(topic_word_dist)
    
    # Compute Jensen-Shannon distances
    from scipy.spatial.distance import jensenshannon
    
    n_topics = topic_word_dist.shape[0]
    distance_matrix = np.zeros((n_topics, n_topics))
    distances = []
    
    for i in range(n_topics):
        for j in range(i + 1, n_topics):
            dist = jensenshannon(topic_word_dist[i], topic_word_dist[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
            distances.append(dist)
    
    mean_distance = np.mean(distances)
    
    return mean_distance, distance_matrix

def evaluate_topic_model_diversity(model, topk=10):
    """Evaluate diversity of topics from a topic model
    
    Args:
        model: Topic model
        topk: Number of top words to consider
        
    Returns:
        Dictionary with diversity metrics
    """
    # Get topic-word distribution
    if hasattr(model, 'get_topic_word_dist'):
        topic_word_dist = model.get_topic_word_dist()
    elif hasattr(model, 'topic_word_dist'):
        topic_word_dist = model.topic_word_dist
    else:
        raise ValueError("Model does not have topic_word_dist attribute or get_topic_word_dist method")
    
    # Get topic words
    if hasattr(model, 'topic_words'):
        topic_words = model.topic_words
    elif hasattr(model, 'get_topic_words'):
        # Try calling get_topic_words if it exists
        topic_words = model.get_topic_words(topk=topk)
    else:
        # Extract top words from distribution
        if isinstance(topic_word_dist, np.ndarray):
            # Try to get vocabulary from model
            if hasattr(model, 'vocab'):
                vocab = model.vocab
            elif hasattr(model, 'vectorizer') and hasattr(model.vectorizer, 'get_feature_names_out'):
                vocab = model.vectorizer.get_feature_names_out()
            else:
                raise ValueError("Cannot extract vocabulary from model")
                
            top_indices = np.argsort(-topic_word_dist, axis=1)[:, :topk]
            topic_words = [[vocab[idx] for idx in indices] for indices in top_indices]
        else:
            # For PyTorch tensors
            import torch
            if isinstance(topic_word_dist, torch.Tensor):
                # Try to get vocabulary from model
                if hasattr(model, 'vocab'):
                    vocab = model.vocab
                elif hasattr(model, 'vectorizer') and hasattr(model.vectorizer, 'get_feature_names_out'):
                    vocab = model.vectorizer.get_feature_names_out()
                else:
                    raise ValueError("Cannot extract vocabulary from model")
                    
                top_indices = torch.topk(topic_word_dist, k=topk, dim=1).indices.cpu().numpy()
                topic_words = [[vocab[idx] for idx in indices] for indices in top_indices]
    
    # Compute diversity metrics
    diversity = compute_topic_diversity(topic_words, topk=topk)
    mean_jaccard, _ = compute_pairwise_jaccard_distance(topic_words, topk=topk)
    mean_js, _ = compute_topic_word_distribution_distance(topic_word_dist)
    
    # Compute embedding distance if available
    if hasattr(model, 'get_topic_embeddings'):
        topic_embeddings = model.get_topic_embeddings()
        mean_emb_dist, _ = compute_topic_embedding_distance(topic_embeddings)
    elif hasattr(model, 'topic_embeddings_detached'):
        mean_emb_dist, _ = compute_topic_embedding_distance(model.topic_embeddings_detached)
    elif hasattr(model, 'topic_embeddings'):
        mean_emb_dist, _ = compute_topic_embedding_distance(model.topic_embeddings)
    else:
        mean_emb_dist = None
    
    # Compile results
    diversity_scores = {
        'unique_word_proportion': diversity,
        'mean_jaccard_distance': mean_jaccard,
        'mean_js_distance': mean_js
    }
    
    if mean_emb_dist is not None:
        diversity_scores['mean_embedding_distance'] = mean_emb_dist
    
    return diversity_scores 