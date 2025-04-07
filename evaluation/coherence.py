import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import warnings

def compute_coherence_cv(topic_words, documents, coherence='c_v'):
    """Compute topic coherence using C_v metric (based on semantic similarity)
    
    Args:
        topic_words: List of lists of top words for each topic
        documents: List of document strings or list of tokenized documents
        coherence: Coherence metric ('c_v', 'u_mass', 'c_npmi')
        
    Returns:
        coherence_score: Topic coherence score
    """
    # Convert strings to tokens if needed
    tokenized_docs = []
    for doc in documents:
        if isinstance(doc, str):
            tokenized_docs.append(doc.lower().split())
        else:
            tokenized_docs.append(doc)
    
    # Create dictionary
    dictionary = Dictionary(tokenized_docs)
    
    # Compute corpus
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    
    # Compute coherence
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokenized_docs,
            corpus=corpus,
            dictionary=dictionary,
            coherence=coherence
        )
        coherence_score = coherence_model.get_coherence()
    
    return coherence_score

def compute_coherence_umass(topic_words, documents):
    """Compute topic coherence using UMass metric (based on document co-occurrence)
    
    Args:
        topic_words: List of lists of top words for each topic
        documents: List of document strings or list of tokenized documents
        
    Returns:
        coherence_score: Topic coherence score
    """
    return compute_coherence_cv(topic_words, documents, coherence='u_mass')

def compute_coherence_npmi(topic_words, documents):
    """Compute topic coherence using NPMI metric (normalized PMI)
    
    Args:
        topic_words: List of lists of top words for each topic
        documents: List of document strings or list of tokenized documents
        
    Returns:
        coherence_score: Topic coherence score
    """
    return compute_coherence_cv(topic_words, documents, coherence='c_npmi')

def compute_topic_coherence_scores(topic_words, documents):
    """Compute all coherence metrics for topics
    
    Args:
        topic_words: List of lists of top words for each topic
        documents: List of document strings or list of tokenized documents
        
    Returns:
        coherence_scores: Dictionary of coherence scores
    """
    coherence_scores = {
        'c_v': compute_coherence_cv(topic_words, documents),
        'u_mass': compute_coherence_umass(topic_words, documents),
        'c_npmi': compute_coherence_npmi(topic_words, documents)
    }
    
    return coherence_scores

def compute_coherence_per_topic(topic_words, documents, coherence='c_v'):
    """Compute coherence for each topic individually
    
    Args:
        topic_words: List of lists of top words for each topic
        documents: List of document strings or list of tokenized documents
        coherence: Coherence metric ('c_v', 'u_mass', 'c_npmi')
        
    Returns:
        coherence_per_topic: List of coherence scores for each topic
    """
    coherence_scores = []
    
    for i, words in enumerate(topic_words):
        single_topic = [words]  # CoherenceModel expects list of lists
        
        try:
            score = compute_coherence_cv(single_topic, documents, coherence=coherence)
            coherence_scores.append(score)
        except:
            # If coherence computation fails, assign NaN
            coherence_scores.append(float('nan'))
    
    return coherence_scores

def evaluate_topic_model_coherence(model, documents, vectorizer=None, top_n=10):
    """Evaluate topic model coherence
    
    Args:
        model: Fitted topic model with get_topic_word_dist method
        documents: List of document strings
        vectorizer: Optional vectorizer with get_feature_names_out method
        top_n: Number of top words per topic
        
    Returns:
        coherence_scores: Dictionary of coherence scores
        topic_words: List of lists of top words for each topic
    """
    # Get topic-word distribution
    topic_word_dist = model.get_topic_word_dist()
    
    # Get vocabulary
    if vectorizer is not None:
        vocab = vectorizer.get_feature_names_out()
    elif hasattr(model, 'vectorizer') and model.vectorizer is not None:
        vocab = model.vectorizer.get_feature_names_out()
    else:
        raise ValueError("Vectorizer must be provided or accessible through model")
    
    # Get top words for each topic
    if isinstance(topic_word_dist, np.ndarray):
        top_indices = np.argsort(-topic_word_dist, axis=1)[:, :top_n]
        topic_words = [[vocab[idx] for idx in indices] for indices in top_indices]
    else:
        # For PyTorch tensors
        import torch
        top_indices = torch.topk(topic_word_dist, k=top_n, dim=1).indices.cpu().numpy()
        topic_words = [[vocab[idx] for idx in indices] for indices in top_indices]
    
    # Compute coherence
    coherence_scores = compute_topic_coherence_scores(topic_words, documents)
    
    return coherence_scores, topic_words 