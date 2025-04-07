import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
from wordcloud import WordCloud

def plot_generated_text_length_distribution(texts, bins=20, figsize=(10, 6)):
    """Plot distribution of generated text lengths
    
    Args:
        texts: List of generated texts
        bins: Number of histogram bins
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    # Compute text lengths
    lengths = [len(text.split()) for text in texts]
    
    # Plot histogram
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(lengths, bins=bins, ax=ax)
    
    ax.set_title("Distribution of Generated Text Lengths", fontsize=16)
    ax.set_xlabel("Text Length (words)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    
    plt.tight_layout()
    return fig

def plot_topic_influenced_texts(texts_by_topic, topic_labels=None, max_chars=200, figsize=(14, 10)):
    """Plot examples of texts generated with different topic influences
    
    Args:
        texts_by_topic: Dictionary mapping topic indices to lists of texts,
                        or list of lists of texts
        topic_labels: Optional topic labels
        max_chars: Maximum number of characters to display
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    # Convert to dict if list
    if isinstance(texts_by_topic, list):
        texts_by_topic = {i: texts for i, texts in enumerate(texts_by_topic)}
    
    # Get topics and number of examples per topic
    topics = list(texts_by_topic.keys())
    n_topics = len(topics)
    
    # Create topic labels if not provided
    if topic_labels is None:
        topic_labels = [f"Topic {i+1}" for i in range(n_topics)]
    
    # Create figure
    fig, axes = plt.subplots(n_topics, 1, figsize=figsize)
    if n_topics == 1:
        axes = [axes]
    
    for i, topic_idx in enumerate(topics):
        ax = axes[i]
        
        # Get texts for this topic
        topic_texts = texts_by_topic[topic_idx]
        
        if topic_texts:
            # Get a representative example
            example = topic_texts[0]
            
            # Truncate if too long
            if len(example) > max_chars:
                example = example[:max_chars] + "..."
            
            # Add text to plot
            ax.text(0.01, 0.5, example, wrap=True, ha='left', va='center',
                   fontsize=10, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No texts available", ha='center', va='center',
                   fontsize=12, transform=ax.transAxes)
        
        # Set title and remove axes
        ax.set_title(topic_labels[i], fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def plot_topic_word_usage(texts_by_topic, topic_words, topic_labels=None, figsize=(14, 8)):
    """Plot frequency of topic words in generated texts
    
    Args:
        texts_by_topic: Dictionary mapping topic indices to lists of texts,
                        or list of lists of texts
        topic_words: List of lists of top words for each topic
        topic_labels: Optional topic labels
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    # Convert to dict if list
    if isinstance(texts_by_topic, list):
        texts_by_topic = {i: texts for i, texts in enumerate(texts_by_topic)}
    
    # Get topics and number of examples per topic
    topics = list(texts_by_topic.keys())
    n_topics = len(topics)
    
    # Create topic labels if not provided
    if topic_labels is None:
        topic_labels = [f"Topic {i+1}" for i in range(n_topics)]
    
    # Count word frequencies in each topic's texts
    word_counts = {}
    for topic_idx in topics:
        texts = texts_by_topic[topic_idx]
        words = topic_words[topic_idx]
        
        # Count occurrences of each word
        counts = {word: 0 for word in words}
        for text in texts:
            for word in words:
                # Case-insensitive count
                counts[word] += text.lower().count(word.lower())
        
        # Normalize by number of texts
        if texts:
            counts = {word: count / len(texts) for word, count in counts.items()}
        
        word_counts[topic_idx] = counts
    
    # Create dataframe for plotting
    data = []
    for topic_idx in topics:
        counts = word_counts[topic_idx]
        for word, count in counts.items():
            data.append({
                'Topic': topic_labels[topic_idx],
                'Word': word,
                'Frequency': count
            })
    
    df = pd.DataFrame(data)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, x='Word', y='Frequency', hue='Topic', ax=ax)
    
    ax.set_title("Topic Word Usage in Generated Texts", fontsize=16)
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel("Average Frequency per Text", fontsize=12)
    
    # Rotate x labels if many words
    if len(topic_words[0]) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def plot_generation_wordcloud(texts, title="Generated Text WordCloud", figsize=(10, 8)):
    """Plot wordcloud of generated texts
    
    Args:
        texts: List of generated texts
        title: Plot title
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    # Combine all texts
    all_text = " ".join(texts)
    
    # Create wordcloud
    wordcloud = WordCloud(
        background_color='white',
        width=800,
        height=400,
        max_words=200,
        colormap='viridis',
        relative_scaling=0.5,
        normalize_plurals=False
    ).generate(all_text)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def plot_perplexity_comparison(perplexity_by_topic, model_names=None, figsize=(12, 8)):
    """Plot perplexity comparison across topics or models
    
    Args:
        perplexity_by_topic: Dictionary mapping model names to dictionaries
                            mapping topic indices to perplexity scores,
                            or single dictionary if comparing topics
        model_names: Optional model names (if comparing multiple models)
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    # Check if comparing multiple models
    if all(isinstance(v, dict) for v in perplexity_by_topic.values()):
        # Multiple models
        comparing_models = True
        if model_names is None:
            model_names = list(perplexity_by_topic.keys())
    else:
        # Single model, multiple topics
        comparing_models = False
        if model_names is None:
            model_names = ["Model"]
    
    # Create dataframe for plotting
    data = []
    
    if comparing_models:
        # Comparing multiple models
        for model_name, perplexities in perplexity_by_topic.items():
            for topic_idx, perplexity in perplexities.items():
                if topic_idx != 'average':  # Skip average if present
                    data.append({
                        'Model': model_name,
                        'Topic': f"Topic {topic_idx}" if isinstance(topic_idx, (int, str)) else topic_idx,
                        'Perplexity': perplexity
                    })
    else:
        # Single model, multiple topics
        for topic_idx, perplexity in perplexity_by_topic.items():
            if topic_idx != 'average':  # Skip average if present
                data.append({
                    'Model': model_names[0],
                    'Topic': f"Topic {topic_idx}" if isinstance(topic_idx, (int, str)) else topic_idx,
                    'Perplexity': perplexity
                })
    
    df = pd.DataFrame(data)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if comparing_models:
        sns.barplot(data=df, x='Topic', y='Perplexity', hue='Model', ax=ax)
        ax.set_title("Perplexity Comparison Across Models by Topic", fontsize=16)
    else:
        sns.barplot(data=df, x='Topic', y='Perplexity', ax=ax)
        ax.set_title("Perplexity by Topic", fontsize=16)
    
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
    
    # Rotate x labels if many topics
    if len(df['Topic'].unique()) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def create_text_examples_table(texts_by_topic, topic_labels=None, max_chars=200):
    """Create a dataframe of text examples by topic
    
    Args:
        texts_by_topic: Dictionary mapping topic indices to lists of texts,
                        or list of lists of texts
        topic_labels: Optional topic labels
        max_chars: Maximum number of characters to display
        
    Returns:
        df: Pandas DataFrame with text examples
    """
    # Convert to dict if list
    if isinstance(texts_by_topic, list):
        texts_by_topic = {i: texts for i, texts in enumerate(texts_by_topic)}
    
    # Get topics
    topics = list(texts_by_topic.keys())
    n_topics = len(topics)
    
    # Create topic labels if not provided
    if topic_labels is None:
        topic_labels = [f"Topic {i+1}" for i in range(n_topics)]
    
    # Create data for dataframe
    data = []
    for i, topic_idx in enumerate(topics):
        topic_texts = texts_by_topic[topic_idx]
        
        # Get up to 3 examples
        for j, text in enumerate(topic_texts[:3]):
            # Truncate if too long
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            
            data.append({
                'Topic': topic_labels[i],
                'Example': j + 1,
                'Text': text
            })
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    return df 