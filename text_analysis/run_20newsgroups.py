import os
import sys
import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

# Add project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from data.data_loader import load_20newsgroups, NewsGroupsDataset, get_dataloaders
from data.preprocessing import preprocess_documents
from models.vae_ntm import VAE_NTM, train_vae_ntm, get_topic_words, get_document_topic_dist
from models.etm import ETM, train_etm, load_pretrained_embeddings, get_nearest_words_to_topics
from models.graph_ntm import GraphNTM, train_graph_ntm, build_word_cooccurrence_graph
from models.clustering_ntm import ClusteringNTM, compute_coherence, compute_diversity
from evaluation.coherence import compute_topic_coherence_scores
from evaluation.diversity import evaluate_topic_model_diversity
from visualization.topic_vis import (
    plot_topic_word_heatmap, plot_wordclouds, plot_document_topic_heatmap,
    plot_topic_embeddings, plot_document_embeddings
)

def parse_args():
    parser = argparse.ArgumentParser(description='Run neural topic models on 20 Newsgroups dataset')
    parser.add_argument('--model', type=str, default='all', 
                      choices=['vae', 'etm', 'graph', 'clustering', 'all'],
                      help='Model to run (default: all)')
    parser.add_argument('--num_topics', type=int, default=20,
                      help='Number of topics (default: 20)')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs (default: 50)')
    parser.add_argument('--vocab_size', type=int, default=2000,
                      help='Vocabulary size (default: 2000)')
    parser.add_argument('--embedding_dim', type=int, default=300,
                      help='Embedding dimension (default: 300)')
    parser.add_argument('--hidden_size', type=int, default=500,
                      help='Hidden layer size (default: 500)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (default: cuda if available, else cpu)')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Output directory (default: results)')
    return parser.parse_args()

def run_vae_ntm(args, train_loader, val_loader, test_loader, vocab):
    """Run VAE-based Neural Topic Model"""
    print("\n===== Running VAE Neural Topic Model =====")
    
    # Create model
    model = VAE_NTM(
        vocab_size=args.vocab_size,
        num_topics=args.num_topics,
        hidden_size=args.hidden_size,
        dropout=0.2
    )
    
    # Train model
    start_time = time.time()
    model, train_losses, val_losses = train_vae_ntm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=0.002,
        beta=1.0,
        device=args.device
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Get topic-word distribution
    topic_word_dist = model.get_topic_word_dist()
    
    # Get top words for each topic
    topic_words = get_topic_words(topic_word_dist, vocab, k=10)
    
    # Print top words for each topic
    print("\nTop 10 words for each topic:")
    for i, words in enumerate(topic_words):
        print(f"Topic {i+1}: {', '.join(words)}")
    
    # Get document-topic distribution
    doc_topic_dist = get_document_topic_dist(model, test_loader, device=args.device)
    
    return model, topic_word_dist, topic_words, doc_topic_dist, training_time

def run_etm(args, train_loader, val_loader, test_loader, vocab, raw_documents):
    """Run Embeddings-based Topic Model"""
    print("\n===== Running Embeddings-based Topic Model =====")
    
    # Try to load pretrained embeddings if available
    try:
        # Check if pretrained embeddings file exists
        embed_file = os.path.join('data', 'pretrained', 'GoogleNews-vectors-negative300.bin')
        if os.path.exists(embed_file):
            pretrained_embeddings = load_pretrained_embeddings(
                vocab, embed_file, embedding_dim=args.embedding_dim
            )
        else:
            print("Pretrained embeddings file not found, initializing randomly")
            pretrained_embeddings = None
    except Exception as e:
        print(f"Error loading pretrained embeddings: {e}")
        pretrained_embeddings = None
    
    # Create model
    model = ETM(
        vocab_size=args.vocab_size,
        num_topics=args.num_topics,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        pretrained_embeddings=pretrained_embeddings,
        trainable_embeddings=True
    )
    
    # Train model
    start_time = time.time()
    model, train_losses, val_losses = train_etm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=0.002,
        device=args.device
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Get topic-word distribution
    topic_word_dist = model.get_topic_word_dist()
    
    # Get top words for each topic
    topic_words = get_nearest_words_to_topics(model, vocab, k=10)
    
    # Print top words for each topic
    print("\nTop 10 words for each topic:")
    for i, words in enumerate(topic_words):
        print(f"Topic {i+1}: {', '.join(words)}")
    
    # Get document-topic distribution using encoder
    doc_topic_dist = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                x = batch['bow'].to(args.device)
            else:
                x = batch.to(args.device)
            
            theta = model.get_document_topic_dist(x)
            doc_topic_dist.append(theta.cpu().numpy())
    
    doc_topic_dist = np.vstack(doc_topic_dist)
    
    return model, topic_word_dist, topic_words, doc_topic_dist, training_time

def run_graph_ntm(args, train_loader, val_loader, test_loader, vocab, raw_documents):
    """Run Graph-based Neural Topic Model"""
    print("\n===== Running Graph-based Neural Topic Model =====")
    
    # Create model
    model = GraphNTM(
        vocab_size=args.vocab_size,
        num_topics=args.num_topics,
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        dropout=0.2
    )
    
    # Train model
    start_time = time.time()
    model, train_losses, val_losses = train_graph_ntm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=0.002,
        device=args.device,
        with_graph_structure=False  # Using standard training for better efficiency
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
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
    elif hasattr(vocab, 'get_feature_names_out'):
        vocab_list = vocab.get_feature_names_out()
    else:
        vocab_list = vocab
    
    # Get top words
    topic_words = []
    for indices in top_indices:
        topic_words.append([vocab_list[idx] for idx in indices])
    
    # Print top words for each topic
    print("\nTop 10 words for each topic:")
    for i, words in enumerate(topic_words):
        print(f"Topic {i+1}: {', '.join(words)}")
    
    # Get document-topic distribution
    doc_topic_dist = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                x = batch['bow'].to(args.device)
            else:
                x = batch.to(args.device)
            
            _, theta = model(x)
            doc_topic_dist.append(theta.cpu().numpy())
    
    doc_topic_dist = np.vstack(doc_topic_dist)
    
    return model, topic_word_dist, topic_words, doc_topic_dist, training_time

def run_clustering_ntm(args, train_loader, val_loader, test_loader, vocab, raw_documents):
    """Run Clustering-based Neural Topic Model"""
    print("\n===== Running Clustering-based Neural Topic Model =====")
    
    # Create model
    model = ClusteringNTM(
        embedding_model='all-MiniLM-L6-v2',
        num_topics=args.num_topics,
        umap_dim=5,
        min_cluster_size=5,
        random_state=42
    )
    
    # Train model
    start_time = time.time()
    model.fit(raw_documents, method='kmeans')
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Get topic-word distribution
    topic_word_dist = model.get_topic_word_dist()
    
    # Get top words for each topic
    topic_words = model.get_topics()
    
    # Print top words for each topic
    print("\nTop 10 words for each topic:")
    for i, words in enumerate(topic_words):
        print(f"Topic {i+1}: {', '.join(words)}")
    
    # Get document-topic distribution
    doc_topic_dist = model.get_document_topic_dist(raw_documents)
    
    return model, topic_word_dist, topic_words, doc_topic_dist, training_time

def evaluate_model(model_name, model, topic_words, topic_word_dist, doc_topic_dist, 
                 processed_docs, raw_documents, vectorizer):
    """Evaluate topic model and generate visualizations"""
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join('results', '20newsgroups', model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n===== Evaluating {model_name} =====")
    
    # Add vocabulary to model for diversity evaluation
    if not hasattr(model, 'vocab'):
        model.vocab = vectorizer.get_feature_names_out()
    
    # If model has topic_embeddings with requires_grad=True, detach them
    if hasattr(model, 'topic_embeddings') and hasattr(model.topic_embeddings, 'requires_grad'):
        model.topic_embeddings_detached = model.topic_embeddings.detach()
    
    # Compute coherence
    coherence_scores = compute_topic_coherence_scores(topic_words, raw_documents)
    print(f"Coherence scores:")
    print(f"  C_v: {coherence_scores['c_v']:.4f}")
    print(f"  U_mass: {coherence_scores['u_mass']:.4f}")
    print(f"  NPMI: {coherence_scores['c_npmi']:.4f}")
    
    # Save coherence scores
    pd.DataFrame({
        'metric': list(coherence_scores.keys()),
        'score': list(coherence_scores.values())
    }).to_csv(os.path.join(output_dir, 'coherence_scores.csv'), index=False)
    
    # Compute diversity
    diversity_scores = evaluate_topic_model_diversity(model, topk=10)
    print(f"Diversity scores:")
    for metric, score in diversity_scores.items():
        print(f"  {metric}: {score:.4f}")
    
    # Save diversity scores
    pd.DataFrame({
        'metric': list(diversity_scores.keys()),
        'score': list(diversity_scores.values())
    }).to_csv(os.path.join(output_dir, 'diversity_scores.csv'), index=False)
    
    # Generate visualizations
    
    # 1. Topic-word heatmap
    fig, _ = plot_topic_word_heatmap(topic_word_dist, vectorizer, top_n=10)
    fig.savefig(os.path.join(output_dir, 'topic_word_heatmap.png'))
    plt.close(fig)
    
    # 2. Wordclouds
    fig = plot_wordclouds(topic_words)
    fig.savefig(os.path.join(output_dir, 'topic_wordclouds.png'))
    plt.close(fig)
    
    # 3. Document-topic heatmap
    fig = plot_document_topic_heatmap(doc_topic_dist, n_docs=20)
    fig.savefig(os.path.join(output_dir, 'document_topic_heatmap.png'))
    plt.close(fig)
    
    # 4. Topic embeddings (if available)
    if hasattr(model, 'topic_embeddings') or hasattr(model, 'get_topic_embeddings'):
        try:
            if hasattr(model, 'get_topic_embeddings'):
                topic_embeddings = model.get_topic_embeddings()
            else:
                topic_embeddings = model.topic_embeddings.detach().cpu().numpy()
            
            fig = plot_topic_embeddings(topic_embeddings, method='tsne')
            fig.savefig(os.path.join(output_dir, 'topic_embeddings_tsne.png'))
            plt.close(fig)
            
            fig = plot_topic_embeddings(topic_embeddings, method='umap')
            fig.savefig(os.path.join(output_dir, 'topic_embeddings_umap.png'))
            plt.close(fig)
        except Exception as e:
            print(f"Error generating topic embedding plots: {e}")
    
    # 5. Document embeddings (if available)
    if hasattr(model, 'document_embeddings') or model_name == 'clustering_ntm':
        try:
            if model_name == 'clustering_ntm':
                doc_embeddings = model.document_embeddings
                doc_labels = model.cluster_labels
            elif hasattr(model, 'document_embeddings'):
                doc_embeddings = model.document_embeddings
                doc_labels = np.argmax(doc_topic_dist, axis=1)
            
            fig = plot_document_embeddings(doc_embeddings, doc_labels, n_docs=1000, method='umap')
            fig.savefig(os.path.join(output_dir, 'document_embeddings.png'))
            plt.close(fig)
        except Exception as e:
            print(f"Error generating document embedding plots: {e}")
    
    # Save model information
    with open(os.path.join(output_dir, 'model_info.txt'), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Number of topics: {len(topic_words)}\n")
        f.write(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}\n")
        f.write(f"Coherence (C_v): {coherence_scores['c_v']:.4f}\n")
        f.write(f"Diversity: {diversity_scores['unique_word_proportion']:.4f}\n")
        f.write("\nTop words for each topic:\n")
        for i, words in enumerate(topic_words):
            f.write(f"Topic {i+1}: {', '.join(words)}\n")
    
    return coherence_scores, diversity_scores

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, '20newsgroups'), exist_ok=True)
    
    # Load and preprocess data
    print("Loading 20 Newsgroups dataset...")
    documents, labels, label_names = load_20newsgroups()
    
    # Preprocess documents
    processed_docs, vectorizer, bow_matrix = preprocess_documents(
        documents, max_features=args.vocab_size
    )
    
    # Create dataset and dataloaders
    dataset = NewsGroupsDataset(processed_docs, labels, bow_matrix)
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset, batch_size=args.batch_size
    )
    
    # Get vocabulary
    vocab = vectorizer.get_feature_names_out()
    
    # Store results
    results = {}
    
    # Run models
    if args.model in ['vae', 'all']:
        model, topic_word_dist, topic_words, doc_topic_dist, training_time = run_vae_ntm(
            args, train_loader, val_loader, test_loader, vocab
        )
        
        coherence, diversity = evaluate_model(
            'vae_ntm', model, topic_words, topic_word_dist, doc_topic_dist,
            processed_docs, documents, vectorizer
        )
        
        results['vae_ntm'] = {
            'model': model,
            'topic_words': topic_words,
            'coherence': coherence,
            'diversity': diversity,
            'training_time': training_time
        }
    
    if args.model in ['etm', 'all']:
        model, topic_word_dist, topic_words, doc_topic_dist, training_time = run_etm(
            args, train_loader, val_loader, test_loader, vocab, documents
        )
        
        coherence, diversity = evaluate_model(
            'etm', model, topic_words, topic_word_dist, doc_topic_dist,
            processed_docs, documents, vectorizer
        )
        
        results['etm'] = {
            'model': model,
            'topic_words': topic_words,
            'coherence': coherence,
            'diversity': diversity,
            'training_time': training_time
        }
    
    if args.model in ['graph', 'all']:
        model, topic_word_dist, topic_words, doc_topic_dist, training_time = run_graph_ntm(
            args, train_loader, val_loader, test_loader, vocab, documents
        )
        
        coherence, diversity = evaluate_model(
            'graph_ntm', model, topic_words, topic_word_dist, doc_topic_dist,
            processed_docs, documents, vectorizer
        )
        
        results['graph_ntm'] = {
            'model': model,
            'topic_words': topic_words,
            'coherence': coherence,
            'diversity': diversity,
            'training_time': training_time
        }
    
    if args.model in ['clustering', 'all']:
        model, topic_word_dist, topic_words, doc_topic_dist, training_time = run_clustering_ntm(
            args, train_loader, val_loader, test_loader, vocab, documents
        )
        
        coherence, diversity = evaluate_model(
            'clustering_ntm', model, topic_words, topic_word_dist, doc_topic_dist,
            processed_docs, documents, vectorizer
        )
        
        results['clustering_ntm'] = {
            'model': model,
            'topic_words': topic_words,
            'coherence': coherence,
            'diversity': diversity,
            'training_time': training_time
        }
    
    # Comparative analysis
    if len(results) > 1:
        print("\n===== Comparative Analysis =====")
        
        # Prepare data for comparison
        models = list(results.keys())
        metrics = {
            'C_v Coherence': [results[m]['coherence']['c_v'] for m in models],
            'U_Mass Coherence': [results[m]['coherence']['u_mass'] for m in models],
            'NPMI Coherence': [results[m]['coherence']['c_npmi'] for m in models],
            'Diversity': [results[m]['diversity']['unique_word_proportion'] for m in models],
            'Training Time (s)': [results[m]['training_time'] for m in models]
        }
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(metrics, index=models)
        print(comparison_df)
        
        # Save comparison
        comparison_df.to_csv(os.path.join(args.output_dir, '20newsgroups', 'model_comparison.csv'))
        
        # Create plots
        os.makedirs(os.path.join(args.output_dir, '20newsgroups', 'comparison'), exist_ok=True)
        
        # Plot coherence comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df[['C_v Coherence', 'NPMI Coherence']].plot(kind='bar', ax=ax)
        ax.set_title('Coherence Comparison', fontsize=16)
        ax.set_ylabel('Coherence Score', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, '20newsgroups', 'comparison', 'coherence_comparison.png'))
        plt.close(fig)
        
        # Plot diversity comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df[['Diversity']].plot(kind='bar', ax=ax)
        ax.set_title('Diversity Comparison', fontsize=16)
        ax.set_ylabel('Diversity Score', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, '20newsgroups', 'comparison', 'diversity_comparison.png'))
        plt.close(fig)
        
        # Plot training time comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df[['Training Time (s)']].plot(kind='bar', ax=ax)
        ax.set_title('Training Time Comparison', fontsize=16)
        ax.set_ylabel('Training Time (s)', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, '20newsgroups', 'comparison', 'training_time_comparison.png'))
        plt.close(fig)
        
        # Create topic word overlap analysis
        print("\nTopic Word Overlap Analysis:")
        
        # Flatten all topic words
        all_topic_words = {}
        for model_name in models:
            all_words = []
            for topic_words in results[model_name]['topic_words']:
                all_words.extend(topic_words)
            all_topic_words[model_name] = set(all_words)
        
        # Compute overlap
        overlap_matrix = np.zeros((len(models), len(models)))
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i == j:
                    overlap_matrix[i, j] = 1.0
                else:
                    intersection = len(all_topic_words[model1].intersection(all_topic_words[model2]))
                    union = len(all_topic_words[model1].union(all_topic_words[model2]))
                    overlap_matrix[i, j] = intersection / union
        
        # Create dataframe
        overlap_df = pd.DataFrame(overlap_matrix, index=models, columns=models)
        print(overlap_df)
        
        # Save overlap matrix
        overlap_df.to_csv(os.path.join(args.output_dir, '20newsgroups', 'comparison', 'topic_word_overlap.csv'))
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(overlap_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
        ax.set_title("Topic Word Overlap Between Models", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, '20newsgroups', 'comparison', 'topic_word_overlap.png'))
        plt.close(fig)
    
    print("\nAnalysis complete. Results saved to:", os.path.join(args.output_dir, '20newsgroups'))

if __name__ == "__main__":
    main() 