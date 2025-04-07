import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add project directory to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze and compare topic model results')
    parser.add_argument('--results_dir', type=str, default='results',
                      help='Directory containing results (default: results)')
    parser.add_argument('--output_dir', type=str, default='results/comparison',
                      help='Output directory for comparison (default: results/comparison)')
    parser.add_argument('--top_words', type=int, default=10,
                      help='Number of top words to display per topic (default: 10)')
    parser.add_argument('--models', type=str, nargs='+', 
                      default=['vae', 'etm', 'graph', 'clustering'],
                      help='Models to compare (default: all)')
    return parser.parse_args()

def load_results(results_dir, models):
    """Load results for each model from the results directory"""
    print("Loading results...")
    
    results = {}
    
    for model in models:
        model_dir = os.path.join(results_dir, model)
        
        if not os.path.exists(model_dir):
            print(f"Warning: Results for model {model} not found in {model_dir}")
            continue
        
        # Load evaluation metrics
        metrics_file = os.path.join(model_dir, 'metrics.csv')
        if os.path.exists(metrics_file):
            metrics = pd.read_csv(metrics_file)
        else:
            print(f"Warning: Metrics file for model {model} not found")
            metrics = None
        
        # Load top words
        top_words_file = os.path.join(model_dir, 'top_words.csv')
        if os.path.exists(top_words_file):
            top_words = pd.read_csv(top_words_file)
        else:
            print(f"Warning: Top words file for model {model} not found")
            top_words = None
        
        # Load coherence per topic
        coherence_file = os.path.join(model_dir, 'coherence_per_topic.csv')
        if os.path.exists(coherence_file):
            coherence = pd.read_csv(coherence_file)
        else:
            print(f"Warning: Coherence file for model {model} not found")
            coherence = None
        
        # Store results
        results[model] = {
            'metrics': metrics,
            'top_words': top_words,
            'coherence': coherence
        }
    
    return results

def compare_metrics(results, output_dir):
    """Compare metrics across models"""
    print("Comparing metrics...")
    
    # Extract metrics for all models
    metrics_data = []
    for model, data in results.items():
        if data['metrics'] is not None:
            metrics = data['metrics'].copy()
            metrics['model'] = model
            metrics_data.append(metrics)
    
    if not metrics_data:
        print("No metrics data found for comparison")
        return
    
    # Combine metrics
    combined_metrics = pd.concat(metrics_data, ignore_index=True)
    
    # Save combined metrics
    os.makedirs(output_dir, exist_ok=True)
    combined_metrics.to_csv(os.path.join(output_dir, 'metrics_comparison.csv'), index=False)
    
    # Plot metrics comparison
    metric_cols = [col for col in combined_metrics.columns if col != 'model']
    
    for metric in metric_cols:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='model', y=metric, data=combined_metrics)
        
        # Add value labels
        for i, v in enumerate(combined_metrics[metric]):
            ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
        
        plt.title(f'Comparison of {metric.replace("_", " ").title()} Across Models')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'), dpi=300)
        plt.close()
    
    # Create summary table
    summary = combined_metrics.set_index('model')
    
    # Plot a radar chart for all metrics
    plt.figure(figsize=(10, 10))
    
    # Normalize metrics for radar chart
    normalized_metrics = summary.copy()
    for col in normalized_metrics.columns:
        normalized_metrics[col] = (normalized_metrics[col] - normalized_metrics[col].min()) / \
                                 (normalized_metrics[col].max() - normalized_metrics[col].min())
    
    # Radar chart
    categories = normalized_metrics.columns
    N = len(categories)
    
    # Create angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create radar plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw the y-axis labels (0-1)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], size=10)
    plt.ylim(0, 1)
    
    # Plot each model
    for model in normalized_metrics.index:
        values = normalized_metrics.loc[model].values.tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Normalized Metrics Comparison', size=16)
    
    # Save the radar chart
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def compare_coherence(results, output_dir):
    """Compare coherence across models"""
    print("Comparing coherence...")
    
    # Extract coherence data for all models
    coherence_data = {}
    for model, data in results.items():
        if data['coherence'] is not None:
            coherence_data[model] = data['coherence']
    
    if not coherence_data:
        print("No coherence data found for comparison")
        return
    
    # Create and save boxplot for coherence comparison
    plt.figure(figsize=(12, 8))
    
    coherence_values = []
    model_names = []
    
    for model, coherence in coherence_data.items():
        for col in coherence.columns:
            if col.startswith('coherence_'):
                values = coherence[col].dropna().values
                coherence_values.extend(values)
                model_names.extend([model] * len(values))
    
    coherence_df = pd.DataFrame({
        'model': model_names,
        'coherence': coherence_values
    })
    
    sns.boxplot(x='model', y='coherence', data=coherence_df)
    plt.title('Topic Coherence Comparison Across Models')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coherence_comparison.png'), dpi=300)
    plt.close()
    
    # Mean coherence per model
    mean_coherence = coherence_df.groupby('model')['coherence'].mean().reset_index()
    mean_coherence.to_csv(os.path.join(output_dir, 'mean_coherence.csv'), index=False)
    
    # Plot mean coherence
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='model', y='coherence', data=mean_coherence)
    
    # Add value labels
    for i, v in enumerate(mean_coherence['coherence']):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.title('Mean Topic Coherence Across Models')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_coherence.png'), dpi=300)
    plt.close()

def compare_top_words(results, output_dir, top_n=10):
    """Compare top words across models"""
    print("Comparing top words...")
    
    # Extract top words for all models
    top_words_data = {}
    for model, data in results.items():
        if data['top_words'] is not None:
            top_words_data[model] = data['top_words']
    
    if not top_words_data:
        print("No top words data found for comparison")
        return
    
    # Find common topics across models
    topics = defaultdict(list)
    
    for model, df in top_words_data.items():
        for i, row in df.iterrows():
            topic_id = row['topic_id']
            top_words = row['top_words'].split(', ')[:top_n]
            topics[topic_id].append((model, top_words))
    
    # Create a comparison table
    with open(os.path.join(output_dir, 'top_words_comparison.md'), 'w') as f:
        f.write("# Top Words Comparison\n\n")
        
        for topic_id, model_words in topics.items():
            f.write(f"## Topic {topic_id}\n\n")
            f.write("| Model | Top Words |\n")
            f.write("|-------|----------|\n")
            
            for model, words in model_words:
                f.write(f"| {model} | {', '.join(words)} |\n")
            
            f.write("\n")

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.results_dir, args.models)
    
    # Compare metrics
    compare_metrics(results, args.output_dir)
    
    # Compare coherence
    compare_coherence(results, args.output_dir)
    
    # Compare top words
    compare_top_words(results, args.output_dir, args.top_words)
    
    print(f"Comparison completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 