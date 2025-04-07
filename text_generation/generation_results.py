import os
import sys
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add project directory to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze text generation results')
    parser.add_argument('--results_dir', type=str, default='results',
                      help='Directory containing results (default: results)')
    parser.add_argument('--output_dir', type=str, default='results/generation_comparison',
                      help='Output directory for comparison (default: results/generation_comparison)')
    parser.add_argument('--models', type=str, nargs='+', 
                      default=['vae', 'etm', 'graph', 'clustering'],
                      help='Models to compare (default: all)')
    return parser.parse_args()

def load_generation_results(results_dir, models):
    """Load text generation results for each model from the results directory"""
    print("Loading generation results...")
    
    results = {}
    
    for model in models:
        model_dir = os.path.join(results_dir, model + '_generation')
        
        if not os.path.exists(model_dir):
            print(f"Warning: Results for model {model} not found in {model_dir}")
            continue
        
        # Load perplexity scores
        perplexity_file = os.path.join(model_dir, 'perplexity.csv')
        if os.path.exists(perplexity_file):
            perplexity = pd.read_csv(perplexity_file)
        else:
            print(f"Warning: Perplexity file for model {model} not found")
            perplexity = None
        
        # Load generated texts
        generated_texts_file = os.path.join(model_dir, 'generated_texts.csv')
        if os.path.exists(generated_texts_file):
            generated_texts = pd.read_csv(generated_texts_file)
        else:
            print(f"Warning: Generated texts file for model {model} not found")
            generated_texts = None
        
        # Load diversity metrics
        diversity_file = os.path.join(model_dir, 'diversity_metrics.csv')
        if os.path.exists(diversity_file):
            diversity = pd.read_csv(diversity_file)
        else:
            print(f"Warning: Diversity metrics file for model {model} not found")
            diversity = None
        
        # Store results
        results[model] = {
            'perplexity': perplexity,
            'generated_texts': generated_texts,
            'diversity': diversity
        }
    
    return results

def compare_perplexity(results, output_dir):
    """Compare perplexity across models"""
    print("Comparing perplexity...")
    
    # Extract perplexity data for all models
    perplexity_data = []
    
    for model, data in results.items():
        if data['perplexity'] is not None:
            perplexity = data['perplexity'].copy()
            perplexity['model'] = model
            perplexity_data.append(perplexity)
    
    if not perplexity_data:
        print("No perplexity data found for comparison")
        return
    
    # Combine perplexity data
    combined_perplexity = pd.concat(perplexity_data, ignore_index=True)
    
    # Save combined perplexity data
    os.makedirs(output_dir, exist_ok=True)
    combined_perplexity.to_csv(os.path.join(output_dir, 'perplexity_comparison.csv'), index=False)
    
    # Plot average perplexity comparison
    if 'average_perplexity' in combined_perplexity.columns:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='model', y='average_perplexity', data=combined_perplexity)
        
        # Add value labels
        for i, v in enumerate(combined_perplexity['average_perplexity']):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
        
        plt.title('Average Perplexity Across Models')
        plt.ylabel('Average Perplexity')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'average_perplexity.png'), dpi=300)
        plt.close()
    
    # Plot per-topic perplexity comparison if available
    topic_perplexity_cols = [col for col in combined_perplexity.columns 
                          if col.startswith('topic_') and col.endswith('_perplexity')]
    
    if topic_perplexity_cols:
        # Reshape data for topic perplexity comparison
        topic_perplexity_data = []
        
        for model, data in results.items():
            if data['perplexity'] is not None:
                perplexity = data['perplexity']
                
                for col in topic_perplexity_cols:
                    topic_id = col.split('_')[1]
                    if col in perplexity.columns:
                        topic_perplexity_data.append({
                            'model': model,
                            'topic': f'Topic {topic_id}',
                            'perplexity': perplexity[col].iloc[0]
                        })
        
        topic_perplexity_df = pd.DataFrame(topic_perplexity_data)
        
        # Plot per-topic perplexity
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='topic', y='perplexity', hue='model', data=topic_perplexity_df)
        plt.title('Per-Topic Perplexity Across Models')
        plt.ylabel('Perplexity')
        plt.legend(title='Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'topic_perplexity.png'), dpi=300)
        plt.close()

def compare_diversity(results, output_dir):
    """Compare text generation diversity across models"""
    print("Comparing diversity...")
    
    # Extract diversity data for all models
    diversity_data = []
    
    for model, data in results.items():
        if data['diversity'] is not None:
            diversity = data['diversity'].copy()
            diversity['model'] = model
            diversity_data.append(diversity)
    
    if not diversity_data:
        print("No diversity data found for comparison")
        return
    
    # Combine diversity data
    combined_diversity = pd.concat(diversity_data, ignore_index=True)
    
    # Save combined diversity data
    combined_diversity.to_csv(os.path.join(output_dir, 'diversity_comparison.csv'), index=False)
    
    # Plot diversity metrics comparison
    metric_cols = [col for col in combined_diversity.columns 
                if col != 'model' and col != 'topic_id']
    
    for metric in metric_cols:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='model', y=metric, data=combined_diversity)
        
        # Add value labels
        for i, v in enumerate(combined_diversity[metric]):
            ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
        
        plt.title(f'Comparison of {metric.replace("_", " ").title()} Across Models')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'), dpi=300)
        plt.close()
    
    # Create a radar chart to compare all diversity metrics
    if len(metric_cols) > 1:
        # Normalize metrics for radar chart
        normalized_metrics = combined_diversity.set_index('model')
        for col in metric_cols:
            normalized_metrics[col] = (normalized_metrics[col] - normalized_metrics[col].min()) / \
                                     (normalized_metrics[col].max() - normalized_metrics[col].min() + 1e-10)
        
        # Create radar chart
        from matplotlib.path import Path
        from matplotlib.spines import Spine
        from matplotlib.transforms import Affine2D
        
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        
        # Set number of variables
        categories = metric_cols
        N = len(categories)
        
        # Create angle for each variable
        angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
        angles += angles[:1]  # Complete the loop
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], categories, size=12)
        
        # Draw the y-axis labels (0-1)
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], size=10)
        plt.ylim(0, 1)
        
        # Plot each model
        for model in normalized_metrics.index.unique():
            values = normalized_metrics.loc[model, metric_cols].values.flatten().tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Normalized Diversity Metrics Comparison', size=16)
        
        # Save radar chart
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'diversity_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()

def create_text_examples_table(results, output_dir):
    """Create a table of generated text examples for each model and topic"""
    print("Creating text examples table...")
    
    # Collect generated texts for all models
    all_texts = []
    
    for model, data in results.items():
        if data['generated_texts'] is not None:
            texts = data['generated_texts']
            
            # Add model information
            texts['model'] = model
            
            # Select a subset of columns if needed
            if 'topic_id' in texts.columns and 'generated_text' in texts.columns:
                selected_texts = texts[['model', 'topic_id', 'generated_text']]
                all_texts.append(selected_texts)
    
    if not all_texts:
        print("No generated texts found for comparison")
        return
    
    # Combine texts
    combined_texts = pd.concat(all_texts, ignore_index=True)
    
    # Save combined texts
    combined_texts.to_csv(os.path.join(output_dir, 'generated_texts_comparison.csv'), index=False)
    
    # Create a markdown table with examples
    with open(os.path.join(output_dir, 'text_examples.md'), 'w') as f:
        f.write("# Generated Text Examples\n\n")
        
        # Group by topic
        topics = combined_texts['topic_id'].unique()
        
        for topic in sorted(topics):
            f.write(f"## Topic {topic}\n\n")
            f.write("| Model | Generated Text |\n")
            f.write("|-------|---------------|\n")
            
            # Get texts for this topic
            topic_texts = combined_texts[combined_texts['topic_id'] == topic]
            
            for model in topic_texts['model'].unique():
                model_texts = topic_texts[topic_texts['model'] == model]['generated_text']
                
                if not model_texts.empty:
                    # Get a sample text (first one)
                    sample_text = model_texts.iloc[0]
                    
                    # Truncate if too long
                    if len(sample_text) > 500:
                        sample_text = sample_text[:500] + "..."
                    
                    # Replace newlines and escape pipes
                    sample_text = sample_text.replace('\n', ' ').replace('|', '\\|')
                    
                    f.write(f"| {model} | {sample_text} |\n")
            
            f.write("\n")

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load generation results
    results = load_generation_results(args.results_dir, args.models)
    
    # Compare perplexity
    compare_perplexity(results, args.output_dir)
    
    # Compare diversity
    compare_diversity(results, args.output_dir)
    
    # Create text examples table
    create_text_examples_table(results, args.output_dir)
    
    print(f"Generation results comparison completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 