import os
import sys
import argparse
import importlib
import random
import numpy as np
import torch

# Import our fix module
try:
    from fix_scipy_linalg import patch_scipy_linalg, reload_gensim
    # Apply the patch
    patch_scipy_linalg()
    # Reload affected modules
    reload_gensim()
except ImportError:
    # Fallback to the original patch
    try:
        import scipy.linalg
        if not hasattr(scipy.linalg, 'triu'):
            # Add triu function from numpy to scipy.linalg
            from numpy import triu
            scipy.linalg.triu = triu
            print("Patched scipy.linalg.triu with numpy.triu")
    except ImportError:
        print("Warning: Could not import scipy.linalg")

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Topic Models Project')
    parser.add_argument('--task', type=str, default='both', choices=['analysis', 'generation', 'both'],
                      help='Task to run (default: both)')
    
    # 20 Newsgroups (text analysis) settings
    parser.add_argument('--analysis_model', type=str, default='all', 
                      choices=['vae', 'etm', 'graph', 'clustering', 'all'],
                      help='Model for text analysis (default: all)')
    parser.add_argument('--analysis_topics', type=int, default=20,
                      help='Number of topics for analysis (default: 20)')
    parser.add_argument('--analysis_batch_size', type=int, default=64,
                      help='Batch size for analysis (default: 64)')
    parser.add_argument('--analysis_epochs', type=int, default=50,
                      help='Number of epochs for analysis (default: 50)')
    parser.add_argument('--analysis_vocab_size', type=int, default=2000,
                      help='Vocabulary size for analysis (default: 2000)')
    
    # WikiText (text generation) settings
    parser.add_argument('--gen_model', type=str, default='vae', 
                      choices=['vae', 'etm', 'graph', 'clustering'],
                      help='Topic model for generation (default: vae)')
    parser.add_argument('--gen_topics', type=int, default=20,
                      help='Number of topics for generation (default: 20)')
    parser.add_argument('--gen_batch_size', type=int, default=64,
                      help='Batch size for generation (default: 64)')
    parser.add_argument('--topic_epochs', type=int, default=30,
                      help='Number of epochs for topic model (default: 30)')
    parser.add_argument('--gen_epochs', type=int, default=10,
                      help='Number of epochs for generator (default: 10)')
    parser.add_argument('--gen_vocab_size', type=int, default=10000,
                      help='Vocabulary size for generation (default: 10000)')
    parser.add_argument('--gen_layers', type=int, default=2,
                      help='Number of LSTM layers (default: 2)')
    parser.add_argument('--seq_length', type=int, default=64,
                      help='Sequence length for generation (default: 64)')
    
    # Common settings
    parser.add_argument('--embedding_dim', type=int, default=300,
                      help='Embedding dimension (default: 300)')
    parser.add_argument('--hidden_size', type=int, default=500,
                      help='Hidden layer size (default: 500)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (default: cuda if available, else cpu)')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Output directory (default: results)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    
    return parser.parse_args()

def run_text_analysis(args):
    """Run text analysis on 20 Newsgroups dataset"""
    # Import text analysis module
    try:
        from text_analysis.run_20newsgroups import main as analysis_main
    except ImportError:
        print("Could not import text analysis module")
        return
    
    # Prepare arguments for text analysis
    sys.argv = [
        'run_20newsgroups.py',
        f'--model={args.analysis_model}',
        f'--num_topics={args.analysis_topics}',
        f'--batch_size={args.analysis_batch_size}',
        f'--epochs={args.analysis_epochs}',
        f'--vocab_size={args.analysis_vocab_size}',
        f'--embedding_dim={args.embedding_dim}',
        f'--hidden_size={args.hidden_size}',
        f'--device={args.device}',
        f'--output_dir={args.output_dir}'
    ]
    
    # Run text analysis
    print("\n" + "="*80)
    print("RUNNING TEXT ANALYSIS ON 20 NEWSGROUPS DATASET")
    print("="*80 + "\n")
    analysis_main()

def run_text_generation(args):
    """Run text generation on WikiText dataset"""
    # Import text generation module
    try:
        from text_generation.run_wikitext import main as generation_main
    except ImportError:
        print("Could not import text generation module")
        return
    
    # Prepare arguments for text generation
    sys.argv = [
        'run_wikitext.py',
        f'--topic_model={args.gen_model}',
        f'--num_topics={args.gen_topics}',
        f'--batch_size={args.gen_batch_size}',
        f'--topic_epochs={args.topic_epochs}',
        f'--gen_epochs={args.gen_epochs}',
        f'--vocab_size={args.gen_vocab_size}',
        f'--embedding_dim={args.embedding_dim}',
        f'--hidden_size={args.hidden_size}',
        f'--num_layers={args.gen_layers}',
        f'--seq_length={args.seq_length}',
        f'--device={args.device}',
        f'--output_dir={args.output_dir}'
    ]
    
    # Run text generation
    print("\n" + "="*80)
    print("RUNNING TEXT GENERATION ON WIKITEXT DATASET")
    print("="*80 + "\n")
    generation_main()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run selected tasks
    if args.task in ['analysis', 'both']:
        run_text_analysis(args)
    
    if args.task in ['generation', 'both']:
        run_text_generation(args)
    
    print("\nAll tasks completed!")

if __name__ == "__main__":
    main() 