# Neural Topic Models

This project implements four different neural topic modeling approaches for text analysis and generation:

1. VAE-based Neural Topic Models
2. Embeddings-based Neural Topic Models
3. Graph-based Neural Topic Models
4. Clustering-based Neural Topic Models

Each method is applied to two datasets:
- 20 Newsgroups (for text analysis)
- WikiText (for text generation)

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
neural_topic_models/
├── data/                       # Data handling code
│   ├── data_loader.py          # Functions to load datasets
│   └── preprocessing.py        # Text preprocessing utilities
├── models/                     # Implementation of topic models
│   ├── vae_ntm.py              # VAE-based Neural Topic Model
│   ├── etm.py                  # Embeddings-based Neural Topic Model
│   ├── graph_ntm.py            # Graph-based Neural Topic Model
│   └── clustering_ntm.py       # Clustering-based Neural Topic Model
├── evaluation/                 # Evaluation metrics
│   ├── coherence.py            # Topic coherence metrics
│   ├── diversity.py            # Topic diversity metrics
│   └── perplexity.py           # Perplexity for text generation
├── visualization/              # Visualization utilities
│   ├── topic_vis.py            # Topic visualization
│   └── text_vis.py             # Text generation visualization
├── text_analysis/              # Text analysis experiments
│   └── run_20newsgroups.py     # Run on 20 Newsgroups dataset
├── text_generation/            # Text generation experiments
│   └── run_wikitext.py         # Run on WikiText dataset
└── main.py                     # Main entry point
```

## Usage

Run the main script to execute all experiments:

```bash
python main.py
```

Or run individual experiments:

```bash
# For text analysis on 20 Newsgroups
python -m text_analysis.run_20newsgroups

# For text generation on WikiText
python -m text_generation.run_wikitext
```

## Evaluation Metrics

- **Topic Coherence**: Measures semantic coherence of discovered topics
- **Topic Diversity**: Quantifies uniqueness of discovered topics
- **Perplexity**: Measures quality of text generation

## Visualization

- Topic word clouds
- t-SNE/UMAP plots of document embeddings
- Heatmaps of topic-word distributions
- Generated text examples with topic influence visualization 