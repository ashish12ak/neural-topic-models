# Neural Topic Models - Installation Guide

This project requires several Python packages. Follow these steps to set up the environment and run the project.

## Requirements

The following packages are required to run this project:

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Pandas
- Scikit-learn
- Gensim
- NLTK
- Wordcloud
- SciPy
- UMAP-learn
- Seaborn
- tqdm

## Installation Steps

### 1. Clone the repository

```bash
git clone <repository-url>
cd neural-topic-models
```

### 2. Create a virtual environment (recommended)

```bash
# Using virtualenv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n ntm python=3.8
conda activate ntm
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Troubleshooting

### ImportError issues

If you encounter import errors for packages like `numpy`, `torch`, etc., make sure you have:

1. Installed all required packages from `requirements.txt`
2. Activated your virtual environment
3. Run the code from the project root directory

### CUDA issues

If you're using PyTorch with CUDA and encounter issues:

1. Verify your CUDA version: `nvidia-smi`
2. Install PyTorch with the matching CUDA version from [PyTorch's official website](https://pytorch.org/get-started/locally/)

### Pre-trained word embeddings

Some models require pre-trained word embeddings. Follow these steps to prepare them:

```bash
# Download GloVe embeddings
mkdir -p data/embeddings
cd data/embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
cd ../..
```

## Running the Code

After installation, you can run the main script:

```bash
python neural_topic_models/main.py
```

To see all available options:

```bash
python neural_topic_models/main.py --help
```

For more detailed usage instructions, see the [README.md](README.md) file. 