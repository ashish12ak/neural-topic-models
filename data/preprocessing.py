import re
import string
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def clean_text(text, remove_stopwords=True, lemmatize=True):
    """Clean text by removing special characters, numbers, and optionally stopwords
    
    Args:
        text: String text to clean
        remove_stopwords: Whether to remove stopwords
        lemmatize: Whether to lemmatize words
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Rejoin
    text = ' '.join(tokens)
    
    return text

def create_bow_matrix(documents, max_features=5000, min_df=5):
    """Create bag-of-words matrix from documents
    
    Args:
        documents: List of document strings
        max_features: Maximum number of features (vocabulary size)
        min_df: Minimum document frequency
        
    Returns:
        vectorizer: CountVectorizer object
        X: Document-term matrix
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words='english'
    )
    X = vectorizer.fit_transform(documents)
    
    return vectorizer, X

def create_tfidf_matrix(documents, max_features=5000, min_df=5):
    """Create TF-IDF matrix from documents
    
    Args:
        documents: List of document strings
        max_features: Maximum number of features (vocabulary size)
        min_df: Minimum document frequency
        
    Returns:
        vectorizer: TfidfVectorizer object
        X: Document-term matrix
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words='english'
    )
    X = vectorizer.fit_transform(documents)
    
    return vectorizer, X

def preprocess_documents(documents, clean=True, vectorize=True, max_features=5000):
    """Preprocess documents with cleaning and vectorization
    
    Args:
        documents: List of document strings
        clean: Whether to clean text
        vectorize: Whether to create BoW matrix
        max_features: Maximum vocabulary size
        
    Returns:
        processed_docs: List of cleaned documents
        vectorizer: Vectorizer object (if vectorize=True)
        bow_matrix: Document-term matrix (if vectorize=True)
    """
    # Clean documents
    if clean:
        print("Cleaning documents...")
        processed_docs = [clean_text(doc) for doc in documents]
    else:
        processed_docs = documents
    
    # Create BoW matrix
    if vectorize:
        print(f"Creating BoW matrix (max_features={max_features})...")
        vectorizer, bow_matrix = create_bow_matrix(
            processed_docs, max_features=max_features
        )
        return processed_docs, vectorizer, bow_matrix
    
    return processed_docs

def get_most_frequent_words(documents, top_n=100):
    """Get most frequent words in corpus
    
    Args:
        documents: List of document strings
        top_n: Number of top words to return
        
    Returns:
        List of top words
    """
    all_words = []
    for doc in documents:
        words = doc.split()
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    top_words = [word for word, count in word_counts.most_common(top_n)]
    
    return top_words 