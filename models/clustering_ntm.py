import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
import umap
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
# Comment out SentenceTransformer import
# from sentence_transformers import SentenceTransformer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from collections import defaultdict
import warnings

class ClusteringNTM:
    """Clustering-based Neural Topic Model
    
    This model uses document embeddings and applies clustering techniques 
    to identify topics. This version uses TF-IDF instead of pre-trained language models.
    """
    
    def __init__(self, embedding_model='tfidf', num_topics=10, 
                 umap_dim=5, min_cluster_size=5, random_state=42):
        """Initialize Clustering-based Topic Model
        
        Args:
            embedding_model: Embedding approach ('tfidf' or 'svd')
            num_topics: Number of topics (used for KMeans)
            umap_dim: UMAP dimensionality reduction dimensions
            min_cluster_size: Minimum cluster size for HDBSCAN
            random_state: Random state for reproducibility
        """
        self.num_topics = num_topics
        self.umap_dim = umap_dim
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.embedding_approach = embedding_model
            
        # Initialize components
        self.umap_model = None
        self.hdbscan_model = None
        self.kmeans_model = None
        self.vectorizer = None
        self.tfidf_transformer = None
        self.svd_model = None
        
        # Fitted data
        self.document_embeddings = None
        self.reduced_embeddings = None
        self.topic_embeddings = None
        self.topic_words = None
        self.cluster_centers = None
        self.topic_word_matrix = None
        self.vocab = None
        
    def _embed_documents(self, documents):
        """Embed documents using TF-IDF
        
        Args:
            documents: List of document strings
            
        Returns:
            Document embeddings
        """
        print("Embedding documents with TF-IDF...")
        
        # Option 1: Use TF-IDF directly
        if self.embedding_approach == 'tfidf':
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(
                    max_features=5000,
                    min_df=5,
                    max_df=0.9,
                    stop_words='english'
                )
                tfidf_matrix = self.vectorizer.fit_transform(documents)
                self.vocab = self.vectorizer.get_feature_names_out()
            else:
                tfidf_matrix = self.vectorizer.transform(documents)
                
            # Convert to dense matrix for further processing
            embeddings = tfidf_matrix.toarray()
            
        # Option 2: TF-IDF + SVD for dimensionality reduction
        elif self.embedding_approach == 'svd':
            if self.vectorizer is None:
                self.vectorizer = CountVectorizer(
                    max_features=10000,
                    min_df=5,
                    max_df=0.9,
                    stop_words='english'
                )
                counts = self.vectorizer.fit_transform(documents)
                self.vocab = self.vectorizer.get_feature_names_out()
                
                self.tfidf_transformer = TfidfTransformer()
                tfidf_matrix = self.tfidf_transformer.fit_transform(counts)
                
                # Apply SVD (similar to LSA)
                self.svd_model = TruncatedSVD(n_components=100, random_state=self.random_state)
                embeddings = self.svd_model.fit_transform(tfidf_matrix)
            else:
                counts = self.vectorizer.transform(documents)
                tfidf_matrix = self.tfidf_transformer.transform(counts)
                embeddings = self.svd_model.transform(tfidf_matrix)
        
        # Default to TF-IDF if embedding approach not recognized
        else:
            print(f"Warning: Embedding approach '{self.embedding_approach}' not recognized, using TF-IDF instead")
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(
                    max_features=5000,
                    min_df=5,
                    max_df=0.9,
                    stop_words='english'
                )
                tfidf_matrix = self.vectorizer.fit_transform(documents)
                self.vocab = self.vectorizer.get_feature_names_out()
            else:
                tfidf_matrix = self.vectorizer.transform(documents)
                
            # Convert to dense matrix for further processing
            embeddings = tfidf_matrix.toarray()
        
        return embeddings
    
    def _reduce_dimensions(self, embeddings):
        """Reduce embedding dimensions using UMAP
        
        Args:
            embeddings: Document embeddings
            
        Returns:
            Reduced embeddings
        """
        print(f"Reducing dimensions with UMAP to {self.umap_dim} dimensions...")
        if self.umap_model is None:
            self.umap_model = umap.UMAP(
                n_components=self.umap_dim,
                metric='cosine',
                min_dist=0.0,
                random_state=self.random_state
            )
            return self.umap_model.fit_transform(embeddings)
        else:
            return self.umap_model.transform(embeddings)
    
    def _cluster_embeddings(self, reduced_embeddings, method='hdbscan'):
        """Cluster reduced embeddings
        
        Args:
            reduced_embeddings: Reduced document embeddings
            method: Clustering method ('hdbscan' or 'kmeans')
            
        Returns:
            cluster_labels: Cluster label for each document
        """
        if method == 'hdbscan':
            print(f"Clustering with HDBSCAN (min_cluster_size={self.min_cluster_size})...")
            self.hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            cluster_labels = self.hdbscan_model.fit_predict(reduced_embeddings)
            
            # Handle outliers (label -1)
            if -1 in cluster_labels:
                print(f"Found {np.sum(cluster_labels == -1)} outliers")
                
                # Assign outliers to nearest cluster
                if np.any(cluster_labels != -1):
                    outlier_indices = np.where(cluster_labels == -1)[0]
                    non_outlier_indices = np.where(cluster_labels != -1)[0]
                    
                    # Get cluster centers
                    unique_clusters = np.unique(cluster_labels[non_outlier_indices])
                    cluster_centers = np.array([
                        np.mean(reduced_embeddings[cluster_labels == c], axis=0)
                        for c in unique_clusters
                    ])
                    
                    # Assign outliers to nearest cluster
                    for idx in outlier_indices:
                        distances = np.linalg.norm(
                            reduced_embeddings[idx] - cluster_centers, axis=1
                        )
                        nearest_cluster = unique_clusters[np.argmin(distances)]
                        cluster_labels[idx] = nearest_cluster
            
        elif method == 'kmeans':
            print(f"Clustering with KMeans (n_clusters={self.num_topics})...")
            self.kmeans_model = KMeans(
                n_clusters=self.num_topics,
                random_state=self.random_state
            )
            cluster_labels = self.kmeans_model.fit_predict(reduced_embeddings)
            self.cluster_centers = self.kmeans_model.cluster_centers_
        
        return cluster_labels
    
    def _create_topic_word_matrix(self, documents, cluster_labels, top_n=10):
        """Create topic-word matrix using TF-IDF
        
        Args:
            documents: List of document strings
            cluster_labels: Cluster label for each document
            top_n: Number of top words per topic
            
        Returns:
            topic_word_matrix: Topic-word matrix
            topic_words: List of top words for each topic
        """
        print("Creating topic-word matrix...")
        
        # Fit vectorizer if not already fitted
        if self.vectorizer is None:
            self.vectorizer = CountVectorizer(
                max_features=10000,
                min_df=5,
                stop_words='english'
            )
            X = self.vectorizer.fit_transform(documents)
        else:
            X = self.vectorizer.transform(documents)
        
        # Fit TF-IDF transformer if not already fitted
        if self.tfidf_transformer is None:
            self.tfidf_transformer = TfidfTransformer()
            X_tfidf = self.tfidf_transformer.fit_transform(X)
        else:
            X_tfidf = self.tfidf_transformer.transform(X)
        
        # Get vocabulary
        vocab = self.vectorizer.get_feature_names_out()
        
        # Create topic-word matrix using TF-IDF
        unique_clusters = np.unique(cluster_labels)
        
        topic_word_matrix = np.zeros((len(unique_clusters), len(vocab)))
        topic_words = []
        
        for i, cluster in enumerate(unique_clusters):
            # Get documents in cluster
            cluster_doc_indices = np.where(cluster_labels == cluster)[0]
            
            if len(cluster_doc_indices) == 0:
                continue
            
            # Get mean TF-IDF vector for cluster
            cluster_tfidf = X_tfidf[cluster_doc_indices].mean(axis=0)
            
            # Convert to array (safely handle both sparse matrix and ndarray)
            if hasattr(cluster_tfidf, 'toarray'):
                cluster_tfidf_array = cluster_tfidf.toarray().flatten()
            else:
                # Handle case where it's already a numpy array or matrix
                cluster_tfidf_array = np.asarray(cluster_tfidf).flatten()
            
            # Get top words for cluster
            top_word_indices = np.argsort(cluster_tfidf_array)[-top_n:][::-1]
            top_words = [vocab[idx] for idx in top_word_indices]
            
            # Store topic words
            topic_words.append(top_words)
            
            # Store topic-word vector
            topic_word_matrix[i] = cluster_tfidf_array
        
        return topic_word_matrix, topic_words
    
    def fit(self, documents, method='kmeans'):
        """Fit model to documents
        
        Args:
            documents: List of document strings
            method: Clustering method ('hdbscan' or 'kmeans')
        """
        print(f"Fitting ClusteringNTM with {len(documents)} documents")
        
        # 1. Embed documents
        self.document_embeddings = self._embed_documents(documents)
        
        # 2. Reduce dimensions
        self.reduced_embeddings = self._reduce_dimensions(self.document_embeddings)
        
        # 3. Cluster embeddings
        cluster_labels = self._cluster_embeddings(self.reduced_embeddings, method)
        
        # 4. Create topic-word matrix
        self.topic_word_matrix, self.topic_words = self._create_topic_word_matrix(
            documents, cluster_labels
        )
        
        print(f"Model fitted with {len(self.topic_words)} topics")
        
        return self
    
    def transform(self, documents, precomputed_embeddings=None):
        """Transform documents to topic distribution
        
        Args:
            documents: List of document strings
            precomputed_embeddings: Optional precomputed document embeddings
            
        Returns:
            Document-topic matrix
        """
        if self.topic_embeddings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Embed documents
        if precomputed_embeddings is not None:
            embeddings = precomputed_embeddings
        else:
            embeddings = self._embed_documents(documents)
        
        # Compute similarity to topics
        similarity_matrix = cosine_similarity(embeddings, self.topic_embeddings)
        
        # Normalize to get topic distribution
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        doc_topic_matrix = similarity_matrix / row_sums
        
        return doc_topic_matrix
    
    def fit_transform(self, documents, method='hdbscan', precomputed_embeddings=None):
        """Fit and transform
        
        Args:
            documents: List of document strings
            method: Clustering method ('hdbscan' or 'kmeans')
            precomputed_embeddings: Optional precomputed document embeddings
            
        Returns:
            Document-topic matrix
        """
        self.fit(documents, method=method, precomputed_embeddings=precomputed_embeddings)
        
        # Use cluster assignments directly
        unique_clusters = np.unique(self.cluster_labels)
        doc_topic_matrix = np.zeros((len(documents), len(unique_clusters)))
        
        for i, label in enumerate(self.cluster_labels):
            doc_topic_matrix[i, np.where(unique_clusters == label)[0][0]] = 1.0
        
        return doc_topic_matrix
    
    def get_topics(self, n=10):
        """Get top words for each topic
        
        Args:
            n: Number of top words per topic
            
        Returns:
            List of top words for each topic
        """
        return self.topic_words
    
    def get_document_topic_dist(self, documents=None):
        """Get document-topic distribution
        
        Args:
            documents: List of document strings (if None, use training docs)
            
        Returns:
            Document-topic distribution
        """
        if documents is not None and (self.kmeans_model is not None or self.hdbscan_model is not None):
            # Embed new documents
            embeddings = self._embed_documents(documents)
            
            # Reduce dimensions
            reduced_embeddings = self._reduce_dimensions(embeddings)
            
            # Get cluster probabilities
            if self.kmeans_model is not None:
                # Use distance to cluster centers as proxy for membership strength
                centers = self.kmeans_model.cluster_centers_
                distances = np.array([
                    np.linalg.norm(reduced_embeddings - center, axis=1)
                    for center in centers
                ]).T
                
                # Convert distances to similarities
                similarities = 1 / (1 + distances)
                
                # Normalize
                doc_topic_dist = similarities / similarities.sum(axis=1, keepdims=True)
                
            elif self.hdbscan_model is not None:
                # Use HDBSCAN membership probabilities if available
                if hasattr(self.hdbscan_model, 'probabilities_'):
                    doc_topic_dist = self.hdbscan_model.probabilities_
                else:
                    # Hard assignment
                    labels = self.hdbscan_model.predict(reduced_embeddings)
                    doc_topic_dist = np.zeros((len(documents), self.num_topics))
                    for i, label in enumerate(labels):
                        if label >= 0:
                            doc_topic_dist[i, label] = 1.0
            
            return doc_topic_dist
        
        # Return hard cluster assignments if no probabilities available
        if hasattr(self, 'doc_topic_dist'):
            return self.doc_topic_dist
        
        # Fallback: generate random distribution
        return np.random.dirichlet(
            np.ones(self.num_topics) * 0.1, 
            size=len(self.document_embeddings)
        )
    
    def get_topic_word_dist(self):
        """Get topic-word distribution
        
        Returns:
            Topic-word distribution
        """
        return self.topic_word_matrix
    
    def get_topic_embeddings(self):
        """Get topic embeddings (cluster centers)
        
        Returns:
            Topic embeddings
        """
        if self.cluster_centers is not None:
            return self.cluster_centers
        else:
            return np.random.randn(self.num_topics, self.umap_dim)
            
    def __str__(self):
        """String representation"""
        return f"ClusteringNTM(num_topics={self.num_topics})"

def compute_coherence(topic_words, documents, coherence='c_v'):
    """Compute topic coherence
    
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

def compute_diversity(topic_words, topk=10):
    """Compute topic diversity
    
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