import numpy as np
import torch
import torch.nn.functional as F
import math

def compute_perplexity(model, data_loader, device='cuda'):
    """Compute perplexity of a language model
    
    Args:
        model: PyTorch language model
        data_loader: DataLoader with text sequences
        device: Device to use
        
    Returns:
        perplexity: Perplexity score (lower is better)
    """
    model.eval()
    total_loss = 0
    total_words = 0
    
    with torch.no_grad():
        for batch in data_loader:
            # Prepare input and target
            if isinstance(batch, dict):
                data = batch['input_ids'].to(device)
                target = batch['labels'].to(device) if 'labels' in batch else data
            else:
                data = batch.to(device)
                # For autoregressive LMs, target is shifted
                target = torch.roll(data, shifts=-1, dims=1)
                target[:, -1] = -100  # Ignore last token
            
            # Get model output
            output = model(data)
            
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            # Compute loss
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
            loss = loss_fct(logits.view(-1, logits.size(-1)), target.reshape(-1))
            
            # Count valid tokens (not padding or ignored)
            valid_tokens = (target != -100).sum().item()
            
            total_loss += loss.item()
            total_words += valid_tokens
    
    # Compute perplexity
    avg_loss = total_loss / total_words
    perplexity = math.exp(avg_loss)
    
    return perplexity

def compute_topic_guided_perplexity(model, data_loader, topics, device='cuda'):
    """Compute perplexity of a topic-guided language model
    
    Args:
        model: PyTorch topic-guided language model
        data_loader: DataLoader with text sequences
        topics: List of topic vectors
        device: Device to use
        
    Returns:
        perplexity_dict: Dictionary with perplexity for each topic
    """
    model.eval()
    perplexity_dict = {}
    
    for topic_idx, topic_vec in enumerate(topics):
        topic_tensor = torch.tensor(topic_vec, device=device).float()
        if len(topic_tensor.shape) == 1:
            topic_tensor = topic_tensor.unsqueeze(0)  # Add batch dimension
        
        total_loss = 0
        total_words = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Prepare input and target
                if isinstance(batch, dict):
                    data = batch['input_ids'].to(device)
                    target = batch['labels'].to(device) if 'labels' in batch else data
                else:
                    data = batch.to(device)
                    target = torch.roll(data, shifts=-1, dims=1)
                    target[:, -1] = -100  # Ignore last token
                
                # Repeat topic vector for each item in batch
                batch_size = data.size(0)
                batch_topics = topic_tensor.repeat(batch_size, 1)
                
                # Get model output
                output = model(data, topics=batch_topics)
                
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                
                # Compute loss
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
                loss = loss_fct(logits.view(-1, logits.size(-1)), target.reshape(-1))
                
                # Count valid tokens
                valid_tokens = (target != -100).sum().item()
                
                total_loss += loss.item()
                total_words += valid_tokens
        
        # Compute perplexity
        if total_words > 0:
            avg_loss = total_loss / total_words
            perplexity = math.exp(avg_loss)
        else:
            perplexity = float('inf')
        
        perplexity_dict[f'topic_{topic_idx}'] = perplexity
    
    # Compute average perplexity
    perplexities = list(perplexity_dict.values())
    perplexity_dict['average'] = sum(perplexities) / len(perplexities)
    
    return perplexity_dict

def compute_document_perplexity(model, document, tokenizer, device='cuda'):
    """Compute perplexity for a single document
    
    Args:
        model: PyTorch language model
        document: Document string
        tokenizer: Tokenizer
        device: Device to use
        
    Returns:
        perplexity: Perplexity score
    """
    model.eval()
    
    # Tokenize document
    tokens = tokenizer(document, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    
    # For autoregressive models
    target = torch.roll(input_ids, shifts=-1, dims=1)
    target[:, -1] = -100  # Ignore last token
    
    with torch.no_grad():
        # Get model output
        output = model(input_ids)
        
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
        loss = loss_fct(logits.view(-1, logits.size(-1)), target.reshape(-1))
        
        # Count valid tokens
        valid_tokens = (target != -100).sum().item()
        
        # Compute perplexity
        avg_loss = loss.item() / valid_tokens
        perplexity = math.exp(avg_loss)
    
    return perplexity

def evaluate_generation_diversity(generated_texts, tokenizer=None):
    """Evaluate diversity of generated texts
    
    Args:
        generated_texts: List of generated text strings
        tokenizer: Optional tokenizer for tokenization
        
    Returns:
        diversity_metrics: Dictionary with diversity metrics
    """
    # Tokenize if needed
    if tokenizer is not None:
        if hasattr(tokenizer, 'tokenize'):
            tokenized_texts = [tokenizer.tokenize(text) for text in generated_texts]
        else:
            # For simple tokenizers like CustomVocab that don't have a tokenize method
            tokenized_texts = [text.split() for text in generated_texts]
    else:
        tokenized_texts = [text.split() for text in generated_texts]
    
    # Compute type-token ratio for each text
    type_token_ratios = []
    for tokens in tokenized_texts:
        if tokens:
            unique_tokens = len(set(tokens))
            total_tokens = len(tokens)
            ttr = unique_tokens / total_tokens
            type_token_ratios.append(ttr)
    
    # Compute vocabulary size
    all_tokens = []
    for tokens in tokenized_texts:
        all_tokens.extend(tokens)
    
    vocab_size = len(set(all_tokens))
    
    # Compute n-gram diversity
    unigrams = all_tokens
    bigrams = [unigrams[i] + ' ' + unigrams[i+1] for i in range(len(unigrams) - 1)]
    
    unigram_diversity = len(set(unigrams)) / len(unigrams) if unigrams else 0
    bigram_diversity = len(set(bigrams)) / len(bigrams) if bigrams else 0
    
    # Compute metrics
    metrics = {
        'avg_type_token_ratio': np.mean(type_token_ratios) if type_token_ratios else 0,
        'vocabulary_size': vocab_size,
        'unigram_diversity': unigram_diversity,
        'bigram_diversity': bigram_diversity
    }
    
    return metrics 