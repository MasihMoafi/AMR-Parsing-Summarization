# models/rl.py
import torch
import torch.nn.functional as F

def compute_reward(generated_ids, reference_ids, vocab):
    """
    Computes a simple ROUGE-like reward based on token overlap.
    """
    # Convert IDs to text, ignoring special tokens
    gen_tokens = {vocab.idx2word.get(idx.item(), "") for idx in generated_ids if idx.item() > 3}
    ref_tokens = {vocab.idx2word.get(idx.item(), "") for idx in reference_ids if idx.item() > 3}
    
    overlap = len(gen_tokens & ref_tokens)
    return overlap / (len(ref_tokens) + 1e-8)

def self_critical_loss(model, src_graphs, ref_ids, vocab):
    """
    Calculates the self-critical policy gradient loss.
    """
    model.eval() # Use eval mode for sampling and greedy decoding
    
    # Greedy search for baseline reward
    with torch.no_grad():
        greedy_summaries = model.generate(src_graphs, vocab)
        
    # Sample from distribution for action
    model.train() # Back to train mode for gradient calculation
    sampled_summaries, log_probs = model.sample(src_graphs, vocab)
    
    # Calculate rewards
    r_greedy = torch.tensor([compute_reward(g, r, vocab) for g, r in zip(greedy_summaries, ref_ids)]).to(log_probs.device)
    r_sampled = torch.tensor([compute_reward(s, r, vocab) for s, r in zip(sampled_summaries, ref_ids)]).to(log_probs.device)
    
    # Calculate loss
    reward_baseline = r_sampled - r_greedy
    loss = - (reward_baseline * log_probs).mean()
    
    return loss
