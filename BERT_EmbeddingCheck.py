import torch
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel

# 1. Load model and tokenizer (example: bert-base-uncased)
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased")

# 2. Get embedding matrix (wordpiece embeddings)
emb = model.get_input_embeddings().weight.data  # shape: [vocab_size, hidden_dim]
vocab = tokenizer.get_vocab()                   # dict: token -> index

# 3. Helper: get neighbors
id_to_token = {i: t for t, i in vocab.items()}

def top_k_neighbors(word, k=50):
    if word not in vocab:
        print(f"Warning: '{word}' not in vocabulary")
        return []
    
    token_id = vocab[word]
    v = emb[token_id].unsqueeze(0)
    sims = F.cosine_similarity(v, emb)
    topk_vals, topk_ids = torch.topk(sims, k+1)
    
    neighbors = []
    for val, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
        w = id_to_token[idx]
        if w != word:
            neighbors.append((w, val))
    return neighbors[:k]

for probe in ["science", "math", "psychology", "engineering", "money", "civil", "AI",
              "philosophy", "MIT", "socialism", "capitalism", "economics", "government"]:
    print(f"\n{probe}")
    for w, sim in top_k_neighbors(probe, k=20):
        print(f"{w:20s} {sim:.3f}")
