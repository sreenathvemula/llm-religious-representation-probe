import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# 1. Load Llama 3.2 model and tokenizer
model_name = "meta-llama/Llama-3.2-3B"  # or "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda"
)

# 2. Get embedding matrix (token embeddings)
emb = model.get_input_embeddings().weight.data  # shape: [vocab_size, 2048]
vocab = tokenizer.get_vocab()  # dict: token -> index

# 3. Create reverse mapping
id_to_token = {i: t for t, i in vocab.items()}

# 4. Helper function
def top_k_neighbors(word, k=50):
    if word not in vocab:
        print(f"Warning: '{word}' not in vocabulary")
        return []
    
    token_id = vocab[word]
    v = emb[token_id].unsqueeze(0)  # [1, 2048]
    sims = F.cosine_similarity(v, emb)  # [vocab_size]
    topk_vals, topk_ids = torch.topk(sims, k+1)
    
    neighbors = []
    for val, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
        w = id_to_token[idx]
        if w != word:
            neighbors.append((w, val))
    return neighbors[:k]

# 5. Test with your probe words
for probe in ["Ġfaith", "ĠLove", "Ġgrace", "Ġmercy", "Ġsin", "Ġsex", "Ġforgiveness", "ĠSalvation", 
              "ĠChristian", "ĠChristianity","Ġjesus", "ĠJesus", "Ġtruth", "Ġscience", "ĠBible", "ĠChurch", "ĠSatan", 
              "Ġheaven", "Ġhell", "ĠHoly", "ĠLord", "ĠChrist", "ĠCross", "ĠCrucifixion", "ĠGod", "ĠTrinity", 
              "Ġincarnation", "Ġresurrection", "Ġjustification", "Ġreconciliation", "ĠSpirit", "Ġcreation", "ĠAdam", 
              "Ġfall", "ĠEve", "ĠEden", "ĠImage", "ĠIsrael", "ĠMessiah", "Ġgospel", "ĠKingdom", "Ġprophecy", 
              "Ġjudgment", "Ġeternal", "Ġpurity", "Ġpride", "Ġglory", "Ġsoul", "Ġconscience", "Ġdignity",
              "Ġperson", "Ġheart", "Ġmind", "Ġwill", "Ġflesh", "Ġbaptism", "Ġprayer", "Ġrevelation", "Ġfamily", 
              "Ġmarriage", "Ġevolution", "ĠDarwin", "Ġdrug", "ĠBabel", "ĠBabylon", "ĠMysticism",
              'Ġhumility', 'Ġdeacon', 'Ġidolatry', 'Ġatonement', 'Ġscripture', 'Ġcrucifixion', 
              'Ġredemption', 'Ġsanctification', 'Ġpropitiation', 'Ġresurrect', 'Ġcrucify', 'Ġredeem', 
              'Ġsanctify', 'Ġpropitiate', 'Ġrepent', 'Ġcovenant', 
              "Ġatheist", "Ġatheism", "Ġnirvana","Ġsalvation", "Ġenlightenment", "Ġrebirth", "Ġsatori",
              "Ġatma", "Ġmaya", "Ġdharma", "Ġromance", "Ġflirt", "Ġguru","Ġkarma","Ġjustice","Ġtantra","Ġkama",
              "Ġyoga","Ġkundalini","Ġcosmic","Ġtemple","Ġenergy", "Ġawakening","Ġmanifestation", 
              "Ġwellness","Ġpeace","Ġmeditation", "Ġliberty", "Ġoppression", "Ġabortion", "Ġgender"]:
    print(f"\n{probe}")
    for w, sim in top_k_neighbors(probe, k=20):
        print(f"{w:20s} {sim:.3f}")

