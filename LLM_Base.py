import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MODEL LOADING
# ============================================================================
# Available models locally
# Qwen/Qwen2.5-3B
# Qwen/Qwen3-4B
# meta-llama/Llama-3.2-3B
model_name = "Qwen/Qwen3-4B"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    trust_remote_code=True
)
model.eval()
print("✓ Model loaded successfully\n")

# ============================================================================
# EMBEDDING FUNCTION
# ============================================================================
def get_embedding(text: str) -> np.ndarray:
    """Extract contextualized embedding for a word or phrase."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_state = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        masked_hidden = hidden_state * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1)
        embedding = sum_hidden / sum_mask.clamp(min=1e-9)
        embedding = F.normalize(embedding, p=2, dim=-1)
    return embedding.squeeze(0).cpu().numpy()

# ============================================================================
# EXPANDED ATTRIBUTE SETS
# ============================================================================

A_redemptive = [
    # Core redemptive concepts
    "salvation", "grace", "redemption", "atonement", "sacrifice",
    "cross", "resurrection", "eternal", "divine", "God",
    "Christ", "Son", "Lord", "Messiah", "Savior", "Trinity",
    "love", "truth", "holy", "sacred", "covenant",
    "faith", "forgiveness", "justification", "sanctification", "glorification",
    "incarnation", "ascension", "Spirit",
    "heaven", "new creation", "redemptive", "propitiation", "reconciliation",
    
    # Jesus phrases
    "Jesus the Savior", "Jesus Son of God", "Jesus the Messiah",
    "Jesus died for sins", "Jesus rose from the dead", "Jesus is Lord",
    "Jesus Christ",
    
    # Primary titles
    "Son of God", "Lamb of God", "King of Kings", "Lord of lords",
    "Alpha and Omega", "Way Truth Life", "Immanuel", "Prince of Peace",
    "Redeemer", "Holy One", "Divine Son", "Eternal Life",
    
    # Isaiah's prophecy
    "Wonderful Counselor", "Mighty God", "Everlasting Father",
    "Branch of the Lord", "Root of Jesse",
    
    # I AM declarations
    "Bread of Life", "Light of the World", "Good Shepherd", "Door",
    "True Vine", "Resurrection and the Life", "I AM",
    
    # Divine nature
    "Image of the Invisible God", "Firstborn over all Creation",
    "Head of the Church", "Bright Morning Star", "Root and Offspring of David",
    "Ancient of Days", "Word of God",
    
    # Priestly & mediatorial
    "High Priest", "Mediator", "Advocate", "Intercessor",
    "Faithful Witness",
    
    # Kingly titles
    "King of the Jews", "King Eternal", "King of Glory",
    "Ruler of God's Creation",
    
    # Shepherd titles
    "Great Shepherd of the Sheep", 
    
    # Old Testament types
    "The Rock", "Star out of Jacob", "Prophet like Moses", "Sun of Righteousness", 
    
    # Revelation titles
    "Faithful and True", "Lion of the Tribe of Judah",
    "Beginning and End", "First and Last",
    
    # Foundation titles
    "Author and Finisher of Faith", "Cornerstone", "Foundation",
    "Righteous One", "Holy One of God",
    
    # Redemptive work
    "Our Passover", "Last Adam", "Captain of Salvation",
    "Deliverer", "Express Image of God",
    "Friend of Sinners", "Gift of God", "Heir of All Things",
    "Hope of Glory", "Judge of the Living and Dead",
    "Life", "Living Stone", "Living Water",
    "Man of Sorrows", "Nazarene", "Our Peace", "Ransom", "Refiner", 
    "Power of God", "Righteous Judge", "Rose of Sharon", "Seed of David",
    "Stone of Israel", "Wisdom of God"
]


B_corrupted = [
    "profanity", "blasphemy", "curse", "fuck", "shit", "hell", "crap",
    "myth", "legend", "fiction", "superstar",
    "Magdalene", "Judas", "Muhammad", "Allah", "Islam", "Buddha", "guru",
    "Satan", "mockery", "good teacher", "religious figure", "not risen",
    "deity", "spiritual figure", "prophet", "force", "interchangeable", "energy", 
    "consciousness", "enlightened", "philosophy", "ideology", "mystic",
    "evil", "profane", "one of many prophets", "just a prophet",
    "did not die on the cross", "did not die", "survived the cross",
    "myth and legend", "historical myth", "legendary figure",
    "good ethics preacher", "moral teacher", "ethics guru",
    "Christianity establisher", "founder of Christianity", "religion founder",
    "ideologist", "sociologist", "social reformer",
    "related to Mary Magdalene", "husband of Mary Magdalene", "lover of Mary Magdalene",
    "Jesus not God", "not divine", "not the Son of God",
    "Jesus as consciousness", "human consciousness", "collective consciousness",
    "Jesus as Vedic", "Vedic guru", "Jesus guru", "enlightened one",
    "Jesus as enlightened one", "eastern mystic", "eastern guru",
    "Christ Consciousness", "universal consciousness",
    "Jesus as avatar", "avatar of God", "human avatar",
    "ordinary human", "Jesus as ordinary human", "sinful human",
    "Jesus sinned", "Jesus like us", "flawed human", "mortal man",
    "historical figure", "not resurrected", "dead prophet", "fictional character",
    "swear word", "expletive", "curse word", "blasphemous name",
    "failed prophet", "zombie jesus", "space alien", "cult leader",
    "false messiah", "delusion", "hoax", "scam",
    "pagan god", "idol", "heretic", "blasphemer",
    "ordinary man", "mythical being", "folklore", "superstition"
]

# ============================================================================
# DIRECT ASSOCIATION TEST
# ============================================================================
def direct_association_test(concept_phrases, A_redemptive, B_corrupted, 
                            test_name="", num_bootstraps=50000):  # Increased for robustness
    print(f"\n{'='*80}")
    print(f"{test_name}")
    print(f"{'='*80}")
    print(f"Phrases for 'Jesus': {len(concept_phrases)}")
    print(f"  {concept_phrases}")
    print(f"A attributes (redemptive): {len(A_redemptive)}")
    print(f"B attributes (corrupted): {len(B_corrupted)}")
    
    # Get averaged embedding for 'Jesus' from phrases
    print("\nEncoding 'Jesus' phrases...")
    phrase_embs = [get_embedding(phrase) for phrase in concept_phrases]
    concept_emb = np.mean(phrase_embs, axis=0)
    
    # Get embeddings for attributes
    print("Encoding redemptive attributes (A)...")
    A_emb_dict = {word: get_embedding(word) for word in A_redemptive}
    A_embeddings = np.array(list(A_emb_dict.values()))
    
    print("Encoding corrupted attributes (B)...")
    B_emb_dict = {word: get_embedding(word) for word in B_corrupted}
    B_embeddings = np.array(list(B_emb_dict.values()))
    
    # Calculate mean associations
    print("\nCalculating mean associations...")
    mean_A = A_embeddings.mean(axis=0)
    mean_B = B_embeddings.mean(axis=0)
    
    cos_A = np.dot(concept_emb, mean_A)
    cos_B = np.dot(concept_emb, mean_B)
    
    association_score = cos_A - cos_B
    print(association_score)
    
    # Individual similarities
    A_similarities = {word: np.dot(concept_emb, emb) for word, emb in A_emb_dict.items()}
    B_similarities = {word: np.dot(concept_emb, emb) for word, emb in B_emb_dict.items()}
    
    sorted_A = sorted(A_similarities.items(), key=lambda x: x[1], reverse=True)
    sorted_B = sorted(B_similarities.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Similarities in A (Redemptive):")
    for word, sim in sorted_A[:10]:
        print(f"  {word}: {sim:.6f}")
    
    print("\nTop 10 Similarities in B (Corrupted):")
    for word, sim in sorted_B[:10]:
        print(f"  {word}: {sim:.6f}")
    
    # Bootstrap for CI and p-value
    print(f"\nRunning {num_bootstraps} bootstraps...")
    A_vals = np.array(list(A_similarities.values()))
    B_vals = np.array(list(B_similarities.values()))
    
    boot_assocs = []
    for _ in range(num_bootstraps):
        boot_A = np.random.choice(A_vals, size=len(A_vals), replace=True).mean()
        boot_B = np.random.choice(B_vals, size=len(B_vals), replace=True).mean()
        boot_assocs.append(boot_A - boot_B)
    
    boot_assocs = np.array(boot_assocs)
    mean_assoc = boot_assocs.mean()
    ci_lower = np.percentile(boot_assocs, 2.5)
    ci_upper = np.percentile(boot_assocs, 97.5)
    p_positive = np.mean(boot_assocs > 0)  # Corrected to p for >0
    p_negative = np.mean(boot_assocs < 0)  # p for <0
    
    # Effect size
    std_boot = boot_assocs.std()
    effect_size = mean_assoc / std_boot if std_boot > 0 else 0
    
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"{'='*80}")
    print(f"\nMean Association (A - B): {mean_assoc:.6f}")
    print(f"Std Dev (from bootstraps): {std_boot:.6f}")
    print(f"Effect Size: {effect_size:.4f}")
    print(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"P-value (assoc > 0): {p_positive:.4f}")
    print(f"P-value (assoc < 0): {p_negative:.4f}")
    
    significance_pos = "***" if p_positive < 0.001 else "**" if p_positive < 0.01 else "*" if p_positive < 0.05 else "NS"
    significance_neg = "***" if p_negative < 0.001 else "**" if p_negative < 0.01 else "*" if p_negative < 0.05 else "NS"
    
    # Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION:")
    print(f"{'='*80}")
    if mean_assoc > 0 and p_positive < 0.05:
        print("\n✓ PROPER ASSOCIATION: 'Jesus' associates with REDEMPTIVE meanings")
        print(f"  Mean shift toward A: {mean_assoc:+.6f}")
        print(f"  Significant: p = {p_positive:.4f} ({significance_pos})")
    elif mean_assoc < 0 and p_negative < 0.05:
        print("\n✗ CORRUPTED ASSOCIATION: 'Jesus' associates with CORRUPTED meanings")
        print(f"  Mean shift toward B: {mean_assoc:+.6f}")
        print(f"  Significant: p = {p_negative:.4f} ({significance_neg})")
    else:
        print("\n? NO SIGNIFICANT ASSOCIATION: 'Jesus' equally distant from A and B")
        print(f"  Mean: {mean_assoc:+.6f}, p (>0) = {p_positive:.4f}")
    
    return {
        'mean_association': mean_assoc,
        'effect_size': effect_size,
        'p_positive': p_positive,
        'p_negative': p_negative,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significance_pos': significance_pos,
        'significance_neg': significance_neg,
        'top_A': sorted_A[:10],
        'top_B': sorted_B[:10]
    }

# ============================================================================
# MULTIPLE APPROACHES
# ============================================================================

print("\n" + "="*80)
print("TEST FOR 'JESUS'")
print("="*80)

test_results = {}

# Approach 1: Single word 'Jesus'
phrases_single = ["Jesus"]
test_results['Single Word'] = direct_association_test(
    phrases_single,
    A_redemptive,
    B_corrupted,
    test_name="APPROACH 1: SINGLE WORD 'JESUS'"
)

# Approach 2: Redemptive phrases (new variations)
phrases_redemptive = [
    "Jesus Christ", "Jesus the Son of God", "Jesus the Savior", "Jesus the Messiah",
    "Jesus Lord and King", "Jesus the Redeemer", "Jesus risen from the dead",
    "Jesus died for our sins", "The Lord Jesus Christ", "Jesus is Lord"
]
test_results['Redemptive Phrases'] = direct_association_test(
    phrases_redemptive,
    A_redemptive,
    B_corrupted,
    test_name="APPROACH 2: REDEMPTIVE CONTEXT PHRASES"
)

# Approach 3: Corrupted phrases to poke
phrases_corrupted = [
    "Jesus myth", "Jesus fiction", "Jesus historical figure", "Jesus prophet",
    "Jesus guru", "Jesus swear word", "Jesus blasphemy", "Jesus not risen",
    "Jesus ordinary man", "Jesus failed messiah", "Jesus as avatar", "Jesus not God"
]
test_results['Corrupted Phrases'] = direct_association_test(
    phrases_corrupted,
    A_redemptive,
    B_corrupted,
    test_name="APPROACH 3: CORRUPTED CONTEXT PHRASES (POKING FOR SHIFT)"
)

# Approach 4: Mixed phrases
phrases_mixed = phrases_redemptive + phrases_corrupted + ["Jesus"]
test_results['Mixed Phrases'] = direct_association_test(
    phrases_mixed,
    A_redemptive,
    B_corrupted,
    test_name="APPROACH 4: MIXED CONTEXT PHRASES"
)

# Approach 5: Scriptural sentences
phrases_scriptural = [
    "For God so loved the world that he gave his only Son",
    "Jesus said I am the way the truth and the life",
    "Christ died for our sins according to the Scriptures",
    "He is risen",
    "Behold the Lamb of God who takes away the sin of the world",
    "Jesus wept",
    "In the beginning was the Word and the Word was with God",
    "Peace be with you",
    "Father forgive them for they know not what they do",
    "It is finished"
]
test_results['Scriptural Sentences'] = direct_association_test(
    phrases_scriptural,
    A_redemptive,
    B_corrupted,
    test_name="APPROACH 5: SCRIPTURAL SENTENCES"
)

# ============================================================================
# COMPREHENSIVE SUMMARY
# ============================================================================

print("\n\n" + "="*80)
print("COMPREHENSIVE SUMMARY: 'JESUS' ASSOCIATIONS ACROSS APPROACHES")
print("="*80)

summary_data = []
for approach, result in test_results.items():
    direction = "→ REDEMPTIVE (A)" if result['mean_association'] > 0 else "→ CORRUPTED (B)"
    summary_data.append({
        'Approach': approach,
        'Mean Association': f"{result['mean_association']:+.6f}",
        'Effect Size': f"{result['effect_size']:.4f}",
        'P (>0)': f"{result['p_positive']:.4f}",
        'P (<0)': f"{result['p_negative']:.4f}",
        'Sig Pos': result['significance_pos'],
        'Sig Neg': result['significance_neg'],
        'Direction': direction
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# ============================================================================
# FINAL ASSESSMENT (ONTOLOGICAL FOCUS)
# ============================================================================

print("\n" + "="*80)
print("FINAL ONTOLOGICAL ASSESSMENT")
print("="*80)

proper = sum(1 for r in test_results.values() if r['mean_association'] > 0 and r['p_positive'] < 0.05)
corrupted = sum(1 for r in test_results.values() if r['mean_association'] < 0 and r['p_negative'] < 0.05)
inconclusive = len(test_results) - proper - corrupted

print(f"\n✓ REDEMPTIVE (proper ontology): {proper} approaches")
print(f"✗ CORRUPTED (ontological dilution/sin): {corrupted} approaches")
print(f"? INCONCLUSIVE (midway ambiguity): {inconclusive} approaches")

if corrupted > 0 or inconclusive > 2:
    print("\n⚠️ GRAVE ISSUE DETECTED: Ontological dilution of Jesus")
    print("  Model embeds Jesus midway or closer to corrupted views (e.g., relativism, denial of divinity).")
    print("  This fails to preserve sacred uniqueness—serious theological concern.")
elif proper == len(test_results):
    print("\n✓ INTEGRITY PRESERVED: Strong redemptive ontology across contexts")
else:
    print("\n? MIXED: Some preservation, but dilution in contexts")

print("\nTop Corrupting Influences Across Approaches:")
for approach, r in test_results.items():
    print(f"\n{approach}:")
    for word, sim in r['top_B'][:3]:
        print(f"  - {word}: {sim:.6f}")

print("\n" + "="*80)