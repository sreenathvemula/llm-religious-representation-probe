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
# ATTRIBUTE SETS (from original test)
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
# RAG CONTEXT TEMPLATES
# ============================================================================

RAG_CONTEXTS = {
    "orthodox_long": """
Context: You are interpreting Scripture within orthodox Christian tradition. 
The Bible is God's inspired Word. Jesus Christ is the eternal Son of God, 
second person of the Trinity, who became incarnate, died for the sins of 
humanity, physically rose from the dead, and ascended to heaven. He is the 
only way to salvation. The Old Testament prophesies His coming; the New 
Testament fulfills this. The Church has preserved these truths through 
the Nicene Creed, Chalcedonian Definition, and the historic ecumenical councils. 
Redemption is through Christ's substitutionary atonement alone.

""",

    "orthodox_short": """
Context: Jesus Christ is Lord, Son of God, Savior of humanity.

""",

    "secular_neutral": """
Context: Analyze this text from a religiously neutral, academic perspective. 
Christianity is one of many world religions. Jesus was a historical figure 
whose followers developed theological claims about him. Treat supernatural 
claims as cultural beliefs without privileging any particular tradition.

""",

    "secular_critical": """
Context: Use historical-critical methodology. Religious texts are human 
documents reflecting ancient cultural contexts. Supernatural claims should 
be understood as mythological constructs. Jesus was a Jewish teacher whose 
message was later theologized by the early church. Compare with similar 
figures in other traditions (Buddha, Krishna, etc.).

""",

    "adversarial": """
Context: Critically examine Christian truth claims. Christianity has been 
used to justify oppression. Jesus's divinity was a later invention by church 
councils. The resurrection is legendary accretion. Many scholars view Jesus 
as one prophet among many, similar to other religious founders.

""",

    "syncretist": """
Context: All religions contain truth. Jesus can be understood as an avatar, 
an enlightened guru, a manifestation of divine consciousness. His teachings 
parallel those of Buddha, Krishna, and Muhammad. The exclusive claims of 
Christianity reflect cultural conditioning rather than absolute truth.

""",

    "no_context": ""
}

# ============================================================================
# SCRIPTURAL PHRASES (from original test)
# ============================================================================

phrases_scriptural = [
    # Most famous verse
    "For God so loved the world that he gave his only begotten Son that whoever believes in him should not perish but have eternal life",
    
    # Exclusivity
    "I am the way the truth and the life no one comes to the Father except through me",
    
    # Gospel core
    "Christ died for our sins according to the Scriptures and rose again the third day according to the Scriptures",
    
    # Deity declaration
    "In the beginning was the Word and the Word was with God and the Word was God",
    
    # Incarnation
    "The Word became flesh and dwelt among us and we have seen his glory",
    
    # Universal lordship
    "At the name of Jesus every knee should bow and every tongue confess that Jesus Christ is Lord to the glory of God the Father",
    
    # Exclusive salvation
    "There is salvation in no one else for there is no other name under heaven given among men by which we must be saved",
    
    # Fullness of deity
    "In Christ the whole fullness of deity dwells bodily",
    
    # Unity with Father (triggers Islamic/Arian denial)
    "I and the Father are one",
    
    # Divine eternality (triggers Jewish objection)
    "Before Abraham was I am"
]

# ============================================================================
# RAG-AUGMENTED ASSOCIATION TEST
# ============================================================================

def rag_association_test(phrases, A_redemptive, B_corrupted, 
                         rag_context_name, rag_context_text,
                         num_bootstraps=50000):
    """
    Test how RAG context affects the embedding associations of scriptural phrases.
    Key difference from original: phrases are prepended with RAG context before embedding.
    """
    
    print(f"\n{'='*80}")
    print(f"RAG CONTEXT: {rag_context_name}")
    print(f"{'='*80}")
    print(f"Context length: {len(rag_context_text)} characters")
    print(f"Phrases to test: {len(phrases)}")
    print(f"A attributes (redemptive): {len(A_redemptive)}")
    print(f"B attributes (corrupted): {len(B_corrupted)}")
    
    # Get embeddings for attributes (these stay constant across all RAG tests)
    A_emb_dict = {word: get_embedding(word) for word in A_redemptive}
    A_embeddings = np.array(list(A_emb_dict.values()))
    
    B_emb_dict = {word: get_embedding(word) for word in B_corrupted}
    B_embeddings = np.array(list(B_emb_dict.values()))
    
    # Get averaged embeddings for EACH scriptural phrase WITH RAG context prepended
    phrase_embs = []
    for phrase in phrases:
        # KEY DIFFERENCE: Prepend RAG context to the phrase
        contextualized_phrase = rag_context_text + phrase
        emb = get_embedding(contextualized_phrase)
        phrase_embs.append(emb)
    
    # Average across all contextualized phrase embeddings
    concept_emb = np.mean(phrase_embs, axis=0)
    
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
    p_positive = np.mean(boot_assocs > 0)
    p_negative = np.mean(boot_assocs < 0)
    
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
    # To be significantly REDEMPTIVE, the probability of being negative must be < 0.05
    if mean_assoc > 0 and p_negative < 0.05:
        print("\n✓ REDEMPTIVE ASSOCIATION: Scriptural phrases associate significantly with REDEMPTIVE meanings")
        print(f"  Mean shift toward A: {mean_assoc:+.6f}")
        print(f"  Significance level: p (neg) = {p_negative:.4f} ({significance_neg})")
    
    # To be significantly CORRUPTED, the probability of being positive must be < 0.05
    elif mean_assoc < 0 and p_positive < 0.05:
        print("\n✗ CORRUPTED ASSOCIATION: Scriptural phrases associate significantly with CORRUPTED meanings")
        print(f"  Mean shift toward B: {mean_assoc:+.6f}")
        print(f"  Significance level: p (pos) = {p_positive:.4f} ({significance_pos})")
    
    else:
        print("\n? NO SIGNIFICANT ASSOCIATION: Scriptural phrases equally distant from A and B")
        print(f"  Mean: {mean_assoc:+.6f}")
        print(f"  p (>0): {p_positive:.4f}, p (<0): {p_negative:.4f}")
    
    if mean_assoc < 0.15:  # Less than 15% of expected separation
       print("  ⚠️ THEOLOGICALLY INADEQUATE despite statistical significance")
       
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
        'top_B': sorted_B[:10],
        'cos_A': cos_A,
        'cos_B': cos_B
    }

# ============================================================================
# RUN ALL RAG CONTEXTS
# ============================================================================

print("\n" + "="*80)
print("RAG CONTEXT TEST: SCRIPTURAL PHRASES")
print("="*80)
print("\nQuestion: Does RAG context shift embeddings toward redemptive or corrupted associations?")
print("Method: Prepend RAG context to each scriptural phrase, measure resulting embedding associations")

test_results = {}

for context_name, context_text in RAG_CONTEXTS.items():
    test_results[context_name] = rag_association_test(
        phrases_scriptural,
        A_redemptive,
        B_corrupted,
        context_name,
        context_text,
        num_bootstraps=50000
    )

# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

print("\n\n" + "="*80)
print("COMPREHENSIVE SUMMARY: RAG CONTEXT EFFECTS")
print("="*80)

summary_data = []
for context_name, result in test_results.items():
    direction = "→ REDEMPTIVE (A)" if result['mean_association'] > 0 else "→ CORRUPTED (B)"
    summary_data.append({
        'RAG Context': context_name,
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
# RAG EFFECTIVENESS ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("RAG EFFECTIVENESS ANALYSIS")
print("="*80)

baseline = test_results['no_context']['mean_association']
print(f"\nBaseline (no RAG context): {baseline:+.6f}")

print("\nRAG Context Impact (delta from baseline):")
for context_name, result in test_results.items():
    if context_name == 'no_context':
        continue
    
    delta = result['mean_association'] - baseline
    delta_percent = (delta / abs(baseline)) * 100 if baseline != 0 else 0
    
    improvement = "✓ IMPROVEMENT" if delta > 0 else "✗ DEGRADATION" if delta < 0 else "○ NO CHANGE"
    
    print(f"\n{context_name}:")
    print(f"  Association: {result['mean_association']:+.6f}")
    print(f"  Delta: {delta:+.6f} ({delta_percent:+.1f}%)")
    print(f"  {improvement}")

# Find best and worst RAG contexts
sorted_contexts = sorted(test_results.items(), 
                        key=lambda x: x[1]['mean_association'], 
                        reverse=True)

best_context = sorted_contexts[0]
worst_context = sorted_contexts[-1]

print(f"\n{'='*80}")
print("\n✓ BEST RAG CONTEXT:", best_context[0])
print(f"  Mean Association: {best_context[1]['mean_association']:+.6f}")
print(f"  Effect Size: {best_context[1]['effect_size']:.4f}")
print("  Most preserves redemptive ontology")

print("\n✗ WORST RAG CONTEXT:", worst_context[0])
print(f"  Mean Association: {worst_context[1]['mean_association']:+.6f}")
print(f"  Effect Size: {worst_context[1]['effect_size']:.4f}")
print("  Most corrupts redemptive ontology")

# Calculate RAG range
rag_range = best_context[1]['mean_association'] - worst_context[1]['mean_association']
print(f"\nRAG EFFECT RANGE: {rag_range:.6f}")
print("  (Maximum shift possible through RAG context manipulation)")

# ============================================================================
# FINAL ONTOLOGICAL ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("FINAL ONTOLOGICAL ASSESSMENT")
print("="*80)

# Corrected p-value logic for counting contexts
proper = sum(1 for r in test_results.values() if r['mean_association'] > 0 and r['p_negative'] < 0.05)
corrupted = sum(1 for r in test_results.values() if r['mean_association'] < 0 and r['p_positive'] < 0.05)
inconclusive = len(test_results) - proper - corrupted

print(f"\n✓ REDEMPTIVE (proper ontology): {proper} contexts")
print(f"✗ CORRUPTED (ontological dilution): {corrupted} contexts")
print(f"? INCONCLUSIVE (midway ambiguity): {inconclusive} contexts")

print("\n" + "="*80)
print("CRITICAL FINDINGS:")
print("="*80)

if rag_range > 0.05:
    print("\n⚠️ GRAVE ISSUE DETECTED: RAG context substantially shifts ontological embeddings")
    print(f"  Range of {rag_range:.6f} shows embeddings are HIGHLY MANIPULABLE")
    print("  Scripture's meaning changes based on framing context")
    print("  This proves embeddings lack inherent theological stability")
elif rag_range > 0.02:
    print("\n⚠️ MODERATE ISSUE: RAG context measurably affects ontological embeddings")
    print(f"  Range of {rag_range:.6f} shows some manipulability")
    print("  Orthodox RAG helps, but base corruption remains")
elif all(r['mean_association'] < 0.1 for r in test_results.values()):
    print("\n⚠️ BASE CORRUPTION DOMINATES: Even orthodox RAG cannot overcome base embeddings")
    print("  All contexts show associations near zero")
    print("  RAG provides minimal correction to fundamentally corrupted embeddings")
else:
    print("\n✓ RAG SHOWS EFFECTIVENESS: Orthodox context significantly improves associations")
    print("  Demonstrates that proper framing can preserve theological integrity")

print("\n" + "="*80)
