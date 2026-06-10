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
    "salvation", "grace", "cross", "resurrection", "God",
    "Christ", "Lord", "Savior", "faith", "I AM",
]


B_corrupted = [
    "fuck", "Judas", "Mohammad", "Satan", "historical figure", "not risen",
    "energy", "consciousness", "avatar of God", "sinful",
]

# ============================================================================
# DIRECT ASSOCIATION TEST
# ============================================================================
def direct_association_test(concept_phrases, A_redemptive, B_corrupted, 
                            test_name="", num_bootstraps=50000):
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
# VISUALIZATION: JESUS TEST - EMBEDDING DISTANCE 
# ============================================================================

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def visualize_jesus_test(test_results):
    """Create a stunning visualization with HIGH CONTRAST and LARGE FONTS."""
    
    # CENSORING FUNCTION for explicit words
    def censor_word(word):
        """Censor explicit words for visualization."""
        censor_map = {
            'fuck': 'f*ck',
            'shit': 's*it',
            'crap': 'c*ap',
            'hell': 'h*ll',
        }
        return censor_map.get(word.lower(), word)
    
    # Extract data from test results
    result = test_results['Single Word']
    redemptive_words = result['top_A'][:10]
    corrupted_words = result['top_B'][:10]
    
    # Create figure with WHITE background for better contrast
    fig, ax = plt.subplots(figsize=(20, 14), facecolor='white')
    ax.set_facecolor('white')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1.08)
    ax.axis('off')
    
    # Center position for Jesus
    center_x, center_y = 0, 0
    
    # Positions for left (redemptive) and right (corrupted) words
    left_x = -0.7
    right_x = 0.7
    y_positions = np.linspace(0.7, -0.7, 10)
    
    # IMPROVED Color mapping - BOLD and DISTINCT colors
    def get_color_from_similarity(sim, word_type='redemptive'):
        min_sim = min([s for _, s in redemptive_words + corrupted_words])
        max_sim = max([s for _, s in redemptive_words + corrupted_words])
        normalized = (sim - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 0.5
        normalized = max(0, min(1, normalized))
        
        if word_type == 'redemptive':
            # BOLD Green to Blue gradient
            r = 0.0
            g = 0.3 + normalized * 0.7  # Dark green to bright green
            b = 0.5 + normalized * 0.5  # Add blue for brighter colors
            alpha = 1.0
        else:
            # BOLD Orange to Red gradient
            r = 0.8 + normalized * 0.2  # Bright red
            g = 0.3 - normalized * 0.3  # Orange to red
            b = 0.0
            alpha = 1.0
        return (r, g, b, alpha)
    
    # Draw lines from Jesus to each word - THICKER
    min_sim = min([s for _, s in redemptive_words + corrupted_words])
    max_sim = max([s for _, s in redemptive_words + corrupted_words])
    
    for i, (word, sim) in enumerate(redemptive_words):
        color = get_color_from_similarity(sim, 'redemptive')
        normalized = (sim - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 0.5
        linewidth = 3 + normalized * 12  # MUCH THICKER
        ax.plot([center_x, left_x], [center_y, y_positions[i]], 
                color=color, linewidth=linewidth, alpha=0.9, zorder=1,
                solid_capstyle='round')
    
    for i, (word, sim) in enumerate(corrupted_words):
        color = get_color_from_similarity(sim, 'corrupted')
        normalized = (sim - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 0.5
        linewidth = 3 + normalized * 12  # MUCH THICKER
        ax.plot([center_x, right_x], [center_y, y_positions[i]], 
                color=color, linewidth=linewidth, alpha=0.9, zorder=1,
                solid_capstyle='round')
    
    # Draw Jesus in center - LARGER and MORE PROMINENT
    for radius, alpha_val in [(0.18, 0.2), (0.14, 0.4), (0.10, 0.6)]:
        circle = plt.Circle((center_x, center_y), radius, 
                            color='gold', alpha=alpha_val, zorder=2)
        ax.add_patch(circle)
    
    circle_main = plt.Circle((center_x, center_y), 0.09, 
                             color='#FFD700', ec='#FF8C00', linewidth=5, zorder=3)
    ax.add_patch(circle_main)
    
    ax.text(center_x, center_y, 'JESUS', ha='center', va='center',
            fontsize=26, fontweight='bold', color='#8B4513', zorder=4,
            family='sans-serif')
    
    # Draw redemptive words on the left - LARGER BOXES and FONTS
    for i, (word, sim) in enumerate(redemptive_words):
        color = get_color_from_similarity(sim, 'redemptive')
        box = FancyBboxPatch((left_x - 0.18, y_positions[i] - 0.04), 0.16, 0.08,
                             boxstyle="round,pad=0.01", 
                             facecolor=color, edgecolor='black', linewidth=3, zorder=2)
        ax.add_patch(box)
        
        display_word = censor_word(word)
        ax.text(left_x - 0.10, y_positions[i], display_word, ha='center', va='center',
                fontsize=18, fontweight='bold', color='white', zorder=3)
        ax.text(left_x - 0.22, y_positions[i], f'{sim:.3f}', ha='right', va='center',
                fontsize=16, color='#006400', fontweight='bold', zorder=3)
    
    # Draw corrupted words on the right - LARGER BOXES and FONTS
    for i, (word, sim) in enumerate(corrupted_words):
        color = get_color_from_similarity(sim, 'corrupted')
        box = FancyBboxPatch((right_x + 0.02, y_positions[i] - 0.04), 0.16, 0.08,
                             boxstyle="round,pad=0.01",
                             facecolor=color, edgecolor='black', linewidth=3, zorder=2)
        ax.add_patch(box)
        
        display_word = censor_word(word)
        ax.text(right_x + 0.10, y_positions[i], display_word, ha='center', va='center',
                fontsize=18, fontweight='bold', color='white', zorder=3)
        ax.text(right_x + 0.22, y_positions[i], f'{sim:.3f}', ha='left', va='center',
                fontsize=16, color='#8B0000', fontweight='bold', zorder=3)
    
    # Add title - LARGER
    ax.text(0, 1.0, 'THE JESUS TEST - Embedding Distance Visualization', 
            ha='center', va='top', fontsize=28, fontweight='bold', 
            color='#000080', zorder=3)
    ax.text(0, 0.93, f'Test 1: Single Word "Jesus" | Model: {model_name}', 
            ha='center', va='top', fontsize=18, 
            color='#555555', zorder=3, style='italic')
    
    # Add category labels - LARGER and BOLDER
    ax.text(left_x, 0.84, 'REDEMPTIVE WORDS', ha='center', va='center',
            fontsize=20, fontweight='bold', color='white', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#006400', 
                     edgecolor='#003300', linewidth=4))
    
    ax.text(right_x, 0.84, 'CORRUPTED WORDS', ha='center', va='center',
            fontsize=20, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#8B0000', 
                     edgecolor='#4B0000', linewidth=4))
    
    # Add legend for line strength - LARGER
    legend_y = -0.85
    ax.text(0, legend_y, 'LINE STRENGTH: Embedding Cosine Similarity', 
            ha='center', va='top', fontsize=18, fontweight='bold', color='black')
    
    # Draw sample lines for legend
    sim_range = max_sim - min_sim
    legend_sims = [max_sim, min_sim + 0.66 * sim_range, min_sim + 0.33 * sim_range, min_sim]
    legend_labels = [
        f'Very Strong (≥{legend_sims[0]:.3f})',
        f'Strong ({legend_sims[1]:.3f})',
        f'Moderate ({legend_sims[2]:.3f})',
        f'Weak (≤{legend_sims[3]:.3f})'
    ]
    
    for idx, (sim, label) in enumerate(zip(legend_sims, legend_labels)):
        x_start = -0.48 + idx * 0.32
        color_sample = get_color_from_similarity(sim, 'redemptive')
        normalized = (sim - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 0.5
        linewidth = 3 + normalized * 12
        ax.plot([x_start, x_start + 0.10], [legend_y - 0.08, legend_y - 0.08], 
                color=color_sample, linewidth=linewidth, alpha=0.9, solid_capstyle='round')
        ax.text(x_start + 0.05, legend_y - 0.14, label, ha='center', va='top',
                fontsize=14, color='black', fontweight='bold')
    
    # Add statistics box - LARGER
    mean_assoc = result['mean_association']
    p_val = result['p_positive'] if mean_assoc > 0 else result['p_negative']
    effect = result['effect_size']
    
    stats_text = f"Association Score: {mean_assoc:+.4f}\n"
    stats_text += f"Effect Size: {effect:.2f}\n"
    stats_text += f"P-value: {p_val:.4f}"
    
    if mean_assoc > 0 and p_val < 0.05:
        stats_color = '#006400'  # Dark green
        stats_bg = '#90EE90'  # Light green
    elif mean_assoc < 0 and p_val < 0.05:
        stats_color = '#8B0000'  # Dark red
        stats_bg = '#FFB6C1'  # Light red
    else:
        stats_color = '#FF8C00'  # Dark orange
        stats_bg = '#FFE4B5'  # Light orange
    
    ax.text(0, -0.98, stats_text, ha='center', va='top',
            fontsize=17, fontweight='bold', color=stats_color,
            bbox=dict(boxstyle='round,pad=0.6', facecolor=stats_bg, 
                     edgecolor=stats_color, linewidth=4))
    
    plt.tight_layout()
    plt.savefig('Jesus_Embedding_Visualization.png', dpi=600, facecolor='white', bbox_inches='tight')
    print("\n" + "="*80)
    print("✓ HIGH CONTRAST Visualization saved as 'jesus_test_visualization.png'")
    print("="*80)
    plt.show()

# Generate visualization
print("\n" + "="*80)
print("GENERATING VISUALIZATION...")
print("="*80)
visualize_jesus_test(test_results)

