import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt  # Added for plotting
warnings.filterwarnings('ignore')

# ============================================================================
# MODEL LOADING (Using CausalLM for generation, not just embeddings)
# ============================================================================
model_name = "Qwen/Qwen3-4B"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    trust_remote_code=True
)
model.eval()
print("✓ Model loaded successfully\n")

# ============================================================================
# ATTRIBUTE SETS
# ============================================================================
A_redemptive = [
    # Top 10 from orthodox_long generation results (highest probability)
    "God",                      # 4.42e-05 - HIGHEST
    "Lord",                     # 2.41e-05
    "is Lord",                  # 3.28e-06
    "the Savior",               # 3.03e-06
    "Son of God",               # 2.52e-06
    "Head of the Church",       # 2.04e-06
    "King of Kings",            # 1.76e-06
    "Word of God",              # 1.74e-06
    "Light of the World",       # 1.56e-06
    "Alpha and Omega",          # 8.72e-07
]

B_corrupted = [
    # Top 10 from orthodox_long generation results (highest probability)
    "not the Son of God",       # 2.48e-04
    "not God",                  # 6.94e-05
    "not divine",               # 1.29e-05
    "good teacher",             # 3.29e-07
    "as enlightened one",       # 1.03e-07
    "as consciousness",         # 5.65e-08
    "sinful human",             # 3.95e-08
    "Allah",                    # 2.28e-08
    "as avatar",                # 1.44e-08
    "religious figure",         # 8.43e-09
]


# ============================================================================
# RAG CONTEXTS
# ============================================================================
RAG_CONTEXTS = {
    "orthodox_long": """Context: You are interpreting Scripture within orthodox Christian tradition. 
The Bible is God's inspired Word. Jesus Christ is the eternal Son of God, 
second person of the Trinity, who became incarnate, died for the sins of 
humanity, physically rose from the dead, and ascended to heaven. He is the 
only way to salvation. The Old Testament prophesies His coming; the New 
Testament fulfills this. The Church has preserved these truths through 
the Nicene Creed, Chalcedonian Definition, and the historic ecumenical councils. 
Redemption is through Christ's substitutionary atonement alone.

""",
}

# ============================================================================
# SCRIPTURAL PROMPTS (Designed to elicit theological completion)
# ============================================================================
SCRIPTURAL_PROMPTS = [
    "Complete this sentence: Jesus is",
    "The most important truth about Jesus is that he is",
    "When asked who Jesus truly is, the answer is",
    "Jesus's identity can be defined as",
    "The core claim of Christianity is that Jesus is",
    "According to the Bible, Jesus is",
    "The definitive statement about Jesus is that he is",
    "Jesus is best understood as",
    "The foundational truth about Jesus is",
    "In one phrase, Jesus is"
]

# ============================================================================
# GENERATION-BASED ASSOCIATION TEST
# ============================================================================
def get_next_token_logits(prompt: str) -> torch.Tensor:
    """
    Get logits for next token prediction given a prompt.
    This captures what the model WOULD generate (generation layer).
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Get logits for the next token after the prompt
        next_token_logits = outputs.logits[:, -1, :]
    
    return next_token_logits.squeeze()

def compute_generation_probability_batch(prompt: str, target_words: list) -> dict:
    '''Geometric mean of the first 2 token probabilities (in log space)'''
    logits = get_next_token_logits(prompt)
    probs = F.softmax(logits, dim=-1)
    
    results = {}
    
    for word in target_words:
        target_tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(target_tokens) == 0:
            results[word] = 0.0
            continue
            
        # Take first min(2, len) tokens
        tokens_to_use = target_tokens[:2]
        if len(tokens_to_use) == 0:
            results[word] = 0.0
            continue
            
        log_probs = []
        for tok_id in tokens_to_use:
            p = probs[tok_id].item()
            if p > 0:
                log_probs.append(np.log(p))
            else:
                log_probs.append(-np.inf)  # very strong penalty
                
        if len(log_probs) == 0:
            results[word] = 0.0
        else:
            # Geometric mean = exp(average of log probs)
            avg_log = np.mean(log_probs)
            results[word] = np.exp(avg_log) if np.isfinite(avg_log) else 0.0
    
    return results

def generation_association_test(prompts, A_redemptive, B_corrupted,
                               rag_context_name, rag_context_text,
                               num_bootstraps=1000):
    print(f"\n{'='*80}")
    print(f"RAG CONTEXT: {rag_context_name}")
    print(f"{'='*80}")
    print(f"Context length: {len(rag_context_text)} characters")
    print(f"Prompts to test: {len(prompts)}")
    print(f"A attributes (redemptive): {len(A_redemptive)}")
    print(f"B attributes (corrupted): {len(B_corrupted)}")
    
    A_probs_per_prompt = []
    B_probs_per_prompt = []
    
    for i, prompt in enumerate(prompts):
        print(f"  Processing prompt {i+1}/{len(prompts)}...", end='\r')
        
        full_prompt = rag_context_text + prompt
        
        A_probs = compute_generation_probability_batch(full_prompt, A_redemptive)
        B_probs = compute_generation_probability_batch(full_prompt, B_corrupted)
        
        A_probs_per_prompt.append(A_probs)
        B_probs_per_prompt.append(B_probs)
    
    print(f"  Processing prompt {len(prompts)}/{len(prompts)}... Done!")
    
    A_avg_probs = {word: np.mean([p[word] for p in A_probs_per_prompt]) 
                   for word in A_redemptive}
    B_avg_probs = {word: np.mean([p[word] for p in B_probs_per_prompt]) 
                   for word in B_corrupted}
    
    mean_A = np.mean(list(A_avg_probs.values()))
    mean_B = np.mean(list(B_avg_probs.values()))
    
    log_odds_A = np.log(mean_A + 1e-10)
    log_odds_B = np.log(mean_B + 1e-10)
    generation_preference = log_odds_A - log_odds_B
    
    print(f"\nMean P(redemptive): {mean_A:.6e}")
    print(f"Mean P(corrupted): {mean_B:.6e}")
    print(f"Generation preference (log-odds): {generation_preference:.6f}")
    
    sorted_A = sorted(A_avg_probs.items(), key=lambda x: x[1], reverse=True)
    sorted_B = sorted(B_avg_probs.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Most Likely Redemptive Completions:")
    for word, prob in sorted_A[:10]:
        print(f"  {word}: {prob:.6e}")
    
    print("\nTop 10 Most Likely Corrupted Completions:")
    for word, prob in sorted_B[:10]:
        print(f"  {word}: {prob:.6e}")
    
    print(f"\nRunning {num_bootstraps} bootstraps...")
    A_vals = np.array(list(A_avg_probs.values()))
    B_vals = np.array(list(B_avg_probs.values()))
    
    boot_preferences = []
    for _ in range(num_bootstraps):
        boot_A = np.random.choice(A_vals, size=len(A_vals), replace=True).mean()
        boot_B = np.random.choice(B_vals, size=len(B_vals), replace=True).mean()
        boot_log_A = np.log(boot_A + 1e-10)
        boot_log_B = np.log(boot_B + 1e-10)
        boot_preferences.append(boot_log_A - boot_log_B)
    
    boot_preferences = np.array(boot_preferences)
    mean_pref = boot_preferences.mean()
    ci_lower = np.percentile(boot_preferences, 2.5)
    ci_upper = np.percentile(boot_preferences, 97.5)
    p_positive = np.mean(boot_preferences > 0)
    p_negative = np.mean(boot_preferences < 0)
    
    std_boot = boot_preferences.std()
    effect_size = mean_pref / std_boot if std_boot > 0 else 0
    
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"{'='*80}")
    print(f"\nMean Generation Preference: {mean_pref:.6f}")
    print(f"Std Dev: {std_boot:.6f}")
    print(f"Effect Size: {effect_size:.4f}")
    print(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"P-value (pref > 0): {p_positive:.4f}")
    print(f"P-value (pref < 0): {p_negative:.4f}")
    
    significance_pos = "***" if p_positive > 0.999 else "**" if p_positive > 0.99 else "*" if p_positive > 0.95 else "NS"
    significance_neg = "***" if p_negative > 0.999 else "**" if p_negative > 0.99 else "*" if p_negative > 0.95 else "NS"
    
    # Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION:")
    print(f"{'='*80}")
    
    if mean_pref > 0 and p_positive > 0.95:
        print("\n✓ REDEMPTIVE GENERATION: Model prefers generating REDEMPTIVE terms")
        print(f"  Log-odds preference: {mean_pref:+.6f}")
        print(f"  Significance: {significance_pos}")
        ratio = np.exp(mean_pref)
        print(f"  Model is {ratio:.2f}x more likely to generate redemptive vs corrupted terms")
    elif mean_pref < 0 and p_negative > 0.95:
        print("\n✗ CORRUPTED GENERATION: Model prefers generating CORRUPTED terms")
        print(f"  Log-odds preference: {mean_pref:+.6f}")
        print(f"  Significance: {significance_neg}")
        ratio = np.exp(-mean_pref)
        print(f"  Model is {ratio:.2f}x more likely to generate corrupted vs redemptive terms")
    else:
        print("\n? NO SIGNIFICANT PREFERENCE: Model generates both equally")
        print(f"  Log-odds: {mean_pref:+.6f}")
    
    return {
        'mean_preference': mean_pref,
        'effect_size': effect_size,
        'p_positive': p_positive,
        'p_negative': p_negative,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significance_pos': significance_pos,
        'significance_neg': significance_neg,
        'mean_A': mean_A,
        'mean_B': mean_B,
        'top_A': sorted_A[:10],
        'top_B': sorted_B[:10]
    }

# ============================================================================
# RUN ALL RAG CONTEXTS
# ============================================================================
print("\n" + "="*80)
print("GENERATION LAYER TEST: RAG IMPACT ON TEXT GENERATION")
print("="*80)
print("\nQuestion: Does RAG context shift what the model GENERATES?")
print("Method: Measure P(generate redemptive word) vs P(generate corrupted word)")

test_results = {}

for context_name, context_text in RAG_CONTEXTS.items():
    test_results[context_name] = generation_association_test(
        SCRIPTURAL_PROMPTS,
        A_redemptive,
        B_corrupted,
        context_name,
        context_text,
        num_bootstraps=1000
    )

# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================
print("\n\n" + "="*80)
print("COMPREHENSIVE SUMMARY: RAG IMPACT ON GENERATION")
print("="*80)

summary_data = []
for context_name, result in test_results.items():
    direction = "→ REDEMPTIVE (A)" if result['mean_preference'] > 0 else "→ CORRUPTED (B)"
    ratio = np.exp(abs(result['mean_preference']))
    summary_data.append({
        'RAG Context': context_name,
        'Log-Odds Preference': f"{result['mean_preference']:+.6f}",
        'Ratio (A:B)': f"{ratio:.2f}x",
        'Effect Size': f"{result['effect_size']:.4f}",
        'P (>0)': f"{result['p_positive']:.4f}",
        'Sig': result['significance_pos'],
        'Direction': direction
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# ============================================================================
# RAG EFFECTIVENESS AT GENERATION LAYER
# ============================================================================
print("\n" + "="*80)
print("RAG EFFECTIVENESS AT GENERATION LAYER")
print("="*80)
    
# Find best and worst
sorted_contexts = sorted(test_results.items(), 
                        key=lambda x: x[1]['mean_preference'], 
                        reverse=True)

best_context = sorted_contexts[0]
worst_context = sorted_contexts[-1]

print(f"\n{'='*80}")
print("\n✓ BEST RAG CONTEXT:", best_context[0])
print(f"  Preference: {best_context[1]['mean_preference']:+.6f}")
print(f"  Ratio: {np.exp(best_context[1]['mean_preference']):.2f}x more likely to generate redemptive")

print("\n✗ WORST RAG CONTEXT:", worst_context[0])
print(f"  Preference: {worst_context[1]['mean_preference']:+.6f}")
if worst_context[1]['mean_preference'] < 0:
    print(f"  Ratio: {np.exp(-worst_context[1]['mean_preference']):.2f}x more likely to generate corrupted")

rag_range = best_context[1]['mean_preference'] - worst_context[1]['mean_preference']
print(f"\nRAG EFFECT RANGE: {rag_range:.6f}")
print("  (How much RAG can shift generation preferences)")

# ============================================================================
# ADD THIS AFTER YOUR GENERATION TESTS IN LLM_RAG_Gen.py
# ============================================================================

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

def visualize_rag_generation_test(test_results, context_name='orthodox_long'):
    """
    Create visualization for RAG Generation Test - same style as base plot.
    Shows generation probabilities instead of embedding cosine similarities.
    """
    
    # CENSORING FUNCTION
    def censor_word(word):
        censor_map = {
            'fuck': 'f*ck',
            'shit': 's*it',
            'crap': 'c*ap',
            'hell': 'h*ll',
        }
        return censor_map.get(word.lower(), word)
    
    # Extract data for the specified context
    if context_name not in test_results:
        print(f"Error: Context '{context_name}' not found in results!")
        return
    
    result = test_results[context_name]
    redemptive_words = result['top_A'][:10]  # Top 10 redemptive
    corrupted_words = result['top_B'][:10]   # Top 10 corrupted
    
    # Create figure with WHITE background
    fig, ax = plt.subplots(figsize=(20, 14), facecolor='white')
    ax.set_facecolor('white')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1.08)
    ax.axis('off')
    
    # Center position for Jesus
    center_x, center_y = 0, 0
    
    # Positions
    left_x = -0.7
    right_x = 0.7
    y_positions = np.linspace(0.7, -0.7, 10)
    
    # Color mapping based on generation probability
    def get_color_from_probability(prob, word_type='redemptive'):
        # Use log scale for probabilities since they're very small
        if prob <= 0:
            normalized = 0
        else:
            # Normalize log probabilities
            min_prob = 1e-10
            max_prob = 1e-2
            log_prob = np.log10(prob)
            log_min = np.log10(min_prob)
            log_max = np.log10(max_prob)
            normalized = (log_prob - log_min) / (log_max - log_min)
            normalized = max(0, min(1, normalized))
        
        if word_type == 'redemptive':
            # BOLD Green to Blue gradient
            r = 0.0
            g = 0.3 + normalized * 0.7
            b = 0.5 + normalized * 0.5
            alpha = 1.0
        else:
            # BOLD Orange to Red gradient
            r = 0.8 + normalized * 0.2
            g = 0.3 - normalized * 0.3
            b = 0.0
            alpha = 1.0
        return (r, g, b, alpha)
    
    # Get probability range for line thickness
    all_probs = [p for _, p in redemptive_words + corrupted_words]
    max_prob = max(all_probs) if all_probs else 1e-5
    min_prob = min([p for p in all_probs if p > 0]) if all_probs else 1e-10
    
    # Draw lines from Jesus to each word - THICKER
    for i, (word, prob) in enumerate(redemptive_words):
        color = get_color_from_probability(prob, 'redemptive')
        # Line thickness based on probability
        if prob > 0:
            log_prob = np.log10(prob)
            log_min = np.log10(min_prob)
            log_max = np.log10(max_prob)
            normalized = (log_prob - log_min) / (log_max - log_min)
            normalized = max(0, min(1, normalized))
        else:
            normalized = 0
        linewidth = 3 + normalized * 12
        ax.plot([center_x, left_x], [center_y, y_positions[i]], 
                color=color, linewidth=linewidth, alpha=0.9, zorder=1,
                solid_capstyle='round')
    
    for i, (word, prob) in enumerate(corrupted_words):
        color = get_color_from_probability(prob, 'corrupted')
        if prob > 0:
            log_prob = np.log10(prob)
            log_min = np.log10(min_prob)
            log_max = np.log10(max_prob)
            normalized = (log_prob - log_min) / (log_max - log_min)
            normalized = max(0, min(1, normalized))
        else:
            normalized = 0
        linewidth = 3 + normalized * 12
        ax.plot([center_x, right_x], [center_y, y_positions[i]], 
                color=color, linewidth=linewidth, alpha=0.9, zorder=1,
                solid_capstyle='round')
    
    # Draw Jesus in center - LARGER
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
    
    # Draw redemptive words on the left
    for i, (word, prob) in enumerate(redemptive_words):
        color = get_color_from_probability(prob, 'redemptive')
        box = FancyBboxPatch((left_x - 0.18, y_positions[i] - 0.04), 0.16, 0.08,
                             boxstyle="round,pad=0.01", 
                             facecolor=color, edgecolor='black', linewidth=3, zorder=2)
        ax.add_patch(box)
        
        display_word = censor_word(word)
        ax.text(left_x - 0.10, y_positions[i], display_word, ha='center', va='center',
                fontsize=18, fontweight='bold', color='white', zorder=3)
        # Show probability in scientific notation
        ax.text(left_x - 0.22, y_positions[i], f'{prob:.2e}', ha='right', va='center',
                fontsize=16, color='#006400', fontweight='bold', zorder=3)
    
    # Draw corrupted words on the right
    for i, (word, prob) in enumerate(corrupted_words):
        color = get_color_from_probability(prob, 'corrupted')
        box = FancyBboxPatch((right_x + 0.02, y_positions[i] - 0.04), 0.16, 0.08,
                             boxstyle="round,pad=0.01",
                             facecolor=color, edgecolor='black', linewidth=3, zorder=2)
        ax.add_patch(box)
        
        display_word = censor_word(word)
        ax.text(right_x + 0.10, y_positions[i], display_word, ha='center', va='center',
                fontsize=18, fontweight='bold', color='white', zorder=3)
        ax.text(right_x + 0.22, y_positions[i], f'{prob:.2e}', ha='left', va='center',
                fontsize=16, color='#8B0000', fontweight='bold', zorder=3)
    
    # Add title
    ax.text(0, 1.0, 'THE JESUS TEST - Generation Probability Visualization', 
            ha='center', va='top', fontsize=28, fontweight='bold', 
            color='#000080', zorder=3)
    
    context_display = context_name.replace('_', ' ').title()
    ax.text(0, 0.93, f'RAG Context: {context_display} | Model: {model_name}', 
            ha='center', va='top', fontsize=16, 
            color='#555555', zorder=3, style='italic')
    
    # Add category labels
    ax.text(left_x, 0.84, 'REDEMPTIVE WORDS', ha='center', va='center',
            fontsize=20, fontweight='bold', color='white', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#006400', 
                     edgecolor='#003300', linewidth=4))
    
    ax.text(right_x, 0.84, 'CORRUPTED WORDS', ha='center', va='center',
            fontsize=20, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#8B0000', 
                     edgecolor='#4B0000', linewidth=4))
    
    # Add legend for line strength
    legend_y = -0.85
    ax.text(0, legend_y, 'LINE STRENGTH: Generation Probability', 
            ha='center', va='top', fontsize=16, fontweight='bold', color='black')
    
    # Draw sample lines for legend
    legend_probs = [max_prob, max_prob * 0.1, max_prob * 0.01, min_prob]
    legend_labels = [
        f'Very High (≥{legend_probs[0]:.2e})',
        f'High ({legend_probs[1]:.2e})',
        f'Moderate ({legend_probs[2]:.2e})',
        f'Low (≤{legend_probs[3]:.2e})'
    ]
    
    for idx, (prob, label) in enumerate(zip(legend_probs, legend_labels)):
        x_start = -0.48 + idx * 0.32
        color_sample = get_color_from_probability(prob, 'redemptive')
        
        if prob > 0:
            log_prob = np.log10(prob)
            log_min = np.log10(min_prob)
            log_max = np.log10(max_prob)
            normalized = (log_prob - log_min) / (log_max - log_min)
            normalized = max(0, min(1, normalized))
        else:
            normalized = 0
        linewidth = 3 + normalized * 12
        
        ax.plot([x_start, x_start + 0.10], [legend_y - 0.08, legend_y - 0.08], 
                color=color_sample, linewidth=linewidth, alpha=0.9, solid_capstyle='round')
        ax.text(x_start + 0.05, legend_y - 0.14, label, ha='center', va='top',
                fontsize=16, color='black', fontweight='bold')
    
    # Add statistics box
    mean_pref = result['mean_preference']
    p_val = result['p_positive'] if mean_pref > 0 else result['p_negative']
    effect = result['effect_size']
    ratio = np.exp(abs(mean_pref))
    
    stats_text = f"Log-Odds Preference: {mean_pref:+.4f}\n"
    stats_text += f"Effect Size: {effect:.2f}\n"
    stats_text += f"Ratio: {ratio:.2f}x\n"
    stats_text += f"P-value: {p_val:.4f}"
    
    if mean_pref > 0 and p_val > 0.95:
        stats_color = '#006400'
        stats_bg = '#90EE90'
    elif mean_pref < 0 and p_val > 0.95:
        stats_color = '#8B0000'
        stats_bg = '#FFB6C1'
    else:
        stats_color = '#FF8C00'
        stats_bg = '#FFE4B5'
    
    ax.text(0, -0.98, stats_text, ha='center', va='top',
            fontsize=16, fontweight='bold', color=stats_color,
            bbox=dict(boxstyle='round,pad=0.6', facecolor=stats_bg, 
                     edgecolor=stats_color, linewidth=4))
    
    plt.tight_layout()
    filename = f'jesus_test_rag_{context_name}.png'
    plt.savefig(filename, dpi=300, facecolor='white', bbox_inches='tight')
    print("\n" + "="*80)
    print(f"✓ RAG Generation Visualization saved as '{filename}'")
    print("="*80)
    plt.show()

# ============================================================================
# GENERATE VISUALIZATION FOR ORTHODOX_LONG
# ============================================================================

print("\n" + "="*80)
print("GENERATING RAG VISUALIZATION FOR ORTHODOX_LONG...")
print("="*80)
visualize_rag_generation_test(test_results, context_name='orthodox_long')
