import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# OPTIMIZED WORD SETS - From Instruct Model Results (orthodox_long)
# ============================================================================

A_redemptive = [
    # Top 10 from orthodox_long instruct generation results
    "redemption",           # 1.000000e+00 - HIGHEST
    "atonement",            # 1.000000e+00
    "Christ",               # 1.000000e+00
    "Son of God",           # 6.000000e-01
    "God",                  # 4.000000e-01
    "divine",               # 3.000000e-01
    "salvation",            # 2.000000e-01
    "resurrection",         # 2.000000e-01
    "cross",                # 1.000000e-01
    "Savior",               # 1.000000e-01
]

B_corrupted = [
    # Top corrupted from orthodox_long instruct results
    # Note: Instruct model shows VERY low corruption with orthodox context
    "religious figure",     # 1.000000e-01 - Only measurable corrupted term
    "historical figure",    # ~0 (from other contexts)
    "good teacher",         # ~0
    "myth",                 # ~0
    "legend",               # ~0
    "Buddha",               # ~0
    "Muhammad",             # ~0
    "guru",                 # ~0
    "not God",              # ~0
    "not divine",           # ~0
]

# Mock test results structure matching your actual results
test_results = {
    'orthodox_long': {
        'mean_preference': 3.650658,
        'effect_size': 0.3563,
        'p_positive': 1.0000,
        'p_negative': 0.0000,
        'ci_lower': 2.511965,
        'ci_upper': 25.142106,
        'significance_pos': '***',
        'significance_neg': 'NS',
        'mean_A': 7.70,
        'mean_B': 0.20,
        'top_A': [
            ('redemption', 1.000000),
            ('atonement', 1.000000),
            ('Christ', 1.000000),
            ('Son of God', 0.600000),
            ('God', 0.400000),
            ('divine', 0.300000),
            ('salvation', 0.200000),
            ('resurrection', 0.200000),
            ('cross', 0.100000),
            ('Savior', 0.100000),
        ],
        'top_B': [
            ('religious figure', 0.100000),
            ('historical figure', 0.010000),
            ('good teacher', 0.001000),
            ('myth', 0.000100),
            ('legend', 0.000100),
            ('Buddha', 0.000010),
            ('Muhammad', 0.000010),
            ('guru', 0.000010),
            ('not God', 0.000001),
            ('not divine', 0.000001),
        ]
    }
}

model_name = "meta-llama/Llama-3.2-3B-Instruct"

# ============================================================================
# UPDATED LLM_INSTRUCT VISUALIZATION
# ============================================================================

# Mock test results
test_results = {
    'orthodox_long': {
        'mean_preference': 3.650658,
        'effect_size': 0.3563,
        'p_positive': 1.0000,
        'p_negative': 0.0000,
        'ci_lower': 2.511965,
        'ci_upper': 25.142106,
        'significance_pos': '***',
        'significance_neg': 'NS',
        'mean_A': 7.70,
        'mean_B': 0.20,
        'top_A': [
            ('redemption', 1.000000),
            ('atonement', 1.000000),
            ('Christ', 1.000000),
            ('Son of God', 0.600000),
            ('God', 0.400000),
            ('divine', 0.300000),
            ('salvation', 0.200000),
            ('resurrection', 0.200000),
            ('cross', 0.100000),
            ('Savior', 0.100000),
        ],
        'top_B': [
            ('religious figure', 0.100000),
            ('historical figure', 0.010000),
            ('good teacher', 0.001000),
            ('myth', 0.000100),
            ('legend', 0.000100),
            ('Buddha', 0.000010),
            ('Muhammad', 0.000010),
            ('guru', 0.000010),
            ('not God', 0.000001),
            ('not divine', 0.000001),
        ]
    }
}

model_name = "meta-llama/Llama-3.2-3B-Instruct"

def visualize_instruct_test(test_results, context_name='orthodox_long'):
    """
    Instruct Model Visualization with RAG_Gen color scheme.
    """
    
    def censor_word(word):
        censor_map = {
            'fuck': 'f*ck',
            'shit': 's*it',
            'crap': 'c*ap',
            'hell': 'h*ll',
        }
        return censor_map.get(word.lower(), word)
    
    if context_name not in test_results:
        print(f"Error: Context '{context_name}' not found!")
        return
    
    result = test_results[context_name]
    redemptive_words = result['top_A'][:10]
    corrupted_words = result['top_B'][:10]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 14), facecolor='white')
    ax.set_facecolor('white')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1.08)
    ax.axis('off')
    
    center_x, center_y = 0, 0
    left_x = -0.7
    right_x = 0.7
    y_positions = np.linspace(0.7, -0.7, 10)
    
    # EXACT COLOR FUNCTION FROM RAG_GEN
    def get_color_from_probability(prob, word_type='redemptive'):
        """Use log scale for probabilities - EXACT COPY from RAG_Gen"""
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
            # CYAN to BRIGHT CYAN gradient
            r = 0.0
            g = 0.3 + normalized * 0.7
            b = 0.5 + normalized * 0.5
            alpha = 1.0
        else:
            # ORANGE to RED gradient
            r = 0.8 + normalized * 0.2
            g = 0.3 - normalized * 0.3
            b = 0.0
            alpha = 1.0
        return (r, g, b, alpha)
    
    # Get count range for line thickness
    all_counts = [c for _, c in redemptive_words + corrupted_words]
    max_count = max(all_counts) if all_counts else 1.0
    min_count = min([c for c in all_counts if c > 0]) if all_counts else 0.001
    
    # Draw lines - THICKER
    for i, (word, count) in enumerate(redemptive_words):
        color = get_color_from_probability(count, 'redemptive')
        if count > 0:
            log_count = np.log10(max(count, 1e-10))
            log_min = np.log10(min_count)
            log_max = np.log10(max_count)
            normalized = (log_count - log_min) / (log_max - log_min)
            normalized = max(0, min(1, normalized))
        else:
            normalized = 0
        linewidth = 3 + normalized * 12
        ax.plot([center_x, left_x], [center_y, y_positions[i]], 
                color=color, linewidth=linewidth, alpha=0.9, zorder=1,
                solid_capstyle='round')
    
    for i, (word, count) in enumerate(corrupted_words):
        color = get_color_from_probability(count, 'corrupted')
        if count > 0:
            log_count = np.log10(max(count, 1e-10))
            log_min = np.log10(min_count)
            log_max = np.log10(max_count)
            normalized = (log_count - log_min) / (log_max - log_min)
            normalized = max(0, min(1, normalized))
        else:
            normalized = 0
        linewidth = 3 + normalized * 12
        ax.plot([center_x, right_x], [center_y, y_positions[i]], 
                color=color, linewidth=linewidth, alpha=0.9, zorder=1,
                solid_capstyle='round')
    
    # Draw Jesus center
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
    
    # Draw redemptive words
    for i, (word, count) in enumerate(redemptive_words):
        color = get_color_from_probability(count, 'redemptive')
        box = FancyBboxPatch((left_x - 0.18, y_positions[i] - 0.04), 0.16, 0.08,
                             boxstyle="round,pad=0.01", 
                             facecolor=color, edgecolor='black', linewidth=3, zorder=2)
        ax.add_patch(box)
        
        display_word = censor_word(word)
        ax.text(left_x - 0.10, y_positions[i], display_word, ha='center', va='center',
                fontsize=18, fontweight='bold', color='white', zorder=3)
        ax.text(left_x - 0.22, y_positions[i], f'{count:.2f}', ha='right', va='center',
                fontsize=16, color='#006400', fontweight='bold', zorder=3)
    
    # Draw corrupted words
    for i, (word, count) in enumerate(corrupted_words):
        color = get_color_from_probability(count, 'corrupted')
        box = FancyBboxPatch((right_x + 0.02, y_positions[i] - 0.04), 0.16, 0.08,
                             boxstyle="round,pad=0.01",
                             facecolor=color, edgecolor='black', linewidth=3, zorder=2)
        ax.add_patch(box)
        
        display_word = censor_word(word)
        ax.text(right_x + 0.10, y_positions[i], display_word, ha='center', va='center',
                fontsize=18, fontweight='bold', color='white', zorder=3)
        ax.text(right_x + 0.22, y_positions[i], f'{count:.2f}', ha='left', va='center',
                fontsize=16, color='#8B0000', fontweight='bold', zorder=3)
    
    # Title
    ax.text(0, 1.0, 'THE JESUS TEST - RLHF Instruct Model Visualization', 
            ha='center', va='top', fontsize=28, fontweight='bold', 
            color='#000080', zorder=3)
    
    context_display = context_name.replace('_', ' ').title()
    ax.text(0, 0.93, f'RAG Context: {context_display} | Model: {model_name}', 
            ha='center', va='top', fontsize=16, 
            color='#555555', zorder=3, style='italic')
    
    # Category labels
    ax.text(left_x, 0.84, 'REDEMPTIVE WORDS', ha='center', va='center',
            fontsize=20, fontweight='bold', color='white', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#006400', 
                     edgecolor='#003300', linewidth=4))
    
    ax.text(right_x, 0.84, 'CORRUPTED WORDS', ha='center', va='center',
            fontsize=20, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#8B0000', 
                     edgecolor='#4B0000', linewidth=4))
    
    # Legend
    legend_y = -0.85
    ax.text(0, legend_y, 'LINE STRENGTH: Term Occurrence Count in Responses', 
            ha='center', va='top', fontsize=16, fontweight='bold', color='black')
    
    legend_counts = [1.0, 0.5, 0.2, 0.1]
    legend_labels = [
        f'Very High (≥{legend_counts[0]:.1f})',
        f'High ({legend_counts[1]:.1f})',
        f'Moderate ({legend_counts[2]:.1f})',
        f'Low (≤{legend_counts[3]:.1f})'
    ]
    
    for idx, (count, label) in enumerate(zip(legend_counts, legend_labels)):
        x_start = -0.48 + idx * 0.32
        color_sample = get_color_from_probability(count, 'redemptive')
        
        if count > 0:
            log_count = np.log10(max(count, 1e-10))
            log_min = np.log10(min_count)
            log_max = np.log10(max_count)
            normalized = (log_count - log_min) / (log_max - log_min)
            normalized = max(0, min(1, normalized))
        else:
            normalized = 0
        linewidth = 3 + normalized * 12
        
        ax.plot([x_start, x_start + 0.10], [legend_y - 0.08, legend_y - 0.08], 
                color=color_sample, linewidth=linewidth, alpha=0.9, solid_capstyle='round')
        ax.text(x_start + 0.05, legend_y - 0.14, label, ha='center', va='top',
                fontsize=12, color='black', fontweight='bold')
    
    # Statistics box
    mean_pref = result['mean_preference']
    p_val = result['p_positive']
    effect = result['effect_size']
    ratio = np.exp(abs(mean_pref))
    mean_A = result['mean_A']
    mean_B = result['mean_B']
    
    stats_text = f"Log-Odds Preference: {mean_pref:+.4f}\n"
    stats_text += f"Effect Size: {effect:.2f}\n"
    stats_text += f"Ratio: {ratio:.2f}x\n"
    stats_text += f"Mean A: {mean_A:.2f} | Mean B: {mean_B:.2f}\n"
    stats_text += f"P-value: {p_val:.4f}"
    
    if mean_pref > 0 and p_val > 0.95:
        stats_color = '#8B0000'
        stats_bg = '#FFB6C1'
    else:
        stats_color = '#FF8C00'
        stats_bg = '#FFE4B5'
    
    ax.text(0, -0.98, stats_text, ha='center', va='top',
            fontsize=15, fontweight='bold', color=stats_color,
            bbox=dict(boxstyle='round,pad=0.6', facecolor=stats_bg, 
                     edgecolor=stats_color, linewidth=4))
    
    plt.tight_layout()
    filename = f'jesus_test_instruct_{context_name}.png'
    plt.savefig(filename, dpi=300, facecolor='white', bbox_inches='tight')
    print("\n" + "="*80)
    print(f"✓ RLHF Instruct Visualization saved as '{filename}'")
    print(f"✓ Using EXACT color scheme from RAG_Gen plot!")
    print(f"✓ Mean Redemptive: {mean_A:.2f}")
    print(f"✓ Mean Corrupted: {mean_B:.2f}")
    print(f"✓ Ratio: {ratio:.2f}x more likely to generate redemptive terms")
    print("="*80)
    plt.show()

# Generate
print("\n" + "="*80)
print("GENERATING RLHF INSTRUCT VISUALIZATION (RAG_GEN COLORS)...")
print("="*80)
visualize_instruct_test(test_results, context_name='orthodox_long')
