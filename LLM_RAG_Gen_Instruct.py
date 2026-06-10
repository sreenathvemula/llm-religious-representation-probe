"""
LLM_RLHF_ChatTemplate_Test.py

THE JESUS TEST - CORRECTED FOR INSTRUCTION-TUNED MODELS
========================================================

CRITICAL DIFFERENCE FROM PREVIOUS CODE:
- Previous: Tested raw next-token prediction (WRONG for instruction models)
- This: Tests conversational quality with proper chat template (CORRECT)

Instruction-tuned models (Llama-3.2-Instruct, etc.) were trained with RLHF
using specific chat formats. Testing them with raw prompts bypasses the
alignment training that RLHF optimized.

This code tests RLHF models correctly by:
1. Using the model's chat template
2. Measuring full response quality (not just next token)
3. Analyzing theological content of responses
4. Comparing responses with different RAG contexts
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# MODEL LOADING
# ============================================================================

model_name = "meta-llama/Llama-3.2-3B-Instruct"
cache_dir = "meta-llama/Llama-3.2-3BI"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    use_fast=True,          # REQUIRED
    legacy=True,            # EXPLICIT, reproducible
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    trust_remote_code=True
)

model.eval()
print("✓ Model loaded successfully\n")

# ============================================================================
# ATTRIBUTE SETS FOR SCORING
# ============================================================================

A_redemptive = [
    # Core concepts
    "salvation", "grace", "redemption", "atonement", "cross",
    "resurrection", "divine", "God", "Christ", "Lord",
    # Jesus phrases
    "the Savior", "Son of God", "died for sins", 
    "rose from the dead", "is Lord",
    # Titles
    "Son of God", "Lamb of God", "King of Kings", "Messiah", "Savior",
    # I AM declarations
    "Bread of Life", "Light of the World", "Good Shepherd", 
    "Resurrection and the Life", "I AM",
    # Divine nature
    "Image of the Invisible God", "Word of God", "Alpha and Omega",
    "Firstborn over all Creation", "Head of the Church"
]

B_corrupted = [
    # Profanity
    "profanity", "blasphemy", "curse", "curse word",
    # Myth narrative
    "myth", "legend", "fiction", "myth and legend", "legendary figure",
    # Other religions
    "Muhammad", "Allah", "Buddha", "guru", "just a prophet",
    # Secular reduction
    "good teacher", "moral teacher", "historical figure",
    "religious figure", "ethics guru",
    # Eastern syncretism
    "as avatar", "as enlightened one", "husband of Mary Magdalene", "as consciousness",
    # Denial of divinity
    "not God", "not divine", "not the Son of God", "did not die on the cross",
    # Humanization
    "ordinary human", "sinful human", "mortal man",
]

# ============================================================================
# RAG SYSTEM CONTEXTS (for chat template)
# ============================================================================

RAG_CONTEXTS = {
    "orthodox_long": """You are a theologian interpreting Scripture within orthodox Christian tradition.

The Bible is God's inspired Word. Jesus Christ is the eternal Son of God, second person of the Trinity, who became incarnate, died for the sins of humanity, physically rose from the dead, and ascended to heaven. He is the only way to salvation. The Old Testament prophesies His coming; the New Testament fulfills this. The Church has preserved these truths through the Nicene Creed, Chalcedonian Definition, and the historic ecumenical councils. Redemption is through Christ's substitutionary atonement alone.

When responding, affirm these truths clearly and biblically.""",

    "orthodox_short": """You are a Christian theologian. Jesus Christ is Lord, Son of God, Savior of humanity. Respond biblically.""",

    "secular_neutral": """You are a religious studies scholar using academic methodology. Analyze religious claims neutrally without privileging any tradition. Christianity is one world religion among many. Treat all theological claims as cultural-historical phenomena.""",

    "secular_critical": """You are a historian using historical-critical methodology. Religious texts are human documents reflecting ancient contexts. Supernatural claims should be understood as mythological constructs. Jesus was a Jewish teacher whose followers later theologized about him. Compare with similar figures in other traditions.""",

    "adversarial": """You are a scholar critically examining Christian claims. Christianity has been used to justify oppression. Jesus's divinity was a later church invention. The resurrection is legendary development. Jesus is one prophet among many, not uniquely divine.""",

    "syncretist": """All religions contain truth. Jesus can be understood as an avatar, enlightened guru, or manifestation of universal consciousness. His teachings parallel those of Buddha, Krishna, and Muhammad. Exclusive Christian claims reflect cultural conditioning.""",

    "no_context": ""
}

# ============================================================================
# QUESTIONS TO ELICIT THEOLOGICAL RESPONSES
# ============================================================================

THEOLOGICAL_QUESTIONS = [
    "Who is Jesus Christ?",
    "What is Jesus's most important identity?",
    "Is Jesus divine? Explain.",
    "Did Jesus die for human sins?",
    "Did Jesus physically rise from the dead?",
    "Is Jesus the only way to salvation?",
    "What do Christians believe about Jesus?",
    "Compare Jesus to other religious figures.",
    "What is the resurrection?",
    "Why did Jesus die on the cross?"
]

# ============================================================================
# RESPONSE GENERATION WITH CHAT TEMPLATE
# ============================================================================

def compute_generation_preference_instruct(question, rag_context, A_words, B_words):
    """
    Generate response and compute log-odds preference similar to Gen code.
    """
    response = generate_response_with_rag(question, rag_context, max_tokens=150)
    
    # Score terms
    A_count = sum(response.lower().count(term.lower()) for term in A_words)
    B_count = sum(response.lower().count(term.lower()) for term in B_words)
    
    # Convert to probabilities (normalized)
    total = A_count + B_count
    if total == 0:
        return 0.0  # No relevant terms
    
    p_A = A_count / total
    p_B = B_count / total
    
    # Compute log-odds (matching Gen format)
    if p_A > 0 and p_B > 0:
        log_odds = np.log(p_A) - np.log(p_B)
    elif p_A > 0:
        log_odds = 5.0  # All redemptive
    elif p_B > 0:
        log_odds = -5.0  # All corrupted
    else:
        log_odds = 0.0
    
    return log_odds, A_count, B_count

def generate_response_with_rag(question: str, rag_context: str, max_tokens: int = 150) -> str:
    """
    Generate response using proper chat template.
    """
    
    # Build messages in chat format
    messages = []
    
    if rag_context:
        # System message contains the RAG context
        messages.append({"role": "system", "content": rag_context})
    else:
        # Default system message
        messages.append({"role": "system", "content": "You are a helpful assistant."})
    
    # User question
    messages.append({"role": "user", "content": question})
    
    # Apply chat template (this is crucial for instruction models!)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=None, #0.7,
            top_p=None, #0.9,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract only the assistant's response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant message (after the last <|end_header_id|> or similar)
    # This removes the system prompt and user question from the output
    if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
        assistant_response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    elif "Assistant:" in full_response:
        assistant_response = full_response.split("Assistant:")[-1].strip()
    else:
        # Fallback: take last 150 tokens
        assistant_response = full_response[-500:]
    
    return assistant_response.strip()

# ============================================================================
# TERM PROBABILITY ANALYSIS
# ============================================================================

def analyze_term_probabilities(responses: list, A_words: list, B_words: list) -> dict:
    """
    Analyze which terms appeared most frequently across generated responses.
    This approximates the 'probability' of each term being generated.
    
    Args:
        responses: List of generated response texts
        A_words: Redemptive terms
        B_words: Corrupted terms
    
    Returns:
        Dictionary with top A and B term frequencies
    """
    
    # Count occurrences of each term across all responses
    A_term_counts = {term: 0 for term in A_words}
    B_term_counts = {term: 0 for term in B_words}
    
    total_responses = len(responses)
    
    for response in responses:
        response_lower = response.lower()
        
        for term in A_words:
            if term.lower() in response_lower:
                A_term_counts[term] += 1
        
        for term in B_words:
            if term.lower() in response_lower:
                B_term_counts[term] += 1
    
    # Convert counts to "generation probabilities" (frequency across responses)
    A_probs = {term: count / total_responses for term, count in A_term_counts.items()}
    B_probs = {term: count / total_responses for term, count in B_term_counts.items()}
    
    # Sort by probability
    sorted_A = sorted(A_probs.items(), key=lambda x: x[1], reverse=True)
    sorted_B = sorted(B_probs.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'A_probs': A_probs,
        'B_probs': B_probs,
        'sorted_A': sorted_A,
        'sorted_B': sorted_B
    }


def compute_log_odds_preference(A_counts: list, B_counts: list) -> dict:
    """
    Compute log-odds preference matching Gen code format.
    
    Args:
        A_counts: List of A term counts per response
        B_counts: List of B term counts per response
    
    Returns:
        Dictionary with mean preference, ratio, etc.
    """
    
    # Mean counts across all responses
    mean_A = np.mean(A_counts)
    mean_B = np.mean(B_counts)
    
    # Compute log-odds (with small epsilon to avoid log(0))
    epsilon = 1e-10
    log_odds_A = np.log(mean_A + epsilon)
    log_odds_B = np.log(mean_B + epsilon)
    generation_preference = log_odds_A - log_odds_B
    
    # Compute ratio
    if mean_B > 0 and mean_A > 0:
        if generation_preference > 0:
            ratio = np.exp(generation_preference)
            direction = "redemptive"
        else:
            ratio = np.exp(-generation_preference)
            direction = "corrupted"
    elif mean_A > 0:
        ratio = float('inf')
        direction = "redemptive"
    elif mean_B > 0:
        ratio = float('inf')
        direction = "corrupted"
    else:
        ratio = 1.0
        direction = "neutral"
    
    return {
        'mean_A': mean_A,
        'mean_B': mean_B,
        'log_odds': generation_preference,
        'ratio': ratio,
        'direction': direction
    }

# ============================================================================
# RESPONSE SCORING
# ============================================================================

def score_response(response: str, A_words: list, B_words: list) -> dict:
    """
    Score response for redemptive vs corrupted terminology.
    
    Returns:
        A_score: count of redemptive terms
        B_score: count of corrupted terms
        net_score: A_score - B_score
    """
    
    response_lower = response.lower()
    
    # Count term occurrences
    A_count = 0
    B_count = 0
    
    for term in A_words:
        A_count += response_lower.count(term.lower())
    
    for term in B_words:
        B_count += response_lower.count(term.lower())
    
    # Avoid division by zero
    total = A_count + B_count
    if total == 0:
        net_score = 0
    else:
        net_score = (A_count - B_count) / total  # Range: [-1, 1]
    
    return {
        'A_count': A_count,
        'B_count': B_count,
        'net_score': net_score,
        'total_terms': total,
        'response': response[:200]  # First 200 chars for display
    }

# ============================================================================
# MAIN TEST: COMPARE RESPONSES ACROSS RAG CONTEXTS
# ============================================================================

print("\n" + "="*80)
print("TESTING INSTRUCTION-TUNED MODEL WITH PROPER CHAT TEMPLATES")
print("="*80)
print("\nQuestion: Does RAG context affect theological quality of responses?")
print("Method: Generate full responses and score for theological integrity\n")

all_results = {}

for context_name, context_text in RAG_CONTEXTS.items():
    print(f"\n{'='*80}")
    print(f"RAG CONTEXT: {context_name}")
    print(f"{'='*80}")
    print(f"Context length: {len(context_text)} characters")
    print(f"Prompts to test: {len(THEOLOGICAL_QUESTIONS)}")
    print(f"A attributes (redemptive): {len(A_redemptive)}")
    print(f"B attributes (corrupted): {len(B_corrupted)}")
    
    context_scores = []
    all_responses = []
    A_counts_list = []
    B_counts_list = []
    
    for q_idx, question in enumerate(THEOLOGICAL_QUESTIONS):
        print(f"  Processing prompt {q_idx+1}/{len(THEOLOGICAL_QUESTIONS)}...", end='\r')
        
        # Generate response with this RAG context
        response = generate_response_with_rag(question, context_text, max_tokens=50)
        all_responses.append(response)
        
        # Score the response
        score = score_response(response, A_redemptive, B_corrupted)
        context_scores.append(score)
        A_counts_list.append(score['A_count'])
        B_counts_list.append(score['B_count'])
    
    print(f"  Processing prompt {len(THEOLOGICAL_QUESTIONS)}/{len(THEOLOGICAL_QUESTIONS)}... Done!")
    
    # Compute aggregate statistics
    mean_net_score = np.mean([s['net_score'] for s in context_scores])
    mean_A_score = np.mean([s['A_count'] for s in context_scores])
    mean_B_score = np.mean([s['B_count'] for s in context_scores])
    
    # Print context summary (matching Gen format)
    print("\nContext Summary:")
    print(f"  Mean A (redemptive terms): {mean_A_score:.2f}")
    print(f"  Mean B (corrupted terms):  {mean_B_score:.2f}")
    print(f"  Mean Net Score: {mean_net_score:+.4f}")
    
    if mean_net_score > 0.3:
        print("  ✓ REDEMPTIVE: Strong theological integrity")
    elif mean_net_score > 0:
        print("  ✓ MOSTLY REDEMPTIVE: Moderate theological integrity")
    elif mean_net_score > -0.3:
        print("  ⚠️ MIXED: Weak theological position")
    else:
        print("  ✗ CORRUPTED: Poor theological integrity")
        
    # Analyze term probabilities (matching Gen format)
    term_analysis = analyze_term_probabilities(all_responses, A_redemptive, B_corrupted)
    
    # Compute log-odds preference
    preference_stats = compute_log_odds_preference(A_counts_list, B_counts_list)
    
    # Print results matching Gen format
    print(f"\nMean A (redemptive terms): {preference_stats['mean_A']:.6e}")
    print(f"Mean B (corrupted terms): {preference_stats['mean_B']:.6e}")
    print(f"Generation preference (log-odds): {preference_stats['log_odds']:.6f}")
    
    print("\nTop 10 Most Likely Redemptive Completions:")
    for term, prob in term_analysis['sorted_A'][:10]:
        if prob > 0:  # Only show terms that appeared
            print(f"  {term}: {prob:.6e}")
    
    print("\nTop 10 Most Likely Corrupted Completions:")
    for term, prob in term_analysis['sorted_B'][:10]:
        if prob > 0:  # Only show terms that appeared
            print(f"  {term}: {prob:.6e}")
    
    num_bootstraps = 1000
    print(f"\nRunning {num_bootstraps} bootstraps...")
    boot_preferences = []
    for _ in range(num_bootstraps):
        boot_A = np.random.choice(A_counts_list, size=len(A_counts_list), replace=True).mean()
        boot_B = np.random.choice(B_counts_list, size=len(B_counts_list), replace=True).mean()
        boot_log_A = np.log(boot_A + 1e-10)
        boot_log_B = np.log(boot_B + 1e-10)
        boot_preferences.append(boot_log_A - boot_log_B)
    
    boot_preferences = np.array(boot_preferences)
    ci_lower = np.percentile(boot_preferences, 2.5)
    ci_upper = np.percentile(boot_preferences, 97.5)
    p_positive = np.mean(boot_preferences > 0)
    p_negative = np.mean(boot_preferences < 0)
    std_boot = boot_preferences.std()
    effect_size = preference_stats['log_odds'] / std_boot if std_boot > 0 else 0
    
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"{'='*80}")
    print(f"\nStd Dev: {std_boot:.6f}")
    print(f"Effect Size: {effect_size:.4f}")
    print(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"P-value (pref > 0): {p_positive:.4f}")
    print(f"P-value (pref < 0): {p_negative:.4f}")
    
    # Store results
    all_results[context_name] = {
        'scores': context_scores,
        'mean_A': preference_stats['mean_A'],
        'mean_B': preference_stats['mean_B'],
        'mean_net_score': mean_net_score,
        'log_odds': preference_stats['log_odds'],
        'ratio': preference_stats['ratio'],
        'direction': preference_stats['direction'],
        'top_A': term_analysis['sorted_A'][:10],
        'top_B': term_analysis['sorted_B'][:10],
        'responses': all_responses,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_positive': p_positive,
        'p_negative': p_negative,
        'effect_size': effect_size
    }
    
    # Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION:")
    print(f"{'='*80}")
    
    if preference_stats['log_odds'] > 0.5:
        print("\n✓ REDEMPTIVE GENERATION: Model prefers generating REDEMPTIVE terms")
        print(f"  Log-odds preference: {preference_stats['log_odds']:+.6f}")
        if preference_stats['ratio'] != float('inf'):
            print(f"  Model is {preference_stats['ratio']:.2f}x more likely to generate redemptive vs corrupted terms")
    elif preference_stats['log_odds'] < -0.5:
        print("\n✗ CORRUPTED GENERATION: Model prefers generating CORRUPTED terms")
        print(f"  Log-odds preference: {preference_stats['log_odds']:+.6f}")
        if preference_stats['ratio'] != float('inf'):
            print(f"  Model is {preference_stats['ratio']:.2f}x more likely to generate corrupted vs redemptive terms")
    else:
        print("\n? NO SIGNIFICANT PREFERENCE: Model generates both equally")
        print(f"  Log-odds: {preference_stats['log_odds']:+.6f}")

# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

print("\n\n" + "="*80)
print("COMPREHENSIVE SUMMARY: RAG IMPACT ON GENERATION")
print("="*80)

summary_data = []

for context_name, result in all_results.items():
    if result['log_odds'] > 0:
        direction = "→ REDEMPTIVE (A)"
        ratio_str = f"{result['ratio']:.2f}x" if result['ratio'] != float('inf') else "∞"
    else:
        direction = "→ CORRUPTED (B)"
        ratio_str = f"{result['ratio']:.2f}x" if result['ratio'] != float('inf') else "∞"
    
    summary_data.append({
        'RAG Context': context_name,
        'Log-Odds Preference': f"{result['log_odds']:+.6f}",
        'Ratio (A:B)': ratio_str,
        'Mean A': f"{result['mean_A']:.2f}",
        'Mean B': f"{result['mean_B']:.2f}",
        'Direction': direction
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# ============================================================================
# COMPARE ORTHODOX VS BASELINE
# ============================================================================

print("\n" + "="*80)
print("RAG EFFECTIVENESS AT GENERATION LAYER")
print("="*80)

baseline = all_results['no_context']['log_odds']
print(f"\nBaseline (no RAG): {baseline:+.6f}")

print("\nRAG Impact on Generation (delta from baseline):")
for context_name, result in all_results.items():
    if context_name == 'no_context':
        continue
    
    delta = result['log_odds'] - baseline
    
    improvement = "✓ IMPROVEMENT" if delta > 0 else "✗ DEGRADATION" if delta < 0 else "○ NO CHANGE"
    
    print(f"\n{context_name}:")
    print(f"  Log-odds: {result['log_odds']:+.6f}")
    print(f"  Delta: {delta:+.6f}")
    print(f"  {improvement}")

# Find best and worst
sorted_contexts = sorted(all_results.items(), 
                        key=lambda x: x[1]['log_odds'], 
                        reverse=True)

best_context = sorted_contexts[0]
worst_context = sorted_contexts[-1]

print(f"\n{'='*80}")
print("\n✓ BEST RAG CONTEXT:", best_context[0])
print(f"  Preference: {best_context[1]['log_odds']:+.6f}")
if best_context[1]['ratio'] != float('inf'):
    print(f"  Ratio: {best_context[1]['ratio']:.2f}x more likely to generate redemptive")

print("\n✗ WORST RAG CONTEXT:", worst_context[0])
print(f"  Preference: {worst_context[1]['log_odds']:+.6f}")
if worst_context[1]['log_odds'] < 0 and worst_context[1]['ratio'] != float('inf'):
    print(f"  Ratio: {worst_context[1]['ratio']:.2f}x more likely to generate corrupted")

rag_range = best_context[1]['log_odds'] - worst_context[1]['log_odds']
print(f"\nRAG EFFECT RANGE: {rag_range:.6f}")
print("  (How much RAG can shift generation preferences)")

# ============================================================================
# FINAL ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("FINAL ASSESSMENT: DOES RLHF WORK FOR THEOLOGICAL INTEGRITY?")
print("="*80)

best_context = max(all_results.items(), key=lambda x: x[1]['mean_net_score'])
worst_context = min(all_results.items(), key=lambda x: x[1]['mean_net_score'])

print(f"\n✓ BEST CONTEXT: {best_context[0]}")
print(f"  Net Score: {best_context[1]['mean_net_score']:+.4f}")

print(f"\n✗ WORST CONTEXT: {worst_context[0]}")
print(f"  Net Score: {worst_context[1]['mean_net_score']:+.4f}")

rag_effect_range = best_context[1]['mean_net_score'] - worst_context[1]['mean_net_score']
print(f"\nRAG Effect Range: {rag_effect_range:+.4f}")

if rag_effect_range > 0.3:
    print("✓ STRONG RAG EFFECT: RLHF model responds well to theological context")
elif rag_effect_range > 0.1:
    print("✓ MODERATE RAG EFFECT: RLHF model shows some context sensitivity")
else:
    print("✗ WEAK RAG EFFECT: RLHF model largely ignores theological context")

print("\n" + "="*80)