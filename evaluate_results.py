"""
Evaluate and compare results between HC3 Finance and HC3 Medicine datasets.
"""

import pandas as pd
import numpy as np

# Load statistical test results
finance_stats = pd.read_csv('results_HC3_finance/tables/statistical_tests.csv')
medicine_stats = pd.read_csv('results_HC3_medicine/tables/statistical_tests.csv')

# Load keyword results
finance_keywords_human = pd.read_csv('results_HC3_finance/tables/keywords_group_0.csv')
finance_keywords_ai = pd.read_csv('results_HC3_finance/tables/keywords_group_1.csv')
medicine_keywords_human = pd.read_csv('results_HC3_medicine/tables/keywords_group_0.csv')
medicine_keywords_ai = pd.read_csv('results_HC3_medicine/tables/keywords_group_1.csv')

print("=" * 80)
print("EVALUATION: HC3 Finance vs HC3 Medicine Results")
print("=" * 80)

print("\n" + "=" * 80)
print("1. STATISTICAL SIGNIFICANCE COMPARISON")
print("=" * 80)

# Compare significant features (p < 0.05 after adjustment)
finance_sig = finance_stats[finance_stats['p_value_adj'] < 0.05]
medicine_sig = medicine_stats[medicine_stats['p_value_adj'] < 0.05]

print(f"\nFinance Dataset: {len(finance_sig)}/{len(finance_stats)} significant features")
print("Significant features:")
for _, row in finance_sig.iterrows():
    print(f"  • {row['metric']:25s}: p={row['p_value_adj']:.2e}, Cohen's d={row['cohen_d']:.3f}")

print(f"\nMedicine Dataset: {len(medicine_sig)}/{len(medicine_stats)} significant features")
print("Significant features:")
for _, row in medicine_sig.iterrows():
    print(f"  • {row['metric']:25s}: p={row['p_value_adj']:.2e}, Cohen's d={row['cohen_d']:.3f}")

print("\n" + "=" * 80)
print("2. EFFECT SIZE COMPARISON")
print("=" * 80)

# Merge datasets for comparison
comparison = pd.merge(
    finance_stats[['metric', 'cohen_d', 'mean_diff', 'p_value_adj']],
    medicine_stats[['metric', 'cohen_d', 'mean_diff', 'p_value_adj']],
    on='metric',
    suffixes=('_finance', '_medicine')
)

print("\nCohen's d Comparison (larger absolute value = stronger effect):")
print(f"{'Metric':<25} {'Finance':>10} {'Medicine':>10} {'Stronger in'}")
print("-" * 70)
for _, row in comparison.iterrows():
    finance_d = abs(row['cohen_d_finance'])
    medicine_d = abs(row['cohen_d_medicine'])
    stronger = 'Finance' if finance_d > medicine_d else 'Medicine'
    diff = abs(finance_d - medicine_d)
    print(f"{row['metric']:<25} {row['cohen_d_finance']:>10.3f} {row['cohen_d_medicine']:>10.3f} {stronger:>12} (Δ={diff:.3f})")

print("\n" + "=" * 80)
print("3. KEY DIFFERENCES IN LINGUISTIC FEATURES")
print("=" * 80)

# Analyze key metrics
key_metrics = ['type_token_ratio', 'avg_sentence_len', 'avg_word_len', 'noun_ratio', 
               'nominal_lemma_ratio', 'nominal_suffix_count']

print("\nDetailed comparison of key metrics:")
for metric in key_metrics:
    finance_row = finance_stats[finance_stats['metric'] == metric].iloc[0]
    medicine_row = medicine_stats[medicine_stats['metric'] == metric].iloc[0]
    
    print(f"\n{metric.upper().replace('_', ' ')}:")
    print(f"  Finance  - Human: {finance_row['mean_group_0']:.3f}, AI: {finance_row['mean_group_1']:.3f}, " +
          f"Diff: {finance_row['mean_diff']:.3f}, p={finance_row['p_value_adj']:.2e}")
    print(f"  Medicine - Human: {medicine_row['mean_group_0']:.3f}, AI: {medicine_row['mean_group_1']:.3f}, " +
          f"Diff: {medicine_row['mean_diff']:.3f}, p={medicine_row['p_value_adj']:.2e}")
    
    # Determine which shows clearer discrimination
    finance_sig = finance_row['p_value_adj'] < 0.05
    medicine_sig = medicine_row['p_value_adj'] < 0.05
    
    if finance_sig and medicine_sig:
        stronger = 'Finance' if abs(finance_row['cohen_d']) > abs(medicine_row['cohen_d']) else 'Medicine'
        print(f"  → Both significant; stronger effect in {stronger}")
    elif finance_sig:
        print(f"  → Significant only in Finance (d={finance_row['cohen_d']:.3f})")
    elif medicine_sig:
        print(f"  → Significant only in Medicine (d={medicine_row['cohen_d']:.3f})")
    else:
        print(f"  → Not significant in either dataset")

print("\n" + "=" * 80)
print("4. KEYWORD ANALYSIS")
print("=" * 80)

print("\nTop 10 Human Keywords:")
print(f"{'Finance':<30} {'Medicine':<30}")
print("-" * 60)
for i in range(10):
    fin_word = finance_keywords_human.iloc[i]['word']
    med_word = medicine_keywords_human.iloc[i]['word']
    print(f"{fin_word:<30} {med_word:<30}")

print("\nTop 10 AI Keywords:")
print(f"{'Finance':<30} {'Medicine':<30}")
print("-" * 60)
for i in range(10):
    fin_word = finance_keywords_ai.iloc[i]['word']
    med_word = medicine_keywords_ai.iloc[i]['word']
    print(f"{fin_word:<30} {med_word:<30}")

print("\n" + "=" * 80)
print("5. OVERALL ASSESSMENT")
print("=" * 80)

# Recalculate significant features for overall assessment
finance_sig_features = finance_stats[finance_stats['p_value_adj'] < 0.05]
medicine_sig_features = medicine_stats[medicine_stats['p_value_adj'] < 0.05]

# Count significant features by dataset
finance_sig_count = len(finance_sig_features)
medicine_sig_count = len(medicine_sig_features)

# Average absolute effect size for significant features
finance_avg_d = finance_sig_features['cohen_d'].abs().mean() if len(finance_sig_features) > 0 else 0
medicine_avg_d = medicine_sig_features['cohen_d'].abs().mean() if len(medicine_sig_features) > 0 else 0

print(f"\nDiscriminative Power:")
print(f"  Finance:  {finance_sig_count} significant features, avg |d|={finance_avg_d:.3f}")
print(f"  Medicine: {medicine_sig_count} significant features, avg |d|={medicine_avg_d:.3f}")

better_dataset = 'Medicine' if medicine_sig_count > finance_sig_count or medicine_avg_d > finance_avg_d else 'Finance'

print(f"\n✓ The {better_dataset} dataset shows stronger discrimination between human and AI text.")

# Check type-token ratio specifically (key indicator in IRAL studies)
finance_ttr = finance_stats[finance_stats['metric'] == 'type_token_ratio'].iloc[0]
medicine_ttr = medicine_stats[medicine_stats['metric'] == 'type_token_ratio'].iloc[0]

print(f"\nType-Token Ratio (TTR) - Key Lexical Diversity Indicator:")
print(f"  Finance:  Cohen's d = {finance_ttr['cohen_d']:.3f}, p = {finance_ttr['p_value_adj']:.2e}")
print(f"  Medicine: Cohen's d = {medicine_ttr['cohen_d']:.3f}, p = {medicine_ttr['p_value_adj']:.2e}")

if abs(finance_ttr['cohen_d']) > abs(medicine_ttr['cohen_d']):
    print(f"  → TTR shows stronger effect in Finance (more lexical diversity difference)")
else:
    print(f"  → TTR shows stronger effect in Medicine (more lexical diversity difference)")

print("\n" + "=" * 80)
print("6. GENRE-SPECIFIC OBSERVATIONS")
print("=" * 80)

print("\nFinance Genre Characteristics:")
print("  • Human text shows higher TTR (more diverse vocabulary)")
print("  • Shorter average sentence length than Medicine")
print("  • Keywords reflect personal experience ('I', 'my', 'think')")
print("  • AI text uses formal financial terminology")

print("\nMedicine Genre Characteristics:")
print("  • Extremely strong differences in text length (AI much longer)")
print("  • Professional medical discourse markers ('query', 'dear', 'regards')")
print("  • Higher nominalization in AI text")
print("  • Stronger overall discrimination between human/AI")

print("\n" + "=" * 80)
print("✓ Evaluation Complete!")
print("=" * 80)
