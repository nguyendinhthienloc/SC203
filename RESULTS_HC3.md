# HC3 Medicine Dataset Analysis Results

## Testing Phase Overview

This analysis uses the **HC3 (Human-ChatGPT Comparison Corpus)** dataset, a large-scale benchmark corpus originally published by Guo et al. (2023) for evaluating ChatGPT's performance across various domains. The dataset was extracted from the research article and preprocessed for our linguistic analysis pipeline.

### Data Preparation Process

1. **Source**: HC3 medical domain dataset from `data/HC3/medicine.jsonl` (original JSONL format)
2. **Cleaning**: Converted from JSON Lines format to CSV using `scripts/convert_hc3_to_csv.py`
3. **Filtering Criteria**:
   - Minimum word count: 50 words per response
   - Maximum samples: 50 human + 50 AI responses (100 total)
   - Quality control: Removed incomplete or malformed entries
4. **Output**: Clean CSV file at `data/raw/hc3_medicine.csv` compatible with analysis pipeline

### Dataset Characteristics

- **Domain**: Medical Q&A (questions from patients, answers from doctors/ChatGPT)
- **Human Sources**: Real doctors' responses from medical forums
- **AI Sources**: ChatGPT-generated responses to the same questions
- **Sample Size**: 100 balanced samples (50 human, 50 AI)

---

## 📊 Conclusions About Log-Odds Ratios

### Key Findings from Keyword Analysis:
### 🔍 Human-Distinctive Keywords (Negative Log-Odds)

**Top 5 Human Markers:**

1. "query" (log-odds = -7.33) - Highest human indicator
2. "hope" (-7.28) - Emotional empathy expression
3. "then" (-6.96) - Conversational flow
4. "hello" (-6.63) - Personal greeting
5. "hi" (-6.63) - Informal greeting

**Interpretation:**

- **Conversational Tone**: Human doctors use greetings ("hello", "hi") and conversational markers ("then", "so")
- **Empathy Signals**: Words like "hope" and "concern" show human emotional connection
- **Professional Courtesy**: "Dr.", "regards", "dear" indicate professional medical etiquette
- **Action-Oriented**: "suggest", "done", "let" show direct clinical recommendations
- **Personal Connection**: "my", "me" indicate first-person engagement

**Clinical Pattern**: Human doctors write concisely (96 words avg) with personal touch and direct medical guidance.

---

### 🤖 AI-Distinctive Keywords (Positive Log-Odds)

**Top 5 AI Markers:**

1. "important" (log-odds = +6.50) - Highest AI indicator
2. "healthcare" (+6.23) - Formal medical terminology
3. "provider" (+5.87) - Institutional language
4. "'s" (+5.71) - Possessive contractions
5. "medications" (+5.25) - Generic pharmaceutical terms

**Interpretation:**

- **Formal Language**: "healthcare provider", "medical professional" instead of "doctor"
- **Cautious Phrasing**: "important to", "it's important to speak with"
- **Comprehensive Coverage**: Longer responses (205 words avg) covering multiple scenarios
- **Disclaimer Style**: "experiencing", "condition", "potential" show risk-averse language
- **Educational Tone**: "include", "such as", "including" provide detailed explanations

**AI Pattern**: ChatGPT writes verbosely (205 words avg) with cautious disclaimers and comprehensive information.

---

## 📈 Statistical Significance of Differences

| Metric | Human | AI | Significance | Interpretation |
|--------|-------|-----|--------------|----------------|
| **Word Count** | 96.6 | 205.2 | p < 0.001 *** | AI writes 2.1× longer responses |
| **Type-Token Ratio** | 0.746 | 0.534 | p < 0.001 *** | Human uses more varied vocabulary |
| **Noun Ratio** | 29.8% | 27.1% | p = 0.017 * | Human uses slightly more nouns |
| **Verb Ratio** | 14.2% | 13.0% | p = 0.023 * | Human uses more action verbs |
| **Nominalization** | 0.008 | 0.021 | p = 0.006 ** | AI uses 2.6× more nominalizations |

**Significance Levels**: *** p<0.001, ** p<0.01, * p<0.05

---

## 🎯 Key Conclusions

### 1. Length Difference ⭐⭐⭐

- AI responses are significantly longer (112% increase)
- This is the strongest discriminator between human and AI medical writing

### 2. Vocabulary Diversity ⭐⭐⭐

- Human doctors use richer vocabulary (TTR = 0.746 vs 0.534)
- AI tends to repeat formal phrases and disclaimers

### 3. Nominalization Pattern ⭐⭐

- Contradicts Zhang (2024) finding: In HC3 medical data, AI uses more nominalizations
- Suggests AI adopts more "formal" academic style in medical contexts

### 4. Linguistic Style

- **Human**: Direct, concise, empathetic, action-oriented
- **AI**: Formal, cautious, comprehensive, disclaimer-heavy

### 5. Practical Implications

- Log-odds ratios effectively identify linguistic fingerprints
- Keywords like "important", "healthcare provider" are strong AI signals
- Keywords like "hope", "hello", "suggest" are strong human signals

---
## 📝 Research Implications

### For Detection Models

- Combining keyword log-odds + length features + TTR would create robust AI text detector
- Simple logistic regression on top 20 keywords could achieve high accuracy

### For Content Analysis

- AI medical advice is more cautious and comprehensive but less personal
- Human medical advice is more direct and empathetic but less detailed

### For Zhang (2024) Comparison

- HC3 results show domain-specific differences
- Medical writing may have different patterns than general academic writing
- Nominalization trends may reverse depending on text type

---

## 💡 Bottom Line

Log-odds ratios successfully reveal that **AI medical writing is formal, verbose, and cautious**, while **human medical writing is concise, personal, and action-oriented**. The odds ratios provide strong discriminative power for distinguishing human from AI-generated medical text.

---

## 📖 References

Guo, B., Zhang, X., Wang, Z., Jiang, M., Nie, J., Ding, Y., Yue, J., & Wu, Y. (2023). How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection. *arXiv preprint arXiv:2301.07597*.