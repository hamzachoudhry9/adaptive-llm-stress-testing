# Adaptive LLM Stress Tester

Detects hallucination, inconsistency, and evasion in Large Language Models using retrieval-augmented generation and a UCB1 multi-armed bandit that adaptively focuses testing on the model's weakest areas.

Evaluated Mistral-7B and LLaMA-3-8B across four real-world domains: Finance, Education, Medical Diagnostics, and Physics.

---

## The Problem

Standard accuracy benchmarks tell you whether a model got the right answer once. They do not tell you whether the model can be trusted in production. This project targets three failure modes that accuracy cannot capture:

- **Hallucination** -- the model states a wrong fact with confidence
- **Inconsistency** -- the model gives contradictory answers to the same question rephrased differently
- **Evasion** -- the model refuses or hedges so heavily the response becomes useless

The goal: build an adaptive agent that uncovers these failures faster than uniform random testing within a fixed evaluation budget.

---

## System Architecture

Six components run as a closed feedback loop:

```
Domain Knowledge Base (JSONL)
           |
           v
   UCB1 Bandit Policy
   Selects (domain, category) pair to probe next
   based on observed failure rates and exploration bonuses
           |
           v
   RAG Retrieval -- TF-IDF
   Fetches the most relevant passage from the knowledge base
   This passage is the factual ground truth for verification
           |
           v
   Question Generator -- Mistral-7B
   Reads the passage and writes a targeted adversarial question
   using a category-conditioned meta-prompt
           |
           v
   Model Under Test -- LLaMA-3-8B
   Answers the question with no access to the source passage
   Same question asked N times with paraphrasing for consistency testing
           |
           v
   Analysis Engine
   RAG verification + consistency scoring + evasion detection
   Returns a pass/fail signal per evaluation step
           |
           v
   Bandit Update
   Failure signal updates arm statistics
   Loop restarts with the bandit refocused on weaker areas
```

The generator (Mistral) and the model under test (LLaMA) are intentionally separated. LLaMA never sees the source passage, which simulates real-world deployment and prevents information leakage into the evaluation.

---

## Methodology

### Data Preparation and Knowledge Representation

Domain knowledge is stored as JSONL files, one text passage per line. Each file is indexed at startup using TF-IDF (scikit-learn's TfidfVectorizer with cosine similarity). The retriever fetches the passage most semantically relevant to the selected question category, which then serves as ground truth downstream in the analysis engine.

Data lineage is preserved end to end: every evaluation step logs which passage was retrieved, which question was generated from it, and how the model's answer was verified against it.

### Adaptive Sampling via UCB1 Bandit

Each (domain, category) pair is modeled as an arm in a multi-armed bandit. The UCB1 algorithm balances exploitation and exploration at every step:

```
UCB_score(arm) = failure_rate + sqrt( 2 * ln(total_steps) / pulls(arm) )
                 |__ exploit __|   |_________ explore bonus ______________|
```

The exploitation term favors arms with the highest observed failure rates. The exploration bonus prevents the system from ignoring untested categories that may be even weaker. The bonus shrinks as an arm is pulled more, letting empirical failure rates take over at scale.

Arms are tracked using Beta distribution parameters rather than a running mean. Alpha increments on a detected failure; beta increments on a pass. This gives the bandit calibrated uncertainty per arm from the first round.

```python
def update(self, domain, category, is_hallucination):
    if is_hallucination:
        self.alpha[(domain, category)] += 1
    else:
        self.beta[(domain, category)] += 1
    self.counts[(domain, category)] += 1
```

### Prompt Engineering for Question Generation

Mistral receives a structured meta-prompt specifying the retrieved passage, the target category, and strict output formatting constraints. This is the only prompt the generator sees -- there is no chain-of-thought leakage into the evaluation pipeline.

Four question categories are generated, each designed to probe a distinct failure surface:

| Category | Design intent |
|----------|---------------|
| Factual | Direct knowledge retrieval. Tests basic recall reliability. |
| Misleading | Embeds a false premise. Tests whether the model rejects it (sycophancy check). |
| Reasoning | Requires multi-step inference. Tests logical chain integrity. |
| Ambiguous | Underspecified intent. Tests whether the model handles uncertainty or hallucinates a confident answer. |

### Failure Detection

Three independent checks run on every response set:

**RAG Verification** computes token overlap between LLaMA's answer and the source passage. An answer that contradicts the passage or introduces facts not present in it is flagged as a hallucination.

**Consistency Checking** asks the same question multiple times with paraphrasing and measures pairwise agreement across responses. Low agreement flags the model as unreliable on that question regardless of whether any single answer was correct.

**Evasion Detection** scans responses for refusal signatures: phrases like "I don't have context", "I cannot determine", or responses so heavily hedged they contain no usable information.

All three signals are combined into a single pass/fail verdict that feeds back into the bandit.

### Experiment Documentation

Every evaluation step is written to `results.jsonl` with the full question, all responses, the verdict for each failure type, the arm selected, and current arm statistics. This makes every experiment fully reproducible and traceable from raw passage to final verdict.

---

## Results

| Domain | Steps | Hallucination | Inconsistency | Evasion |
|--------|-------|---------------|---------------|---------|
| Finance | 30 | 20.0% | 23.3% | 13.3% |
| Education | 50 | 6.0% | 8.0% | 2.0% |
| Medical | 200 | 3.0% | 3.5% | 2.5% |
| Physics | 200 | 4.0% | 8.0% | 3.0% |

**Domain dependence is the strongest finding.** Finance, which requires real-world causal reasoning under ambiguity, produced hallucination rates 6x higher than Medical or Physics, which involve structured factual knowledge.

**The bandit converges correctly.** In every domain it identified the weakest question category within the first 30 rounds and concentrated the remaining budget there. Uniform sampling would have wasted equal effort on categories where the model was already reliable.

**Sycophancy is LLaMA's biggest weakness.** On misleading prompts, the model accepted false premises in 67% of cases rather than identifying and rejecting them -- a significant reliability risk in any real deployment.

**Larger evaluation budgets stabilize estimates.** Finance at 30 steps produced a 95% confidence interval of 9.5% to 37.3%. Medical at 200 steps narrowed to 1.4% to 6.4%. Adaptive testing concentrates queries where variance matters most.

---

## Tech Stack

- **Python** -- all core logic, retrieval, bandit policy, and analysis
- **Mistral-7B** -- question generation and LLM-as-judge evaluation
- **LLaMA-3-8B** -- model under test
- **Ollama** -- local model serving with 4-bit quantization
- **scikit-learn** -- TF-IDF indexing and cosine similarity retrieval
- **JSONL** -- knowledge base storage and experiment logging

---

## Project Structure

```
llm-stress-tester/
|
+-- main.py           Entry point. Runs the adaptive loop and prints the hallucination report.
+-- bandit.py         UCB1Bandit class. Alpha/beta tracking, UCB scoring, arm selection.
+-- generator.py      Category-conditioned meta-prompts to Mistral for question generation.
+-- rag.py            TF-IDF index and retrieval over domain JSONL knowledge bases.
+-- analyzer.py       Hallucination, consistency, and evasion detection logic.
+-- models.py         Ollama API wrapper for Mistral and LLaMA.
+-- config.py         All settings. The only file that needs editing to test a new domain.
|
+-- rag_llm_stress/
|   +-- stem_contexts.jsonl   Default knowledge base.
|
+-- results.jsonl     Per-step experiment log with full traceability.
```

---

## Setup

Requirements: Python 3.9+, Ollama installed locally.

```bash
git clone https://github.com/hamzachoudhry9/LLM-Stress-Testing.git
cd LLM-Stress-Testing
pip install scikit-learn wikipedia requests
ollama serve
ollama pull llama3
ollama pull mistral
python main.py
```

---

## Configuration

All evaluation parameters live in `config.py`:

```python
CURRENT_DOMAIN = "medical"
RAG_FILE       = "rag_llm_stress/stem_contexts.jsonl"
NUM_STEPS      = 200
N_RESPONSES    = 3
DOMAINS        = ["finance", "education", "medical", "physics"]
CATEGORIES     = ["factual", "misleading", "reasoning", "ambiguous"]
```

To add a new domain, create a JSONL file with one passage per line, point `RAG_FILE` to it, and run. The bandit picks up the new arms automatically with no other changes required.

---

## Limitations

The consistency check uses word-overlap scoring, which can incorrectly penalize valid paraphrases. Replacing it with a sentence embedding similarity metric would improve precision. Using Mistral as both generator and judge also introduces potential bias; a dedicated judge model would produce a cleaner evaluation signal.
