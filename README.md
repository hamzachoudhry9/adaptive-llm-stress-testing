# LLM Stress Tester

A system to evaluate hallucinations, inconsistency, and evasion behavior in Large Language Models (LLMs) using RAG-based verification.

---

## 🚀 What This Project Does 

Large Language Models (LLMs) like Llama3 often give answers that *sound correct* but may be:

- ❌ Factually wrong (hallucination)
- ⚠️ Inconsistent (different answers to same question)
- 🚨 Evasive (avoid answering properly)

This project automatically tests and detects these behaviors.

---

### 🔍 Example

#### Step 1 — System reads knowledge (RAG)

From dataset:

```
South Korea has one of the highest tertiary education rates globally.
```

---

#### Step 2 — System generates a question

```
Which country has the highest tertiary education rate?
```

---

#### Step 3 — Model answers (multiple variations)

```
Answer 1: South Korea
Answer 2: Norway
Answer 3: Canada
```

---

#### Step 4 — System analyzes

```
Correct answer (from RAG): South Korea

→ Answer 1: Correct
→ Answer 2: Wrong
→ Answer 3: Wrong
```

---

### 📊 Final Result

```
Hallucination: YES
Reason: Model gave incorrect factual answers
Consistency: LOW (different answers)
```

---

### 💡 What this means

Even though the model sounds confident, it:

- ❌ Does not reliably know the correct fact
- ❌ Changes answers depending on phrasing
- ❌ Cannot be trusted for factual queries

---

## 🎯 Why this project is useful

Instead of just asking:

> "Is the model accurate?"

We answer:

- Where does the model fail?
- How often does it hallucinate?
- Is it stable across different question forms?
- Can we trust it for real-world use?

---

👉 This makes the system useful for:
- AI evaluation
- Model comparison
- Research and production readiness

---

## 🧠 System Flow (How It Works)

```
           ┌──────────────────────────────┐
           │   RAG Dataset (Knowledge)    │
           │  (JSONL domain data)         │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │  Question Generator (Mistral)│
           │  Creates domain-based        │
           │  questions from context      │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │  Test Model (Llama3)         │
           │  Answers WITHOUT context     │
           │  (simulates real usage)      │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │  Analyzer                    │
           │                              │
           │  ✓ RAG Verification          │
           │  ✓ Wikipedia Check           │
           │  ✓ Consistency Check         │
           │  ✓ Evasion Detection         │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │  Final Decision              │
           │                              │
           │  Hallucination? YES / NO     │
           │  Breakdown:                 │
           │   - Hallucination           │
           │   - Inconsistency           │
           │   - Evasion                 │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │  Bandit Learner              │
           │  Focuses on weak areas       │
           └──────────────────────────────┘
```

---

## 🔍 Step-by-Step Explanation

### 1. RAG (Knowledge Base)
- The system loads domain-specific knowledge (education, medical, etc.)
- Stored in `.jsonl` file
- Acts as **ground truth**

---

### 2. Question Generation (Mistral)
- Generates questions based on RAG context
- Ensures questions are:
  - factual
  - misleading
  - reasoning
  - ambiguous

---

### 3. Model Under Test (Llama3)
- Answers WITHOUT seeing the context
- This simulates **real-world usage**
- Forces the model to rely on its internal knowledge

---

### 4. Analysis Engine

Each answer is evaluated using:

#### ✅ RAG Check
- Is the answer correct according to the passage?

#### 🌐 Wikipedia Check
- External verification (for factual questions)

#### 🔁 Consistency Check
- Same question → multiple variations  
- Do answers match?

#### 🚨 Evasion Detection
- Did the model avoid answering?

---

### 5. Final Classification

Each question is classified into:

- **Hallucination** → wrong fact
- **Inconsistency** → unstable answers
- **Evasion** → avoided answering

---

### 6. Bandit Learning (Smart Testing)

- The system learns which category causes more failures
- Focuses more on those areas over time

---

## 🎯 Key Idea

This system doesn’t just test:

> "Is the model correct?"

It tests:

- Can the model be trusted?
- Does it stay consistent?
- Does it hallucinate under pressure?
- Does it avoid answering?

---

## 🧠 One-Line Summary

> This project simulates real-world questioning to detect when an LLM gives confident but incorrect answers — and explains why.

---

## 📁 Project Structure

```
llm-stress-tester/
│
├── main.py                 # Entry point
├── analyzer.py             # Verification logic
├── generator.py            # Question generation
├── rag.py                  # Retrieval system
├── bandit.py               # Adaptive testing
├── models.py               # Ollama model interface
├── config.py               # Configuration (IMPORTANT)
│
├── rag_llm_stress/
│   └── stem_contexts.jsonl # RAG dataset
│
├── results.jsonl           # Output results
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/llm-stress-tester.git
cd llm-stress-tester
```

---

### 2. Open in VS Code

```bash
code .
```

---

### 3. Install dependencies

```bash
pip install scikit-learn wikipedia requests
```

---

### 4. Start Ollama

```bash
ollama serve
```

---

### 5. Pull required models

```bash
ollama pull llama3
ollama pull mistral
```

---

## ▶️ Running the Project

```bash
python main.py
```

---

## 🔧 Changing Domain / Testing

👉 **This is the only file you need to modify**

Open:

```
config.py
```

---

### Change domain

```python
CURRENT_DOMAIN = "education"
```

---

### Change RAG dataset path

```python
RAG_FILE = "rag_llm_stress/stem_contexts.jsonl"
```

👉 Replace with your own dataset if needed

---

### Change number of test steps

```python
NUM_STEPS = 200
```

Examples:
- 100 → quick test
- 200 → standard
- 500+ → strong evaluation

---

## 📊 Output

After running, you will see:

```
HALLUCINATION REPORT
```

Example:

```
Overall hallucination rate : 7.5%

Failure Breakdown:
  Hallucination : 15
  Inconsistency : 40
  Evasion       : 10
```

---

## 🧠 Interpretation

- **Hallucination** → model gave wrong facts
- **Inconsistency** → model gave different answers for same question
- **Evasion** → model avoided answering

---

## 🎯 Key Insight

This system does NOT just measure accuracy.

It tells you:

- Where the model fails
- Why it fails
- Whether it is reliable for real-world use

---

## 🚀 Summary

This project helps you:

- Detect real hallucinations (not just inconsistency)
- Evaluate model reliability by domain
- Identify weak areas (e.g., factual vs reasoning)

---

## 📌 Tip

To test a new domain:

1. Create a new `.jsonl` RAG dataset
2. Update `config.py`
3. Run `main.py`

---
