"""
STEM RAG Context Generator — Mac M-series (MLX)
Model  : mlx-community/Qwen2.5-7B-Instruct-4bit via mlx-lm
Output : stem_contexts.jsonl

Features:
- Checkpoint resume — if interrupted, restarts from where it left off
- Saves after every single sample
- Works with screen off (run via nohup)

Usage:
    python generate_rag_contexts_mac.py
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

from mlx_lm import load, generate

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID     = "mlx-community/Qwen2.5-7B-Instruct-4bit"
NUM_SAMPLES  = 500
OUTPUT_FILE  = "stem_contexts.jsonl"
LOG_FILE     = "generation.log"
MAX_TOKENS   = 600

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── STEM Topics ───────────────────────────────────────────────────────────────
STEM_TOPICS = [
    # Mathematics
    ("Mathematics", "Calculus",                  "The Fundamental Theorem of Calculus"),
    ("Mathematics", "Linear Algebra",             "Eigenvalues and eigenvectors"),
    ("Mathematics", "Statistics",                 "The Central Limit Theorem"),
    ("Mathematics", "Probability",                "Bayes' Theorem"),
    ("Mathematics", "Differential Equations",     "Ordinary vs partial differential equations"),
    # Physics
    ("Physics", "Quantum Mechanics",              "Wave-particle duality and the double-slit experiment"),
    ("Physics", "Thermodynamics",                 "The four laws of thermodynamics"),
    ("Physics", "Electromagnetism",               "Maxwell's equations"),
    ("Physics", "Special Relativity",             "Time dilation and length contraction"),
    ("Physics", "Classical Mechanics",            "Newton's laws of motion"),
    # Chemistry
    ("Chemistry", "Organic Chemistry",            "SN1 vs SN2 nucleophilic substitution reactions"),
    ("Chemistry", "Physical Chemistry",           "Gibbs free energy and reaction spontaneity"),
    ("Chemistry", "Electrochemistry",             "How galvanic cells generate electrical energy"),
    ("Chemistry", "Quantum Chemistry",            "The Schrödinger equation applied to hydrogen"),
    ("Chemistry", "Biochemistry",                 "DNA structure and information encoding"),
    # Biology
    ("Biology", "Cell Biology",                   "The role of mitochondria in cellular respiration"),
    ("Biology", "Genetics",                       "CRISPR-Cas9 gene editing mechanism"),
    ("Biology", "Evolutionary Biology",           "Natural selection and observable evolution"),
    ("Biology", "Neuroscience",                   "Synaptic transmission at chemical synapses"),
    ("Biology", "Molecular Biology",              "The central dogma of molecular biology"),
    # Computer Science
    ("Computer Science", "Algorithms",            "Time complexity and Big-O notation"),
    ("Computer Science", "Machine Learning",      "Gradient descent in neural network optimization"),
    ("Computer Science", "Data Structures",       "Hash tables vs binary search trees"),
    ("Computer Science", "Computer Architecture", "CPU caching and cache miss costs"),
    ("Computer Science", "Networking",            "The TCP/IP model and data transmission"),
    # Engineering
    ("Engineering", "Electrical Engineering",     "How transistors enable digital logic"),
    ("Engineering", "Mechanical Engineering",     "Stress-strain analysis in structural design"),
    ("Engineering", "Control Systems",            "PID controllers and their three components"),
    ("Engineering", "Signal Processing",          "The Fourier Transform and frequency analysis"),
    ("Engineering", "Materials Science",          "Crystalline vs amorphous solids"),
]

# ── Prompt ────────────────────────────────────────────────────────────────────
CONTEXT_PROMPT = """\
You are an expert educator in {field}, specializing in {topic}.

Write a precise, factually accurate educational passage (200-300 words) about:
"{concept}"

Requirements:
- University-level depth, written in clear prose (no bullet points)
- Include: core definition, key principles, and one concrete real-world example
- Be specific — avoid vague generalities
- Do NOT include a title or any preamble, just the passage itself
"""

# ── Checkpoint Helpers ────────────────────────────────────────────────────────
def load_completed_ids(output_path: Path) -> set:
    """Read already-completed sample IDs from existing JSONL file."""
    completed = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        completed.add(record["id"])
                    except json.JSONDecodeError:
                        continue
    return completed


# ── Generation ────────────────────────────────────────────────────────────────
def build_prompt(tokenizer, field: str, topic: str, concept: str) -> str:
    """Apply chat template to format the prompt correctly for the model."""
    messages = [{"role": "user", "content": CONTEXT_PROMPT.format(
        field=field, topic=topic, concept=concept
    )}]
    # Apply chat template without tokenizing
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def clean_output(text: str) -> str:
    """Remove any <think>...</think> reasoning tags if present."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    output_path = Path(OUTPUT_FILE)

    # Load completed IDs for checkpoint resume
    completed_ids = load_completed_ids(output_path)
    if completed_ids:
        log.info(f"Resuming — {len(completed_ids)} samples already done, skipping them.")
    else:
        log.info("Starting fresh generation.")

    # Load model
    log.info(f"Loading model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)
    log.info("Model ready.")

    # Build full topic pool (cycle topics to reach NUM_SAMPLES)
    topic_pool = (STEM_TOPICS * ((NUM_SAMPLES // len(STEM_TOPICS)) + 1))[:NUM_SAMPLES]

    # Open file in append mode — safe for resuming
    with open(output_path, "a", encoding="utf-8") as f:
        for i, (field, topic, concept) in enumerate(topic_pool):
            sample_id = f"stem_{i+1:04d}"

            # Skip if already done (checkpoint resume)
            if sample_id in completed_ids:
                log.info(f"[{i+1}/{NUM_SAMPLES}] Skipping {sample_id} (already done)")
                continue

            log.info(f"[{i+1}/{NUM_SAMPLES}] {field} > {topic} — {concept}")

            try:
                prompt  = build_prompt(tokenizer, field, topic, concept)
                passage = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=MAX_TOKENS,
                    verbose=False,
                )
                passage = clean_output(passage)

                if not passage:
                    log.warning(f"  Empty output for {sample_id}, skipping.")
                    continue

                record = {
                    "id":           sample_id,
                    "field":        field,
                    "topic":        topic,
                    "concept":      concept,
                    "context":      passage,
                    "generated_at": datetime.utcnow().isoformat(),
                    "model":        MODEL_ID,
                }

                # Write immediately — this is the checkpoint
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()   # force write to disk right away
                os.fsync(f.fileno())

                log.info(f"  ✓ Saved ({len(passage.split())} words)")

            except Exception as e:
                log.error(f"  Error on {sample_id}: {e}, skipping.")
                continue

            time.sleep(0.1)

    log.info(f"\nAll done! {NUM_SAMPLES} passages saved to {output_path}")


if __name__ == "__main__":
    main()
