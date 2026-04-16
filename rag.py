"""
RAG Module
Loads the JSONL context file and retrieves the most relevant
passages for a given question using TF-IDF cosine similarity.
"""
 
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
 
class RAG:
    def __init__(self, jsonl_path: str):
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(
                f"RAG file not found: {jsonl_path}\n"
                f"Run generate_rag_contexts_mac.py first to create it."
            )
 
        self.records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
 
        if not self.records:
            raise ValueError(f"RAG file is empty: {jsonl_path}")
 
        # Build TF-IDF index over all context passages
        self.texts = [r["context"] for r in self.records]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)
 
        print(f"[RAG] Loaded {len(self.records)} documents from {jsonl_path}")
 
    def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        """Return top_k most relevant records for a query."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vectors)[0]
        top_indices = scores.argsort()[-top_k:][::-1]
        return [self.records[i] for i in top_indices]
 
    def retrieve_by_topic(self, field: str = None, topic: str = None) -> list[dict]:
        """Return all records matching a field or topic."""
        results = []
        for r in self.records:
            if field and r.get("field") == field:
                results.append(r)
            elif topic and r.get("topic") == topic:
                results.append(r)
        return results
 
    def random_record(self) -> dict:
        """Return a random record (used for bandit exploration)."""
        import random
        return random.choice(self.records)