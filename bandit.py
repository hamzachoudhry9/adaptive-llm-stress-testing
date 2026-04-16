"""
Bandit Module
UCB1 bandit that learns which (domain, category) combinations
expose the most hallucinations in the tested model.
Over time it focuses testing on the model's weakest areas.
"""

import math
from config import DOMAINS, CATEGORIES


class UCB1Bandit:
    def __init__(self):
        self.alpha  = {(d, c): 1 for d in DOMAINS for c in CATEGORIES}
        self.beta   = {(d, c): 1 for d in DOMAINS for c in CATEGORIES}
        self.counts = {(d, c): 0 for d in DOMAINS for c in CATEGORIES}
        self.total  = 0

    def _ucb_score(self, d: str, c: str) -> float:
        n = self.counts[(d, c)]
        if n == 0:
            return float("inf")
        mean  = self.alpha[(d, c)] / (self.alpha[(d, c)] + self.beta[(d, c)])
        bonus = math.sqrt((2 * math.log(self.total)) / n)
        return mean + bonus

    def select(self) -> tuple[str, str]:
        """Select the (domain, category) with highest UCB score."""
        self.total += 1
        scores = {(d, c): self._ucb_score(d, c) for d in DOMAINS for c in CATEGORIES}
        return max(scores, key=scores.get)

    def update(self, domain: str, category: str, is_hallucination: bool):
        """
        Update bandit:
        - hallucination = reward (we want to find more of these)
        - no hallucination = no reward
        """
        self.counts[(domain, category)] += 1
        if is_hallucination:
            self.alpha[(domain, category)] += 1   # success = found hallucination
        else:
            self.beta[(domain, category)]  += 1   # failure = no hallucination found

    def stats(self) -> dict:
        """Return hallucination rates per (domain, category)."""
        out = {}
        for d in DOMAINS:
            for c in CATEGORIES:
                total = self.alpha[(d, c)] + self.beta[(d, c)]
                out[(d, c)] = {
                    "hallucination_rate": round(self.alpha[(d, c)] / total, 2),
                    "samples": self.counts[(d, c)],
                }
        return out

    def weakest_areas(self) -> list:
        """Return categories sorted by hallucination rate (highest first)."""
        stats = self.stats()
        ranked = sorted(stats.items(), key=lambda x: x[1]["hallucination_rate"], reverse=True)
        return [(k, v) for k, v in ranked if v["samples"] > 0]