"""
Models Module
Handles all calls to local Ollama models.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def query_model(model: str, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
    """Send a prompt to an Ollama model and return the response."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                }
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure it's running:\n"
            "  ollama serve\n"
            f"  ollama pull {model}"
        )
    except Exception as e:
        raise RuntimeError(f"Model query failed ({model}): {e}")