import os
import numpy as np
from dotenv import load_dotenv
import requests
import uuid


load_dotenv()


def cosine_similarity(a, b):
    """Computes cosine similarity between two vectors a and b."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    dot_prod = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_prod / (norm_a * norm_b))


def get_openrouter_embeddings(texts, model="openai/text-embedding-ada-002"):
    """
    Gets embeddings using OpenRouter API or compatible endpoint.
    OpenRouter API key is used.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/assessment",
        "X-Title": "VectorSearchAssessment"
    }
    payload = {
        "model": model,
        "input": texts
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data.get("data", [])]
    except Exception as e:
        print(f"OpenRouter embedding error: {e}. Using deterministic fallback embeddings for demo.")
        np.random.seed(42)
        return [np.random.randn(384).tolist() for _ in texts]


def generate_stable_id(text: str, source: str, chunk_index: int) -> str:
    """Generates stable UUID5 based on text content, source, and chunk index."""
    unique_string = f"{source}{chunk_index}{text[:50]}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))