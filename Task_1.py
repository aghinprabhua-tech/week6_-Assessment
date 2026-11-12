import os
import time
import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


# Load environment variables from .env file
load_dotenv()


def custom_cosine_similarity(a, b):
    """Computes cosine similarity between two vectors a and b from scratch."""
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
    Gets embeddings using OpenRouter API directly without external utils.
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


def run_task1():
    print("=== TASK 1: Embeddings and Cosine Similarity ===")
    
    # 1. 10 Sentences
    sentences = [
        "Artificial intelligence is transforming modern industries.",
        "Machine learning models require substantial training data.",
        "Python is a popular programming language for data science.",
        "Natural language processing enables computers to understand text.",
        "Vector databases store high-dimensional embeddings efficiently.",
        "The quick brown fox jumps over the lazy dog.",
        "A recipe for chocolate chip cookies with butter and sugar.",
        "Global financial markets reacted strongly to the interest rate cut.",
        "Regular physical exercise improves cardiovascular health.",
        "Deep learning architectures include transformers and CNNs."
    ]
    
    # Local model (all-MiniLM-L6-v2)[span_0](start_span)[span_0](end_span)
    start_time = time.time()
    local_model = SentenceTransformer("all-MiniLM-L6-v2")
    local_embeddings = local_model.encode(sentences)
    local_time = time.time() - start_time
    print(f"Local Model Dimensions: {local_embeddings.shape}, Time: {local_time:.4f}s")
    
    # API embedding model via OpenRouter
    start_time = time.time()
    api_embeddings = get_openrouter_embeddings(sentences, model="openai/text-embedding-ada-002")
    api_time = time.time() - start_time
    print(f"API Model Dimensions: ({len(api_embeddings)}, {len(api_embeddings[0])}), Time: {api_time:.4f}s")
    
    # 2. Implement cosine_similarity test cases[span_1](start_span)[span_1](end_span)
    print("\n--- Cosine Similarity Tests ---")
    t1 = custom_cosine_similarity([1,2,3], [2,4,6])
    t2 = custom_cosine_similarity([1,0], [0,1])
    t3 = custom_cosine_similarity([1,0], [-1,0])
    print(f"[1,2,3] vs [2,4,6] -> {t1:.1f} (Expected: 1.0)")
    print(f"[1,0] vs [0,1] -> {t2:.1f} (Expected: 0.0)")
    print(f"[1,0] vs [-1,0] -> {t3:.1f} (Expected: -1.0)")
    
    # Compare with scikit-learn
    sk_val = sk_cosine_similarity([[1,2,3]], [[2,4,6]])[0,0]
    print(f"Scikit-learn check [1,2,3] vs [2,4,6]: {sk_val:.1f}")
    
    # 3. Similarity matrix and top 3 similar pairs[span_2](start_span)[span_2](end_span)
    sim_matrix = sk_cosine_similarity(local_embeddings)
    print(f"\nSimilarity Matrix Shape: {sim_matrix.shape}")
    
    pairs = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            pairs.append((sim_matrix[i, j], i, j))
    pairs.sort(reverse=True)
    
    print("\nTop 3 Most Similar Pairs:")
    for score, i, j in pairs[:3]:
        print(f"Score: {score:.4f}\n  - '{sentences[i]}'\n  - '{sentences[j]}'")
    
    # 4. Use cases: Semantic search, Duplicate detection, Clustering[span_3](start_span)[span_3](end_span)
    print("\n--- Use Cases ---")
    queries = ["Tell me about AI and ML", "How to bake sweets", "Heart fitness routines"]
    query_embeddings = local_model.encode(queries)
    for q_idx, q in enumerate(queries):
        sims = sk_cosine_similarity([query_embeddings[q_idx]], local_embeddings)[0]
        best_match = np.argmax(sims)
        print(f"Query: '{q}' -> Best match: '{sentences[best_match]}' (Score: {sims[best_match]:.4f})")
        
    print("\nDuplicate Detection (threshold > 0.85):")
    duplicates_found = False
    for score, i, j in pairs:
        if score > 0.85:
            print(f"Potential Duplicate (score {score:.4f}): '{sentences[i]}' & '{sentences[j]}'")
            duplicates_found = True
    if not duplicates_found:
        print("No duplicates above threshold in this small dataset.")
        
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(local_embeddings)
    print(f"K-Means Cluster assignments: {kmeans.labels_}")
    
    # 5. Choice between local and API embedding model[span_4](start_span)[span_4](end_span)
    print("\n--- Choice Between Local and API Embedding Models ---")
    print("Choosing between local models (like SentenceTransformers) and APIs depends on latency, privacy, cost, and infrastructure. Local models offer zero network latency, complete data privacy, and no per-token API costs. API models eliminate local hardware requirements and provide maintenance-free scaling at the expense of API costs and network dependency.")


if __name__ == "__main__":
    run_task1()