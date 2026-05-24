import numpy as np
import time
import faiss
from turbovec import TurboQuantIndex
from memory import get_memory_usage, calculate_overlap

def run_comparison():
    print("--- Vector Search Comparison: FAISS vs TurboVec ---")
    
    # Load embeddings
    try:
        embeddings = np.load("embeddings.npy").astype("float32")
    except FileNotFoundError:
        print("Error: embeddings.npy not found. Please run main.py first.")
        return

    num_vectors = len(embeddings)
    dim = embeddings.shape[1]
    print(f"Dataset: {num_vectors} vectors, {dim} dimensions\n")

    # --- FAISS Section ---
    print("[1] Building FAISS Index (IndexFlatIP)...")
    start_mem = get_memory_usage()
    faiss_index = faiss.IndexFlatIP(dim)
    start_time = time.time()
    faiss_index.add(embeddings)
    faiss_build_time = time.time() - start_time
    faiss_mem = get_memory_usage() - start_mem
    
    print(f"    Build Time: {faiss_build_time:.4f}s")
    print(f"    Memory Used: {faiss_mem:.2f} MB")

    # --- TurboVec Section ---
    print("\n[2] Building TurboVec Index (TurboQuantIndex 4-bit)...")
    start_mem = get_memory_usage()
    turbo_index = TurboQuantIndex(dim=dim, bit_width=4)
    start_time = time.time()
    turbo_index.add(embeddings)
    turbo_build_time = time.time() - start_time
    turbo_mem = get_memory_usage() - start_mem
    
    print(f"    Build Time: {turbo_build_time:.4f}s")
    print(f"    Memory Used: {turbo_mem:.2f} MB")

    # --- Search Comparison ---
    query = embeddings[0:1]
    k = 10
    print(f"\n[3] Searching Top {k} Neighbors for Query #0...")

    # FAISS Search
    start_time = time.time()
    _, faiss_indices = faiss_index.search(query, k)
    faiss_search_time = time.time() - start_time

    # TurboVec Search
    start_time = time.time()
    _, turbo_indices = turbo_index.search(query, k)
    turbo_search_time = time.time() - start_time

    print(f"    FAISS Search Time: {faiss_search_time:.6f}s")
    print(f"    TurboVec Search Time: {turbo_search_time:.6f}s")
    
    overlap = calculate_overlap(faiss_indices, turbo_indices)
    print(f"\n[4] Result Accuracy (Overlap with FAISS): {overlap:.2f}%")
    print(f"    FAISS Indices: {faiss_indices[0]}")
    print(f"    TurboVec Indices: {turbo_indices[0]}")

if __name__ == "__main__":
    run_comparison()
