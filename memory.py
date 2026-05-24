import psutil
import os
import numpy as np
import time

def get_memory_usage():
    """Returns the current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def calculate_overlap(ids1, ids2):
    """Calculates the percentage of overlap between two sets of indices."""
    if not ids1 or not ids2:
        return 0.0
    set1 = set(ids1.flatten() if isinstance(ids1, np.ndarray) else ids1)
    set2 = set(ids2.flatten() if isinstance(ids2, np.ndarray) else ids2)
    overlap = len(set1 & set2) / len(set1) * 100
    return overlap

if __name__ == "__main__":
    print(f"Current Process Memory Usage: {get_memory_usage():.2f} MB")
    
    # This script can be expanded to run both and compare memory
    print("\nNote: To compare memory usage between FAISS and TurboVec,")
    print("      import these functions in a comparison script.")