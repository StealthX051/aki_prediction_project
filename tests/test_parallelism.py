
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from joblib import Parallel, delayed

def benchmark_lr(n_jobs, n_samples=2000, n_features=10000):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
    
    start = time.time()
    clf = LogisticRegression(n_jobs=n_jobs, solver='lbfgs', max_iter=1000)
    clf.fit(X, y)
    end = time.time()
    return end - start

def benchmark_parallel_loop(n_jobs_loop, n_jobs_lr, n_iter=8):
    X, y = make_classification(n_samples=2000, n_features=10000, random_state=42)
    
    def run_one():
        clf = LogisticRegression(n_jobs=n_jobs_lr, solver='lbfgs', max_iter=1000)
        clf.fit(X, y)
        
    start = time.time()
    Parallel(n_jobs=n_jobs_loop)(delayed(run_one)() for _ in range(n_iter))
    end = time.time()
    return end - start

if __name__ == "__main__":
    print("Benchmarking LogisticRegression(n_jobs)...")
    t_seq = benchmark_lr(n_jobs=1)
    print(f"LR(n_jobs=1): {t_seq:.2f}s")
    
    t_par = benchmark_lr(n_jobs=-1)
    print(f"LR(n_jobs=-1): {t_par:.2f}s")
    
    speedup = t_seq / t_par
    print(f"Speedup: {speedup:.2f}x")
    
    if speedup < 1.5:
        print("\nWARNING: LR parallelization is ineffective for single fit.")
        print("Testing Parallel Loop vs Sequential Loop...")
        
        # Simulate HPO
        t_loop_seq = benchmark_parallel_loop(n_jobs_loop=1, n_jobs_lr=1, n_iter=4)
        print(f"Sequential Loop (n_jobs=1, LR_jobs=1): {t_loop_seq:.2f}s")
        
        t_loop_par = benchmark_parallel_loop(n_jobs_loop=-1, n_jobs_lr=1, n_iter=4)
        print(f"Parallel Loop (n_jobs=-1, LR_jobs=1): {t_loop_par:.2f}s")
        
        loop_speedup = t_loop_seq / t_loop_par
        print(f"Loop Speedup: {loop_speedup:.2f}x")
    
