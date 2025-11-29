import time
import numpy as np
import pycatch22

def benchmark():
    lengths = [1000, 10000, 100000, 500000, 1000000]
    times = []
    
    print(f"{'Length':<15} | {'Time (s)':<15} | {'Rate (pts/s)':<15}")
    print("-" * 50)
    
    for N in lengths:
        # Generate random signal
        sig = np.random.randn(N)
        
        start = time.perf_counter()
        # Run catch22
        _ = pycatch22.catch22_all(sig, catch24=True)
        end = time.perf_counter()
        
        duration = end - start
        times.append(duration)
        rate = N / duration
        
        print(f"{N:<15} | {duration:<15.4f} | {rate:<15.2f}")

    # Check scaling
    print("\nScaling Analysis:")
    for i in range(1, len(lengths)):
        n_ratio = lengths[i] / lengths[i-1]
        t_ratio = times[i] / times[i-1]
        print(f"Increase N by {n_ratio:.1f}x -> Time increases by {t_ratio:.1f}x (Expected Linear: {n_ratio:.1f}x)")

if __name__ == "__main__":
    benchmark()
