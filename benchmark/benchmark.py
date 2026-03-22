"""
RAM Coffers Benchmark Feature (#49)
"""
import time

def run_benchmark(iterations=1000):
    start = time.time()
    for i in range(iterations):
        pass
    end = time.time()
    return {"duration": end - start, "iterations": iterations}

if __name__ == "__main__":
    result = run_benchmark()
    print(result)
