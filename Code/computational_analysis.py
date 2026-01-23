import time
import psutil
import os

def measure_training_time_and_memory(model_fn, *args, **kwargs):
    """
    Measure training time and peak memory usage.
    Essential for discussing training budget and computational cost.
    """
    process = psutil.Process(os.getpid())
    
    # Measure memory before
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Time the training
    start_time = time.time()
    result = model_fn(*args, **kwargs)
    end_time = time.time()
    
    # Measure memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    training_time = end_time - start_time
    memory_used = mem_after - mem_before
    
    print(f"\n⏱️  Computational Cost:")
    print(f"   Training time: {training_time:.2f} seconds")
    print(f"   Memory usage: {memory_used:.2f} MB")
    
    return result, training_time, memory_used
