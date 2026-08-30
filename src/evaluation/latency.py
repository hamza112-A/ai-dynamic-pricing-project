import time
import numpy as np
import torch
import os
import sys
from pathlib import Path
import json
import psutil


def measure_inference_time(model, X_test, n_runs=5, warmup_runs=2, batch_size=1):
    
    print(f"\nMeasuring inference time ({n_runs} runs, batch_size={batch_size})...")
    
    n_samples = len(X_test)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Warmup runs
    print("  Warming up...")
    for _ in range(warmup_runs):
        _ = model.predict(X_test[:100])
    
    # Timed runs
    print("  Timing...")
    run_times = []
    per_sample_times = []
    
    for run in range(n_runs):
        batch_times = []
        
        for i in range(0, n_samples, batch_size):
            batch = X_test[i:i+batch_size]
            
            start = time.perf_counter()
            _ = model.predict(batch)
            end = time.perf_counter()
            
            batch_time = end - start
            batch_times.append(batch_time)
        
        total_time = sum(batch_times)
        run_times.append(total_time)
        
        # Compute per-sample time
        per_sample_time = total_time / n_samples
        per_sample_times.extend([per_sample_time * 1000] * n_samples)  # Convert to ms
    
    per_sample_times = np.array(per_sample_times)
    
    results = {
        'mean_time_ms': float(per_sample_times.mean()),
        'std_time_ms': float(per_sample_times.std()),
        'median_time_ms': float(np.median(per_sample_times)),
        'p95_time_ms': float(np.percentile(per_sample_times, 95)),
        'p99_time_ms': float(np.percentile(per_sample_times, 99)),
        'min_time_ms': float(per_sample_times.min()),
        'max_time_ms': float(per_sample_times.max()),
        'total_samples': n_samples,
        'n_runs': n_runs,
        'batch_size': batch_size
    }
    
    return results


def measure_throughput(model, X_test, duration=10.0):
    
    print(f"\nMeasuring throughput ({duration}s test)...")
    
    n_samples = len(X_test)
    samples_processed = 0
    
    start = time.perf_counter()
    elapsed = 0
    
    while elapsed < duration:
        # Process samples in batches
        batch_size = min(1000, n_samples)
        idx = samples_processed % n_samples
        batch = X_test[idx:idx+batch_size]
        
        _ = model.predict(batch)
        samples_processed += len(batch)
        
        elapsed = time.perf_counter() - start
    
    throughput = samples_processed / elapsed
    
    print(f"  Processed {samples_processed:,} samples in {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.2f} flows/s")
    
    return {
        'throughput_fps': float(throughput),
        'samples_processed': samples_processed,
        'duration_seconds': elapsed
    }


def measure_model_size(model_path):
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        return {'size_bytes': 0, 'size_mb': 0, 'size_human': '0 B'}
    
    total_size = 0
    
    if model_path.is_file():
        total_size = model_path.stat().st_size
    else:
        for file in model_path.rglob('*'):
            if file.is_file():
                total_size += file.stat().st_size
    
    size_mb = total_size / (1024 * 1024)
    
    # Human readable
    if total_size < 1024:
        size_human = f"{total_size} B"
    elif total_size < 1024 * 1024:
        size_human = f"{total_size / 1024:.2f} KB"
    elif total_size < 1024 * 1024 * 1024:
        size_human = f"{size_mb:.2f} MB"
    else:
        size_human = f"{total_size / (1024 * 1024 * 1024):.2f} GB"
    
    return {
        'size_bytes': total_size,
        'size_mb': size_mb,
        'size_human': size_human
    }


def measure_memory_usage(model, X_sample):
    
    process = psutil.Process()
    
    # Get baseline memory
    baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Run inference
    _ = model.predict(X_sample)
    
    # Get peak memory
    peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    memory_used = peak_memory - baseline_memory
    
    return {
        'baseline_mb': baseline_memory,
        'peak_mb': peak_memory,
        'used_mb': memory_used
    }


def comprehensive_latency_benchmark(
    model,
    X_test,
    model_path=None,
    model_name="Model",
    save_path=None
):
    
    print("\n" + "=" * 70)
    print(f"Comprehensive Latency Benchmark: {model_name}")
    print("=" * 70)
    
    results = {'model_name': model_name}
    
    # 1. Per-flow inference time (batch_size=1)
    print("\n1. Per-Flow Inference Time")
    print("-" * 70)
    timing_single = measure_inference_time(model, X_test, n_runs=5, batch_size=1)
    results['per_flow_inference'] = timing_single
    
    print(f"  Mean: {timing_single['mean_time_ms']:.4f} ms")
    print(f"  P95:  {timing_single['p95_time_ms']:.4f} ms")
    print(f"  P99:  {timing_single['p99_time_ms']:.4f} ms")
    
    # 2. Batch inference time (batch_size=512)
    print("\n2. Batch Inference Time (batch_size=512)")
    print("-" * 70)
    timing_batch = measure_inference_time(model, X_test, n_runs=5, batch_size=512)
    results['batch_inference'] = timing_batch
    
    print(f"  Mean: {timing_batch['mean_time_ms']:.4f} ms per sample")
    
    # 3. Throughput
    print("\n3. Throughput")
    print("-" * 70)
    throughput = measure_throughput(model, X_test, duration=10.0)
    results['throughput'] = throughput
    
    print(f"  Throughput: {throughput['throughput_fps']:.2f} flows/s")
    
    # 4. Model size
    if model_path is not None:
        print("\n4. Model Size")
        print("-" * 70)
        size = measure_model_size(model_path)
        results['model_size'] = size
        
        print(f"  Disk size: {size['size_human']}")
    
    # 5. Memory usage
    print("\n5. Memory Usage")
    print("-" * 70)
    memory = measure_memory_usage(model, X_test[:1000])
    results['memory'] = memory
    
    print(f"  Peak memory: {memory['peak_mb']:.2f} MB")
    print(f"  Memory used: {memory['used_mb']:.2f} MB")
    
    # Save results
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {save_path}")
    
    return results


def compare_latency(results_list):
    
    import pandas as pd
    
    comparison = []
    
    for result in results_list:
        comparison.append({
            'Model': result['model_name'],
            'Mean Latency (ms)': result['per_flow_inference']['mean_time_ms'],
            'P95 Latency (ms)': result['per_flow_inference']['p95_time_ms'],
            'Throughput (flows/s)': result['throughput']['throughput_fps'],
            'Model Size (MB)': result.get('model_size', {}).get('size_mb', 0),
            'Memory (MB)': result['memory']['used_mb']
        })
    
    df = pd.DataFrame(comparison)
    
    print("\n" + "=" * 100)
    print("Latency Comparison")
    print("=" * 100)
    print(df.to_string(index=False))
    
    return df


def print_latency_summary(results):
    
    print("\n" + "=" * 70)
    print(f"Latency Summary: {results['model_name']}")
    print("=" * 70)
    
    print(f"\nInference Time (per flow):")
    print(f"  Mean:   {results['per_flow_inference']['mean_time_ms']:.4f} ms")
    print(f"  Median: {results['per_flow_inference']['median_time_ms']:.4f} ms")
    print(f"  P95:    {results['per_flow_inference']['p95_time_ms']:.4f} ms")
    print(f"  P99:    {results['per_flow_inference']['p99_time_ms']:.4f} ms")
    
    print(f"\nThroughput:")
    print(f"  {results['throughput']['throughput_fps']:.2f} flows/second")
    
    if 'model_size' in results:
        print(f"\nModel Size:")
        print(f"  {results['model_size']['size_human']}")
    
    print(f"\nMemory Usage:")
    print(f"  {results['memory']['used_mb']:.2f} MB")


if __name__ == "__main__":
    # Quick test
    print("Testing latency benchmarking...")
    
    # Create a dummy model
    class DummyModel:
        def predict(self, X):
            # Simulate some computation
            time.sleep(0.0001 * len(X))
            return np.random.randint(0, 5, len(X))
    
    model = DummyModel()
    X_test = np.random.randn(1000, 50)
    
    # Run benchmark
    results = comprehensive_latency_benchmark(
        model,
        X_test,
        model_name="Dummy Model"
    )
    
    # Print summary
    print_latency_summary(results)
    
    print("\n✓ Latency benchmarking test passed")
