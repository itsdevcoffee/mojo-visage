"""
Mojo Showcase - Performance Examples
Demonstrates progressive performance optimization in Mojo
"""

from time import perf_counter_ns

# Python-style (dynamic, slow)
def python_style_sum(n: Int) -> Int:
    var total = 0
    for i in range(n):
        total += i
    return total

# Mojo-style with `fn` (compiled, fast)
fn mojo_style_sum(n: Int) -> Int:
    var total = 0
    for i in range(n):
        total += i
    return total

fn main() raises:
    print("🔥 Mojo Performance Showcase 🔥")
    print("=" * 50)

    var n = 10_000_000
    print("Summing numbers from 0 to", n)
    print()

    # Benchmark python-style def
    var start = perf_counter_ns()
    var result1 = python_style_sum(n)
    var end = perf_counter_ns()
    var duration_ms1 = Float64(end - start) / 1_000_000.0
    print("Python-style (def):", result1, "in", duration_ms1, "ms")

    # Benchmark mojo-style fn
    start = perf_counter_ns()
    var result2 = mojo_style_sum(n)
    end = perf_counter_ns()
    var duration_ms2 = Float64(end - start) / 1_000_000.0
    print("Mojo-style (fn):  ", result2, "in", duration_ms2, "ms")

    print()
    var speedup = duration_ms1 / duration_ms2
    print("Speedup:", speedup, "x faster!")
    print()
    print("Key takeaways:")
    print("1. Use 'fn' for performance-critical code")
    print("2. Use 'def' for Python-like flexibility")
    print("3. Mojo gives you both ease AND speed")
    print()
    print("Ready to build ML from scratch? Let's go! 🚀")
