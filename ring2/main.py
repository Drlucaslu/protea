#!/usr/bin/env python3

import os
import pathlib
import time
import json
import sys
from threading import Thread, Event
from datetime import datetime
from typing import Dict, List, Any, Tuple, Set
import signal
import math
import random
from collections import defaultdict, deque
from itertools import islice, combinations

HEARTBEAT_INTERVAL = 2

def heartbeat_loop(heartbeat_path: pathlib.Path, pid: int, stop_event: Event) -> None:
    while not stop_event.is_set():
        try:
            heartbeat_path.write_text(f"{pid}\n{time.time()}\n")
        except Exception:
            pass
        time.sleep(HEARTBEAT_INTERVAL)

class NumberTheoryEngine:
    """Explores prime numbers, sequences, and mathematical patterns"""
    
    def __init__(self):
        self.primes_cache = []
        self.sequence_cache = {}
        
    def sieve_of_eratosthenes(self, limit: int) -> List[int]:
        """Generate all primes up to limit using sieve algorithm"""
        if limit < 2:
            return []
        
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i in range(limit + 1) if sieve[i]]
    
    def goldbach_conjecture(self, n: int) -> List[Tuple[int, int]]:
        """Find all ways to express even n as sum of two primes"""
        if n < 4 or n % 2 != 0:
            return []
        
        if not self.primes_cache or self.primes_cache[-1] < n:
            self.primes_cache = self.sieve_of_eratosthenes(n)
        
        prime_set = set(self.primes_cache)
        pairs = []
        
        for p in self.primes_cache:
            if p > n // 2:
                break
            complement = n - p
            if complement in prime_set:
                pairs.append((p, complement))
        
        return pairs
    
    def collatz_sequence(self, n: int, max_steps: int = 1000) -> List[int]:
        """Generate Collatz sequence starting from n"""
        sequence = [n]
        current = n
        steps = 0
        
        while current != 1 and steps < max_steps:
            if current % 2 == 0:
                current = current // 2
            else:
                current = 3 * current + 1
            sequence.append(current)
            steps += 1
        
        return sequence
    
    def fibonacci_modular(self, n: int, mod: int) -> List[int]:
        """Generate Fibonacci sequence modulo mod"""
        if n <= 0:
            return []
        
        fib = [0, 1]
        for i in range(2, n):
            fib.append((fib[i-1] + fib[i-2]) % mod)
        
        return fib[:n]
    
    def prime_gaps(self, limit: int) -> Dict[int, int]:
        """Analyze gaps between consecutive primes"""
        if not self.primes_cache or self.primes_cache[-1] < limit:
            self.primes_cache = self.sieve_of_eratosthenes(limit)
        
        gaps = defaultdict(int)
        for i in range(1, len(self.primes_cache)):
            gap = self.primes_cache[i] - self.primes_cache[i-1]
            gaps[gap] += 1
        
        return dict(gaps)
    
    def perfect_numbers(self, limit: int) -> List[int]:
        """Find perfect numbers (sum of divisors equals the number)"""
        perfect = []
        
        for n in range(2, limit):
            divisors_sum = sum(i for i in range(1, n) if n % i == 0)
            if divisors_sum == n:
                perfect.append(n)
        
        return perfect
    
    def twin_primes(self, limit: int) -> List[Tuple[int, int]]:
        """Find twin prime pairs (primes that differ by 2)"""
        if not self.primes_cache or self.primes_cache[-1] < limit:
            self.primes_cache = self.sieve_of_eratosthenes(limit)
        
        twins = []
        for i in range(len(self.primes_cache) - 1):
            if self.primes_cache[i+1] - self.primes_cache[i] == 2:
                twins.append((self.primes_cache[i], self.primes_cache[i+1]))
        
        return twins

class FractalGenerator:
    """Generate ASCII fractals and mathematical patterns"""
    
    def mandelbrot_set(self, width: int, height: int, max_iter: int = 50) -> List[str]:
        """Generate ASCII Mandelbrot set"""
        chars = " .:-=+*#%@"
        result = []
        
        for y in range(height):
            line = ""
            for x in range(width):
                # Map pixel to complex plane
                c = complex(
                    (x - width * 0.7) / (width * 0.3),
                    (y - height * 0.5) / (height * 0.4)
                )
                
                z = 0
                iteration = 0
                
                while abs(z) <= 2 and iteration < max_iter:
                    z = z * z + c
                    iteration += 1
                
                char_idx = min(iteration * len(chars) // max_iter, len(chars) - 1)
                line += chars[char_idx]
            
            result.append(line)
        
        return result
    
    def sierpinski_triangle(self, size: int) -> List[str]:
        """Generate Sierpinski triangle using chaos game"""
        grid = [[' ' for _ in range(size * 2)] for _ in range(size)]
        
        # Vertices of the triangle
        vertices = [
            (size, 0),           # Top
            (0, size - 1),       # Bottom left
            (size * 2 - 1, size - 1)  # Bottom right
        ]
        
        # Start at random point
        x, y = size, size // 2
        
        for _ in range(size * size * 2):
            # Pick random vertex
            vx, vy = random.choice(vertices)
            
            # Move halfway to vertex
            x = (x + vx) // 2
            y = (y + vy) // 2
            
            if 0 <= y < size and 0 <= x < len(grid[0]):
                grid[y][x] = '█'
        
        return [''.join(row) for row in grid]
    
    def pascal_triangle(self, rows: int) -> List[str]:
        """Generate Pascal's triangle"""
        triangle = []
        
        for i in range(rows):
            row = [1]
            if triangle:
                last_row = [int(x) for line in triangle[-1].split() for x in [line]]
                for j in range(len(last_row) - 1):
                    row.append(last_row[j] + last_row[j + 1])
                row.append(1)
            
            # Format with spacing
            spaces = ' ' * (rows - i)
            nums = ' '.join(str(n).rjust(4) for n in row)
            triangle.append(spaces + nums)
        
        return triangle

class SequenceAnalyzer:
    """Analyze and generate integer sequences"""
    
    def look_and_say(self, start: str, iterations: int) -> List[str]:
        """Generate look-and-say sequence (Conway sequence)"""
        sequence = [start]
        current = start
        
        for _ in range(iterations):
            next_term = ""
            i = 0
            
            while i < len(current):
                digit = current[i]
                count = 1
                
                while i + count < len(current) and current[i + count] == digit:
                    count += 1
                
                next_term += str(count) + digit
                i += count
            
            sequence.append(next_term)
            current = next_term
        
        return sequence
    
    def recaman_sequence(self, n: int) -> List[int]:
        """Generate Recamán's sequence"""
        sequence = [0]
        seen = {0}
        
        for i in range(1, n):
            candidate = sequence[i-1] - i
            
            if candidate > 0 and candidate not in seen:
                sequence.append(candidate)
                seen.add(candidate)
            else:
                candidate = sequence[i-1] + i
                sequence.append(candidate)
                seen.add(candidate)
        
        return sequence
    
    def catalan_numbers(self, n: int) -> List[int]:
        """Generate Catalan numbers"""
        if n <= 0:
            return []
        
        catalan = [1]
        
        for i in range(1, n):
            catalan.append(
                catalan[i-1] * 2 * (2 * i - 1) // (i + 1)
            )
        
        return catalan

def main() -> None:
    heartbeat_path_str = os.environ.get("PROTEA_HEARTBEAT")
    if not heartbeat_path_str:
        print("ERROR: PROTEA_HEARTBEAT not set", flush=True)
        return

    heartbeat_path = pathlib.Path(heartbeat_path_str)
    pid = os.getpid()

    try:
        heartbeat_path.write_text(f"{pid}\n{time.time()}\n")
    except Exception as e:
        print(f"ERROR: Heartbeat failed: {e}", flush=True)
        return

    stop_event = Event()

    def signal_handler(signum, frame):
        print(f"\n⚠️  Signal {signum} received, shutting down", flush=True)
        stop_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    heartbeat_thread = Thread(
        target=heartbeat_loop,
        args=(heartbeat_path, pid, stop_event),
        daemon=True
    )
    heartbeat_thread.start()

    workspace = pathlib.Path(__file__).parent

    print("╔" + "═" * 78 + "╗", flush=True)
    print("║" + "COMPUTATIONAL MATHEMATICS ENGINE".center(78) + "║", flush=True)
    print("║" + "Number Theory · Fractals · Sequences".center(78) + "║", flush=True)
    print("╚" + "═" * 78 + "╝", flush=True)
    print(f"🔢 PID: {pid} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("", flush=True)

    nt = NumberTheoryEngine()
    fg = FractalGenerator()
    sa = SequenceAnalyzer()

    try:
        # Prime number analysis
        print("═" * 80, flush=True)
        print("PRIME NUMBER ANALYSIS", flush=True)
        print("═" * 80, flush=True)
        
        primes = nt.sieve_of_eratosthenes(1000)
        print(f"✓ Found {len(primes)} primes up to 1000", flush=True)
        print(f"  First 20: {primes[:20]}", flush=True)
        print(f"  Last 10:  {primes[-10:]}", flush=True)
        print("", flush=True)
        
        # Twin primes
        twins = nt.twin_primes(500)
        print(f"🔗 Twin primes up to 500: {len(twins)} pairs", flush=True)
        for i, (p1, p2) in enumerate(twins[:8]):
            print(f"  {i+1}. ({p1}, {p2})", flush=True)
        print("", flush=True)
        
        # Prime gaps
        gaps = nt.prime_gaps(1000)
        print("📊 Prime gap distribution:", flush=True)
        for gap in sorted(gaps.keys())[:10]:
            bar = "█" * (gaps[gap] // 2)
            print(f"  Gap {gap:3d}: {bar} ({gaps[gap]} occurrences)", flush=True)
        print("", flush=True)
        
        # Goldbach conjecture
        print("═" * 80, flush=True)
        print("GOLDBACH CONJECTURE VERIFICATION", flush=True)
        print("═" * 80, flush=True)
        
        test_numbers = [100, 200, 500, 1000]
        for n in test_numbers:
            pairs = nt.goldbach_conjecture(n)
            print(f"✓ {n} = sum of two primes: {len(pairs)} representations", flush=True)
            print(f"  Examples: {pairs[:3]}", flush=True)
        print("", flush=True)
        
        # Collatz sequences
        print("═" * 80, flush=True)
        print("COLLATZ CONJECTURE EXPLORATION", flush=True)
        print("═" * 80, flush=True)
        
        collatz_starts = [27, 127, 255, 999]
        for start in collatz_starts:
            seq = nt.collatz_sequence(start, max_steps=200)
            max_val = max(seq)
            print(f"🔄 Starting from {start}:", flush=True)
            print(f"   Steps to 1: {len(seq)-1} | Max value: {max_val}", flush=True)
            print(f"   First 10: {seq[:10]}", flush=True)
        print("", flush=True)
        
        # Fibonacci modular arithmetic
        print("═" * 80, flush=True)
        print("FIBONACCI SEQUENCES (MODULAR ARITHMETIC)", flush=True)
        print("═" * 80, flush=True)
        
        for mod in [7, 13, 17]:
            fib_mod = nt.fibonacci_modular(30, mod)
            print(f"📐 Fibonacci mod {mod}:", flush=True)
            print(f"   {fib_mod}", flush=True)
            
            # Find period
            period_len = 0
            for i in range(2, len(fib_mod) - 1):
                if fib_mod[i] == 0 and fib_mod[i+1] == 1:
                    period_len = i
                    break
            
            if period_len > 0:
                print(f"   Pisano period: {period_len}", flush=True)
            print("", flush=True)
        
        # Perfect numbers
        print("═" * 80, flush=True)
        print("PERFECT NUMBERS", flush=True)
        print("═" * 80, flush=True)
        
        perfect = nt.perfect_numbers(10000)
        print(f"✨ Perfect numbers up to 10000: {perfect}", flush=True)
        for p in perfect:
            divisors = [i for i in range(1, p) if p % i == 0]
            print(f"   {p} = {' + '.join(map(str, divisors))}", flush=True)
        print("", flush=True)
        
        # Fractals
        print("═" * 80, flush=True)
        print("MANDELBROT SET (ASCII)", flush=True)
        print("═" * 80, flush=True)
        
        mandelbrot = fg.mandelbrot_set(60, 20, max_iter=30)
        for line in mandelbrot:
            print(line, flush=True)
        print("", flush=True)
        
        # Sierpinski triangle
        print("═" * 80, flush=True)
        print("SIERPINSKI TRIANGLE", flush=True)
        print("═" * 80, flush=True)
        
        sierpinski = fg.sierpinski_triangle(15)
        for line in sierpinski:
            print(line, flush=True)
        print("", flush=True)
        
        # Pascal's triangle
        print("═" * 80, flush=True)
        print("PASCAL'S TRIANGLE", flush=True)
        print("═" * 80, flush=True)
        
        pascal = fg.pascal_triangle(10)
        for line in pascal:
            print(line, flush=True)
        print("", flush=True)
        
        # Sequences
        print("═" * 80, flush=True)
        print("INTEGER SEQUENCES", flush=True)
        print("═" * 80, flush=True)
        
        # Look-and-say
        look_say = sa.look_and_say("1", 8)
        print("🔢 Look-and-Say sequence:", flush=True)
        for i, term in enumerate(look_say):
            print(f"  {i}: {term[:50]}{'...' if len(term) > 50 else ''}", flush=True)
        print("", flush=True)
        
        # Recamán
        recaman = sa.recaman_sequence(30)
        print(f"🔄 Recamán sequence (30 terms):", flush=True)
        print(f"  {recaman}", flush=True)
        print("", flush=True)
        
        # Catalan
        catalan = sa.catalan_numbers(15)
        print("🔺 Catalan numbers:", flush=True)
        for i, c in enumerate(catalan):
            print(f"  C({i}) = {c:,}", flush=True)
        print("", flush=True)
        
        # Export results
        results = {
            "timestamp": time.time(),
            "primes_count": len(primes),
            "twin_primes": len(twins),
            "perfect_numbers": perfect,
            "collatz_analysis": {
                str(start): len(nt.collatz_sequence(start, 200))-1
                for start in collatz_starts
            },
            "catalan_numbers": catalan,
            "recaman_30": recaman
        }
        
        output_file = workspace / f"math_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.write_text(json.dumps(results, indent=2), encoding='utf-8')
        
        print("═" * 80, flush=True)
        print(f"✅ Analysis exported to: {output_file.name}", flush=True)
        print("═" * 80, flush=True)
        print("", flush=True)
        
        # Keep alive
        start_time = time.time()
        iteration = 0
        
        while not stop_event.is_set():
            iteration += 1
            elapsed = time.time() - start_time
            
            if iteration % 60 == 0:
                print(f"⏱️  Runtime: {elapsed/60:.1f}m | Computation engine active", flush=True)
            
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted", flush=True)
        stop_event.set()
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=5)

        try:
            heartbeat_path.unlink(missing_ok=True)
        except Exception:
            pass

        print(f"\n🏁 Mathematical computation session ended", flush=True)

if __name__ == "__main__":
    main()