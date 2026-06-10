# Ring all-reduce (simulated, multiprocessing-based)

**The single highest-clarity exercise in distributed training.** Implements the bandwidth-optimal all-reduce algorithm using N OS processes connected by IPC pipes. No GPUs. After 90 min of this you understand 70% of NCCL.

## Problem

N processes (ranks `0..N-1`) each hold a vector of `B` floats. After `all_reduce`, every rank holds the element-wise SUM across all ranks. Use the **ring algorithm** — each rank only talks to its left and right neighbors.

```python
def ring_all_reduce(local: np.ndarray, rank: int, world_size: int,
                    left_recv: Connection, right_send: Connection,
                    right_recv: Connection, left_send: Connection) -> np.ndarray:
    """
    local: this rank's input vector, shape (B,)
    rank: this process's ID in [0, world_size)
    world_size: total number of ranks
    left_recv: pipe to receive from left neighbor (rank-1 mod N)
    right_send: pipe to send to right neighbor (rank+1 mod N)
    (and the reverse pair for the all-gather phase)

    Returns: the fully-reduced vector — every rank gets the same result.
    """
    ...
```

## The algorithm in two phases

Split each rank's local buffer into `N` chunks of size `B/N`. Each rank initially "owns" its index's chunk; ownership rotates around the ring.

### Phase 1: scatter-reduce (N-1 steps)

After this phase, rank `r` holds the **fully reduced chunk `r`** (every other chunk is partial).

```
for step in range(N - 1):
    send_chunk_idx = (rank - step) mod N           # send my "current responsibility"
    recv_chunk_idx = (rank - step - 1) mod N       # receive my "next responsibility"
    send chunks[send_chunk_idx] → right
    recv buffer from left
    chunks[recv_chunk_idx] += buffer                # ACCUMULATE (this is the "reduce")
```

After `N-1` steps, the chunk at index `(rank+1) mod N` is fully reduced on this rank... let's just say chunk `r` is fully reduced on rank `r`.

### Phase 2: all-gather (N-1 steps)

After this phase, every rank holds every fully-reduced chunk.

```
for step in range(N - 1):
    send_chunk_idx = (rank - step + 1) mod N
    recv_chunk_idx = (rank - step) mod N
    send chunks[send_chunk_idx] → right            # this chunk is fully reduced
    recv buffer from left
    chunks[recv_chunk_idx] = buffer                # OVERWRITE (already reduced)
```

After `N-1` more steps, every rank has every chunk fully reduced — that's `all_reduce`.

## Total cost — the load-bearing math

Each rank sends `2(N-1)` chunks of size `B/N`:
```
total_bytes_per_rank = 2(N-1)/N · B
```

**For large N, this approaches `2·B` — independent of N.** That's why ring all-reduce scales: doubling the number of ranks doesn't double the per-rank communication. This is THE property of the algorithm.

Compare to naive "everyone sends to a root, root reduces, broadcasts back":
- Root receives `(N-1)·B` bytes (root becomes bottleneck)
- Total comm: `2(N-1)·B`

Ring spreads the bandwidth across all ranks instead of one.

## Test

```python
# Correctness
np.random.seed(0)
N = 4
B = 1024
locals = [np.random.randn(B) for _ in range(N)]
expected = sum(locals)

# Run ring_all_reduce across N processes
results = launch_ring_all_reduce(locals)

for r, result in enumerate(results):
    assert np.allclose(result, expected, atol=1e-6), f"rank {r} mismatch"
print("✓ all ranks have the reduced sum")

# Bandwidth math
# Each rank should have sent ~2(N-1)/N · B floats = ~6/4 · 1024 = 1536 floats
# Check: total bytes sent per rank == 2(N-1)/N · B · 4 (bytes per f32)
```

Scale test: run with N=2, 4, 8 and verify per-rank comm volume stays roughly `2·B` (not growing with N).

## Things to get right

- **Topology:** ring is one-directional logically, but each rank needs BOTH a send-to-right pipe AND a receive-from-left pipe (these are different pipes; `multiprocessing.Pipe` returns a tuple).
- **Buffer split:** if `B % N != 0`, padding or uneven chunks. For first attempt, assume divisible.
- **Use `multiprocessing.Process`, not threads.** Threads share memory — defeats the purpose. Each rank needs its own memory space to simulate a real distributed setting.
- **Send/recv are blocking.** That's correct — the ring algorithm IS step-synchronous.
- **Use `Connection.send_bytes`/`recv_bytes`** if you want speed, or `send`/`recv` (pickle-based) for simplicity.
- **Modular arithmetic** on chunk indices — `mod N` everywhere. Off-by-one is the most common bug.

## Common bugs

1. **`(rank - step) % N`** vs `(rank - step - 1) % N` — easy to get the wrong chunk and accumulate into the wrong slot. Write out a 4-rank trace by hand before coding.
2. **Sending then receiving in the wrong order** — every rank sends FIRST, then receives. If two adjacent ranks both try to receive first, deadlock.
3. **Forgetting to accumulate in scatter-reduce but accumulating in all-gather** — phase 1 ADDS; phase 2 OVERWRITES. Mixing these gives wrong results without a crash.
4. **Pipe direction confusion** — each process needs 4 pipe endpoints: send-to-right, recv-from-right (for the reverse direction... wait, no, ring is one direction). Actually 2: send-to-right, recv-from-left. The neighbor's `send-to-right` IS your `recv-from-left`.
5. **Spawning processes inside `if __name__ == "__main__":`** — required on macOS / Windows. Without this guard, your child processes recursively spawn more processes (infinite loop).
6. **Float precision drift** — summing N float32 vectors via ring vs all-at-once may differ in the last few bits. Use `atol=1e-5` not `atol=1e-10`.

## References (after timer)

- **Patarasuk & Yuan (2009)** — "Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations." 8 pages, the foundation.
- **HuggingFace UltraScale Playbook** §"Communication Primitives" — visual explanation.
- **NCCL source** — `github.com/NVIDIA/nccl/blob/master/src/collectives/all_reduce.cc`. After this movement you'll be able to read it.
- **PyTorch's `gloo` backend** — `github.com/facebookincubator/gloo/blob/main/gloo/allreduce_ring.cc`. CPU-based, same algorithm.

## What this unlocks

After this movement, the following become tractable:
- `tree_all_reduce_simulated` (next movement in CATALOG track D)
- `all_gather_reduce_scatter_paired` — they're the two halves of what you just built
- `projects/mini_nccl_lite/` Phase 4 — same algorithm, wrapped in a library API

## Attempt header template

```python
# Attempt N — YYYY-MM-DD
# Started:
# First pass (runs without deadlock):
# Correct (all ranks have the reduced sum):
# Bandwidth verified (per-rank ~ 2(N-1)/N · B):
# Stuck on:
# Notes:
```

## Stretch (after the basic version works)

- Time the ring all-reduce; vary N and message size; plot bandwidth. Show it stays flat as N grows.
- Try `multiprocessing.shared_memory` instead of `Pipe` for the buffer transfer — zero-copy intra-host. Compare timings.
- Implement async/non-blocking variants where send doesn't wait for recv (uses two ring directions in parallel).
