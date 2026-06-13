# mini_nccl_lite

A toy collective-communications library — NCCL-shaped but over TCP, runnable on a laptop with N processes. The educational core of distributed training, without needing GPUs.

**Why this project:** GPU multi-node training intuition is mostly about the *algorithms* (ring all-reduce, tree broadcast, reduce-scatter), not the wires underneath. Build the algorithms correctly over a slow transport (TCP), and the real-hardware version (NCCL on NVLink/IB) is "swap the bytes-moving layer." This project gives you the algorithm-level fluency.

## What you'll have when done

A Python library with this API:

```python
from mini_nccl_lite import ProcessGroup, AllReduceOp, AllGatherOp, BroadcastOp

pg = ProcessGroup(world_size=4, rank=int(os.environ["RANK"]))

x = np.random.randn(1024)
pg.all_reduce(x, op=AllReduceOp.SUM)   # ring all-reduce
# every rank now holds the sum across ranks

pg.broadcast(x, src=0)                  # tree broadcast
pg.all_gather(x, out_list)              # FSDP-style
pg.reduce_scatter(out, x_list, op=AllReduceOp.SUM)
```

Run with `python -m mini_nccl_lite.launch --nproc 4 your_script.py` — same UX as `torchrun`.

## Phases

### Phase 1 — Process group bootstrap (~1 session)

- Launcher script that spawns N processes via `multiprocessing.Process`.
- Each process reads `RANK` + `WORLD_SIZE` from env.
- Rendezvous: rank 0 listens on a TCP port; ranks 1..N-1 connect; rank 0 sends each peer the full address table.
- Result: each rank has TCP connections to all others (or a subset, for a ring).

**Deliverable:** `pg.barrier()` works — all ranks block until all arrive.

### Phase 2 — Point-to-point send/recv (~1 session)

- Length-prefixed framing over TCP socket (`u32 length` then bytes).
- `pg.send(tensor, dst=...)`, `pg.recv(buffer, src=...)`.
- Bytes-level — accept np arrays, serialize via `tobytes()`, deserialize via `np.frombuffer()`.

**Deliverable:** Round-trip: rank 0 sends 1MB to rank 1, rank 1 echoes it back. Timing.

### Phase 3 — Broadcast (tree) (~1 session)

- Binary tree from rank 0 outward. Each non-leaf rank sends to two children.
- `pg.broadcast(tensor, src=0)`.

**Deliverable:** Rank 0 has a tensor, everyone else has zeros. After `broadcast(src=0)`, all hold the tensor. `O(log N)` rounds.

### Phase 4 — Ring all-reduce (the main course) (~2 sessions)

The bandwidth-optimal algorithm. Each rank holds a slice of the buffer:

1. **Scatter-reduce phase:** for `N-1` steps, each rank sends its current accumulated slice to the next rank and receives from the previous; sums them.
2. **All-gather phase:** for `N-1` more steps, propagate the now-fully-reduced slices around the ring.

After `2(N-1)` steps, every rank holds the fully reduced array.

**Math to internalize:** total bytes per rank = `2(N-1)/N × buffer_size`. For large N, that's `≈ 2 × buffer_size` per rank — independent of N. That's why ring all-reduce scales.

**Deliverable:** `pg.all_reduce(x, op=SUM)` agrees with `sum(x_per_rank)` on every rank to bit equality. Timing: bandwidth doesn't degrade as N goes 2 → 4 → 8.

### Phase 5 — Reduce-scatter + all-gather (~1 session)

The FSDP primitives. Both are halves of the ring all-reduce.

- `reduce_scatter(out, x_list, op=SUM)` — input: list of N slices per rank; output: rank `r` gets the reduced slice `r`.
- `all_gather(out_list, x)` — input: each rank has one slice; output: each rank has all N slices.

**Deliverable:** `all_reduce` decomposed as `reduce_scatter + all_gather` matches the direct ring impl.

### Phase 6 — Tree all-reduce (~1 session)

For comparison: implement double-tree all-reduce. `O(log N) × buffer_size` per rank — slower bandwidth-wise but lower latency for small messages.

**Deliverable:** Plot bandwidth as message size varies from 1 KB to 100 MB for both ring and tree. Show the crossover point.

### Phase 7 — Benchmark suite (~ongoing)

- Measure bandwidth + latency for each op across message sizes.
- Compare ring vs tree.
- Compare 2/4/8 ranks.
- Generate plots → save in this folder.

### Phase 8 — `torchrun`-like launcher (~1 session)

- `python -m mini_nccl_lite.launch --nproc 4 my_script.py`
- Sets `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` env vars.
- Same UX as PyTorch's distributed launcher.

### Phase 9 — Multi-host (optional, ~1 session)

- Same library but across two machines (e.g., your laptop + a cloud VM via Tailscale).
- Verify the ring all-reduce still works.
- This is the "yes, the algorithms are transport-agnostic" proof.

## Stretch goals (after Phase 8)

- **Hierarchical all-reduce** — fast intra-node IPC (Unix sockets / shared memory) + slow inter-node TCP. Simulates NVLink + IB.
- **GPUDirect-style mmap** — use shared memory for intra-host comm; ranks read each other's tensors zero-copy.
- **PyTorch integration** — register as a `torch.distributed` backend. Run actual DDP through your `mini_nccl_lite`. The ultimate "I built this" moment.

## What this proves

- You understand collectives at the algorithmic level, not the API level.
- You can debug "why is my distributed training slow" — bandwidth math is internalized.
- You can read NCCL source and recognize the patterns.
- The project is a concrete artifact you can talk through, end to end, about distributed-training internals.

## Feeding movements (start with these before any phase)

From CATALOG track D — simulation movements:

- `ring_all_reduce_simulated` — single-file version, ~100 lines. Algorithm only, no library. Do this *first*. Then the project is "package + extend."
- `tree_all_reduce_simulated`
- `all_gather_reduce_scatter_paired`
- `broadcast_tree_simulated`

These each take 60-90 min. By the time you've done 2-3, Phase 3-4 of this project is mostly "wrap and add proper API."

## References

- Patarasuk & Yuan, 2009 — "Bandwidth Optimal All-reduce Algorithms" (8 pages, the foundation)
- NCCL source: `github.com/NVIDIA/nccl` — read `src/collectives/`
- HuggingFace UltraScale Playbook — §"Communication Primitives"
- PyTorch's `gloo` source — CPU collective lib, much simpler than NCCL, good reference for TCP-based implementations
- `python-multiprocessing` docs — `Pipe`, `Queue`, `shared_memory`

## Milestone checklist

- [ ] Phase 1: process group bootstrap, barrier works
- [ ] Phase 2: send/recv with timing
- [ ] Phase 3: broadcast in O(log N) rounds
- [ ] Phase 4: ring all-reduce — passes correctness, bandwidth independent of N
- [ ] Phase 5: reduce-scatter + all-gather; compose to all-reduce
- [ ] Phase 6: tree all-reduce; crossover plot vs ring
- [ ] Phase 7: full benchmark suite + plots
- [ ] Phase 8: `torchrun`-like launcher
- [ ] Phase 9: multi-host run
- [ ] **Stretch**: PyTorch backend integration

Use `notes.md` for daily session logs.
