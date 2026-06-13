# mini_pytorch_rust

A tensor library + autograd engine + training loop in Rust. Goal: train an MLP on MNIST, then a tiny transformer on TinyShakespeare, through an API that feels Pythonic-but-Rust.

**Why Rust?** Forces deep understanding of memory ownership, lifetimes, and zero-cost abstractions — the things that make ML systems fast. Real production precedent: HuggingFace's `tokenizers` and `Candle` are Rust. The skills here transfer to contributing to either.

**Scope:** This is a multi-month project, not a weekend. Tackle one phase at a time. Each phase is shippable on its own.

## Prerequisites

- Comfortable with Rust ownership + lifetimes. If not, run through "Rust by Example" + chapters 4, 10, 15 of the Rust Book first (~10 hours).
- Familiarity with PyTorch's API (you have this).
- The `backprop_from_scratch` movement done in Python first — gives you the math reference. Re-derive in Rust.

## Architecture overview

```
mini_pytorch_rust/
├── src/
│   ├── lib.rs          # public API
│   ├── tensor.rs       # Tensor struct: data, shape, stride, dtype
│   ├── ops.rs          # element-wise ops, matmul, reductions
│   ├── autograd.rs     # computation graph + backward pass
│   ├── nn/             # Linear, ReLU, LayerNorm, etc.
│   ├── optim/          # SGD, Adam
│   └── dataset.rs      # Dataset + DataLoader
├── examples/
│   ├── train_mlp_mnist.rs
│   └── train_gpt_tinyshakespeare.rs
└── tests/
    ├── tensor_ops.rs
    └── autograd_correctness.rs   # compare gradients vs numerical diff
```

## Phases (each is a milestone)

### Phase 1 — Tensor struct + basic ops (~1 week)

- `struct Tensor { data: Vec<f32>, shape: Vec<usize>, stride: Vec<usize> }`
- Element-wise add, sub, mul, div
- `Tensor::zeros`, `Tensor::ones`, `Tensor::randn`
- Indexing + reshape (stride manipulation only, no data copy)
- Tests against hand-computed values

**Deliverable:** Add two `(2, 3)` tensors, multiply by scalar, print result. Tests pass.

### Phase 2 — Broadcasting + matmul + reductions (~1 week)

- NumPy-style broadcasting (`(B, D) + (D,)` → `(B, D)`)
- Naive triple-loop matmul, then blocked matmul
- Reductions: sum, mean (over a given axis)
- Transpose (stride swap, no copy)
- Tests against `ndarray` crate as reference

**Deliverable:** `a @ b + c` with broadcasting works. Matmul timing < 10× a `ndarray` reference for `(1024, 1024)`.

### Phase 3 — Autograd: forward graph + backward pass (~2 weeks)

The hard part. Decide:
- **Tape-based (dynamic graph)** — like PyTorch. Record operations during forward; replay backward. Friendlier API, easier in Rust if you accept some `Rc<RefCell<...>>`.
- **Static graph** — like early TF. Less friendly but no runtime overhead.

Recommended: tape-based. Implement:
- `Tensor` carries an optional `Op` reference (the op that produced it + its inputs).
- `tensor.backward()` walks the graph, accumulating gradients.
- Implement backward for: add, sub, mul, div, matmul, transpose, sum, mean, ReLU, sigmoid.

**Verification:** Numerical gradient check. For random inputs and a random scalar loss, compare analytical gradients (from `.backward()`) to central-difference numerical gradients. Should match to `1e-4`.

**Deliverable:** Train a 2-layer MLP on a synthetic binary classification task — loss drops, accuracy > 90%.

### Phase 4 — Modules + optimizers + datasets (~1 week)

- `trait Module` with `forward()` and parameter iteration.
- `Linear`, `ReLU`, `Sequential` structs.
- `Optimizer` trait with `step()` and `zero_grad()`.
- `SGD` (with momentum), `Adam`.
- `Dataset` trait + `DataLoader` (batching, shuffling).

**Deliverable:** Train an MLP on MNIST. Reach >95% test accuracy. End-to-end works.

### Phase 5 — Transformer + training loop (~2 weeks)

- `Embedding`, `LayerNorm`, `MultiHeadAttention`, `TransformerBlock`.
- Cross-entropy loss with softmax.
- Causal masking for autoregressive generation.
- Generate-with-temperature sampling.

**Deliverable:** Train a tiny transformer (4 layers, 64 d_model) on TinyShakespeare characters. Sample reads like vaguely-Shakespeare text after 1000 iterations.

### Phase 6 — Speed (~ongoing)

The "make it production-ish" phase. Optional but instructive:
- Switch matmul to a SIMD intrinsics version (Rust `std::simd` or `packed_simd`).
- Add OpenMP-style parallelism via `rayon`.
- Memory pool / arena allocator to avoid `Vec` allocations in the hot path.
- Compare benchmark vs PyTorch CPU on `(1024, 1024)` matmul.

### Phase 7 — GPU (~major undertaking)

Only attempt after Phases 1-6 are stable. Two paths:
- **`cudarc` or `cust`** — Rust bindings to CUDA. Write kernels in CUDA C++, call from Rust.
- **`wgpu`** — cross-platform GPU compute via WebGPU. Less performant but works on macOS.

Realistically, this is months on its own. Save for "after the CPU version is done and battle-tested."

## Feeding movements

Small Rust reps that build the muscle before tackling each phase. Add these to `movements/` when you start:

- `rust_tensor_struct` — define `Tensor` with `Vec<f32>` storage + shape/stride. Implement indexing.
- `rust_naive_matmul` — triple-loop matmul, verify against `ndarray`.
- `rust_blocked_matmul` — tile-based matmul, benchmark vs naive.
- `rust_broadcasting_helper` — given two shapes, compute the broadcasted result + strides.
- `rust_autograd_minimal` — scalar-only autograd (like micrograd in Rust). Build the graph mental model.
- `rust_computation_graph` — tape-based op recording with `Rc<RefCell<Op>>`.
- `rust_module_trait` — design the `Module` trait + parameter iteration.

Each is a 60-90 min Rust rep. Once a few are done, the project phases become much more tractable.

## What this proves

- Deep ownership of ML internals — you've implemented autograd, not just used it.
- Rust fluency — ownership, lifetimes, traits, performance-aware code.
- Systems thinking — memory layouts, SIMD, GPU-via-FFI.
- Real shipped code — `cargo run --example train_mlp_mnist` works on someone else's machine.

## References (read after each phase, not before)

- **Phase 1-2:** [`ndarray` crate](https://docs.rs/ndarray) — Rust's NumPy. Read source for tensor layout patterns.
- **Phase 3:** Karpathy's [micrograd](https://github.com/karpathy/micrograd) — Python scalar autograd in 200 lines. Port to Rust scalars first, then generalize to tensors.
- **Phase 4-5:** [`Candle` source](https://github.com/huggingface/candle) — read `candle-core/src/tensor.rs`. Real-world Rust tensor library.
- **Phase 6:** [`Burn` source](https://github.com/tracel-ai/burn) — different architectural choices (graph-IR-first); good for comparing designs.
- **Phase 7:** [`cudarc`](https://github.com/coreylowman/cudarc) docs + Triton paper.

## Milestones checkbox

- [ ] Phase 1: tensor struct + element-wise ops
- [ ] Phase 2: broadcasting + matmul + reductions
- [ ] Phase 3: autograd; passes numerical gradient check
- [ ] Phase 4: modules + optimizers; MNIST MLP > 95%
- [ ] Phase 5: transformer + TinyShakespeare generation
- [ ] Phase 6: SIMD + rayon speedup
- [ ] Phase 7: GPU via cudarc/wgpu

## Notes log

Use `notes.md` (alongside this README) to log daily progress: what you built today, what you got stuck on, what the next session should pick up. Don't underestimate this — six months from now you'll want the trail.
