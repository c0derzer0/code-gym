# Plan

The rotation. Six-week backbone. Each Sunday: review, re-prioritize next week's row if signal warrants.

## Day cadence (whichever calendar day)

- **Day 1, 2, 4, 5** — ML days: 45 min main movement + 15 min warmup
- **Day 3, 6** — LC days: 30 min LC medium + 30 min basic deep-dive
- **Day 7** — Rest + Sunday retro

If extra time on a day, do another attempt of a past movement and try to set a new PB.

## Week 1 — Attention foundations

| Day | Main (45 min) | Warmup / LC deep-dive (15 / 30 min) |
|-----|---------------|-------------------------------------|
| 1   | `scaled_dot_product_attention`         | `softmax_stable` |
| 2   | `multi_head_attention` (+ causal mask) | `triangular_mask` |
| 3   | LC: Top K Frequent Elements            | Deep-dive: `adam` |
| 4   | `transformer_block` (MHA+FFN+res+LN)   | `layernorm` |
| 5   | `positional_encodings` (sinusoidal + RoPE) | `rmsnorm` |
| 6   | LC: Number of Islands (graphs)         | Deep-dive: `dropout` |
| 7   | Rest + retro                            | — |

Note: `sgd_momentum` rolls into Week 2's warmup rotation. Order rationale: `triangular_mask` lands on Day 2 because it's the primitive MHA needs that same session; `layernorm` lands on Day 4 because the transformer block uses it.

**LC rotation tilted toward Amazon (graphs + trees front-loaded for the ~1-2 week Annapurna timeline):** Days 6 → 13 cover graphs (Number of Islands, Course Schedule, Clone Graph); Days 17 + 20 cover trees (Level Order, Validate BST). Arrays/strings/DP/sliding-window come back in Weeks 4-6 once Amazon is done. Original arrays stubs (`product_of_array_except_self`) preserved for later.

**Bonus reps available** (rest days, extra-time slots, or pre-Amazon cram):
- `leetcode/graphs/pacific_atlantic_water_flow/` — multi-source DFS, reverse the flow
- `leetcode/graphs/rotting_oranges/` — multi-source BFS with level tracking
- `leetcode/graphs/word_ladder/` — BFS on word graph with pattern bucketing
- `leetcode/trees/lowest_common_ancestor/` — recursive LCA, return-value semantics

**Foundational movements (added based on interview signal):**
- `movements/backprop_from_scratch/` — 2-layer MLP binary classifier, pure numpy, forward + backward + training loop. THE most common "implement X from scratch" ML interview question. Failed one on this — re-rep until under 45 min unaided. Slot into any ML day as the main, or as a warmup re-attempt once you've passed once.

## Week 2 — LLM inference core

| Day | Main (45 min) | Warmup / LC deep-dive |
|-----|---------------|-----------------------|
| 8   | `kv_cache` (extend Week 1's MHA)         | `cross_entropy` |
| 9   | `sampling_greedy_temperature`            | `embedding_lookup` |
| 10  | LC: Course Schedule (topo sort)          | Deep-dive: `sgd_momentum` |
| 11  | `sampling_top_k_top_p`                   | `dataset_dataloader` |
| 12  | `batched_inference_padding`              | `training_loop_skeleton` |
| 13  | LC: Clone Graph (DFS + hashmap)          | Deep-dive: `residual_block` |
| 14  | Rest + retro                             | — |

## Week 3 — Inference advanced (algo + memory, interleaved)

| Day | Main (45 min) | Warmup / LC deep-dive |
|-----|---------------|-----------------------|
| 15  | `paged_kv_cache` (memory: block-table KV)    | `block_indexing` (paging primitive) |
| 16  | `grouped_query_attention` (algo: Llama-3 style) | `repeat_interleave_kv` (GQA primitive) |
| 17  | LC: Binary Tree Level Order Traversal        | Deep-dive: re-attempt `scaled_dot_product_attention` (speed PB) |
| 18  | `quantize_linear_int8` (memory: weight-only) | `int8_quantize_dequantize_pair` |
| 19  | `sliding_window_attention` (algo: Mistral)   | Re-attempt: `triangular_mask` as band-mask variant |
| 20  | LC: Validate BST                             | Deep-dive: `flash_attention_simplified` (online-softmax recurrence) |
| 21  | Rest + retro                                 | — |

Week 3 mixes 2 memory mains + 2 algo mains. Pairings: paged_kv_cache pairs with the GQA day (KV cache memory savings stack); quantization sits independently; sliding window naturally pairs with bounded KV cache. Dropped: beam_search (legacy), speculative_decoding (push to W4/W6), continuous batching (scheduler — discuss as system design, not a kata).

## Weeks 4-7 — mixed weeks (every category every week)

Each week from here pulls from every track:
- **Foundational** (architecture, RL algos, agent loops)
- **Lower-level algorithms** (Welford, online softmax, FA recurrence)
- **Optimizations** (MoE, LoRA, KV variants, quantization)
- **Kernels** (Triton primitives → production Triton kernels)
- **GPU networking** (TP, ZeRO-3, PP, NCCL — mostly conceptual on LC deep-dive days)
- **LeetCode graphs/trees**
- **Interesting algos** (LRU cache, Bloom filter, consistent hashing — occasional LC deep-dive)
- **RL** (REINFORCE → DQN → PPO → GRPO)
- **Speed PBs** (re-attempts of Week 1 movements)

## Week 4 — Llama-3 + first kernels + RL kickoff

| Day | Main (45 min) | Warmup / LC deep-dive |
|-----|---------------|-----------------------|
| 22 | `llama3_block` (foundational arch: RoPE + GQA + RMSNorm + SwiGLU) | `swiglu` (small primitive) |
| 23 | `triton_vector_add` (kernels: first Triton kernel from scratch) | `welford_fused_meanvar` (lower-level alg) |
| 24 | LC: Number of Islands (graphs) | Deep-dive: tensor parallel concepts (GPU networking) |
| 25 | `mixture_of_experts` (optimization: top-k routing) | `online_softmax_recurrence` (lower-level alg, FA-2 building block) |
| 26 | `reinforce_cartpole` (RL: policy gradient end-to-end) | `gym_env_skeleton` (RL primitive) |
| 27 | LC: Clone Graph | Deep-dive: ZeRO-3 memory accounting (GPU networking) |
| 28 | Rest + retro | — |

## Week 5 — Kernels deepen + alignment + GPU networking

| Day | Main (45 min) | Warmup / LC deep-dive |
|-----|---------------|-----------------------|
| 29 | `triton_rmsnorm` (kernels: production-relevant fused kernel; benchmark vs naive PyTorch) | `triton_grid_block_concepts` (kernels primitive) |
| 30 | `dqn_cartpole_minimal` (RL: replay buffer + target net + ε-decay) | `replay_buffer` (RL primitive) |
| 31 | LC: Course Schedule | Deep-dive: pipeline parallel + bubble mitigation (GPU networking) |
| 32 | `lora_adapter` (optimization: low-rank wrap of nn.Linear) | `dpo_loss` (alignment primitive) |
| 33 | `flashattention_full_recurrence` (lower-level: FA-2 algorithm tile-by-tile) | Re-attempt: `multi_head_attention` (speed PB) |
| 34 | LC: Validate BST | Deep-dive: NCCL all-reduce ring vs tree (GPU networking) |
| 35 | Rest + retro | — |

## Week 6 — More kernels + agents + interesting algos

| Day | Main (45 min) | Warmup / LC deep-dive |
|-----|---------------|-----------------------|
| 36 | `triton_fused_softmax` (kernels: stable softmax in Triton; benchmark) | Re-attempt: `softmax_stable` (compare to Triton output) |
| 37 | `ppo_clipped_objective` (RL: clipped policy ratio + GAE) | `vectorized_envs` (RL primitive) |
| 38 | LC: Lowest Common Ancestor | Deep-dive: `lru_cache_from_scratch` (interesting algo, Amazon-frequent) |
| 39 | `grpo_loss` (alignment: group-relative policy opt; DeepSeek style) | `reward_signal_design` (RL concept) |
| 40 | `agent_loop_with_tools` (foundational: orchestration + tool dispatch + reflection) | `chat_template_serialization` (agent primitive) |
| 41 | LC: Pacific Atlantic Water Flow | Deep-dive: `triton_attention_simple` (kernels: simplified FA-2 in Triton) |
| 42 | Rest + retro | — |

## Week 7 — Adaptive synthesis / interview week

By this point real interviews have likely landed. Each session:
- Pick the weakest movement from the last interview, re-attempt under timer.
- Or pick the topic the interviewer pulled hardest on, dive deeper.

If no interview signal, default to:
- Re-attempt the 3 movements with the largest gap between current PB and target.
- Pick 2 LC mediums you haven't touched (Word Ladder, Rotting Oranges from the bonus pool).
- 1 system design conversation aloud (multi-agent serving, LLM inference at scale).

## Future weeks (adjust as interviews / signal demand)

| Week | Headline |
|------|----------|
| 1 | Attention foundations |
| 2 | LLM inference core |
| 3 | Inference advanced — algo + memory mixed |
| 4 | Mixed: Llama-3 + first kernels + RL kickoff |
| 5 | Mixed: kernels deepen + alignment + GPU networking |
| 6 | Mixed: more kernels + agents + interesting algos |
| 7 | Adaptive synthesis / interview week |

Each Sunday: write retro, update next week's row.

---

## Long-term curriculum (Weeks 8+, indefinite)

The gym keeps going past interview season. **The vision: deep CS + AI mastery — system-level, low-level, mathematical, applied.** Topics on deck, sprinkled across future mixed weeks:

### Serving systems (track A)
- vLLM internals: PagedAttention manager, scheduler, continuous batching, prefix caching
- SGLang: RadixAttention, structured generation runtime
- TensorRT-LLM patterns
- Inference graph compilation (torch.compile, TorchDynamo, FX graphs)

### Low-level kernels & systems (track B)
- Triton: tiled matmul, attention kernels, persistent kernels
- Memory hierarchies (HBM / L2 / SRAM / registers) — Roofline analysis
- NCCL internals: ring vs tree all-reduce, NVLink topology, infiniband basics
- GPU scheduling (cooperative groups, async copy)
- Quantization deep dives: GPTQ, AWQ, SmoothQuant, FP8

### Math fundamentals (track C)
- Linear algebra reflexes: SVD, eigendecomp, low-rank approximation
- Probability: KL/JSD, importance sampling, exponential families
- Information theory: entropy, mutual info, channel capacity (for compression / quantization intuition)
- Optimization theory: convexity, KKT, Adam convergence, why GD works
- Calculus reflexes: backprop derivation, gradient estimators (REINFORCE, reparameterization, Gumbel-softmax)

### RL — low-level + applied (track D)
- Theory: policy gradient theorem derivation, value iteration, actor-critic variance reduction
- Distributional RL: C51, QR-DQN, IQN
- Off-policy + importance sampling weighting
- Distributed RL infra: IMPALA-style actor-learner split, Ape-X replay
- Custom env design patterns (POMDP wrappers, frame stacking, reward shaping pitfalls)
- World models (V-JEPA, Cosmos, DINO-based) — cross-pollinate with CMU VLA research

### Production AI (track E)
- Eval harness design (lm-eval, custom rule-based evals)
- RAG patterns at scale (hybrid retrieval, re-ranking, hallucination detection)
- Agent orchestration: tool dispatch, reflection loops, parallel agents
- Multi-modal: VLA action heads, flow matching, latent action models

### Interesting algos + system design (track F)
- LRU cache, LFU, ARC
- Bloom filter, Count-Min sketch, HyperLogLog
- Consistent hashing, Skiplist, Merkle trees
- LSM trees, B+ trees (storage engines)
- Raft consensus (for distributed RL infra background)
- Practical system design: design a serving stack, design a training cluster, design a multi-agent platform

### Systems algorithms — OS, storage, networks (track G — NEW)
All 45-min movements that fit the daily gym. Build the underlying primitives of the `projects/` builds:
- `page_table_walk` — virtual address → physical via multi-level page table; TLB simulation
- `cooperative_scheduler` — yield-based coroutines; context switches without OS threads
- `b_tree_insert_search` — B-tree from scratch; node splits + merges
- `lsm_tree_compaction` — write to memtable, flush to SSTables, leveled compaction
- `consistent_hashing_ring` — ring + virtual nodes + lookup
- `bloom_filter` — bit array + k hash functions; FPR computation
- `count_min_sketch` — streaming frequency, error bounds
- `hyperloglog_cardinality` — probabilistic cardinality estimation
- `merkle_tree_verification` — content-addressed verification + proof
- `skiplist_ops` — probabilistic balanced search structure
- `chroot_jail_demo` — chroot + minimal filesystem isolation (Linux)
- `pid_namespace_isolation` — Linux PID namespace via syscall (mini_docker's primitive)
- `cgroup_memory_limit` — set memory limit on a child process (mini_docker's primitive)
- `tcp_three_way_handshake` — SYN/SYN-ACK/ACK state machine in user space
- `producer_consumer_queue` — bounded queue with lock + condition var
- `mark_sweep_gc` — toy garbage collector; reachability from roots
- `simple_paxos_or_raft_leader` — just the leader election piece
- `content_addressed_blob_store` — Git's object store primitive

### Generative models (track H — NEW)
- `vae_minimal` — encoder + decoder + reparameterization + KL on MNIST. The original generative latent-variable model.
- `ddpm_forward_noising` — forward diffusion: `q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t) · I)`. Linear / cosine schedules.
- `ddpm_reverse_denoising` — reverse process: train a U-Net to predict noise; sample by iterating `x_{t-1} = (1/√α_t)(x_t - β_t·ε_θ(x_t,t)/√(1-ᾱ_t)) + σ_t · z`.
- `ddim_deterministic_sampling` — non-Markovian deterministic sampling; fewer steps.
- `classifier_free_guidance` — train with conditional dropout, sample with `ε_guided = (1+w)·ε_cond − w·ε_uncond`.
- `flow_matching_2d` — modern continuous-time alternative; train velocity field, sample via ODE solve.
- `gan_minimal` — generator + discriminator + JS divergence. Mode collapse demo.

### Vision architectures (track I — NEW)
- `conv2d_from_scratch` — im2col-based convolution; verify against `F.conv2d` to 1e-5.
- `resnet_block` — conv → BN → ReLU → conv → BN → residual.
- `batchnorm_from_scratch` — train/eval stats, running mean/var, momentum.
- `unet_minimal` — encoder-decoder with skip connections; the diffusion backbone.
- `vit_patch_embed` — image → patches → flatten → linear → add CLS token → transformer.

### Sequence model variants (track J — NEW)
- `bpe_tokenizer` — byte-pair encoding from scratch; train on a small corpus, encode/decode.
- `encoder_decoder_transformer` — cross-attention layer; the original "Attention is All You Need" architecture.
- `bert_masked_lm` — bidirectional attention (no causal mask) + MLM head.
- `mamba_ssm_basic` — selective state space model; the linear-attention competitor.
- `linear_attention_perfomer` — approximate softmax attention with feature maps.

### Multimodal (track K — NEW)
- `clip_contrastive_loss` — image-text contrastive learning; symmetric NT-Xent loss.
- `vla_action_head` — language-conditioned action head for robotic control; flow-matching style action prediction (connects to user's CMU world-models research).
- `cross_modal_attention` — attend from text tokens to image patches and vice versa.

### Cloud & distributed systems (track L — NEW)
The primitives behind every AWS/GCP/Azure service. 45-min movements that compose into the `projects/mini_cloud/` builds:
- `http_server_from_sockets` — parse HTTP/1.1 requests + write responses; no framework.
- `dns_query_encoder_decoder` — DNS message format (header + question + RR records) from raw bytes.
- `tls_handshake_concept` — client hello → server hello → key exchange → finished; conceptual + cipher suite negotiation.
- `jwt_sign_verify` — HMAC-SHA256 or RSA signing; header.payload.signature; verify untrusted JWTs.
- `oauth2_authorization_code_flow` — token issuance + refresh; resource server validation.
- `rate_limiter_token_bucket` — token bucket + leaky bucket; per-user + per-IP variants.
- `circuit_breaker_state_machine` — closed/open/half-open with retry budgets.
- `service_discovery_registry` — heartbeat-based registry with eviction.
- `distributed_lock_redlock` — Redis-style distributed lock with fencing tokens.
- `gossip_membership_protocol` — SWIM-style failure detector + membership.
- `vector_clocks` — track causal ordering across nodes.
- `lamport_clocks` — total ordering of events.
- `two_phase_commit` — coordinator + participants; failure handling.
- `quorum_read_write` — N/R/W parameters; read-your-writes consistency.
- `crdt_g_counter` — grow-only counter (state-based CRDT).
- `crdt_lww_register` — last-write-wins register.
- `vnode_consistent_hashing` — partitioning + rebalancing on join/leave.
- `idempotent_request_dedup` — request ID-based dedup; exactly-once semantics.

### From-scratch projects (multi-session builds — lives in `projects/`, not `movements/`)
See `projects/README.md` for the full list. Highlights:
- `mini_docker` — chroot + namespaces + cgroups
- `mini_postgres` — buffer pool + B-tree + executor
- `mini_redis` — KV + AOF + pub/sub
- `mini_git` — object store + refs + commits
- `mini_tcp` — TCP state machine over UDP
- `mini_os` — xv6-style: bootloader + paging + scheduler + FS
- `mini_compiler` — lexer + parser + bytecode VM
- `mini_vllm` — inference server with PagedAttention + continuous batching
- `mini_pytorch` — tensor library with autograd

Future mixed weeks pull from these tracks the same way Weeks 4-7 do: 4 mains + 2 LC days, each touching different tracks. No theme weeks — variety per week, depth per session. Systems-algorithm movements integrate directly into the daily rotation; projects fill rest days / weekends / dedicated sprints.

**The repo's long-term vision: everything from scratch. CS + AI mastery measured by what you've personally built, primitive by primitive.**

## Sunday retro template

1. PBs set this week?
2. Movements that felt slow — schedule a re-attempt next week.
3. Any interview signal (real or upcoming) that should shift next week's theme?
