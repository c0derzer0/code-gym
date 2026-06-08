# Catalog

The full menu of movements and projects to pull from. **Browse, don't grind.** PLAN.md handles the immediate next 1-2 weeks; this file holds the whole library.

How to use:
- At each Sunday retro, browse this catalog. Pick 4 movements + 2 LC mediums for next week.
- Mix tracks. Don't do five from one track in a week — boring and one-dimensional.
- New interests? Add them as a track here, not in PLAN.md.

---

## Track A — ML primitives (Weeks 1-3 covered; revisit for speed PBs)

Attention stack, optimizers, samplers, transformer core. Most of these are in `movements/` already.

- `scaled_dot_product_attention`, `multi_head_attention`, `transformer_block`, `positional_encodings`, `kv_cache`
- `softmax_stable`, `cross_entropy`, `layernorm`, `rmsnorm`, `dropout`
- `sgd_momentum`, `adam`, `adamw`
- `sampling_greedy_temperature`, `sampling_top_k_top_p`, `beam_search`
- `embedding_lookup_tied_projection`, `residual_block`
- `backprop_from_scratch` (the 2-layer MLP foundation)

## Track B — LLM inference & memory

- `paged_kv_cache`, `grouped_query_attention`, `sliding_window_attention`
- `batched_inference_padding`, `continuous_batching_scheduler`
- `quantize_linear_int8`, `int8_quantize_dequantize_pair`
- `flash_attention_simplified`, `flashattention_full_recurrence`, `online_softmax_recurrence`
- `welford_fused_meanvar` (single-pass mean+var algorithm)
- `speculative_decoding`, `prefix_caching`

## Track C — Kernels (Triton, CUDA basics)

- `triton_vector_add`, `triton_grid_block_concepts`
- `triton_rmsnorm`, `triton_fused_softmax`, `triton_attention_simple`
- `triton_tiled_matmul`
- `cuda_kernel_read_explain` (read a real CUDA kernel, annotate)
- `hbm_traffic_measure` (time naive vs fused; count expected bytes)

## Track D — GPU networking & distributed training

Mostly conceptual deep-dives on LC days. A few movements.

- Tensor parallel concepts (column-parallel + row-parallel pattern)
- ZeRO-1/2/3 memory accounting
- Pipeline parallel + bubble mitigation
- NCCL: ring vs tree all-reduce, NVLink topology
- `all_reduce_naive` (concept-level implementation; understand what NCCL does)
- `nccl_collectives_overview` (broadcast, gather, scatter, all-gather)

## Track E — Production architectures (Llama-3 + MoE)

- `swiglu`, `llama3_block`
- `mixture_of_experts`, `moe_load_balancing_loss`
- `lora_adapter`, `qlora_concept`
- `agent_loop_with_tools`, `chat_template_serialization`

## Track F — Fine-tuning & alignment

- SFT loop with proper data masking
- `reward_model_bradley_terry`
- `dpo_loss`, `grpo_loss`, `ppo_clipped_objective`
- `kl_penalty_term`, `reference_model_freezing`

## Track G — RL & environments

- `gym_env_skeleton`, `vectorized_envs`
- `replay_buffer`, `prioritized_replay`
- `reinforce_cartpole`, `dqn_cartpole_minimal`, `a2c_minimal`
- `ppo_clipped_objective` (also fits Track F for RLHF)
- `epsilon_greedy_thompson`, `ucb1_bandit`
- `reward_signal_design`, `reward_hacking_failure_modes`
- `world_model_predictor_loss` (connects to user's CMU VLA work)

## Track H — Generative models

- `vae_minimal`
- `ddpm_forward_noising`, `ddpm_reverse_denoising`, `ddim_deterministic_sampling`
- `classifier_free_guidance`, `flow_matching_2d`
- `gan_minimal` (with mode collapse demo)

## Track I — Vision architectures

- `conv2d_from_scratch` (im2col)
- `batchnorm_from_scratch` (train/eval stats)
- `resnet_block`
- `unet_minimal` (the diffusion backbone)
- `vit_patch_embed`

## Track J — Sequence model variants

- `bpe_tokenizer`
- `encoder_decoder_transformer` (cross-attention)
- `bert_masked_lm` (bidirectional)
- `mamba_ssm_basic`, `linear_attention_performer`

## Track K — Multimodal

- `clip_contrastive_loss` (NT-Xent)
- `vla_action_head` (language-conditioned action prediction; user's CMU VLA work)
- `cross_modal_attention`

## Track L — Systems algorithms (OS, storage, networks)

- `page_table_walk`, `tlb_simulation`
- `cooperative_scheduler` (yield-based coroutines)
- `b_tree_insert_search`, `lsm_tree_compaction`
- `consistent_hashing_ring`, `bloom_filter`, `count_min_sketch`, `hyperloglog_cardinality`
- `merkle_tree_verification`, `skiplist_ops`
- `chroot_jail_demo`, `pid_namespace_isolation`, `cgroup_memory_limit` (mini_docker primitives)
- `tcp_three_way_handshake`
- `producer_consumer_queue`, `mark_sweep_gc`
- `simple_paxos_or_raft_leader`, `content_addressed_blob_store`

## Track M — Cloud & distributed systems

- `http_server_from_sockets`, `dns_query_encoder_decoder`, `tls_handshake_concept`
- `jwt_sign_verify`, `oauth2_authorization_code_flow`
- `rate_limiter_token_bucket`, `circuit_breaker_state_machine`
- `service_discovery_registry`, `distributed_lock_redlock`
- `gossip_membership_protocol`, `vector_clocks`, `lamport_clocks`
- `two_phase_commit`, `quorum_read_write`
- `crdt_g_counter`, `crdt_lww_register`, `vnode_consistent_hashing`
- `idempotent_request_dedup`

## Track N — Math for ML (linear algebra, prob, optimization, info theory)

The math underneath everything. Slot one of these in any week — they're often quick (15-30 min) and reset the foundation.

- `matmul_from_scratch_numpy` (naive triple loop → vectorized → blocked)
- `svd_power_iteration` (top singular vector via power method)
- `eigendecomp_jacobi_method` or `eigendecomp_power_iteration`
- `pca_from_scratch` (center → covariance → eigendecompose → project)
- `cholesky_decomposition`
- `numerical_gradient_check` (verify analytical backprop with central differences)
- `kl_divergence_two_gaussians` (closed form + verify with samples)
- `cross_entropy_softmax_derivation` (log-sum-exp, gradient algebra)
- `reparameterization_trick` (sample `z ~ N(μ,σ²)` as `μ + σ·ε`)
- `reinforce_gradient_estimator` (likelihood ratio; demonstrate high variance)
- `gumbel_softmax_trick` (differentiable argmax)
- `mutual_information_pmi` (from a co-occurrence matrix)
- `taylor_expansion_function_approx` (basis change, polynomial approximation)
- `convex_optimization_kkt` (Lagrangian, simple constrained problem)
- `gradient_descent_convergence_basic_proof` (strongly convex case)
- `low_rank_approximation_via_svd` (matrix compression)

## Track O — Programming fundamentals

Generic CS reflexes. Most are 15-30 min — perfect quick wins.

- `lru_cache_class` (HashMap + doubly-linked list)
- `lfu_cache`, `arc_cache`
- `decorator_from_scratch` (decorator that times calls; chain of decorators)
- `generator_iterator` (`__iter__` + `__next__` from scratch)
- `context_manager_from_scratch` (`__enter__`/`__exit__` + `@contextmanager`)
- `memoization_decorator` (with cache eviction)
- `partial_application_currying` (`functools.partial` from scratch)
- `closure_counter` (closures, free variables)
- `metaclass_basics` (`__new__` vs `__init__`)
- `dataclass_internals` (implement a `@dataclass`-like decorator)
- `recursive_to_iterative` (convert recursion to explicit stack)
- `design_patterns_factory_observer_strategy` (3 patterns, one file each)
- `singleton_thread_safe`
- `event_loop_from_scratch` (build a minimal asyncio loop)
- `big_o_under_timer` (annotate 10 code snippets with time/space in 10 min)

## Track P — State machines

The hidden currency of CS — show up in regex, parsers, async/await, distributed systems, game logic.

- `traffic_light_dfa` (table-driven DFA, simplest case)
- `regex_to_nfa_thompson` (Thompson's construction)
- `nfa_to_dfa_subset_construction`
- `regex_engine_from_scratch` (parser → NFA → match)
- `parser_combinator_library` (`seq`, `alt`, `many`; build a calculator parser)
- `pushdown_automaton_balanced_parens` (PDA for nested structure)
- `async_state_machine_simulation` (show how `async/await` desugars to a state machine)
- `hierarchical_state_machine` (Statecharts pattern; nested states + transitions)
- `saga_distributed_transaction`
- `event_sourced_state` (rebuild state by replaying events)
- `actor_model_basic` (message-passing with mailboxes)

## LeetCode — by category

- **Arrays/hashing**: Top K Frequent (done), Product of Array Except Self
- **Strings/two pointers**: Longest Substring w/o Repeat, 3Sum, Valid Anagram
- **Graphs/BFS-DFS**: Number of Islands, Course Schedule, Clone Graph, Pacific Atlantic, Rotting Oranges, Word Ladder
- **Trees**: Binary Tree Level Order, Validate BST, Lowest Common Ancestor, Binary Tree Max Path Sum
- **DP**: Coin Change, Longest Increasing Subsequence, Word Break, House Robber
- **Heap / Priority Queue**: Kth Largest, Find Median from Data Stream
- **Linked Lists**: LRU Cache (LC 146), Reverse Linked List, Merge K Sorted Lists
- **Stack/queue**: Valid Parens, Min Stack, Daily Temperatures, Sliding Window Maximum

## From-scratch projects (multi-session, lives in `projects/`)

See [`projects/README.md`](projects/README.md) for the full list. Categories:

- **Systems**: mini_docker, mini_postgres, mini_redis, mini_git, mini_tcp, mini_os, mini_compiler
- **ML / serving**: mini_vllm, mini_pytorch, mini_triton
- **Generative pipelines**: mini_ddpm_mnist, mini_vae_mnist, mini_clip, mini_flow_matching, nanoGPT_replica
- **Cloud / distributed**: mini_s3, mini_kubernetes, mini_kafka, mini_zookeeper, mini_dns_authoritative, mini_load_balancer, mini_service_mesh, mini_cdn, mini_iam, mini_terraform, mini_lambda
- **Capstone**: mini_cloud_full — stitch the cloud pieces into a toy AWS

---

## The vision

Everything from scratch. CS + AI mastery measured by what you've personally built, primitive by primitive. The catalog is the menu — every week you pull from a few different tracks, never grinding one. The goal is breadth over depth this year, depth over breadth next year, and both forever.

Don't try to do all of this. **Do one movement a day. Rotate categories. The repo grows quietly over years.**
