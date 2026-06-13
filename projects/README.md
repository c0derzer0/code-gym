# projects/

Multi-week from-scratch builds. Different from `movements/` (which are 45-min reps): projects are extended builds that span sessions or weeks. The daily gym still runs movements; projects fill rest days, weekends, vacation, or get their own dedicated weeks when scheduled.

## The list

Each is a directory containing its own README + design notes + sub-modules. Tackle in any order, none of them block the daily gym.

### Systems / infra

- **`mini_docker/`** — Container runtime from scratch. `chroot` + Linux namespaces (PID, network, mount) + cgroups. Liz Rice's "Containers from Scratch" pattern. Go or Python in ~500 lines. Goal: launch a process in its own isolated filesystem + PID space.
- **`mini_postgres/`** — Page-based DB from scratch. Buffer pool manager + B-tree index + query parser + simple executor. CMU 15-445 scope. C or Go. Goal: support `INSERT`, `SELECT WHERE`, `CREATE INDEX`.
- **`mini_redis/`** — Single-node KV store. In-memory hash + AOF persistence + pub/sub + simple RESP protocol. Python or Rust. Goal: pass `redis-benchmark`'s basic ops.
- **`mini_git/`** — Content-addressed object store. Blobs + trees + commits + refs. The plumbing layer; no porcelain commands. Python in ~300 lines. Goal: `commit`, `log`, `diff` on a local repo.
- **`mini_tcp/`** — TCP state machine over UDP packets. Three-way handshake + retransmission + sliding window. "Build your own TCP" sized project. Goal: reliable bytestream over an unreliable channel.
- **`mini_os/`** — xv6-style OS in C. Bootloader + page tables + process scheduler + filesystem. The biggest project on this list. Tackle after mini_docker + understanding paging in movements.
- **`mini_compiler/`** — Lexer + parser + AST + simple bytecode VM. Crafting Interpreters Part I. Goal: arithmetic, vars, functions, closures.

### ML / serving

- **`mini_vllm/`** — Inference server with PagedAttention + continuous batching + simple HTTP API. End-to-end: tokenizer → KV cache → sampling → response stream.
- **`mini_pytorch/`** — Tensor library with autograd, NumPy backend (Python). Build on `backprop_from_scratch` — generalize to tape-based autodiff. Goal: train MNIST through a `mini_pytorch` API that mirrors PyTorch's.
- **`mini_pytorch_rust/`** — Same goal, but in **Rust**. 7 phases from tensor struct → autograd → MNIST MLP → tiny transformer → SIMD speedups → GPU. Forces deep understanding of ownership + zero-cost abstractions. Precedent: HuggingFace Candle, Burn. **Multi-month — tackle one phase at a time.** See its own README for milestones.
- **`mini_nccl_lite/`** — A toy NCCL-shaped collective communications library over TCP. Ring all-reduce, tree broadcast, reduce-scatter/all-gather, `torchrun`-like launcher. **Runs entirely on a laptop** with N processes — no GPUs needed to build the distributed-training intuition. 9 phases, multi-week. See its own README.
- **`mini_triton/`** — Read-and-explain — port a Triton kernel to CUDA C++, identify what the Triton compiler is doing for you. Less "from scratch," more "decompile and understand."

### Generative models (end-to-end training pipelines)

- **`mini_ddpm_mnist/`** — Train a DDPM end-to-end on MNIST. Tiny U-Net + cosine schedule + classifier-free guidance. Builds on the diffusion movements. Goal: generate recognizable digits in <1 hour of training on a laptop GPU.
- **`mini_vae_mnist/`** — Train a VAE on MNIST, visualize the latent space. Builds on `vae_minimal`.
- **`mini_clip/`** — Tiny CLIP on a small image-caption dataset (CIFAR-10 + class names). Vision encoder + text encoder + symmetric contrastive loss. Goal: zero-shot classification accuracy > random.
- **`mini_flow_matching/`** — Train a flow matching model on a 2D distribution, then a 32x32 image dataset. Compare ODE samplers (Euler, Heun, RK4).
- **`nanoGPT_replica/`** — End-to-end GPT pretraining on TinyShakespeare. Tokenizer + transformer + training loop + sampling. Karpathy's nanoGPT as the reference. Already covered by movements but worth a full integration pass.

### Cloud / distributed systems (multi-week builds)

These are bigger than typical projects — each is a "build a real AWS service from scratch" target. Use after the cloud-systems movements (track L) are warm.

- **`mini_s3/`** — Object storage with bucket/key API. Multipart upload, eventual consistency, metadata sidecar, content-addressed dedup on the backend. HTTP API; clients via `boto3`-like SDK.
- **`mini_kubernetes/`** — Container orchestrator. API server (etcd-backed) + scheduler + kubelet + pod controller. Schedule pods across "nodes" (local processes). YAML-driven manifests.
- **`mini_kafka/`** — Append-only log + partitions + consumer groups + offsets. Built on `mini_redis` patterns; adds partitioning + replication.
- **`mini_zookeeper/`** — Distributed coordination service. Hierarchical key store + ephemeral nodes + watches + Zab consensus. Powers distributed locks.
- **`mini_dns_authoritative/`** — Authoritative DNS server. Parse queries, serve RR records, support zone transfers. Build on `dns_query_encoder_decoder` movement.
- **`mini_load_balancer_l4_l7/`** — TCP (L4) and HTTP (L7) load balancing. Round-robin + least-connections + health checks. Sidecar proxy mode.
- **`mini_service_mesh/`** — Sidecar proxy + control plane. mTLS between services, traffic policies, observability. Envoy-lite.
- **`mini_cdn/`** — Edge cache nodes + origin pull + cache invalidation + geographic routing. Tiered caching.
- **`mini_iam/`** — Auth/AuthN/AuthZ. User + role + policy + STS-style token issuance. Build on `jwt_sign_verify` + `oauth2_authorization_code_flow`.
- **`mini_terraform/`** — IaC tool. Read declarative HCL-like config, diff against state, apply create/update/destroy. Plugin system for "providers" (any HTTP API).
- **`mini_lambda/`** — Function-as-a-service. Sandbox per invocation (chroot + cgroup limits from `mini_docker`), warm pool, billing meter.
- **`mini_cloud_full/`** — The capstone: stitch the above into a working "cloud" — compute (lambda), storage (s3), orchestration (kubernetes), messaging (kafka), auth (iam), networking (load balancer + service mesh). Submit infra via terraform. A toy AWS in your repo.

## Format per project

Each project folder has:
```
projects/<name>/
├── README.md       # design doc: scope, goals, sub-tasks, references
├── notes.md        # daily progress log (what I built, what I'm stuck on)
└── src/            # the code
```

Projects don't have a PB table the way movements do — they're not timed reps. Instead each has a "milestones" section in its README that gets checked off.

## When to tackle a project

- **Weekend / rest day:** start a milestone, push as far as you can in 3-4 hours.
- **Dedicated week:** if a project is the right build for the moment (e.g., focused study of inference systems → `mini_vllm`), swap a regular gym week for it.
- **Bounded learning sprint:** "this Sunday I'm going to write the Docker chroot + PID namespace piece." Single session, single milestone.

Default: don't let projects displace the daily gym. Movements keep the muscle warm; projects build depth on top.
