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
- **`mini_pytorch/`** — Tensor library with autograd, NumPy backend. Build on `backprop_from_scratch` — generalize to tape-based autodiff. Goal: train MNIST through a `mini_pytorch` API that mirrors PyTorch's.
- **`mini_triton/`** — Read-and-explain — port a Triton kernel to CUDA C++, identify what the Triton compiler is doing for you. Less "from scratch," more "decompile and understand."

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
- **Dedicated week:** if a project is the right interview-relevant build (e.g., mini_vllm before an inference-team interview), swap a regular gym week for it.
- **Bounded learning sprint:** "this Sunday I'm going to write the Docker chroot + PID namespace piece." Single session, single milestone.

Default: don't let projects displace the daily gym. Movements keep the muscle warm; projects build depth on top.
