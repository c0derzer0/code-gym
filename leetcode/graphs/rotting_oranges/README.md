# Rotting Oranges

LeetCode 994 · Medium · **bonus rep**

## Problem

Grid of cells: `0` (empty), `1` (fresh orange), `2` (rotten orange). Every minute, fresh oranges 4-adjacent to a rotten one become rotten. Return the minimum minutes until no fresh oranges remain, or `-1` if impossible.

```python
def orangesRotting(grid: list[list[int]]) -> int:
    ...
```

## Approach

**Multi-source BFS.** Start the queue with *every* rotten orange. Track minutes by processing the queue level by level (snapshot `len(q)` before the inner loop, same as Binary Tree Level Order).

```python
from collections import deque
q = deque()
fresh = 0
for i, row in enumerate(grid):
    for j, v in enumerate(row):
        if v == 2: q.append((i, j))
        elif v == 1: fresh += 1

minutes = 0
while q and fresh > 0:
    for _ in range(len(q)):
        i, j = q.popleft()
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]) and grid[ni][nj] == 1:
                grid[ni][nj] = 2
                fresh -= 1
                q.append((ni, nj))
    minutes += 1
return minutes if fresh == 0 else -1
```

## Test

```python
[[2,1,1],[1,1,0],[0,1,1]]      # → 4
[[2,1,1],[0,1,1],[1,0,1]]      # → -1 (bottom-left orange unreachable)
[[0,2]]                         # → 0 (no fresh oranges)
```

## Watch out for

- **Initialize with ALL rotten oranges**, not just one. Multi-source.
- Track fresh count separately so you know if any are stranded.
- Increment minutes *after* processing each level. If `q` empties on the level that makes the last orange rot, that level counted but no future level adds to minutes.
- Edge case: grid starts with zero fresh oranges → return 0, not the number of minutes BFS would take.

## Why this is worth knowing

Multi-source BFS + level tracking + edge cases (impossible, no-fresh, all-fresh). Realistic systems-style problem packaging.
