# Course Schedule

LeetCode 207 · Medium 

## Problem

Given `numCourses` and a list of `prerequisites` (each `[a, b]` meaning a depends on b), return `True` iff you can finish all courses (no cycles).

```python
def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    ...
```

## Approaches

1. **Kahn's algorithm (BFS topological sort)** — O(V + E). Compute in-degrees, BFS from nodes with in-degree 0. If you visit all nodes, no cycle.
2. **DFS with three colors** — O(V + E). White (unvisited), gray (in progress), black (done). If you see a gray node during DFS, cycle detected.

**Preferred approach:** Kahn's is cleaner to write under timer. Mention DFS-coloring as an alternative.

## Test

```python
canFinish(2, [[1, 0]])              # True — take 0 then 1
canFinish(2, [[1, 0], [0, 1]])      # False — cycle
canFinish(4, [[1, 0], [2, 1], [3, 2]])  # True — linear chain
canFinish(3, [[0, 1], [1, 2], [2, 0]])  # False — 3-cycle
```

## Watch out for

- Prerequisite format: `[course, prerequisite]` — direction matters when building the graph. Edge goes from `prereq → course`.
- Empty prerequisite list = trivially possible.
- Self-loops (`[a, a]`) = cycle → False.

## Related (good to mention)

- Course Schedule II (LC 210) — same but return the order. Same algorithm, just collect nodes as they're "finished."
