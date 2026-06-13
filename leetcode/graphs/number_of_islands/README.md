# Number of Islands

LeetCode 200 · Medium 

## Problem

Given a 2D grid of `'1'`s (land) and `'0'`s (water), return the number of islands. An island = 4-connected (up/down/left/right) land cells.

```python
def numIslands(grid: list[list[str]]) -> int:
    ...
```

## Approaches

1. **DFS** — O(MN). Walk the grid; when you hit a `'1'`, DFS to mark the whole island as visited (overwrite to `'0'` or use a visited set), increment count.
2. **BFS** — Same complexity. Use a queue instead of recursion. Better for huge grids (avoids stack overflow).
3. **Union-Find** — O(MN · α(MN)). Overkill for this problem; useful if islands could be added/removed dynamically.

**Preferred approach:** DFS is the expected answer. Mention BFS as an alternative when the grid is too deep for recursion.

## Test

```python
grid = [
    ["1","1","0","0","0"],
    ["1","1","0","0","0"],
    ["0","0","1","0","0"],
    ["0","0","0","1","1"]
]
# expected: 3

grid = [["1","1","1"],["0","1","0"],["1","1","1"]]
# expected: 1 (the middle col connects top to bottom)
```

## Watch out for

- Mutating the grid is OK if the problem allows it — saves O(MN) space. Mention this tradeoff aloud.
- Bounds checking when recursing into neighbors.
- Don't forget to check both initial direction AND that the cell is land before recursing.
