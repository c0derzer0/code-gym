# Pacific Atlantic Water Flow

LeetCode 417 · Medium · **bonus rep**

## Problem

Given an `m × n` grid of heights where the Pacific touches the top + left edges and the Atlantic touches the bottom + right edges, return all cells from which water can flow to *both* oceans. Water flows from a cell to a 4-neighbor only if the neighbor's height ≤ current.

```python
def pacificAtlantic(heights: list[list[int]]) -> list[list[int]]:
    ...
```

## The trick

Don't forward-simulate from each cell (O(M²N²)). Instead, **reverse the flow** and do **multi-source DFS/BFS** from the borders.

- Start a DFS/BFS from every Pacific-border cell (top row + left column). Mark every cell reachable while moving to cells of equal or *higher* height. Call this set `pac`.
- Same from Atlantic border. Call it `atl`.
- Answer = `pac ∩ atl`.

O(MN) total — each cell visited at most twice.

## Test

```python
heights = [
    [1,2,2,3,5],
    [3,2,3,4,4],
    [2,4,5,3,1],
    [6,7,1,4,5],
    [5,1,1,2,4]
]
# expected (any order): [[0,4], [1,3], [1,4], [2,2], [3,0], [3,1], [4,0]]
```

## Watch out for

- Inverted comparison: when moving *backward* from a border, water arrived from a neighbor that was *higher or equal*, so DFS to neighbors where `neighbor.height >= current.height`.
- Edge cells belong to both oceans if they're corner cells — yes, they should be in the result.
- Single-row or single-column grids — every cell is on a border.

## Why Amazon likes this

Tests two patterns at once: multi-source BFS/DFS + inverted-direction thinking. The "don't simulate forward, simulate backward from the borders" insight is the kind of leap they look for.
