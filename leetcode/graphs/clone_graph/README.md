# Clone Graph

LeetCode 133 · Medium 

## Problem

Given a reference to a node in a connected undirected graph, return a deep copy of the graph.

```python
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors or []

def cloneGraph(node: Node) -> Node:
    ...
```

## Approach

DFS or BFS with a `visited` hashmap keyed by original node, value is the cloned node. When you hit an unvisited node, create its clone, register in the map, then recurse into neighbors and append their clones.

The map serves two purposes:
1. Detect visited (avoid infinite loops in cyclic graphs)
2. Get the canonical clone for a given original (avoid creating duplicate clones)

## Test

Hard to test without LeetCode's harness. Build a graph manually:

```python
# Graph: 1—2—3—4—1 (4-cycle), 2 also connects to 4
n1 = Node(1); n2 = Node(2); n3 = Node(3); n4 = Node(4)
n1.neighbors = [n2, n4]
n2.neighbors = [n1, n3, n4]
n3.neighbors = [n2, n4]
n4.neighbors = [n1, n2, n3]

clone = cloneGraph(n1)
# Assert clone is a different object than n1
assert clone is not n1
assert clone.val == 1
# Assert structure preserved
assert len(clone.neighbors) == 2
```

## Watch out for

- `node is None` edge case → return None.
- Both DFS and BFS work. DFS is fewer lines; BFS is iterative (no recursion limit).
- Hashmap key is the original Node object, not the value (graphs can have duplicate values).
