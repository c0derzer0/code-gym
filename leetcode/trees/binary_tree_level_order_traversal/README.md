# Binary Tree Level Order Traversal

LeetCode 102 · Medium 

## Problem

Return the level-order traversal of a binary tree's node values, grouped by level.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def levelOrder(root: TreeNode) -> list[list[int]]:
    ...
```

## Approach

BFS with a queue. Key trick: at each level, grab the **current queue length BEFORE you start the inner loop**. Process exactly that many nodes — those are the current level. Anything you append during the inner loop belongs to the next level.

```python
from collections import deque
q = deque([root])
while q:
    level_size = len(q)
    level = []
    for _ in range(level_size):
        node = q.popleft()
        level.append(node.val)
        if node.left: q.append(node.left)
        if node.right: q.append(node.right)
    result.append(level)
```

## Test

```python
# Tree:
#      3
#     / \
#    9  20
#       / \
#      15  7
root = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
# expected: [[3], [9, 20], [15, 7]]

# Edge: empty tree
# expected: []
```

## Watch out for

- Empty root → return `[]`, not `[[]]`.
- The "snapshot the level size before iterating" trick is the whole point of the problem. Without it you can't separate levels.

## Related (good to mention)

- Right Side View (LC 199) — same BFS, take the last node per level.
- Zigzag Level Order (LC 103) — same BFS, alternate left-to-right and right-to-left.
