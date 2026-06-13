# Validate Binary Search Tree

LeetCode 98 · Medium 

## Problem

Given a binary tree, determine if it's a valid BST. A valid BST: for every node, all values in its left subtree are strictly less, all values in its right subtree are strictly greater. (And both subtrees must themselves be BSTs.)

```python
def isValidBST(root: TreeNode) -> bool:
    ...
```

## The common wrong answer

```python
# DON'T DO THIS — only checks immediate children, misses the global constraint.
def bad(root):
    if not root: return True
    if root.left and root.left.val >= root.val: return False
    if root.right and root.right.val <= root.val: return False
    return bad(root.left) and bad(root.right)
```

Counterexample: `[5, 1, 4, null, null, 3, 6]`. The 3 is in the right subtree of 5 but 3 < 5 — invalid. The check above passes because it only compares to immediate parent.

## Correct approaches

1. **Recursion with bounds** — Pass `(low, high)` to each recursive call. Initial call has `(-inf, +inf)`. Each recursion narrows the bounds.
   ```python
   def helper(node, low, high):
       if not node: return True
       if not (low < node.val < high): return False
       return helper(node.left, low, node.val) and helper(node.right, node.val, high)
   return helper(root, float('-inf'), float('inf'))
   ```

2. **Inorder traversal** — In a valid BST, inorder traversal produces a strictly increasing sequence. Walk inorder, track the previous value, fail if `prev >= current`.

Both are O(N). Bounds-based recursion is cleaner under timer.

## Test

```python
# [2, 1, 3]    → True
# [5, 1, 4, null, null, 3, 6]  → False (the classic trap)
# [1, 1]       → False (equal is not strictly less)
```

## Watch out for

- "Strictly less / greater" — equal values are NOT allowed.
- Empty tree (`root is None`) → True (vacuously a BST).
- Single node → True.
- Use `float('-inf')` and `float('inf')` for initial bounds, NOT integer min/max — the tree values can be arbitrary.
