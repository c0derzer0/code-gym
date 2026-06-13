# Lowest Common Ancestor of a Binary Tree

LeetCode 236 · Medium · **bonus rep**

## Problem

Given a binary tree and two nodes `p` and `q`, return their lowest common ancestor. The LCA is the deepest node that has both `p` and `q` as descendants (a node can be a descendant of itself).

```python
def lowestCommonAncestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    ...
```

## The recursive trick

Recurse into left and right subtrees. Each call returns:
- `None` if neither `p` nor `q` is in this subtree
- The found node (`p`, `q`, or the LCA) if at least one is

At each node:
- If `node is p or node is q`: return `node` (it's at least the LCA from this subtree's perspective)
- Recurse left → `L`, recurse right → `R`
- If both `L` and `R` are non-None: `node` itself is the LCA (one of `p`, `q` is on each side)
- Otherwise return whichever is non-None (or None if both are None)

```python
def lowestCommonAncestor(root, p, q):
    if not root or root is p or root is q:
        return root
    L = lowestCommonAncestor(root.left, p, q)
    R = lowestCommonAncestor(root.right, p, q)
    if L and R:
        return root
    return L or R
```

5 lines. The whole problem.

## Why it works

Think about what the recursion's return value *means*: "the LCA of `p` and `q` in this subtree, if both are here; otherwise whichever one I found (or None)."
- If I find both `p` and `q` in left subtree alone → L is already the LCA, return it.
- If I find one in left and one in right → I'M the LCA. Return self.
- If both are in right → R is the LCA.
- If neither → None.

The base case "`root is p` returns p" handles the "node is a descendant of itself" rule.

## Test

```python
# Tree: [3, 5, 1, 6, 2, 0, 8, null, null, 7, 4]
#        3
#       / \
#      5   1
#     / \ / \
#    6  2 0  8
#      / \
#     7   4
# LCA(5, 1) = 3
# LCA(5, 4) = 5  (5 is ancestor of 4 and of itself)
# LCA(7, 8) = 3
```

## Watch out for

- Identity comparison (`is`), not value comparison (`==`). Two nodes can have the same value but be different nodes.
- The "node can be a descendant of itself" rule is the reason the base case `root is p or root is q` returns immediately.
- This problem has many variants: LCA in BST (LC 235, easier), LCA of multiple nodes, LCA when the tree has parent pointers. Be ready to discuss.

## Why this is worth knowing

The recursion is elegant but easy to get wrong. The return-value semantics test whether you can reason about recursive types — what does a function return, and what does that mean? Trips a lot of people up.
