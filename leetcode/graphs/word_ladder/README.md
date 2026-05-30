# Word Ladder

LeetCode 127 · Medium · **bonus rep**

## Problem

Given `beginWord`, `endWord`, and a `wordList`, return the length of the shortest transformation sequence from begin to end such that:
- Each transformation changes exactly one letter.
- Each intermediate word must be in `wordList`.

Return `0` if no such sequence exists.

```python
def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:
    ...
```

## Approach

**BFS on a word graph.** Two flavors:

### Flavor A — naive (slow)

For each word in the queue, check all words in `wordList` to see if they differ by exactly one letter. O(N²·L) — too slow for big inputs.

### Flavor B — pattern bucketing (fast)

Pre-build a map from "pattern" (e.g., `h*t` for `hot`) to the words matching that pattern. Then each BFS step is O(L) instead of O(N·L).

```python
from collections import defaultdict, deque

if endWord not in wordList: return 0
L = len(beginWord)
buckets = defaultdict(list)
for w in wordList:
    for i in range(L):
        buckets[w[:i] + '*' + w[i+1:]].append(w)

q = deque([(beginWord, 1)])
visited = {beginWord}
while q:
    word, steps = q.popleft()
    if word == endWord: return steps
    for i in range(L):
        pattern = word[:i] + '*' + word[i+1:]
        for neighbor in buckets[pattern]:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append((neighbor, steps + 1))
return 0
```

## Test

```python
ladderLength("hit", "cog", ["hot","dot","dog","lot","log","cog"])
# → 5  (hit → hot → dot → dog → cog)

ladderLength("hit", "cog", ["hot","dot","dog","lot","log"])
# → 0  (cog not in wordList)
```

## Watch out for

- `endWord` not in `wordList` → return 0 immediately.
- All words must be the same length (problem guarantee, but worth stating aloud).
- `visited` set prevents re-processing — critical for performance.
- Use a tuple `(word, steps)` in queue, or track distance via a separate dict.

## Why Amazon likes this

Tests the "transform the problem into a graph" insight. Words aren't a graph until you decide they are. The pattern-bucketing optimization is the kind of detail that separates "I can BFS" from "I can BFS *efficiently*."

## Bidirectional BFS (mention but skip unless time)

Run BFS from both ends simultaneously. When the frontiers meet, you have the answer. ~2x faster in practice. Worth mentioning aloud as a follow-up optimization.
