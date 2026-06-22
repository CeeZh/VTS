"""
Tree state management for hierarchical video segment navigation.

Provides TreeNode, TreeState, tree loading from cache, frame path resolution,
and uniform tree fallback for when no cache is available.
"""

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TreeNode:
    """A node in the hierarchical video segment tree."""
    start_sec: float
    end_sec: float
    caption: str = ''
    children: List['TreeNode'] = field(default_factory=list)
    parent: Optional['TreeNode'] = None
    node_id: int = 0  # index among siblings (0 for root)
    visited: bool = False
    gt_coverage: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def to_tree_dict(self) -> Dict:
        """Recursively serialize the tree for memory rendering."""
        return {
            'node_id': self.node_id,
            'start_sec': self.start_sec,
            'end_sec': self.end_sec,
            'caption': self.caption,
            'visited': self.visited,
            'children': [c.to_tree_dict() for c in self.children],
        }


def _build_tree_recursive(
    node_dict: Dict,
    parent: Optional[TreeNode],
    node_id: int,
) -> TreeNode:
    """Recursively build a TreeNode from a cache dict."""
    node = TreeNode(
        start_sec=node_dict['start_sec'],
        end_sec=node_dict['end_sec'],
        caption=node_dict.get('caption', ''),
        parent=parent,
        node_id=node_id,
    )
    for i, child_dict in enumerate(node_dict.get('children', [])):
        child = _build_tree_recursive(child_dict, parent=node, node_id=i)
        node.children.append(child)
    return node


def load_tree_from_cache(cache_dir: str, video_id: str) -> Optional[TreeNode]:
    """
    Load a pre-built tree from JSON cache.

    Args:
        cache_dir: Directory containing {video_id}.json files.
        video_id: Video identifier.

    Returns:
        Root TreeNode if cache found, None otherwise.
    """
    safe_id = video_id.replace('/', '_')
    cache_file = Path(cache_dir) / f'{safe_id}.json'
    if not cache_file.exists():
        return None

    with open(cache_file) as f:
        data = json.load(f)

    tree_dict = data.get('tree')
    if tree_dict is None:
        return None

    return _build_tree_recursive(tree_dict, parent=None, node_id=0)


def create_uniform_tree(
    duration: float,
    num_children: int = 4,
    depth: int = 3,
) -> TreeNode:
    """
    Create a uniform tree by recursively dividing the video duration.
    Fallback when no tree cache is available.
    """
    root = TreeNode(start_sec=0.0, end_sec=duration, node_id=0)
    _build_uniform_children(root, depth, num_children)
    return root


def _build_uniform_children(
    node: TreeNode,
    remaining_depth: int,
    num_children: int,
) -> None:
    """Recursively create uniform children for a node."""
    if remaining_depth <= 0 or node.duration < 64:
        return
    seg_dur = node.duration / num_children
    for i in range(num_children):
        child = TreeNode(
            start_sec=node.start_sec + i * seg_dur,
            end_sec=node.start_sec + (i + 1) * seg_dur,
            parent=node,
            node_id=i,
        )
        node.children.append(child)
        _build_uniform_children(child, remaining_depth - 1, num_children)


def get_frame_paths(
    frames_base: str,
    video_id: str,
    start_sec: float,
    end_sec: float,
    max_frames: int = 16,
) -> List[str]:
    """
    Get paths to pre-extracted frames for a segment.

    Frames are at {frames_base}/{video_id}/{sec}.jpg where sec is 0-indexed
    integer seconds (1fps extraction).

    Returns list of existing frame paths, subsampled to max_frames.
    """
    frame_dir = os.path.join(frames_base, video_id)
    if not os.path.isdir(frame_dir):
        return []

    first = math.ceil(start_sec)
    last = math.floor(end_sec)
    # Collect all candidate seconds
    candidates = list(range(first, last + 1))
    if not candidates:
        # Very short segment: try the midpoint
        mid = int((start_sec + end_sec) / 2)
        candidates = [mid]

    # Filter to existing files
    paths = []
    for sec in candidates:
        p = os.path.join(frame_dir, f'{sec}.jpg')
        if os.path.exists(p):
            paths.append(p)

    if not paths:
        return []

    # Subsample to max_frames
    if len(paths) > max_frames:
        indices = np.linspace(0, len(paths) - 1, max_frames, dtype=int)
        paths = [paths[i] for i in indices]

    return paths


def allocate_frames(
    children: List[TreeNode],
    max_total: int,
) -> List[int]:
    """
    Distribute max_total frames across children proportionally to duration.
    Each child gets at least 1 frame.
    """
    if not children:
        return []
    total_dur = sum(c.duration for c in children)
    if total_dur <= 0:
        per = max(1, max_total // len(children))
        return [per] * len(children)

    allocation = []
    for c in children:
        n = max(1, int(max_total * c.duration / total_dur))
        allocation.append(n)

    # Trim if over budget
    while sum(allocation) > max_total and max(allocation) > 1:
        idx = allocation.index(max(allocation))
        allocation[idx] -= 1

    return allocation


def populate_gt_coverage(
    root: TreeNode,
    gt_intervals: List[Tuple[float, float]],
) -> None:
    """
    Set gt_coverage on every node in the tree.

    For each node, gt_coverage = max over all GT intervals of
    overlap(node, gt_i) / length(gt_i).
    """
    stack = [root]
    while stack:
        node = stack.pop()
        if gt_intervals:
            node.gt_coverage = max(
                _node_gt_coverage(
                    (node.start_sec, node.end_sec), gt,
                )
                for gt in gt_intervals
            )
        else:
            node.gt_coverage = 0.0
        stack.extend(node.children)


def _node_gt_coverage(
    node_interval: Tuple[float, float],
    gt_interval: Tuple[float, float],
) -> float:
    """Fraction of the GT interval that falls inside the node interval."""
    node_start, node_end = node_interval
    gt_start, gt_end = gt_interval
    gt_length = gt_end - gt_start
    if gt_length <= 0:
        return 0.0
    overlap_start = max(node_start, gt_start)
    overlap_end = min(node_end, gt_end)
    overlap = max(0.0, overlap_end - overlap_start)
    return overlap / gt_length


def compute_tree_distance(
    start_node: TreeNode,
    coverage_threshold: float = 0.5,
) -> Optional[int]:
    """
    Minimum number of actions from start_node to a node with gt_coverage > threshold.

    Actions (each costs 1): ZOOM_IN (child), ZOOM_OUT (parent), SHIFT (sibling).

    Returns 0 if start_node already qualifies, or None if no node qualifies.
    """
    if start_node.gt_coverage > coverage_threshold:
        return 0

    from collections import deque
    visited = {id(start_node)}
    queue = deque([(start_node, 0)])

    while queue:
        node, dist = queue.popleft()

        neighbors: List[TreeNode] = list(node.children)
        if node.parent is not None:
            neighbors.append(node.parent)
            for sibling in node.parent.children:
                if sibling is not node:
                    neighbors.append(sibling)

        for neighbor in neighbors:
            if id(neighbor) not in visited:
                visited.add(id(neighbor))
                if neighbor.gt_coverage > coverage_threshold:
                    return dist + 1
                queue.append((neighbor, dist + 1))

    return None


def _interval_iou(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """Temporal IoU of two intervals."""
    a_start, a_end = a
    b_start, b_end = b
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def find_best_node(
    root: TreeNode,
    gt_intervals: List[Tuple[float, float]],
) -> Tuple[Optional[TreeNode], float]:
    """
    Find the node whose interval best matches the GT, scored as
    max over gt in gt_intervals of [IoU(node, gt) + IoG(node, gt)].

    Tiebreak: deeper node first, then smaller temporal span.
    Returns (best_node, best_score), or (None, 0.0) if gt_intervals is empty.
    """
    if not gt_intervals:
        return None, 0.0

    best_node: Optional[TreeNode] = None
    best_score = -1.0
    best_depth = -1
    best_span = float('inf')

    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()

        node_interval = (node.start_sec, node.end_sec)
        score = 0.0
        for gt in gt_intervals:
            iou = _interval_iou(node_interval, gt)
            iog = _node_gt_coverage(node_interval, gt)
            score = max(score, iou + iog)

        span = node.end_sec - node.start_sec
        if (score > best_score
                or (score == best_score and depth > best_depth)
                or (score == best_score and depth == best_depth and span < best_span)):
            best_node = node
            best_score = score
            best_depth = depth
            best_span = span

        for child in node.children:
            stack.append((child, depth + 1))

    return best_node, best_score


def bfs_distance(
    start_node: TreeNode,
    target_node: TreeNode,
) -> Optional[int]:
    """
    Minimum number of actions (zoom_in / zoom_out / shift) from start to target.

    Each of children / parent / siblings counts as one step. Returns 0 if
    start is target, or None if the two nodes are in disconnected components
    (should not happen within a single tree).
    """
    if start_node is target_node:
        return 0

    from collections import deque
    visited = {id(start_node)}
    queue = deque([(start_node, 0)])

    while queue:
        node, dist = queue.popleft()

        neighbors: List[TreeNode] = list(node.children)
        if node.parent is not None:
            neighbors.append(node.parent)
            for sibling in node.parent.children:
                if sibling is not node:
                    neighbors.append(sibling)

        for neighbor in neighbors:
            if id(neighbor) in visited:
                continue
            if neighbor is target_node:
                return dist + 1
            visited.add(id(neighbor))
            queue.append((neighbor, dist + 1))

    return None


class TreeState:
    """
    Manages the agent's position within a tree during a single rollout episode.
    """

    def __init__(
        self,
        root: TreeNode,
        frames_base: str,
        video_id: str,
    ):
        self.root = root
        self.frames_base = frames_base
        self.video_id = video_id
        self.current_node = root

    def get_children(self) -> List[TreeNode]:
        """Return current node's children."""
        return self.current_node.children

    def get_siblings(self) -> List[TreeNode]:
        """Return siblings of the current node (excluding self)."""
        if self.current_node.parent is None:
            return []
        return [
            s for s in self.current_node.parent.children
            if s is not self.current_node
        ]

    def zoom_in(self, segment_id: int) -> TreeNode:
        """Navigate to a child segment."""
        child = self.current_node.children[segment_id]
        self.current_node = child
        return child

    def zoom_out(self) -> Optional[TreeNode]:
        """Navigate to parent. Returns None if already at root."""
        if self.current_node.parent is None:
            return None
        self.current_node = self.current_node.parent
        return self.current_node

    def shift(self, segment_id: int) -> TreeNode:
        """Navigate to a sibling segment (by parent's children index)."""
        if self.current_node.parent is None:
            raise ValueError('Cannot SHIFT from root node')
        sibling = self.current_node.parent.children[segment_id]
        self.current_node = sibling
        return sibling

    def get_node_frame_paths(
        self,
        node: TreeNode,
        max_frames: int = 16,
    ) -> List[str]:
        """Get frame paths for a given node."""
        return get_frame_paths(
            self.frames_base, self.video_id,
            node.start_sec, node.end_sec,
            max_frames=max_frames,
        )

    def get_all_nodes(self) -> List[TreeNode]:
        """Collect all nodes in DFS order. List index serves as global unique ID."""
        nodes: List[TreeNode] = []
        def _dfs(node: TreeNode) -> None:
            nodes.append(node)
            for child in node.children:
                _dfs(child)
        _dfs(self.root)
        return nodes

    def shift_to(self, node: TreeNode) -> TreeNode:
        """Navigate directly to any node in the tree (free mode)."""
        self.current_node = node
        return node

    def to_tree_dict(self) -> Dict:
        """Serialize the full tree from root."""
        return self.root.to_tree_dict()
