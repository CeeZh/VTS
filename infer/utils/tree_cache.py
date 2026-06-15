"""
Utility for loading pre-built tree cache files produced by scripts/build_tree_cache.py.

Each cache file is a per-video JSON with structure:
{
    "video_id": "...",
    "video_path": "...",
    "duration": 1526.0,
    "tree": {
        "start_sec": 0.0,
        "end_sec": 1526.0,
        "caption": "...",
        "children": [...]
    }
}
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class TreeCache:
    """Lazy-loading cache for pre-built segment trees and captions."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self._cache: Dict[str, Optional[dict]] = {}

    def load(self, video_id: str) -> Optional[dict]:
        """Load a video's tree cache JSON. Returns None if not found."""
        if video_id in self._cache:
            return self._cache[video_id]

        safe_id = video_id.replace("/", "_")
        cache_file = self.cache_dir / f"{safe_id}.json"
        if not cache_file.exists():
            self._cache[video_id] = None
            return None

        with open(cache_file) as f:
            data = json.load(f)
        self._cache[video_id] = data
        return data

    def _find_node(
        self, tree_node: dict, start_sec: float, end_sec: float, tol: float
    ) -> Optional[dict]:
        """DFS to find the node matching (start_sec, end_sec) within tolerance."""
        if (
            abs(tree_node["start_sec"] - start_sec) < tol
            and abs(tree_node["end_sec"] - end_sec) < tol
        ):
            return tree_node
        for child in tree_node.get("children", []):
            result = self._find_node(child, start_sec, end_sec, tol)
            if result is not None:
                return result
        return None

    def get_children(
        self,
        video_id: str,
        start_sec: float,
        end_sec: float,
        tol: float = 1.0,
    ) -> Optional[List[dict]]:
        """Return cached children for a node, or None if not found.

        Returns [] for leaf nodes (no children exist in cache).
        Returns None if the video or node is not in the cache.
        """
        data = self.load(video_id)
        if data is None:
            return None
        node = self._find_node(data["tree"], start_sec, end_sec, tol)
        if node is None:
            return None
        return node.get("children", [])

    def get_caption(
        self,
        video_id: str,
        start_sec: float,
        end_sec: float,
        tol: float = 1.0,
    ) -> Optional[str]:
        """Return the cached caption for a node, or None if not found."""
        data = self.load(video_id)
        if data is None:
            return None
        node = self._find_node(data["tree"], start_sec, end_sec, tol)
        if node is None:
            return None
        return node.get("caption")
