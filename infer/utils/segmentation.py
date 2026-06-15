"""
Segmentation utilities for splitting video intervals into child segments.

Provides factory functions that return a `segment_fn` callable with signature:
    (video_path: str, start_sec: float, end_sec: float) -> List[Tuple[float, float]]

Compatible with `TrajectoryGenerator(segment_fn=...)`.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scene boundary detection helpers (inlined from scripts/scene.py)
# ---------------------------------------------------------------------------


def _extract_frames_uniform(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps: float = 1.0,
    max_frames: int = 64,
    short_side: int = -1,
) -> Tuple[List[float], List[np.ndarray], float]:
    """Sample frames uniformly from [start_sec, end_sec]."""
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    max_end_sec = (total_frames - 1) / video_fps
    if end_sec < 0:
        end_sec = max_end_sec
    else:
        end_sec = min(end_sec, max_end_sec)

    duration = end_sec - start_sec
    num_frames = min(max_frames, max(2, int(duration * fps) + 1))
    timestamps = np.linspace(start_sec, end_sec, num_frames).tolist()
    frames = []
    valid_timestamps = []

    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if short_side > 0:
            h, w = frame_rgb.shape[:2]
            if h < w:
                new_h, new_w = short_side, int(w * short_side / h)
            else:
                new_h, new_w = int(h * short_side / w), short_side
            interp = cv2.INTER_AREA if new_h < h else cv2.INTER_LINEAR
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=interp)
        frames.append(frame_rgb)
        valid_timestamps.append(ts)

    cap.release()
    return valid_timestamps, frames, end_sec


# Public alias so external callers (e.g. scripts/build_tree_cache.py) can decode
# frames once and share them between the VLM caption and scene segmentation.
extract_frames_uniform = _extract_frames_uniform


def _compute_clip_embeddings(frames: List[np.ndarray], clip_client) -> np.ndarray:
    """Compute CLIP embeddings for a list of RGB numpy frames."""
    from docarray import Document

    docs = [Document(tensor=frame) for frame in frames]
    return clip_client.encode(docs).embeddings


def _compute_frame_deltas(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine distance between each consecutive pair of frame embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    cosine_sim = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return 1.0 - cosine_sim


def _detect_boundaries_topk(deltas: np.ndarray, num_boundaries: int) -> List[int]:
    """Return exactly num_boundaries boundary indices (top-k largest deltas)."""
    num_boundaries = min(num_boundaries, len(deltas))
    if num_boundaries <= 0:
        return []
    indices = np.argpartition(deltas, -num_boundaries)[-num_boundaries:]
    return sorted(indices.tolist())


def _detect_boundaries_threshold(
    deltas: np.ndarray,
    k: float = 1.5,
    min_segments: int = 3,
    max_segments: int = 8,
) -> List[int]:
    """Detect scene boundaries where delta > mean + k * std (z-score)."""
    threshold = deltas.mean() + k * deltas.std()
    boundary_indices = np.where(deltas > threshold)[0].tolist()

    num_segments = len(boundary_indices) + 1
    if num_segments > max_segments:
        boundary_indices = _detect_boundaries_topk(deltas, max_segments - 1)
    elif num_segments < min_segments:
        boundary_indices = _detect_boundaries_topk(deltas, min_segments - 1)

    return boundary_indices


def _boundaries_to_segments(
    timestamps: List[float],
    boundary_indices: List[int],
    start_sec: float,
    end_sec: float,
) -> List[Tuple[float, float]]:
    """Convert boundary frame indices to (start_sec, end_sec) time segments."""
    boundary_times = [
        (timestamps[i] + timestamps[i + 1]) / 2.0 for i in boundary_indices
    ]
    segment_starts = [start_sec] + boundary_times
    segment_ends = boundary_times + [end_sec]
    return list(zip(segment_starts, segment_ends))


def _merge_short_segments(
    segments: List[Tuple[float, float]], min_segment_duration: float
) -> List[Tuple[float, float]]:
    """Merge segments shorter than min_segment_duration into their shorter neighbor.

    First, group consecutive short segments together, then merge each group into
    whichever adjacent (non-short) neighbor is shorter.
    """
    if min_segment_duration <= 0 or len(segments) <= 1:
        return list(segments)

    dur = lambda s: s[1] - s[0]

    # Pass 1: group consecutive short segments into single spans
    groups: List[Tuple[Tuple[float, float], bool]] = []
    for seg in segments:
        short = dur(seg) < min_segment_duration
        if groups and groups[-1][1] and short:
            groups[-1] = ((groups[-1][0][0], seg[1]), True)
        else:
            groups.append((seg, short))

    # Pass 2: merge each short group into the shorter neighbor
    i = 0
    while i < len(groups):
        combined, is_short = groups[i]
        if not is_short:
            i += 1
            continue
        left = groups[i - 1][0] if i > 0 and not groups[i - 1][1] else None
        right = groups[i + 1][0] if i + 1 < len(groups) and not groups[i + 1][1] else None
        if left is None and right is None:
            i += 1
        elif left is None:
            groups[i + 1] = ((combined[0], right[1]), False)
            groups.pop(i)
        elif right is None:
            groups[i - 1] = ((left[0], combined[1]), False)
            groups.pop(i)
        elif dur(left) <= dur(right):
            groups[i - 1] = ((left[0], combined[1]), False)
            groups.pop(i)
        else:
            groups[i + 1] = ((combined[0], right[1]), False)
            groups.pop(i)

    return [seg for seg, _ in groups]


def scene_segments_from_frames(
    frames: List[np.ndarray],
    timestamps: List[float],
    start_sec: float,
    end_sec: float,
    clip_client,
    *,
    num_children: Optional[int] = None,
    k: float = 1.5,
    min_segments: int = 2,
    max_segments: int = 16,
    min_segment_duration: float = 0.0,
) -> List[Tuple[float, float]]:
    """Compute scene segments from already-decoded frames (no cv2 work).

    Same boundary logic as `segment_video_by_scene`, but lets the caller decode
    frames once and reuse them for the VLM caption call.
    """
    if len(frames) < 2:
        return [(start_sec, end_sec)]

    embeddings = _compute_clip_embeddings(frames, clip_client)
    deltas = _compute_frame_deltas(embeddings)

    if num_children is not None:
        boundary_indices = _detect_boundaries_topk(deltas, num_children - 1)
    else:
        boundary_indices = _detect_boundaries_threshold(
            deltas, k=k, min_segments=min_segments, max_segments=max_segments
        )

    segments = _boundaries_to_segments(timestamps, boundary_indices, start_sec, end_sec)
    segments = _merge_short_segments(segments, min_segment_duration)

    # Fallback: ensure we return at least 2 segments
    if len(segments) == 1:
        mid = (start_sec + end_sec) / 2.0
        segments = [(start_sec, mid), (mid, end_sec)]

    return segments


def segment_video_by_scene(
    video_path: str,
    start_sec: float,
    end_sec: float,
    clip_client,
    num_children: Optional[int] = None,
    fps: float = 1.0,
    max_frames: int = 64,
    short_side: int = -1,
    k: float = 1.5,
    min_segments: int = 2,
    max_segments: int = 16,
) -> List[Tuple[float, float]]:
    """Segment a video interval into scenes using CLIP frame embeddings."""
    timestamps, frames, _clamped_end = _extract_frames_uniform(
        video_path, start_sec, end_sec, fps=fps, max_frames=max_frames, short_side=short_side
    )

    if len(frames) < 2:
        return [(start_sec, end_sec)]

    embeddings = _compute_clip_embeddings(frames, clip_client)
    deltas = _compute_frame_deltas(embeddings)

    if num_children is not None:
        boundary_indices = _detect_boundaries_topk(deltas, num_children - 1)
    else:
        boundary_indices = _detect_boundaries_threshold(
            deltas, k=k, min_segments=min_segments, max_segments=max_segments
        )

    return _boundaries_to_segments(timestamps, boundary_indices, start_sec, end_sec)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_scene_segmenter(
    clip_client,
    fps: float = 1.0,
    max_frames: int = 64,
    short_side: int = -1,
    num_children: Optional[int] = None,
    k: float = 1.5,
    min_segments: int = 2,
    max_segments: int = 16,
    min_segment_duration: float = 1.0,
) -> Callable[[str, float, float], List[Tuple[float, float]]]:
    """
    Create a segment_fn for TrajectoryGenerator using CLIP scene detection.

    The returned callable splits a video interval into segments at visually
    distinct scene boundaries, detected via cosine distance between consecutive
    CLIP frame embeddings.

    Args:
        clip_client: clip_client.Client instance connected to a CLIP gRPC service
        fps: Frame sampling rate for scene detection
        max_frames: Maximum frames to sample per segment
        short_side: Resize shorter side to this value (-1 for no resize)
        num_children: If set, use top-k strategy for exactly this many segments.
                      If None, use adaptive threshold strategy.
        k: Z-score multiplier for threshold strategy (higher = fewer boundaries)
        min_segments: Minimum segments for threshold strategy
        max_segments: Maximum segments for threshold strategy
        min_segment_duration: Merge segments shorter than this (seconds)

    Returns:
        segment_fn: callable (video_path, start_sec, end_sec) -> List[Tuple[float, float]]
    """
    def segment_fn(
        video_path: str, start_sec: float, end_sec: float
    ) -> List[Tuple[float, float]]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                segments = segment_video_by_scene(
                    video_path,
                    start_sec,
                    end_sec,
                    clip_client,
                    num_children=num_children,
                    fps=fps,
                    max_frames=max_frames,
                    short_side=short_side,
                    k=k,
                    min_segments=min_segments,
                    max_segments=max_segments,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.warning(
                        "Scene segmentation failed after %d retries for %s [%.1f-%.1f]: %s",
                        max_retries, video_path, start_sec, end_sec, e,
                    )
                    return [(start_sec, end_sec)]

        segments = _merge_short_segments(segments, min_segment_duration)

        # Fallback: if only 1 segment remains, uniformly split into 2
        if len(segments) == 1:
            mid = (start_sec + end_sec) / 2.0
            segments = [(start_sec, mid), (mid, end_sec)]

        return segments

    return segment_fn
