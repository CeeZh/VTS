"""
Keyframe sampling strategies for deriving predicted keyframe timestamps from
an evidence interval.

Two strategies are provided:

- `sample_keyframes_uniform(start, end, n)`: uniform linspace sampling.
- `sample_keyframes_clip(video_path, query, start, end, clip_client, ...)`:
  candidate frames are extracted at `fps` (default 0.5) within the evidence
  interval and ranked by CLIP cosine similarity against the query. When the
  FPS-driven candidate pool already exceeds `max_frames`, we fall back to a
  uniform linspace of `max_frames` timestamps (no CLIP call). Returned
  timestamps are sorted by descending query similarity so downstream top-K
  evaluation receives the most relevant frames first.
"""

from typing import List, Optional

import cv2
import numpy as np


def sample_keyframes_uniform(
    start_sec: float, end_sec: float, num_frames: int = 8
) -> List[float]:
    """Uniformly sample `num_frames` timestamps in [start_sec, end_sec]."""
    return np.linspace(start_sec, end_sec, num_frames).tolist()


def _extract_frames_at_timestamps(
    video_path: str,
    timestamps: List[float],
    short_side: int = -1,
) -> tuple[List[float], List[np.ndarray]]:
    """Decode RGB frames at the requested timestamps. Returns (kept_ts, frames)."""
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    max_end_sec = (total_frames - 1) / video_fps if video_fps > 0 else float("inf")

    kept_ts: List[float] = []
    frames: List[np.ndarray] = []
    for ts in timestamps:
        ts = min(ts, max_end_sec)
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
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        frames.append(frame_rgb)
        kept_ts.append(ts)

    cap.release()
    return kept_ts, frames


def sample_keyframes_clip(
    video_path: str,
    query: str,
    start_sec: float,
    end_sec: float,
    clip_client,
    fps: float = 0.5,
    max_frames: int = 64,
    candidate_cap: int = 64,
    short_side: int = -1,
) -> List[float]:
    """
    Select keyframes from [start_sec, end_sec] using CLIP query-frame similarity.

    Candidate extraction density is controlled by `fps` (default 0.5). To bound
    CLIP encoding cost, the candidate pool is capped at `candidate_cap` frames
    (uniform subsample when `fps * duration` exceeds this). Candidates are
    scored against `query` and the top `max_frames` (by cosine similarity) are
    returned in temporal order. When the candidate pool is already at or below
    `max_frames`, all candidates are returned without ranking.

    Args:
        video_path: Path to the video file.
        query: Text query to score frames against.
        start_sec, end_sec: Evidence interval.
        clip_client: `clip_client.Client` instance (accepts text + tensor Documents).
        fps: Candidate sampling rate within the interval.
        max_frames: Final cap on returned timestamps (top-K by CLIP similarity).
        candidate_cap: Hard cap on candidates encoded with CLIP.
        short_side: Resize short side before CLIP encoding (-1 disables resize).

    Returns:
        List of timestamps (seconds), length <= max_frames, in temporal order.
    """
    duration = max(0.0, end_sec - start_sec)
    n_from_fps = max(2, int(duration * fps) + 1)
    n_candidates = min(n_from_fps, candidate_cap)

    candidate_ts = np.linspace(start_sec, end_sec, n_candidates).tolist()
    kept_ts, frames = _extract_frames_at_timestamps(
        video_path, candidate_ts, short_side=short_side
    )

    if len(frames) == 0:
        # Nothing decoded — fall back to a uniform linspace at the requested density.
        return np.linspace(start_sec, end_sec, min(n_candidates, max_frames)).tolist()
    if len(frames) <= max_frames:
        # Candidate pool already within budget; no ranking needed.
        return kept_ts

    from docarray import Document

    query_emb = clip_client.encode([Document(text=query)]).embeddings[0]
    frame_embs = clip_client.encode([Document(tensor=f) for f in frames]).embeddings

    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    f_norms = frame_embs / (np.linalg.norm(frame_embs, axis=1, keepdims=True) + 1e-8)
    sims = f_norms @ q_norm

    top_idx = np.argsort(-sims)[:max_frames]
    top_idx = sorted(top_idx.tolist())  # temporal order
    return [float(kept_ts[i]) for i in top_idx]


def derive_keyframe_timestamps(
    mode: str,
    start_sec: float,
    end_sec: float,
    *,
    video_path: Optional[str] = None,
    query: Optional[str] = None,
    clip_client=None,
    uniform_num_frames: int = 8,
    clip_fps: float = 0.5,
    clip_max_frames: int = 64,
    clip_short_side: int = -1,
) -> List[float]:
    """
    Dispatch to a keyframe sampling strategy.

    mode: "uniform" or "clip".
    """
    if mode == "uniform":
        return sample_keyframes_uniform(start_sec, end_sec, num_frames=uniform_num_frames)
    if mode == "clip":
        if clip_client is None or video_path is None or query is None:
            raise ValueError(
                "mode='clip' requires video_path, query, and clip_client arguments."
            )
        return sample_keyframes_clip(
            video_path=video_path,
            query=query,
            start_sec=start_sec,
            end_sec=end_sec,
            clip_client=clip_client,
            fps=clip_fps,
            max_frames=clip_max_frames,
            short_side=clip_short_side,
        )
    raise ValueError(f"Unknown keyframe sampling mode: {mode!r}")
