"""
Utility functions for trajectory generation.
"""

from typing import List, Tuple


def compute_interval_iou(
    interval1: Tuple[float, float],
    interval2: Tuple[float, float]
) -> float:
    """
    Compute Intersection over Union (IoU) between two temporal intervals.

    Args:
        interval1: First interval as (start_sec, end_sec)
        interval2: Second interval as (start_sec, end_sec)

    Returns:
        IoU value in range [0.0, 1.0]
    """
    start1, end1 = interval1
    start2, end2 = interval2

    # Calculate intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    if intersection_start >= intersection_end:
        return 0.0  # No overlap

    intersection = intersection_end - intersection_start
    union = (end1 - start1) + (end2 - start2) - intersection

    return intersection / union if union > 0 else 0.0


def compute_interval_coverage(
    interval: Tuple[float, float],
    reference: Tuple[float, float]
) -> float:
    """
    Compute what percentage of the reference interval is covered by interval.

    Args:
        interval: The interval being measured
        reference: The reference interval (e.g., GT interval)

    Returns:
        Coverage ratio in range [0.0, 1.0]
    """
    start1, end1 = interval
    ref_start, ref_end = reference

    intersection_start = max(start1, ref_start)
    intersection_end = min(end1, ref_end)

    if intersection_start >= intersection_end:
        return 0.0

    intersection = intersection_end - intersection_start
    ref_duration = ref_end - ref_start

    return intersection / ref_duration if ref_duration > 0 else 0.0


def compute_multi_interval_iou(
    interval: Tuple[float, float],
    gt_intervals: List[Tuple[float, float]]
) -> float:
    """
    Compute IoU between a single predicted interval and multiple GT intervals.

    Intersection is the sum of overlaps between the prediction and each GT interval.
    Union is the total time covered by the prediction OR any GT interval (merged).

    Args:
        interval: Predicted interval as (start_sec, end_sec)
        gt_intervals: List of GT intervals as [(start, end), ...]

    Returns:
        IoU value in range [0.0, 1.0]
    """
    if not gt_intervals:
        return 0.0

    pred_start, pred_end = interval

    # Calculate total intersection: sum of overlaps with each GT segment
    total_intersection = 0.0
    for gt_start, gt_end in gt_intervals:
        inter_start = max(pred_start, gt_start)
        inter_end = min(pred_end, gt_end)
        if inter_start < inter_end:
            total_intersection += inter_end - inter_start

    # Calculate union: merge all segments (pred + all GTs) and sum durations
    all_segments = [(pred_start, pred_end)] + list(gt_intervals)
    all_segments.sort(key=lambda x: x[0])

    merged = []
    for start, end in all_segments:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    total_union = sum(end - start for start, end in merged)

    return total_intersection / total_union if total_union > 0 else 0.0


def compute_multi_interval_coverage(
    interval: Tuple[float, float],
    gt_intervals: List[Tuple[float, float]]
) -> float:
    """
    Compute what percentage of the total GT duration is covered by the predicted interval.

    Args:
        interval: Predicted interval as (start_sec, end_sec)
        gt_intervals: List of GT intervals as [(start, end), ...]

    Returns:
        Coverage ratio in range [0.0, 1.0]
    """
    if not gt_intervals:
        return 0.0

    pred_start, pred_end = interval

    # Calculate total intersection: sum of overlaps with each GT segment
    total_intersection = 0.0
    for gt_start, gt_end in gt_intervals:
        inter_start = max(pred_start, gt_start)
        inter_end = min(pred_end, gt_end)
        if inter_start < inter_end:
            total_intersection += inter_end - inter_start

    # Denominator: total GT duration
    total_gt_duration = sum(end - start for start, end in gt_intervals)

    return total_intersection / total_gt_duration if total_gt_duration > 0 else 0.0
