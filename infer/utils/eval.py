"""
Evaluation metrics for Longvideohaystack benchmark.

Functions copied from the benchmark authors' evaluation code:
/mnt/arc/cezhang/projects/TStar/LVHaystackBench/val_tstar_results.py

This module provides:
- Temporal PRF (Precision, Recall, F1) metrics
- SSIM (Structural Similarity Index) metrics
- ANND (Average Nearest Neighbor Distance) metrics
"""

import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# SSIM Calculation Functions (copied from benchmark)
# -----------------------------------------------------------------------------

def gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    """Creates a 1D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def create_window(window_size: int, channel: int) -> torch.Tensor:
    """Creates a 2D Gaussian kernel window."""
    kernel_1d = gaussian_kernel(window_size, sigma=1.5).unsqueeze(1)
    window_2d = kernel_1d @ kernel_1d.T
    window = window_2d.expand(channel, 1, window_size, window_size)
    return window


def ssim_torch(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11,
               C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """Calculates the SSIM between two images using PyTorch."""
    channel = img1.size(0)
    window = create_window(window_size, channel).to(img1.device)
    mu1 = F.conv2d(img1.unsqueeze(0), window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2.unsqueeze(0), window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1.unsqueeze(0) * img1.unsqueeze(0), window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def pairwise_ssim(gt_frames: List[np.ndarray], pred_frames: List[np.ndarray]) -> np.ndarray:
    """
    Calculates pairwise SSIM between two lists of images.
    Returns a numpy array of shape (num_gt, num_pred) containing SSIM scores.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert images to torch tensors and scale to [0, 1]
    gt_tensors = [torch.tensor(frame, dtype=torch.float32, device=device) / 255.0 for frame in gt_frames]
    pred_tensors = [torch.tensor(frame, dtype=torch.float32, device=device) / 255.0 for frame in pred_frames]

    ssim_results = np.zeros((len(gt_tensors), len(pred_tensors)))
    for i in range(len(gt_tensors)):
        for j in range(len(pred_tensors)):
            ssim_score = ssim_torch(gt_tensors[i], pred_tensors[j])
            ssim_results[i, j] = ssim_score.item()
    return ssim_results


# -----------------------------------------------------------------------------
# Video Frame Extraction (copied from benchmark)
# -----------------------------------------------------------------------------

def load_video_fps(video_path: str) -> float:
    """
    Get the frames per second (FPS) of a video.

    Raises:
        ValueError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        raise ValueError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    logger.debug(f"Video FPS for {video_path}: {fps}")
    return fps


def extract_frames(video_path: str, frame_indices: List[int]) -> List[np.ndarray]:
    """
    Extract specified frames from a video.

    Args:
        video_path: Path to the video file.
        frame_indices: List of frame indices to extract.

    Returns:
        List of extracted frames (RGB format). If a frame is not read successfully,
        an empty numpy array is returned in its place.
    """
    if len(frame_indices) > 20:
        sampled_positions = np.linspace(0, len(frame_indices) - 1, 20).astype(int)
        frame_indices = [frame_indices[i] for i in sampled_positions]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        import os

        # Gather file diagnostics
        exists = os.path.exists(video_path)
        readable = os.access(video_path, os.R_OK) if exists else False
        size = os.path.getsize(video_path) if exists else 0

        error_details = []
        if not exists:
            error_details.append("file does not exist")
        elif not readable:
            error_details.append("file is not readable (permission denied)")
        elif size == 0:
            error_details.append("file is empty (0 bytes)")
        else:
            error_details.append(
                f"file exists ({size} bytes) but OpenCV cannot open it "
                "(possible codec/format issue)"
            )

        logger.error(
            f"Cannot open video file: {video_path}\n"
            f"  Details: {', '.join(error_details)}"
        )
        raise ValueError(f"Cannot open video file: {video_path}")

    # Get video properties for diagnostics
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / fps if fps > 0 else 0.0

    logger.debug(
        f"Video metadata for {video_path}: "
        f"{total_frames} frames, {fps:.2f} fps, {video_duration:.2f}s duration"
    )

    # Validate frame indices
    invalid_indices = [idx for idx in frame_indices if idx >= total_frames or idx < 0]
    if invalid_indices:
        logger.warning(
            f"Invalid frame indices requested for {video_path}: "
            f"{invalid_indices} (video has {total_frames} frames, 0-{total_frames-1}). "
            f"Clamping to valid range."
        )

    frames = []
    for idx in frame_indices:
        # Clamp to valid range as defensive measure
        clamped_idx = max(0, min(idx, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, clamped_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            logger.debug(f"Extracted frame {idx} from {video_path}")
        else:
            # Gather diagnostic information
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            stream_valid = cap.isOpened()
            frame_timestamp = idx / fps if fps > 0 else -1

            # Determine likely failure reason
            if not stream_valid:
                reason = "Video stream became invalid (possible corruption)"
            elif idx >= total_frames:
                reason = f"Frame index out of range (max: {total_frames-1})"
            elif idx < 0:
                reason = "Negative frame index"
            elif abs(current_pos - idx) > 1:
                reason = f"Seeking failed (sought to {idx}, at position {current_pos})"
            else:
                reason = "Frame read failed (codec/decode issue)"

            logger.warning(
                f"Failed to extract frame {idx} from {video_path}\n"
                f"  Reason: {reason}\n"
                f"  Video: {total_frames} frames, {fps:.2f} fps, {video_duration:.2f}s duration\n"
                f"  Requested: frame {idx} ({frame_timestamp:.2f}s)\n"
                f"  Position: frame {current_pos}"
            )

            frames.append(np.array([]))
    cap.release()
    return frames


# -----------------------------------------------------------------------------
# Metrics Calculation Functions (copied from benchmark)
# -----------------------------------------------------------------------------

def calculate_prf(list_gt: List[np.ndarray], list_pred: List[np.ndarray], threshold: int = 5) -> Tuple[float, float, float]:
    """
    Calculate average Temporal Precision, Recall and F1 Score based on frame distances.
    """
    precision_list, recall_list, f1_list = [], [], []
    for gt_array, pred_array in zip(list_gt, list_pred):
        if gt_array.size == 0 or pred_array.size == 0:
            continue
        # Compute the minimum absolute differences between predicted and ground truth frame numbers.
        distances_gt_to_pred = np.min(np.abs(gt_array[:, np.newaxis] - pred_array), axis=1)
        distances_pred_to_gt = np.min(np.abs(pred_array[:, np.newaxis] - gt_array), axis=1)

        covered_gt = np.sum(distances_gt_to_pred <= threshold)
        covered_pred = np.sum(distances_pred_to_gt <= threshold)
        total_gt_frames = len(gt_array)
        total_pred_frames = len(pred_array)

        precision = covered_pred / total_pred_frames if total_pred_frames > 0 else 0.0
        recall = covered_gt / total_gt_frames if total_gt_frames > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    avg_precision = np.mean(precision_list) if precision_list else 0.0
    avg_recall = np.mean(recall_list) if recall_list else 0.0
    avg_f1 = np.mean(f1_list) if f1_list else 0.0
    return avg_precision, avg_recall, avg_f1


def calculate_ssim_scores(list_gt_images: List[List[np.ndarray]], list_pred_images: List[List[np.ndarray]]) -> List[Tuple[float, float]]:
    """
    Calculate SSIM Precision and Recall for each video entry.

    For each video:
      - SSIM Precision is computed as the mean of the maximum SSIM values for each predicted frame (across all ground truth frames).
      - SSIM Recall is computed as the mean of the maximum SSIM values for each ground truth frame (across all predicted frames).
    """
    ssim_list = []
    pairs = list(zip(list_gt_images, list_pred_images))
    for gt_images, pred_images in tqdm(pairs, desc="Calculating SSIM"):
        if not gt_images or not pred_images:
            continue

        # Filter out empty images
        paired_gt = [img for img in gt_images if img.size > 0]
        paired_pred = [img for img in pred_images if img.size > 0]
        if not paired_gt or not paired_pred:
            continue

        ssim_matrix = pairwise_ssim(paired_gt, paired_pred)
        ssim_precision = np.mean(np.max(ssim_matrix, axis=0))
        ssim_recall = np.mean(np.max(ssim_matrix, axis=1))
        ssim_list.append((ssim_precision, ssim_recall))
    return ssim_list


def calculate_annd(list_gt: List[np.ndarray], list_pred: List[np.ndarray]) -> List[Tuple[float, float]]:
    """
    Calculate the Average Nearest Neighbor Distance (ANND) for each video entry.
    Returns a list of (precision, recall) tuples.
    """
    annd_list = []
    for gt_array, pred_array in zip(list_gt, list_pred):
        if gt_array.size == 0 or pred_array.size == 0:
            continue

        distances_gt_to_pred = np.min(np.abs(gt_array[:, np.newaxis] - pred_array), axis=1)
        distances_pred_to_gt = np.min(np.abs(pred_array[:, np.newaxis] - gt_array), axis=1)
        annd_precision = np.mean(distances_pred_to_gt)
        annd_recall = np.mean(distances_gt_to_pred)
        annd_list.append((annd_precision, annd_recall))
    return annd_list


# -----------------------------------------------------------------------------
# Main Evaluation Function
# -----------------------------------------------------------------------------

def evaluate_longvideohaystack(
    result_data: List[Dict[str, Any]],
    threshold: int = 5,
    max_workers: int = 4
) -> Dict[str, float]:
    """
    Calculate Temporal PRF and SSIM metrics for Longvideohaystack dataset.

    Args:
        result_data: List of dictionaries with keys:
            - 'video_path': Path to video file
            - 'keyframe_timestamps': List of predicted timestamps (seconds)
            - 'gt_frame_index': List of ground truth frame indices
        threshold: Distance threshold for PRF calculation (default: 5)
        max_workers: Number of threads for parallel processing (default: 4)

    Returns:
        Dictionary with metric names and values
    """
    logger.info(f"Evaluating {len(result_data)} samples for Longvideohaystack metrics...")

    # Prepare data structures
    list_gt = []
    list_pred = []
    list_gt_images = []
    list_pred_images = []

    # Extract frames in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}

        for idx, item in enumerate(result_data):
            try:
                video_path = item['video_path']
                pred_timestamps = item['keyframe_timestamps']
                gt_frame_indices = item['gt_frame_index']

                # Cap predicted keyframes at 20 via uniform subsampling.
                if len(pred_timestamps) > 20:
                    sampled_positions = np.linspace(0, len(pred_timestamps) - 1, 20).astype(int)
                    pred_timestamps = [pred_timestamps[i] for i in sampled_positions]

                # Get video FPS for conversion
                fps = load_video_fps(video_path)

                # Convert ground truth frame indices to seconds
                gt_seconds = np.array([f / fps for f in gt_frame_indices])

                # Predicted timestamps are already in seconds
                pred_seconds = np.array(pred_timestamps)

                list_gt.append(gt_seconds)
                list_pred.append(pred_seconds)

                # Prepare frame indices for extraction
                gt_frame_nums = np.array(gt_frame_indices, dtype=int)
                pred_frame_nums = np.array([int(ts * fps) for ts in pred_timestamps], dtype=int)
                combined_frames = gt_frame_nums.tolist() + pred_frame_nums.tolist()

                # Submit frame extraction task
                future = executor.submit(extract_frames, video_path, combined_frames)
                future_to_idx[future] = (idx, len(gt_frame_nums), len(pred_frame_nums))

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                list_gt.append(np.array([]))
                list_pred.append(np.array([]))
                list_gt_images.append([])
                list_pred_images.append([])

        # Collect results
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Extracting frames"):
            idx, gt_num, pred_num = future_to_idx[future]
            try:
                frames = future.result()
                if not frames:
                    logger.warning(f"No frames extracted for sample {idx}")
                    list_gt_images.append([])
                    list_pred_images.append([])
                    continue

                # Split frames into GT and predicted
                gt_images = frames[:gt_num]
                pred_images = frames[gt_num:gt_num + pred_num]

                list_gt_images.append(gt_images)
                list_pred_images.append(pred_images)
            except Exception as e:
                logger.error(f"Error extracting frames for sample {idx}: {e}")
                list_gt_images.append([])
                list_pred_images.append([])

    # Calculate Temporal PRF
    logger.info("Calculating Temporal PRF Scores...")
    avg_precision, avg_recall, avg_f1 = calculate_prf(list_gt, list_pred, threshold=threshold)
    logger.info(f"Temporal - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")

    # Calculate SSIM
    logger.info("Calculating SSIM Scores...")
    ssim_scores = calculate_ssim_scores(list_gt_images, list_pred_images)
    if ssim_scores:
        avg_ssim_precision = np.mean([s[0] for s in ssim_scores])
        avg_ssim_recall = np.mean([s[1] for s in ssim_scores])
        if avg_ssim_precision + avg_ssim_recall > 0:
            ssim_f1 = 2 * avg_ssim_precision * avg_ssim_recall / (avg_ssim_precision + avg_ssim_recall)
        else:
            ssim_f1 = 0.0
        logger.info(f"SSIM - Precision: {avg_ssim_precision:.4f}, Recall: {avg_ssim_recall:.4f}, F1: {ssim_f1:.4f}")
    else:
        avg_ssim_precision = avg_ssim_recall = ssim_f1 = 0.0
        logger.warning("No SSIM scores were calculated.")

    return {
        "Average Temporal Precision": avg_precision,
        "Average Temporal Recall": avg_recall,
        "Average Temporal F1 Score": avg_f1,
        "Average SSIM Precision": avg_ssim_precision,
        "Average SSIM Recall": avg_ssim_recall,
        "Average SSIM F1 Score": ssim_f1
    }


# -----------------------------------------------------------------------------
# MomentSeeker R@1 Evaluation Functions
# -----------------------------------------------------------------------------

def _extract_prediction(sample: Dict[str, Any]) -> Tuple[float, float] | None:
    """
    Extract predicted interval from sample data.

    Args:
        sample: Sample dictionary with prediction information

    Returns:
        Tuple of (start, end) timestamps, or None if not found
    """
    # Option 1: Pre-extracted prediction
    if 'predicted_timestamps' in sample and sample['predicted_timestamps']:
        return tuple(sample['predicted_timestamps'])

    # Option 2: Extract from trajectory turns
    if 'turns' in sample and len(sample['turns']) > 0:
        last_turn = sample['turns'][-1]
        if last_turn.get('action') and last_turn['action'].get('evidence'):
            timestamps = last_turn['action']['evidence'].get('timestamps')
            if timestamps:
                return tuple(timestamps)

    return None


def _extract_gt_intervals(sample: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Extract ground truth intervals from sample data.
    Supports multi-interval ground truth.

    Args:
        sample: Sample dictionary with ground truth information

    Returns:
        List of (start, end) tuples for all ground truth intervals
    """
    # Multi-interval support: use gt_all_timestamps if available
    if 'gt_all_timestamps' in sample and sample['gt_all_timestamps']:
        return [tuple(iv) for iv in sample['gt_all_timestamps']]

    # Single interval fallback
    if 'gt_timestamps' in sample and sample['gt_timestamps']:
        return [tuple(sample['gt_timestamps'])]

    return []


def _validate_interval(interval: Tuple[float, float]) -> bool:
    """
    Validate that interval is well-formed.

    Args:
        interval: Tuple of (start, end) timestamps

    Returns:
        True if valid, False otherwise
    """
    start, end = interval
    if start is None or end is None:
        return False
    return start >= 0 and end > start


def evaluate_momentseeker(
    result_data: List[Dict[str, Any]],
    iou_thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]
) -> Dict[str, float]:
    """
    Calculate R@1 metrics for MomentSeeker temporal localization task.

    R@1 (Recall at 1): Percentage of queries where the top-1 prediction
    has IoU >= threshold with ANY ground truth interval.

    This metric supports multi-interval ground truth, where a query can have
    multiple valid temporal moments. The prediction is considered correct if
    it has sufficient overlap (IoU >= threshold) with at least one ground truth.

    Args:
        result_data: List of dictionaries with keys:
            - 'gt_timestamps': Primary ground truth interval [start, end]
            - 'gt_all_timestamps' (optional): All valid intervals [[start1, end1], ...]
            - 'turns': List of trajectory turns (to extract prediction)
            OR
            - 'predicted_timestamps': Pre-extracted prediction [start, end]
        iou_thresholds: List of IoU thresholds (default: [0.1, 0.2, 0.3, 0.4, 0.5])

    Returns:
        Dictionary with keys:
            - 'R@1_IoU=0.1': Recall at IoU threshold 0.1
            - 'R@1_IoU=0.2': Recall at IoU threshold 0.2
            - 'R@1_IoU=0.3': Recall at IoU threshold 0.3 (main metric)
            - 'R@1_IoU=0.4': Recall at IoU threshold 0.4
            - 'R@1_IoU=0.5': Recall at IoU threshold 0.5
            - 'num_samples': Number of samples evaluated
    """
    from utils.temporal import compute_interval_iou

    logger.info(f"Evaluating {len(result_data)} samples for MomentSeeker R@1 metrics...")

    valid_samples = []
    skipped_count = 0

    for idx, sample in enumerate(result_data):
        # Step 1: Extract prediction
        pred_interval = _extract_prediction(sample)
        if pred_interval is None:
            logger.warning(f"Sample {idx}: No valid prediction found, skipping")
            skipped_count += 1
            continue

        # Step 2: Collect GT intervals
        gt_intervals = _extract_gt_intervals(sample)
        if not gt_intervals:
            logger.warning(f"Sample {idx}: No ground truth intervals found, skipping")
            skipped_count += 1
            continue

        # Validate intervals
        if not _validate_interval(pred_interval):
            logger.warning(f"Sample {idx}: Invalid prediction interval {pred_interval}, skipping")
            skipped_count += 1
            continue

        valid_samples.append({
            'pred': pred_interval,
            'gt_intervals': gt_intervals
        })

    if not valid_samples:
        logger.error("No valid samples found for evaluation!")
        return {f'R@1_IoU={t}': 0.0 for t in iou_thresholds} | {'num_samples': 0}

    logger.info(f"Evaluating {len(valid_samples)} valid samples (skipped {skipped_count})")

    # Step 3: Compute max IoU for each sample
    max_ious = []
    for sample in valid_samples:
        max_iou = 0.0
        for gt_interval in sample['gt_intervals']:
            iou = compute_interval_iou(sample['pred'], gt_interval)
            max_iou = max(max_iou, iou)
        max_ious.append(max_iou)

    # Step 4: Calculate R@1 at each threshold
    metrics = {}
    for threshold in iou_thresholds:
        correct_count = sum(1 for iou in max_ious if iou >= threshold)
        recall = correct_count / len(valid_samples)
        metrics[f'R@1_IoU={threshold}'] = recall

    metrics['num_samples'] = len(valid_samples)

    logger.info("MomentSeeker R@1 Metrics:")
    for threshold in iou_thresholds:
        logger.info(f"  R@1@IoU={threshold}: {metrics[f'R@1_IoU={threshold}']:.4f}")

    return metrics


# -----------------------------------------------------------------------------
# Video-MME Evaluation Functions
# -----------------------------------------------------------------------------

def evaluate_videomme(
    result_data: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Calculate accuracy metrics for Video-MME benchmark, broken down by
    duration category (short/medium/long).

    Args:
        result_data: List of dictionaries with keys:
            - 'answer_correct': bool indicating if the predicted answer is correct
            - 'sub_category': duration category string ('short', 'medium', or 'long')

    Returns:
        Dictionary with overall and per-split accuracy metrics.
    """
    logger.info(f"Evaluating {len(result_data)} samples for Video-MME metrics...")

    # Group by duration category
    groups: Dict[str, List[bool]] = {}
    all_correct = []

    for sample in result_data:
        correct = sample.get('answer_correct')
        if correct is None:
            continue
        all_correct.append(correct)
        category = sample.get('sub_category', 'unknown')
        groups.setdefault(category, []).append(correct)

    if not all_correct:
        logger.error("No valid samples found for evaluation!")
        return {'overall_accuracy': 0.0, 'num_samples': 0}

    metrics = {
        'overall_accuracy': sum(all_correct) / len(all_correct),
        'num_samples': len(all_correct),
    }

    for category in ['short', 'medium', 'long']:
        samples = groups.get(category, [])
        acc = sum(samples) / len(samples) if samples else 0.0
        metrics[f'{category}_accuracy'] = acc
        metrics[f'{category}_count'] = len(samples)

    logger.info("Video-MME Accuracy Metrics:")
    logger.info(f"  Overall: {metrics['overall_accuracy']:.4f} ({metrics['num_samples']} samples)")
    for category in ['short', 'medium', 'long']:
        logger.info(f"  {category.capitalize()}: {metrics[f'{category}_accuracy']:.4f} ({metrics[f'{category}_count']} samples)")

    return metrics


# -----------------------------------------------------------------------------
# MLVU Evaluation Functions
# -----------------------------------------------------------------------------

def evaluate_mlvu(
    result_data: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Calculate accuracy metrics for MLVU benchmark, broken down by question_type.

    Args:
        result_data: List of dictionaries with keys:
            - 'answer_correct': bool indicating if the predicted answer is correct
            - 'question_type': question type string (e.g. 'plotQA', 'needleQA', etc.)

    Returns:
        Dictionary with overall and per-question_type accuracy metrics.
    """
    logger.info(f"Evaluating {len(result_data)} samples for MLVU metrics...")

    groups: Dict[str, List[bool]] = {}
    all_correct = []

    for sample in result_data:
        correct = sample.get('answer_correct')
        if correct is None:
            continue
        all_correct.append(correct)
        qtype = sample.get('question_type', 'unknown')
        groups.setdefault(qtype, []).append(correct)

    if not all_correct:
        logger.error("No valid samples found for evaluation!")
        return {'overall_accuracy': 0.0, 'num_samples': 0}

    metrics = {
        'overall_accuracy': sum(all_correct) / len(all_correct),
        'num_samples': len(all_correct),
    }

    for qtype, samples in sorted(groups.items()):
        acc = sum(samples) / len(samples) if samples else 0.0
        metrics[f'{qtype}_accuracy'] = acc
        metrics[f'{qtype}_count'] = len(samples)

    logger.info("MLVU Accuracy Metrics:")
    logger.info(f"  Overall: {metrics['overall_accuracy']:.4f} ({metrics['num_samples']} samples)")
    for qtype in sorted(groups.keys()):
        logger.info(f"  {qtype}: {metrics[f'{qtype}_accuracy']:.4f} ({metrics[f'{qtype}_count']} samples)")

    return metrics


# -----------------------------------------------------------------------------
# LongVideoBench Evaluation Functions
# -----------------------------------------------------------------------------

def evaluate_longvideobench(
    result_data: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Calculate accuracy metrics for LongVideoBench benchmark, broken down by
    question_category.

    Args:
        result_data: List of dictionaries with keys:
            - 'answer_correct': bool indicating if the predicted answer is correct
            - 'question_category': question category string (e.g. 'TOS', 'SSS', etc.)

    Returns:
        Dictionary with overall and per-question_category accuracy metrics.
    """
    logger.info(f"Evaluating {len(result_data)} samples for LongVideoBench metrics...")

    groups: Dict[str, List[bool]] = {}
    all_correct = []

    for sample in result_data:
        correct = sample.get('answer_correct')
        if correct is None:
            continue
        all_correct.append(correct)
        category = sample.get('question_category', 'unknown')
        groups.setdefault(category, []).append(correct)

    if not all_correct:
        logger.error("No valid samples found for evaluation!")
        return {'overall_accuracy': 0.0, 'num_samples': 0}

    metrics = {
        'overall_accuracy': sum(all_correct) / len(all_correct),
        'num_samples': len(all_correct),
    }

    for category, samples in sorted(groups.items()):
        acc = sum(samples) / len(samples) if samples else 0.0
        metrics[f'{category}_accuracy'] = acc
        metrics[f'{category}_count'] = len(samples)

    logger.info("LongVideoBench Accuracy Metrics:")
    logger.info(f"  Overall: {metrics['overall_accuracy']:.4f} ({metrics['num_samples']} samples)")
    for category in sorted(groups.keys()):
        logger.info(f"  {category}: {metrics[f'{category}_accuracy']:.4f} ({metrics[f'{category}_count']} samples)")

    return metrics
