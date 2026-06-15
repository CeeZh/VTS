import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def _probe_video_metadata(video_path: str):
    """Return {'fps', 'duration'} for a video file, or None if unreadable.

    Kept module-level so it can be dispatched to a ThreadPoolExecutor without
    pickling a closure.
    """
    import cv2  # local import to keep dataset import light for non-video paths

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not fps or not frame_count:
        return None
    return {'fps': fps, 'duration': frame_count / fps}


def _probe_videos_parallel(paths_by_id: dict, max_workers: int = 32) -> dict:
    """Probe many videos concurrently, returning {video_id: meta} for readable ones."""
    if not paths_by_id:
        return {}
    ids = list(paths_by_id.keys())
    paths = [paths_by_id[vid] for vid in ids]
    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(ids)))) as ex:
        results = list(ex.map(_probe_video_metadata, paths))
    return {vid: meta for vid, meta in zip(ids, results) if meta is not None}


def load_cgbench_data(anno_path, video_base_path, num_samples=None, shuffle=False, multi_interval=False, add_cannot_determine=False):
    """
    Load CGBench dataset annotations and prepare them for trajectory generation.

    Args:
        anno_path: Path to cgbench.json
        video_base_path: Base path to video files
        num_samples: Number of samples to load (None for all)
        shuffle: Whether to shuffle the samples
        multi_interval: If True, include all GT intervals in 'gt_all_timestamps' key.
                        gt_timestamps is always set to the first interval for backward compat.

    Returns:
        List of dictionaries with video data
    """
    # Load annotations
    with open(anno_path, 'r') as f:
        annotations = json.load(f)

    # Limit number of samples if specified
    if num_samples is not None:
        annotations = annotations[:num_samples]

    # Shuffle samples if specified
    if shuffle:
        random.shuffle(annotations)

    # Convert to format expected by generator
    dataset = []
    for anno in annotations:
        video_uid = anno.get('video_uid', anno.get('video_id'))
        video_path = Path(video_base_path) / f"{video_uid}.mp4"

        if 'clue_intervals' in anno:
            if not anno['clue_intervals']:
                print(f"Warning: No clue intervals for qid {anno['qid']}, skipping")
                continue

            all_intervals = [(iv[0], iv[1]) for iv in anno['clue_intervals']]
            gt_start, gt_end = all_intervals[0]
        elif 'gt_timestamps' in anno:
            all_intervals = [anno['gt_timestamps']]
            gt_start, gt_end = anno['gt_timestamps']
        else:
            print(f"Warning: No clue intervals or gt_timestamps for qid {anno['qid']}, skipping")
            continue

        entry = {
            'video_path': str(video_path),
            'video_id': video_uid,
            'qid': anno['qid'],
            'question': anno['question'],
            'answer': anno['answer'],
            'right_answer': anno['right_answer'],
            'choices': anno['choices'],
            'gt_timestamps': [float(gt_start), float(gt_end)],
            'duration': float(anno['duration']),
            'domain': anno.get('domain', 'Unknown'),
            'sub_category': anno.get('sub_category', 'Unknown'),
        }
        if add_cannot_determine and entry['choices']:
            entry['choices'].append('Given the current information, the question cannot be answered')
        if multi_interval:
            entry['gt_all_timestamps'] = [[float(start), float(end)] for start, end in all_intervals]
        dataset.append(entry)

    return dataset


def load_cgbench_data_gen(anno_path, video_base_path, num_samples=None, shuffle=False, multi_interval=False, add_cannot_determine=False):
    """
    Load CGBench dataset annotations and prepare them for trajectory generation.

    Args:
        anno_path: Path to cgbench.json
        video_base_path: Base path to video files
        num_samples: Number of samples to load (None for all)
        shuffle: Whether to shuffle the samples
        multi_interval: If True, include all GT intervals in 'gt_all_timestamps' key.
                        gt_timestamps is always set to the first interval for backward compat.

    Returns:
        List of dictionaries with video data
    """
    # Load annotations
    with open(anno_path, 'r') as f:
        annotations = json.load(f)

    # Limit number of samples if specified
    if num_samples is not None:
        annotations = annotations[:num_samples]

    # Shuffle samples if specified
    if shuffle:
        random.shuffle(annotations)

    # Convert to format expected by generator
    dataset = []
    for anno in annotations:
        video_uid = anno.get('video_uid', anno.get('video_id'))
        video_path = Path(video_base_path) / f"{video_uid}.mp4"

        entry = {
            'video_path': str(video_path),
            'video_id': video_uid,
            'qid': anno['qid'],
            'question': anno['question'],
            'answer': anno['answer'],
            'right_answer': anno['right_answer'],
            'choices': anno['choices'],
            'gt_timestamps': anno['gt_timestamps'],
            'duration': float(anno['duration']),
            'domain': anno.get('domain', 'Unknown'),
            'sub_category': anno.get('sub_category', 'Unknown'),
        }
        dataset.append(entry)

    return dataset


def load_longvideohaystack_data(anno_path, video_base_path, num_samples=None, shuffle=False, add_cannot_determine=False):
    """
    Load LongVideoHaystack dataset from parquet or JSON file.

    Args:
        anno_path: Path to parquet or JSON file
        video_base_path: Base path to video files
        num_samples: Number of samples to load (None for all)
        shuffle: Whether to shuffle the samples

    Returns:
        List of dictionaries with video data (CGBench-compatible format)
    """
    if str(anno_path).endswith('.json'):
        with open(anno_path, 'r') as f:
            dataset = json.load(f)
        if shuffle:
            random.shuffle(dataset)
        if num_samples is not None:
            dataset = dataset[:num_samples]
        if add_cannot_determine:
            for entry in dataset:
                if entry.get('choices'):
                    entry['choices'] = entry['choices'] + ['Given the current information, the question cannot be answered']
        return dataset

    df = pd.read_parquet(anno_path)

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    if num_samples is not None:
        df = df.head(num_samples)

    dataset = []
    for global_idx, row in df.iterrows():
        frame_indexes = row['frame_indexes_video']

        # Skip if frame_indexes_video is empty
        if frame_indexes is None or len(frame_indexes) == 0:
            print(f"Warning: Empty frame_indexes_video for video {row['video_id']}, skipping")
            continue

        metadata = row['video_metadata']
        frame_rate = float(metadata['frame_rate'])
        video_duration = float(metadata['video_duration'])

        # Build clue interval from frame_indexes_video with ±1s window
        timestamps = np.array(frame_indexes) / frame_rate
        gt_start = max(0.0, float(np.min(timestamps)) - 1.0)
        gt_end = min(video_duration, float(np.max(timestamps)) + 1.0)

        # Build video path
        video_path = Path(video_base_path) / f"{row['video_id']}.mp4"

        # Convert options dict to sorted list of choice strings
        options_dict = row['options']
        choices = [options_dict[k] for k in sorted(options_dict)]
        answer_text = options_dict.get(row['answer'], row['answer'])

        if add_cannot_determine and choices:
            choices = choices + ['Given the current information, the question cannot be answered']

        dataset.append({
            'video_path': str(video_path),
            'video_id': row['video_id'],
            'qid': global_idx,
            'question': row['question'],
            'answer': answer_text,
            'right_answer': row['answer'],
            'choices': choices,
            'gt_timestamps': [gt_start, gt_end],
            'duration': video_duration,
            'frame_indexes_video': [int(x) for x in frame_indexes],
            'frame_rate': frame_rate,
            'domain': 'LongVideoHaystack',
            'sub_category': 'LongVideoHaystack',
        })

    return dataset


def load_tstar_data(anno_path, video_base_path, num_samples=None, shuffle=False, add_cannot_determine=False):
    """
    Load TStar-format dataset from JSON file.

    Args:
        anno_path: Path to TStar JSON annotation file
        video_base_path: Base path to video files
        fps: Frame rate of the videos (used to convert frame indices to timestamps)
        duration: Video duration in seconds
        num_samples: Number of samples to load (None for all)
        shuffle: Whether to shuffle the samples

    Returns:
        List of dictionaries with video data (CGBench-compatible format)
    """
    with open(anno_path, 'r') as f:
        annotations = json.load(f)

    if shuffle:
        random.shuffle(annotations)

    if num_samples is not None:
        annotations = annotations[:num_samples]

    dataset = []
    global_idx = 0
    for entry in annotations:
        frame_indexes = entry.get('gt_frame_index', [])
        if not frame_indexes:
            print(f"Warning: No gt_frame_index for video {entry['video_id']}, skipping")
            continue

        # Parse options string: "A) text\nB) text\n..."
        choices = []
        letter_to_text = {}
        for line in entry['options'].strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            match = re.match(r'^([A-Z])\)\s*(.+)$', line)
            if match:
                letter = match.group(1)
                text = match.group(2).strip()
                choices.append(text)
                letter_to_text[letter] = text
            else:
                choices.append(line)

        answer_letter = entry['answer']
        answer_text = letter_to_text.get(answer_letter, answer_letter)

        # Convert frame indices to timestamps
        timestamps = np.array(frame_indexes) / entry['fps']
        half_duration = 2

        if len(timestamps) == 1:
            center_time = float(timestamps[0])
            gt_start = float(max(0, center_time - half_duration))
            gt_end = float(min(entry['duration'], center_time + half_duration))
        else:
            gt_start = float(np.min(timestamps))
            gt_start = float(max(0, gt_start - half_duration))
            gt_end = float(np.max(timestamps))
            gt_end = float(min(entry['duration'], gt_end + half_duration))

        video_path = Path(video_base_path) / f"{entry['video_id']}.mp4"

        if add_cannot_determine and choices:
            choices = choices + ['Given the current information, the question cannot be answered']

        dataset.append({
            'video_path': str(video_path),
            'video_id': entry['video_id'],
            'qid': global_idx,
            'question': entry['question'],
            'answer': answer_text,
            'right_answer': answer_letter,
            'choices': choices,
            'gt_timestamps': [gt_start, gt_end],
            'duration': entry['duration'],
            'frame_indexes_video': sorted([int(x) for x in frame_indexes]),
            'frame_rate': entry['fps'],
            'domain': 'TStar',
            'sub_category': 'TStar',
        })
        global_idx += 1

    return dataset


def load_momentseeker_data(anno_path, video_base_path, num_samples=None, shuffle=False, multi_interval=False):
    """
    Load MomentSeeker dataset (temporal localization task).

    Args:
        anno_path: Path to t2v.json annotation file
        video_base_path: Base path to video directory
        num_samples: Number of samples to load (None for all)
        shuffle: Whether to shuffle the samples
        multi_interval: If True, include all intervals in 'gt_all_timestamps' key

    Returns:
        List of dictionaries with video data (CGBench-compatible format)
    """
    import cv2

    with open(anno_path, 'r') as f:
        annotations = json.load(f)

    if shuffle:
        random.shuffle(annotations)

    if num_samples is not None:
        annotations = annotations[:num_samples]

    dataset = []
    for idx, item in enumerate(annotations):
        # Extract video ID from path (e.g., "./videos/ego_79.mp4" → "ego_79")
        video_filename = Path(item['src_video_path']).stem
        video_path = Path(video_base_path) / f"{video_filename}.mp4"

        # Skip if video doesn't exist
        if not video_path.exists():
            print(f"Warning: Video not found: {video_path}")
            continue

        # Get video duration using cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        if duration == 0:
            print(f"Warning: Could not determine duration for {video_filename}")
            continue

        intervals = item['answering_time_interval']
        if not intervals or len(intervals) == 0:
            continue

        # Use first interval as primary gt_timestamps
        gt_timestamps = [float(intervals[0][0]), float(intervals[0][1])]

        data_item = {
            'video_path': str(video_path),
            'video_id': video_filename,
            'qid': idx,  # Use index as question ID
            'question': item['qry_text'],
            'answer': 'A',  # Not applicable for localization task
            'right_answer': 'A',  # Not applicable
            'choices': [],  # Not applicable
            'gt_timestamps': gt_timestamps,
            'duration': duration,
            'domain': 'MomentSeeker',
            'sub_category': item.get('task', 'Description Location'),
        }

        # Add all intervals if requested
        if multi_interval and len(intervals) > 1:
            data_item['gt_all_timestamps'] = [
                [float(iv[0]), float(iv[1])] for iv in intervals
            ]

        dataset.append(data_item)

    return dataset


def load_lvbench_data(anno_path, video_base_path, num_samples=None, shuffle=False, add_cannot_determine=False):
    """
    Load LVBench dataset from parquet file.

    Args:
        anno_path: Path to test-00000-of-00001.parquet
        video_base_path: Base path to video directory (contains all_videos/ subfolder)
        num_samples: Number of samples to load (None for all)
        shuffle: Whether to shuffle the samples
        add_cannot_determine: If True, append an extra "cannot determine" choice

    Returns:
        List of dictionaries with video data (CGBench-compatible format)
    """
    df = pd.read_parquet(anno_path)

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    if num_samples is not None:
        df = df.head(num_samples)

    dataset = []
    for _, row in df.iterrows():
        # Parse question text and choices from combined string
        # Format: "Question?\n(A) opt1\n(B) opt2\n(C) opt3\n(D) opt4"
        q_text = row['question']
        parts = re.split(r'\n\(([A-Z])\)\s*', q_text)
        question = parts[0].strip()
        # parts = [question, 'A', 'opt1', 'B', 'opt2', ...]
        letter_to_text = {}
        choices = []
        for i in range(1, len(parts) - 1, 2):
            letter = parts[i]
            text = parts[i + 1].strip()
            letter_to_text[letter] = text
            choices.append(text)

        right_answer = row['answer']
        answer_text = letter_to_text.get(right_answer, right_answer)

        # Duration from video_info (minutes → seconds)
        video_info = row['video_info']
        duration = float(video_info['duration_minutes']) * 60.0

        # Parse time_reference "MM:SS-MM:SS" → [start_sec, end_sec]
        # Use video start/end as fallback for missing values
        time_ref = str(row.get('time_reference', ''))
        parts = time_ref.split('-', 1) if '-' in time_ref else ['None', 'None']
        def mmss_to_sec(s):
            m = re.match(r'(\d+):(\d+)', s.strip())
            return int(m.group(1)) * 60 + int(m.group(2)) if m else None
        gt_start = mmss_to_sec(parts[0])
        gt_end = mmss_to_sec(parts[1]) if len(parts) > 1 else None
        gt_start = float(gt_start) if gt_start is not None else 0.0
        gt_end = float(gt_end) if gt_end is not None else duration

        # Video path
        video_path = Path(video_base_path) / 'all_videos' / f"{row['key']}.mp4"

        # Question type / sub_category
        q_type = row.get('question_type', ['Unknown'])
        sub_category = q_type[0] if hasattr(q_type, '__len__') and len(q_type) > 0 else 'Unknown'

        if add_cannot_determine and choices:
            choices = choices + ['Given the current information, the question cannot be answered']

        dataset.append({
            'video_path': str(video_path),
            'video_id': row['key'],
            'qid': int(row['uid']),
            'question': question,
            'answer': answer_text,
            'right_answer': right_answer,
            'choices': choices,
            'gt_timestamps': [gt_start, gt_end],
            'duration': duration,
            'domain': row.get('type', 'Unknown'),
            'sub_category': sub_category,
        })

    return dataset


def load_videomme_data(anno_path, video_base_path, num_samples=None, shuffle=False):
    """
    Load Video-MME dataset from parquet file.

    Args:
        anno_path: Path to test-00000-of-00001.parquet
        video_base_path: Base path to video directory (contains {videoID}.mp4 files)
        num_samples: Number of samples to load (None for all)
        shuffle: Whether to shuffle the samples

    Returns:
        List of dictionaries with video data (CGBench-compatible format)
    """
    import cv2
    import re

    df = pd.read_parquet(anno_path)

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    if num_samples is not None:
        df = df.head(num_samples)

    # Cache video durations to avoid re-reading for videos with multiple questions
    duration_cache = {}

    dataset = []
    for idx, row in df.iterrows():
        video_id = row['videoID']
        video_path = Path(video_base_path) / f"{video_id}.mp4"

        # Get video duration in seconds (cached per video)
        if video_id not in duration_cache:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Warning: Cannot open video {video_path}, skipping")
                cap.release()
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = frame_count / fps if fps > 0 else 0.0
            cap.release()
            if duration_sec == 0:
                print(f"Warning: Could not determine duration for {video_id}, skipping")
                continue
            duration_cache[video_id] = duration_sec

        duration_sec = duration_cache[video_id]

        # Parse options: 'A. Apples.' → 'Apples.'
        options = list(row['options'])
        choices = [re.sub(r'^[A-Z]\.\s*', '', opt) for opt in options]

        # Map answer letter to full text
        answer_letter = row['answer']
        letter_to_text = {}
        for opt in options:
            match = re.match(r'^([A-Z])\.\s*(.+)$', opt)
            if match:
                letter_to_text[match.group(1)] = match.group(2)
        answer_text = letter_to_text.get(answer_letter, answer_letter)

        dataset.append({
            'video_path': str(video_path),
            'video_id': video_id,
            'qid': idx,
            'question': row['question'],
            'answer': answer_text,
            'right_answer': answer_letter,
            'choices': choices,
            'gt_timestamps': [0.0, duration_sec],
            'duration': duration_sec,
            'domain': row.get('domain', 'Unknown'),
            'sub_category': row.get('duration', 'Unknown'),  # short/medium/long
        })

    return dataset


def load_mlvu_data(anno_path, video_base_path, num_samples=None, shuffle=False):
    """
    Load MLVU Test dataset from JSON annotation file (with ground truth).

    Args:
        anno_path: Path to test_mcq_gt.json (contains 'answer' field)
        video_base_path: Base path to video directory (contains .mp4 files)
        num_samples: Number of samples to load (None for all)
        shuffle: Whether to shuffle the samples

    Returns:
        List of dictionaries with video data (CGBench-compatible format)
    """
    with open(anno_path, 'r') as f:
        annotations = json.load(f)

    if shuffle:
        random.shuffle(annotations)

    if num_samples is not None:
        annotations = annotations[:num_samples]

    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    dataset = []
    for idx, anno in enumerate(annotations):
        video_filename = anno['video']
        video_path = Path(video_base_path) / video_filename
        duration = float(anno['duration'])

        # Build choices list; candidates may contain non-string values (e.g. ints)
        candidates = anno.get('candidates', [])
        choices = [str(c) for c in candidates]

        # Map answer text to letter (A, B, C, ...)
        answer_text = str(anno['answer']) if anno.get('answer') is not None else None
        right_answer = None
        if answer_text is not None:
            for i, c in enumerate(choices):
                if c == answer_text:
                    right_answer = letters[i]
                    break

        dataset.append({
            'video_path': str(video_path),
            'video_id': Path(video_filename).stem,
            'qid': anno.get('question_id', idx),
            'question': anno['question'],
            'answer': answer_text,
            'right_answer': right_answer,
            'choices': choices,
            'gt_timestamps': [0.0, duration],  # dummy: no temporal grounding GT
            'duration': duration,
            'domain': anno.get('question_type', 'Unknown'),
            'sub_category': anno.get('question_type', 'Unknown'),
            'question_type': anno.get('question_type', 'Unknown'),
        })

    return dataset


def load_longvideobench_data(anno_path, video_base_path, num_samples=None, shuffle=False, add_cannot_determine=False):
    """
    Load LongVideoBench dataset from JSON annotation file.

    Args:
        anno_path: Path to lvb_val.json
        video_base_path: Base path to video directory (contains {video_id}.mp4 files)
        num_samples: Number of samples to load (None for all)
        shuffle: Whether to shuffle the samples
        add_cannot_determine: If True, append an extra "cannot determine" choice

    Returns:
        List of dictionaries with video data (CGBench-compatible format)
    """
    with open(anno_path, 'r') as f:
        annotations = json.load(f)

    if shuffle:
        random.shuffle(annotations)

    if num_samples is not None:
        annotations = annotations[:num_samples]

    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    # Probe all unique videos' fps + duration concurrently. Serial cv2.VideoCapture
    # calls over a network mount were a major startup stall on LongVideoBench.
    unique_paths = {}
    for anno in annotations:
        vid = anno['video_id']
        if vid not in unique_paths:
            unique_paths[vid] = str(Path(video_base_path) / anno['video_path'])
    video_cache = _probe_videos_parallel(unique_paths)
    missing = [vid for vid in unique_paths if vid not in video_cache]
    if missing:
        print(f"Warning: could not probe {len(missing)} videos (skipping): {missing[:5]}{'...' if len(missing) > 5 else ''}")

    dataset = []
    for anno in annotations:
        video_id = anno['video_id']
        video_path = Path(video_base_path) / anno['video_path']

        if video_id not in video_cache:
            continue

        meta = video_cache[video_id]
        fps = meta['fps']
        duration = meta['duration']

        # Convert position (frame indices) to timestamps in seconds
        positions = anno['position']
        timestamps = [p / fps for p in positions]
        gt_start = float(min(timestamps))
        gt_end = float(max(timestamps))
        # For single-position items, use a minimal window (single frame duration)
        if gt_start == gt_end:
            gt_end = gt_start + 1.0 / fps

        # correct_choice is 0-indexed
        correct_idx = anno['correct_choice']
        right_answer = letters[correct_idx] if correct_idx < len(letters) else str(correct_idx)

        candidates = anno['candidates']
        choices = list(candidates)
        answer_text = candidates[correct_idx] if correct_idx < len(candidates) else right_answer

        if add_cannot_determine and choices:
            choices = choices + ['Given the current information, the question cannot be answered']

        dataset.append({
            'video_path': str(video_path),
            'video_id': video_id,
            'qid': anno['id'],
            'question': anno['question'],
            'answer': answer_text,
            'right_answer': right_answer,
            'choices': choices,
            'gt_timestamps': [gt_start, gt_end],
            'duration': duration,
            'domain': anno.get('topic_category', 'Unknown'),
            'sub_category': anno.get('question_category', 'Unknown'),
            'question_category': anno.get('question_category', 'Unknown'),
            'level': anno.get('level', 'Unknown'),
            'duration_group': anno.get('duration_group'),
        })

    return dataset


def load_youtube_gen_data(anno_path, video_base_path=None, num_samples=None, shuffle=False, multi_interval=False, add_cannot_determine=False):
    """
    Load YouTube annotations already in the CGBench-compatible generator schema
    (e.g. `output/filters/youtube/after_no_clue.json`).

    Unlike `load_cgbench_data_gen`, the per-example `video_path` field is used
    verbatim — `video_base_path` is ignored. This is required for YouTube data
    where videos live in multiple directories (`long_v1/`, `top30k/`, ...) and
    cannot be reconstructed from a single base.

    Source format per entry:
        {
          "video_id": "...",
          "video_path": "/abs/path/to/video.mp4",
          "qid": ...,
          "question": "...",
          "answer": "...",
          "right_answer": "C",
          "choices": [...],
          "gt_timestamps": [start, end],
          "duration": float,
          "domain": "...",
          "sub_category": "...",
          "gt_all_timestamps": [[s, e], ...]  (optional)
        }
    """
    with open(anno_path, 'r') as f:
        annotations = json.load(f)

    if shuffle:
        random.shuffle(annotations)

    if num_samples is not None:
        annotations = annotations[:num_samples]

    dataset = []
    for anno in annotations:
        video_id = anno.get('video_id') or anno.get('video_uid')
        choices = list(anno.get('choices', []))
        if add_cannot_determine and choices:
            choices = choices + ['Given the current information, the question cannot be answered']

        entry = {
            'video_path': anno['video_path'],
            'video_id': video_id,
            'qid': anno['qid'],
            'question': anno['question'],
            'answer': anno['answer'],
            'right_answer': anno['right_answer'],
            'choices': choices,
            'gt_timestamps': anno['gt_timestamps'],
            'duration': float(anno['duration']),
            'domain': anno.get('domain', 'YouTube'),
            'sub_category': anno.get('sub_category', 'YouTube'),
        }
        if multi_interval and 'gt_all_timestamps' in anno:
            entry['gt_all_timestamps'] = anno['gt_all_timestamps']
        dataset.append(entry)

    return dataset


def load_youtube_data(anno_path, video_base_path=None, num_samples=None, shuffle=False, multi_interval=False, add_cannot_determine=False):
    """
    Load YouTube dataset from JSON annotation file.

    Source format per entry:
        {
          "video": "<video_id>",
          "question": "...",
          "correct": "C",
          "answers": [{"text": "...", "timestamps": [{"start": s, "end": e}, ...]}, ...],
          "video_path": "/abs/path/to/<video_id>.mp4"
        }

    Args:
        anno_path: Path to youtube.json
        video_base_path: Unused (video_path is absolute in the source); kept for
            signature compatibility with other loaders.
        num_samples: Number of samples to load (None for all)
        shuffle: Whether to shuffle the samples
        multi_interval: If True, include all GT intervals in 'gt_all_timestamps'
        add_cannot_determine: If True, append extra "cannot be answered" choice

    Returns:
        List of dictionaries with video data (CGBench-compatible format)
    """
    import cv2

    with open(anno_path, 'r') as f:
        annotations = json.load(f)

    if shuffle:
        random.shuffle(annotations)

    if num_samples is not None:
        annotations = annotations[:num_samples]

    # Cache (fps, duration) per video_id — each video has ~15 QA pairs on average
    video_cache = {}

    dataset = []
    for idx, anno in enumerate(annotations):
        video_id = anno['video']
        video_path = anno['video_path']

        correct_letter = anno['correct']
        correct_idx = ord(correct_letter) - ord('A')
        answers = anno['answers']
        if correct_idx < 0 or correct_idx >= len(answers):
            print(f"Warning: correct letter {correct_letter} out of range for video {video_id}, skipping")
            continue

        correct_answer = answers[correct_idx]
        ts_list = correct_answer.get('timestamps') or []
        if not ts_list:
            print(f"Warning: No timestamps for correct answer in video {video_id} (idx {idx}), skipping")
            continue

        if video_id not in video_cache:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Warning: Cannot open video {video_path}, skipping")
                cap.release()
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0.0
            cap.release()
            if duration == 0:
                print(f"Warning: Could not determine duration for {video_id}, skipping")
                continue
            video_cache[video_id] = {'fps': fps, 'duration': duration}

        if video_id not in video_cache:
            continue

        duration = video_cache[video_id]['duration']

        intervals = sorted((float(t['start']), float(t['end'])) for t in ts_list)
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            prev_s, prev_e = merged[-1]
            if s - prev_e < 10.0:
                merged[-1] = (prev_s, max(prev_e, e))
            else:
                merged.append((s, e))
        intervals = merged
        gt_start, gt_end = intervals[0]

        choices = [a['text'] for a in answers]
        answer_text = choices[correct_idx]

        if add_cannot_determine and choices:
            choices = choices + ['Given the current information, the question cannot be answered']

        entry = {
            'video_path': str(video_path),
            'video_id': video_id,
            'qid': idx,
            'question': anno['question'],
            'answer': answer_text,
            'right_answer': correct_letter,
            'choices': choices,
            'gt_timestamps': [float(gt_start), float(gt_end)],
            'duration': float(duration),
            'domain': 'YouTube',
            'sub_category': 'YouTube',
        }
        if multi_interval:
            entry['gt_all_timestamps'] = [[float(s), float(e)] for s, e in intervals]
        dataset.append(entry)

    return dataset


def make_json_serializable(obj):
    """Convert object to JSON-serializable format."""
    from enum import Enum

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, Enum):
        # Convert enum to its value
        return obj.value
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Skip callables, private attributes, and complex objects
            if callable(v) or k.startswith('_'):
                continue
            # Skip Node objects and other complex types
            if hasattr(v, '__class__') and v.__class__.__name__ in ['Node', 'Action', 'Observation', 'Reasoning']:
                continue
            try:
                result[k] = make_json_serializable(v)
            except (TypeError, ValueError):
                # Skip objects that can't be serialized
                continue
        return result
    elif isinstance(obj, (list, tuple)):
        result = []
        for item in obj:
            try:
                result.append(make_json_serializable(item))
            except (TypeError, ValueError):
                continue
        return result
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    else:
        # Try to convert to string, or skip
        try:
            return str(obj)
        except:
            return None
