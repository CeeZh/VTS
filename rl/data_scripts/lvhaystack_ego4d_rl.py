"""
LVHaystack-Ego4D dataset preparation for RL training.

Converts LVHaystack-Ego4D annotations into JSONL format suitable for Swift
GRPO training with the TreeSearchScheduler.

Usage:
    python rl/data_scripts/lvhaystack_ego4d_rl.py \
        --anno_path /mnt/arc/cezhang/projects/datagen/output/filters/lvhaystack-ego4d/after_no_clue.json \
        --output_path rl/data/lvhaystack_ego4d.jsonl
"""

import argparse
import json
import os
from typing import Dict, List, Optional


class LVHaystackEgo4DRLDataset:
    """
    Load LVHaystack-Ego4D annotations and convert to RL training JSONL.

    Each output line contains the metadata needed by TreeSearchScheduler:
    video_id, duration, question, choices, solution, timestamp, etc.
    The images and messages fields are placeholders — the scheduler rebuilds
    them dynamically each turn.
    """

    def __init__(
        self,
        anno_path: str,
        video_base_path: str = '/mnt/sun/mmiemon/datasets/ego4d/v1/video_540ss',
        frames_base_path: str = '/mnt/arc/cezhang/datasets/lvhaystack_ego4d/frames_1fps',
        tree_cache_dir: Optional[str] = '/mnt/arc/cezhang/projects/datagen/output/tree_cache/lvhaystack_ego4d',
        require_tree_cache: bool = False,
        require_frames: bool = True,
    ):
        self.anno_path = anno_path
        self.video_base_path = video_base_path
        self.frames_base_path = frames_base_path
        self.tree_cache_dir = tree_cache_dir
        self.require_tree_cache = require_tree_cache
        self.require_frames = require_frames

    def load(self) -> List[Dict]:
        """Load and filter LVHaystack-Ego4D annotations."""
        with open(self.anno_path) as f:
            raw_data = json.load(f)

        items = []
        skipped = {'no_tree': 0, 'no_frames': 0, 'no_video': 0}

        for entry in raw_data:
            video_id = entry['video_id']
            video_path = os.path.join(self.video_base_path, f'{video_id}.mp4')

            # Check video exists
            if not os.path.exists(video_path):
                skipped['no_video'] += 1
                continue

            # Check tree cache exists
            if self.require_tree_cache and self.tree_cache_dir:
                cache_file = os.path.join(
                    self.tree_cache_dir, f'{video_id}.json'
                )
                if not os.path.exists(cache_file):
                    skipped['no_tree'] += 1
                    continue

            # Check frames directory exists
            if self.require_frames:
                frames_dir = os.path.join(self.frames_base_path, video_id)
                if not os.path.isdir(frames_dir):
                    skipped['no_frames'] += 1
                    continue

            gt_ts = entry.get('gt_timestamps') or []
            if gt_ts and not isinstance(gt_ts[0], list):
                lo, hi = sorted(gt_ts[:2])
                timestamp = [[float(lo), float(hi)]]
            else:
                timestamp = [[float(lo), float(hi)] for lo, hi in (sorted(iv[:2]) for iv in gt_ts)]

            items.append({
                'qid': entry['qid'],
                'video_id': video_id,
                'video_path': video_path,
                'duration': float(entry['duration']),
                'question': entry['question'],
                'choices': entry['choices'],
                'answer': entry['answer'],
                'solution': entry['right_answer'],
                'data_type': 'qa',
                'timestamp': timestamp,
                'frame_indexes_video': entry.get('frame_indexes_video', []),
                'frame_rate': float(entry.get('frame_rate', 30.0)),
                'domain': entry.get('domain', ''),
                'sub_category': entry.get('sub_category', ''),
                'tree_cache_dir': self.tree_cache_dir,
                'frames_base_dir': self.frames_base_path,
            })

        print(f'Loaded {len(items)} items from {len(raw_data)} total')
        print(f'Skipped: {skipped}')
        return items

    def to_jsonl(self, items: List[Dict], output_path: str) -> None:
        """Write items to JSONL format for Swift."""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with open(output_path, 'w') as f:
            for item in items:
                record = {
                    # Images are populated dynamically by the scheduler
                    'images': [],
                    # Metadata flows into data_dict
                    'question': item['question'],
                    'solution': item['solution'],
                    'answer': item['answer'],
                    'data_type': item['data_type'],
                    'timestamp': item['timestamp'],
                    'video_id': item['video_id'],
                    'video_path': item['video_path'],
                    'duration': item['duration'],
                    'choices': item['choices'],
                    'qid': item['qid'],
                    # LVHaystack-specific fields
                    'frame_indexes_video': item['frame_indexes_video'],
                    'frame_rate': item['frame_rate'],
                    'domain': item['domain'],
                    'sub_category': item['sub_category'],
                    # Per-sample paths for multi-dataset support
                    'tree_cache_dir': item['tree_cache_dir'],
                    'frames_base_dir': item['frames_base_dir'],
                    # Placeholder messages — scheduler rebuilds from scratch
                    'messages': [
                        {
                            'role': 'user',
                            'content': item['question'],
                        }
                    ],
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f'Wrote {len(items)} records to {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate RL training data from LVHaystack-Ego4D'
    )
    parser.add_argument(
        '--anno_path',
        default='/mnt/arc/cezhang/projects/datagen/output/filters/lvhaystack-ego4d/after_no_clue.json',
        help='Path to LVHaystack-Ego4D annotation JSON',
    )
    parser.add_argument(
        '--video_base_path',
        default='/mnt/sun/mmiemon/datasets/ego4d/v1/video_540ss',
        help='Base path for video files',
    )
    parser.add_argument(
        '--frames_base_path',
        default='/mnt/arc/cezhang/datasets/lvhaystack_ego4d/frames_1fps',
        help='Base path for pre-extracted frames',
    )
    parser.add_argument(
        '--tree_cache_dir',
        default='/mnt/arc/cezhang/projects/datagen/output/tree_cache/lvhaystack_ego4d',
        help='Path to tree cache directory',
    )
    parser.add_argument(
        '--output_path',
        default='rl/data/lvhaystack_ego4d.jsonl',
        help='Output JSONL path',
    )
    parser.add_argument(
        '--require_tree_cache',
        action='store_true',
        help='Only include videos with pre-built tree caches',
    )
    parser.add_argument(
        '--require_frames',
        action='store_true',
        default=True,
        help='Only include videos with pre-extracted frames',
    )

    args = parser.parse_args()

    dataset = LVHaystackEgo4DRLDataset(
        anno_path=args.anno_path,
        video_base_path=args.video_base_path,
        frames_base_path=args.frames_base_path,
        tree_cache_dir=args.tree_cache_dir,
        require_tree_cache=args.require_tree_cache,
        require_frames=args.require_frames,
    )

    items = dataset.load()
    dataset.to_jsonl(items, args.output_path)


if __name__ == '__main__':
    main()

'''
python rl/data_scripts/lvhaystack_ego4d_rl.py \
    --video_base_path /mnt/sun/mmiemon/datasets/ego4d/v1/video_540ss \
    --frames_base_path /mnt/arc/cezhang/datasets/lvhaystack_ego4d/frames_1fps \
    --tree_cache_dir /mnt/arc/cezhang/projects/datagen/output/tree_cache/lvhaystack_ego4d \
    --anno_path /mnt/arc/cezhang/projects/datagen/output/filters/lvhaystack-ego4d/after_clue.json \
    --output_path rl/data/lvhaystack_ego4d_after_clue.jsonl
'''

'''
python rl/data_scripts/lvhaystack_ego4d_rl.py \
    --video_base_path /mnt/sun/mmiemon/datasets/ego4d/v1/video_540ss \
    --frames_base_path /mnt/arc/cezhang/datasets/lvhaystack_ego4d/frames_1fps \
    --tree_cache_dir /mnt/arc/cezhang/projects/datagen/output/tree_cache/lvhaystack_ego4d \
    --anno_path /mnt/arc/cezhang/projects/datagen/output/generation/traj_filtered_splits/lvhaystack_ego4d/60.json \
    --output_path rl/data/merged/lvhaystack_ego4d_60.jsonl
'''