"""
YouTube filtered-splits dataset preparation for RL training.

Converts the YouTube generation annotations into JSONL format suitable for
Swift GRPO training with the TreeSearchScheduler.

Usage:
    python rl/data_scripts/youtube_rl.py \
        --anno_path /mnt/arc/cezhang/projects/datagen/output/generation/youtube_filtered_splits/60.json \
        --output_path rl/data/youtube_60.jsonl
"""

import argparse
import json
import os
from typing import Dict, List, Optional


class YouTubeRLDataset:
    """
    Load YouTube generation annotations and convert to RL training JSONL.

    Each entry already contains an absolute `video_path`, so `video_base_path`
    defaults to empty — the path is used as-is. If a base path is provided,
    it is joined with `{video_id}.mp4` (same convention as CGBench).
    """

    def __init__(
        self,
        anno_path: str,
        video_base_path: str = '',
        frames_base_path: str = '/mnt/mir/ce/datasets/youtube/frames_1fps',
        tree_cache_dir: Optional[str] = '/mnt/arc/cezhang/projects/datagen/output/tree_cache/youtube',
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
        """Load and filter YouTube annotations."""
        with open(self.anno_path) as f:
            raw_data = json.load(f)

        items = []
        skipped = {'no_tree': 0, 'no_frames': 0, 'no_video': 0}

        for entry in raw_data:
            video_id = entry['video_id']

            if self.video_base_path:
                video_path = os.path.join(
                    self.video_base_path, f'{video_id}.mp4'
                )
            else:
                video_path = entry['video_path']

            if not os.path.exists(video_path):
                skipped['no_video'] += 1
                continue

            if self.require_tree_cache and self.tree_cache_dir:
                safe_id = video_id.replace('/', '_')
                cache_file = os.path.join(
                    self.tree_cache_dir, f'{safe_id}.json'
                )
                if not os.path.exists(cache_file):
                    skipped['no_tree'] += 1
                    continue

            if self.require_frames:
                frames_dir = os.path.join(self.frames_base_path, video_id)
                if not os.path.isdir(frames_dir):
                    skipped['no_frames'] += 1
                    continue

            items.append({
                'qid': entry['qid'],
                'video_id': video_id,
                'video_path': video_path,
                'duration': float(entry['duration']),
                'question': entry['question'],
                'choices': entry['choices'],
                'solution': entry['right_answer'],
                'data_type': 'qa',
                'timestamp': entry.get('gt_all_timestamps', []),
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
                    'images': [],
                    'question': item['question'],
                    'solution': item['solution'],
                    'data_type': item['data_type'],
                    'timestamp': item['timestamp'],
                    'video_id': item['video_id'],
                    'video_path': item['video_path'],
                    'duration': item['duration'],
                    'choices': item['choices'],
                    'qid': item['qid'],
                    'tree_cache_dir': item['tree_cache_dir'],
                    'frames_base_dir': item['frames_base_dir'],
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
        description='Generate RL training data from YouTube filtered splits'
    )
    parser.add_argument(
        '--anno_path',
        default='/mnt/arc/cezhang/projects/datagen/output/generation/youtube_filtered_splits/60.json',
        help='Path to YouTube annotation JSON',
    )
    parser.add_argument(
        '--video_base_path',
        default='',
        help='Base path for video files (empty → use entry.video_path as-is)',
    )
    parser.add_argument(
        '--frames_base_path',
        default='/mnt/mir/ce/datasets/youtube/frames_1fps',
        help='Base path for pre-extracted frames',
    )
    parser.add_argument(
        '--tree_cache_dir',
        default='/mnt/arc/cezhang/projects/datagen/output/tree_cache/youtube',
        help='Path to tree cache directory',
    )
    parser.add_argument(
        '--output_path',
        default='rl/data/youtube_60.jsonl',
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

    dataset = YouTubeRLDataset(
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
python rl/data_scripts/youtube_rl.py \
    --anno_path /mnt/arc/cezhang/projects/datagen/output/generation/youtube_filtered_splits/60.json \
    --video_base_path "" \
    --frames_base_path /mnt/mir/ce/datasets/youtube/frames_1fps \
    --tree_cache_dir /mnt/arc/cezhang/projects/datagen/output/tree_cache/youtube \
    --output_path rl/data/youtube_60.jsonl
'''

'''
python rl/data_scripts/youtube_rl.py \
    --anno_path /mnt/arc/cezhang/projects/datagen/output/generation/traj_filtered_splits/youtube/60.json \
    --video_base_path "" \
    --frames_base_path /mnt/mir/ce/datasets/youtube/frames_1fps \
    --tree_cache_dir /mnt/arc/cezhang/projects/datagen/output/tree_cache/youtube \
    --output_path rl/data/merged/youtube_60.jsonl
'''

