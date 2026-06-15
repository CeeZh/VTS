"""
Example usage of the inference engine for autonomous video temporal reasoning.
Evaluates the inference algorithm on CGBench or LongVideoHaystack datasets.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import random
import time
import traceback

import cv2
import numpy as np
from tqdm import tqdm

# Limit OpenCV internal threads to prevent thread explosion with many workers
cv2.setNumThreads(2)

from vlm_interface import DummyVLMInterface, GPTVLMInterface, CaptionLLMInterface
from inference import InferenceEngine, InferenceConfig
from utils.dataset import (
    load_cgbench_data, load_longvideohaystack_data, load_tstar_data, load_momentseeker_data, load_lvbench_data, load_videomme_data, load_mlvu_data, load_longvideobench_data, make_json_serializable
)
from utils.keyframe_sampling import derive_keyframe_timestamps
from utils.temporal import compute_interval_iou
from utils.vis import visualize_tree_structure


def _process_sample(
    item, index, total, output_path,
    base_url="http://bumblebee:1234/v1",
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    use_dummy=False,
    direct=False,
    direct_prompt_style="json",
    video_native=False,
    action_mode="segmented",
    keyframe_mode=False,
    clue_mode=False,
    dataset_type="cgbench",
    ## inference config
    max_turns=20,
    fps=1.0,
    max_frames=64,
    short_side=256,
    min_turns=0,
    max_in_loop=1,
    loop_detection_mode="visited",
    loop_detection_window=2,
    ## segment splitting (common)
    segment_mode="uniform",
    min_segment_duration=60.0,
    max_depth=-1,
    clip_url="grpc://localhost:51000",
    ## uniform segmentation
    num_children=4,
    ## uniform fixed-duration segmentation
    uf_child_duration=60.0,
    ## query-agnostic CLIP segmentation
    scene_min_segments=2,
    scene_max_segments=8,
    scene_k=1.5,
    scene_fps=1.0,
    scene_max_frames=64,
    scene_min_duration=4.0,
    ## direct-nonuniform segmentation
    dn_num_segments=64,
    dn_clip_fps=1.0,
    ## prompt logging
    save_prompts=False,
    ## VLM type selection
    vlm_type="gpt",
    caption_base_url=None,
    caption_model="Qwen/Qwen3-VL-8B-Instruct",
    llm_base_url=None,
    llm_model=None,
    llm_api_key="EMPTY",
    captioning_mode="segment",
    memory_frames_per_node=0,
    image_tokens=None,
    separate_caption_generation=False,
    caption_frames_from_parent=False,
    skip_reasoning=False,
    pregenerated_caption_path=None,
    use_frame_captions=False,
    reasoning_video_representation=None,
    tree_cache_dir=None,
    decide_action_type="default",
    enforce_valid_segments=True,
    allowed_actions=("ZOOM_IN", "ZOOM_OUT", "SHIFT", "ANSWER"),
    ## keyframe sampling (lvhaystack eval)
    keyframe_sampling="uniform",
    keyframe_uniform_num=8,
    keyframe_clip_fps=0.5,
    keyframe_clip_max_frames=64,
    ## fixed timestamps for --direct mode
    fixed_timestamps_map=None,
):
    """
    Worker function to process a single sample through the inference engine.
    Creates per-worker instances for thread safety.

    Returns a result dict with timing, status, and metrics.
    """
    # Compute prompt log directory if save_prompts is enabled
    prompt_log_dir = None
    if save_prompts:
        safe_vid = item['video_id'].replace('/', '_')
        prompt_log_dir = str(output_path / f"{index+1:03d}_{safe_vid}_prompts")

    reasoning_video_representation = "continuous" if (direct or clue_mode) else reasoning_video_representation

    if use_dummy:
        vlm = DummyVLMInterface(seed=42 + index)
    elif vlm_type == "caption_llm":
        vlm = CaptionLLMInterface(
            caption_base_url=caption_base_url or base_url,
            caption_model=caption_model,
            caption_api_key="EMPTY",
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            reasoning_use_node_text=False,
            reasoning_use_node_frames=True,
            reasoning_history_turns=-1,
            reasoning_video_representation=reasoning_video_representation,
            reasoning_action_representation=action_mode,
            prompt_log_dir=prompt_log_dir,
            captioning_mode=captioning_mode,
            pregenerated_caption_path=pregenerated_caption_path,
        )
    else:
        vlm = GPTVLMInterface(
            base_url=base_url,
            api_key="EMPTY",
            model=model_name,
            timeout=3600,
            reasoning_use_node_text=False,
            reasoning_use_node_frames=True,
            reasoning_history_turns=-1,
            reasoning_video_representation=reasoning_video_representation,
            reasoning_action_representation=action_mode,
            prompt_log_dir=prompt_log_dir,
            memory_frames_per_node=memory_frames_per_node,
            image_tokens=image_tokens,
        )

    config = InferenceConfig(
        num_children=num_children,
        max_turns=max_turns,
        fps=fps,
        max_frames_per_segment=max_frames,
        short_side=short_side,
        force_answer_on_max_turns=True,
        min_turns=min_turns,
        max_in_loop=max_in_loop,
        loop_detection_mode=loop_detection_mode,
        loop_detection_window=loop_detection_window,
        min_segment_duration=min_segment_duration,
        max_depth=max_depth,
        separate_caption_generation=separate_caption_generation,
        caption_frames_from_parent=caption_frames_from_parent,
        skip_reasoning=skip_reasoning,
        pregenerated_caption_path=pregenerated_caption_path,
        use_frame_captions=use_frame_captions,
        tree_cache_dir=tree_cache_dir,
        decide_action_type=decide_action_type,
        enforce_valid_segments=enforce_valid_segments,
        direct_prompt_style=direct_prompt_style,
        allowed_actions=tuple(allowed_actions),
    )

    # Lazily instantiate a single CLIP client (shared by scene segmentation
    # and CLIP-based keyframe sampling).
    clip_client = None
    def _get_clip_client():
        nonlocal clip_client
        if clip_client is None:
            from clip_client import Client
            clip_client = Client(clip_url)
        return clip_client

    # Set up segmenter based on segment_mode
    segment_fn = None
    if segment_mode == "direct-nonuniform":
        # Non-uniform baseline: single-pass scene segmentation producing exactly
        # dn_num_segments segments (top-k mode), with one center frame each.
        from utils.segmentation import create_scene_segmenter
        segment_fn = create_scene_segmenter(
            _get_clip_client(),
            fps=dn_clip_fps,
            max_frames=256,
            short_side=short_side,
            num_children=dn_num_segments,
        )
    elif segment_mode == "query-agnostic-clip":
        from utils.segmentation import create_scene_segmenter
        segment_fn = create_scene_segmenter(
            _get_clip_client(),
            fps=scene_fps,
            max_frames=scene_max_frames,
            short_side=short_side,
            k=scene_k,
            min_segments=scene_min_segments,
            max_segments=scene_max_segments,
            min_segment_duration=scene_min_duration,
        )
    elif segment_mode == "query-aware-clip":
        # TODO: query-aware CLIP segmentation should consider the question's
        # text embedding when producing the tree structure (e.g., score frames
        # by query-frame CLIP similarity to bias boundary placement).
        raise NotImplementedError(
            "segment-mode 'query-aware-clip' is not yet implemented."
        )
    elif segment_mode == "uniform-fixed":
        fd = uf_child_duration
        def segment_fn(_video_path, start_sec, end_sec):
            if end_sec - start_sec <= fd:
                return []
            segments = []
            cur = start_sec
            while cur < end_sec:
                seg_end = min(cur + fd, end_sec)
                segments.append((cur, seg_end))
                cur = seg_end
            return segments
    elif segment_mode == "no-split":
        def segment_fn(_video_path, _start_sec, _end_sec):
            return []
    # else: segment_mode == "uniform" — leave segment_fn=None for uniform splitting

    # If a fixed-timestamps map is provided (--timestamps-file with --direct),
    # override the default frame_sampler so the root sampling for this video
    # uses the precomputed timestamps instead of uniform fps sampling.
    frame_sampler = None
    if fixed_timestamps_map is not None:
        per_video = fixed_timestamps_map.get(item['video_id'])
        if isinstance(per_video, dict):
            qid_key = item.get('qid')
            ts = per_video.get(qid_key)
            if ts is None and qid_key is not None:
                ts = per_video.get(str(qid_key))
        else:
            ts = per_video
        if ts is None:
            raise ValueError(
                f"--timestamps-file: no entry for video_id={item['video_id']} "
                f"qid={item.get('qid')}"
            )
        fixed_ts = [float(t) for t in ts]
        def frame_sampler(_video_path, _start_sec, _end_sec, _ts=fixed_ts):
            return list(_ts)

    engine = InferenceEngine(vlm, config, segment_fn=segment_fn, frame_sampler=frame_sampler)

    ground_truth = {
        'timestamps': item['gt_timestamps'],
        'answer': item['answer'],
        'right_answer': item['right_answer'],
        'choices': item['choices'],
    }
    if 'gt_all_timestamps' in item:
        ground_truth['all_timestamps'] = item['gt_all_timestamps']

    result = {
        'index': index,
        'video_id': item['video_id'],
        'qid': item.get('qid', index),
        'success': False,
        'time': 0.0,
        'error': None,
        'answer_correct': None,
        'evidence_iou': None,
    }

    try:
        start_time = time.time()

        infer_extra_kwargs = {}
        if clue_mode:
            infer_fn = engine.infer_clue
        elif keyframe_mode:
            infer_fn = engine.infer_direct_keyframes
        elif segment_mode == "direct-nonuniform":
            infer_fn = engine.infer_direct_nonuniform
        elif direct:
            infer_fn = engine.infer_direct
            infer_extra_kwargs['video_native'] = video_native
        else:
            infer_fn = engine.infer
        trajectory = infer_fn(
            video_path=item['video_path'],
            video_id=item['video_id'],
            question=item['question'],
            choices=item['choices'],
            video_duration=item['duration'],
            ground_truth=ground_truth,
            **infer_extra_kwargs,
        )

        elapsed = time.time() - start_time
        result['time'] = elapsed
        result['success'] = not trajectory.metadata.get('forced_termination', False)
        result['forced_termination_reason'] = trajectory.metadata.get('forced_termination_reason')
        result['answer_correct'] = trajectory.metadata.get('answer_correct')
        result['evidence_iou'] = trajectory.metadata.get('evidence_iou')
        result['num_turns'] = len(trajectory)
        result['total_frames'] = trajectory.metadata.get('total_frames')
        result['num_keyframes'] = None

        # Extract keyframe timestamps for Longvideohaystack evaluation
        keyframe_timestamps = None
        if "lvhaystack" in dataset_type and trajectory.is_terminated():
            if keyframe_mode:
                # Keyframe mode: use directly predicted timestamps from metadata
                keyframe_timestamps = trajectory.metadata.get('keyframe_timestamps')
            else:
                # Legacy mode: derive from evidence interval using the selected
                # sampling strategy (uniform linspace or CLIP query-similarity).
                last_turn = trajectory.turns[-1]
                if last_turn.action and last_turn.action.evidence:
                    ev_start, ev_end = last_turn.action.evidence['timestamps']
                    keyframe_timestamps = derive_keyframe_timestamps(
                        mode=keyframe_sampling,
                        start_sec=ev_start,
                        end_sec=ev_end,
                        video_path=item['video_path'],
                        query=item['question'],
                        clip_client=_get_clip_client() if keyframe_sampling == "clip" else None,
                        uniform_num_frames=keyframe_uniform_num,
                        clip_fps=keyframe_clip_fps,
                        clip_max_frames=keyframe_clip_max_frames,
                        clip_short_side=short_side,
                    )

        if keyframe_timestamps is not None:
            result['num_keyframes'] = len(keyframe_timestamps)

        print(f"\n[{index+1}/{total}] {item['video_id']}")
        print(f"  Time: {elapsed:.2f}s | Turns: {len(trajectory)}")
        print(f"  Answer: {trajectory.get_final_answer()} | "
              f"GT: {item['right_answer']} | "
              f"Correct: {result['answer_correct']}")
        if result['evidence_iou'] is not None:
            print(f"  Evidence IoU: {result['evidence_iou']:.3f}")

        # Save per-sample JSON
        safe_id = item['video_id'].replace('/', '_')
        meta_path = output_path / f"{index+1:03d}_{safe_id}_inference.json"

        # Build turns info
        turns_info = []
        for t_idx, turn in enumerate(trajectory.turns):
            # Observation before this action = previous turn's observation
            prev_obs = trajectory.turns[t_idx - 1].observation if t_idx > 0 else None
            turns_info.append({
                'turn': t_idx + 1,
                'observation_before': {
                    'start_sec': prev_obs.node.start_sec,
                    'end_sec': prev_obs.node.end_sec,
                    'level': prev_obs.node.level,
                } if prev_obs else None,
                'compressed_history': (
                    turn.reasoning.context.get('compressed_history')
                    if turn.reasoning and turn.reasoning.context else None
                ),
                'reasoning': turn.reasoning.content if turn.reasoning else None,
                'raw_response': turn.reasoning.raw_response if turn.reasoning else None,
                'action': {
                    'type': turn.action.action_type.value if turn.action else None,
                    'segment_id': turn.action.segment_id if turn.action else None,
                    'start_sec': turn.action.start_sec if turn.action else None,
                    'end_sec': turn.action.end_sec if turn.action else None,
                    'answer': turn.action.answer if turn.action else None,
                    'evidence': turn.action.evidence if turn.action else None,
                } if turn.action else None,
                'observation': {
                    'start_sec': turn.observation.node.start_sec,
                    'end_sec': turn.observation.node.end_sec,
                    'level': turn.observation.node.level,
                } if turn.observation else None,
            })

        # Prepare JSON data
        json_data = {
            'video_id': item['video_id'],
            'qid': item.get('qid'),
            'question': item['question'],
            'choices': item['choices'],
            'gt_answer': item['right_answer'],
            'gt_timestamps': item['gt_timestamps'],
            'predicted_answer': trajectory.get_final_answer(),
            'answer_correct': result['answer_correct'],
            'evidence_iou': result['evidence_iou'],
            'num_turns': len(trajectory),
            'frames_per_turn': trajectory.metadata.get('frames_per_turn'),
            'total_frames': trajectory.metadata.get('total_frames'),
            'generation_time': elapsed,
            'turns': make_json_serializable(turns_info),
            'metadata': make_json_serializable(trajectory.metadata),
        }

        # Add Longvideohaystack-specific fields
        if "lvhaystack" in dataset_type:
            json_data['video_path'] = item['video_path']
            json_data['keyframe_timestamps'] = keyframe_timestamps
            json_data['gt_frame_index'] = item.get('frame_indexes_video')
            json_data['frame_rate'] = item.get('frame_rate')
            # Convert predicted seconds to frame IDs using video fps
            if keyframe_timestamps and item.get('frame_rate'):
                # Get total frames for boundary validation
                cap = cv2.VideoCapture(item['video_path'])
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                # Convert to frame IDs and clamp to valid range
                json_data['predicted_frame_ids'] = [
                    min(int(ts * item['frame_rate']), total_frames - 1)
                    for ts in keyframe_timestamps
                ]

        # Add MomentSeeker-specific fields
        if dataset_type == "momentseeker":
            # Extract predicted interval from last turn for easier evaluation
            if trajectory.turns:
                last_turn = trajectory.turns[-1]
                if last_turn.action and last_turn.action.evidence:
                    json_data['predicted_timestamps'] = list(last_turn.action.evidence['timestamps'])

            # Include all GT intervals if available for multi-interval evaluation
            if 'gt_all_timestamps' in item:
                json_data['gt_all_timestamps'] = item['gt_all_timestamps']

        # Add Video-MME-specific fields
        if dataset_type == "videomme":
            json_data['sub_category'] = item.get('sub_category')
            json_data['domain'] = item.get('domain')

        # Add MLVU-specific fields
        if dataset_type in ["mlvu", "mlvu_dev"]:
            json_data['question_type'] = item.get('question_type')

        # Add LongVideoBench-specific fields
        if dataset_type == "longvideobench":
            json_data['question_category'] = item.get('question_category')

        # Choose save directory: failed/ subfolder for forced terminations
        if trajectory.metadata.get('forced_termination', False):
            save_path = output_path / "failed"
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = output_path

        meta_path = save_path / f"{index+1:03d}_{safe_id}_inference.json"
        with open(meta_path, 'w') as f:
            json.dump(json_data, f, indent=4)

        # Generate tree visualization (skip for single-turn direct/keyframe modes)
        if not direct and not keyframe_mode and not clue_mode:
            try:
                root_node = trajectory.initial_observation.node
                gt_node = trajectory.gt_node
                tree_output = save_path / f"{index+1:03d}_{safe_id}_tree.png"
                visualize_tree_structure(root_node, gt_node, trajectory, str(tree_output))
            except Exception as e:
                print(f"\n[{index+1}/{total}] {item['video_id']} - Tree visualization failed: {e}")

    except Exception as e:
        result['error'] = str(e)
        print(f"\n[{index+1}/{total}] {item['video_id']} - ERROR: {e}")
        traceback.print_exc()

    return result


DATASET_DEFAULTS = {
    "cgbench": {
        "anno_path": "/mnt/arc/cezhang/datasets/CG-Bench/cgbench.json",
        "video_base_path": "/mnt/arc/cezhang/datasets/CG-Bench/cg_videos_720p",
        "tree_cache_dir": None,
    },
    "cgbench_mini": {
        "anno_path": "/mnt/arc/cezhang/projects/datagen/output/filters/cgbench_mini/after_no_clue.json",
        "video_base_path": "/mnt/arc/cezhang/datasets/CG-Bench/cg_videos_720p",
        "tree_cache_dir": "/mnt/arc/cezhang/projects/datagen/output/tree_cache/cgbench_mini_filtered",
    },
    "lvhaystack_ego4d": {
        "anno_path": "/mnt/arc/cezhang/datasets/LongVideoHaystack/data/val-00000-of-00001.parquet",
        "video_base_path": "/mnt/sun/mmiemon/datasets/ego4d/v1/video_540ss",
        "tree_cache_dir": "/mnt/arc/cezhang/projects/datagen/output/tree_cache/lvhaystack_ego4d",
    },
    "lvhaystack_longvideobench": {
        "anno_path": "/mnt/arc/cezhang/projects/TStar/lvb_val_TStarFormat_with_metadata.json",
        # "video_base_path": "/mnt/arc/cezhang/datasets/longvideobench/videos",
        "video_base_path": "/mnt/arc/cezhang/datasets/longvideobench/videos_reencode",
        "tree_cache_dir": "/mnt/arc/cezhang/projects/datagen/output/tree_cache/lvhaystack_longvideobench",
    },
    "momentseeker": {
        "anno_path": "/mnt/arc/cezhang/datasets/MomentSeeker/t2v.json",
        "video_base_path": "/mnt/arc/cezhang/datasets/MomentSeeker/videos",
        "tree_cache_dir": None,
    },
    "lvbench": {
        "anno_path": "/mnt/arc/cezhang/datasets/LVBench/data/test-00000-of-00001.parquet",
        "video_base_path": "/mnt/arc/cezhang/datasets/LVBench/extracted_videos",
        "tree_cache_dir": "/mnt/arc/cezhang/projects/datagen/output/tree_cache/lvbench",
    },
    "mlvu": {
        "anno_path": "/mnt/arc/cezhang/datasets/MLVU_Test/MLVU_Test/test-ground-truth/test_mcq_gt.json",
        "video_base_path": "/mnt/arc/cezhang/datasets/MLVU_Test/MLVU_Test/video",
        "tree_cache_dir": "/mnt/arc/cezhang/projects/datagen/output/tree_cache/mlvu",
    },
    "mlvu_dev": {
        "anno_path": "/mnt/arc/cezhang/datasets/MLVU/MLVU/dev_question.json",
        "video_base_path": "/mnt/arc/cezhang/datasets/MLVU/MLVU/video",
        "tree_cache_dir": "/mnt/arc/cezhang/projects/datagen/output/tree_cache/mlvu",
    },
    "videomme": {
        "anno_path": "/mnt/arc/cezhang/xdg_dirs/cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme/test-00000-of-00001.parquet",
        "video_base_path": "/mnt/arc/cezhang/xdg_dirs/cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/videos/data",
        "tree_cache_dir": "/mnt/arc/cezhang/projects/datagen/output/tree_cache/videomme",
    },
    "longvideobench": {
        "anno_path": "/mnt/arc/cezhang/datasets/longvideobench/lvb_val.json",
        # "video_base_path": "/mnt/opr/yblin/mycache/longvideobench/videos",
        "video_base_path": "/mnt/arc/cezhang/datasets/longvideobench/videos_reencode",
        "tree_cache_dir": "/mnt/arc/cezhang/projects/datagen/output/tree_cache/longvideobench",
    },
}


def run_inference(
    base_url: str = "http://bumblebee:1234/v1",
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    num_examples: int = 10,
    output_dir: str = "./output/inference_results",
    num_workers: int = 4,
    use_dummy: bool = False,
    dataset_type: str = "cgbench",
    anno_path: str | None = None,
    video_base_path: str | None = None,
    eval_only: bool = False,
    resume: bool = False,
    save_prompts: bool = False,
    # Inference config
    max_turns: int = 20,
    fps: float = 1.0,
    max_frames: int = 64,
    short_side: int = 256,
    min_turns: int = 0,
    max_in_loop: int = 1,
    loop_detection_mode: str = "visited",
    loop_detection_window: int = 2,
    direct: bool = False,
    direct_prompt_style: str = "json",
    video_native: bool = False,
    action_mode: str = "segmented",
    keyframe_mode: bool = False,
    clue_mode: bool = False,
    # Segment splitting (common)
    segment_mode: str = "uniform",
    min_segment_duration: float = 60.0,
    max_depth: int = -1,
    clip_url: str = "grpc://localhost:51000",
    # Uniform segmentation
    num_children: int = 4,
    # Uniform fixed-duration segmentation
    uf_child_duration: float = 60.0,
    # Query-agnostic CLIP segmentation
    scene_min_segments: int = 2,
    scene_max_segments: int = 8,
    scene_k: float = 1.5,
    scene_fps: float = 1.0,
    scene_max_frames: int = 64,
    scene_min_duration: float = 4.0,
    # Direct-nonuniform segmentation
    dn_num_segments: int = 64,
    dn_clip_fps: float = 1.0,
    # VLM type selection
    vlm_type: str = "gpt",
    caption_base_url: str | None = None,
    caption_model: str = "Qwen/Qwen3-VL-8B-Instruct",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    llm_api_key: str = "EMPTY",
    captioning_mode: str = "segment",
    memory_frames_per_node: int = 0,
    image_tokens: int | None = None,
    separate_caption_generation: bool = False,
    caption_frames_from_parent: bool = False,
    skip_reasoning: bool = False,
    pregenerated_caption_path: str | None = None,
    use_frame_captions: bool = False,
    reasoning_video_representation: str | None = None,
    tree_cache_dir: str | None = None,
    decide_action_type: str = "default",
    enforce_valid_segments: bool = True,
    allowed_actions: list[str] | tuple[str, ...] = ("ZOOM_IN", "ZOOM_OUT", "SHIFT", "ANSWER"),
    qids: list[int] | None = None,
    # Keyframe sampling (legacy lvhaystack eval)
    keyframe_sampling: str = "uniform",
    keyframe_uniform_num: int = 8,
    keyframe_clip_fps: float = 0.5,
    keyframe_clip_max_frames: int = 64,
    lvh_breakdown: bool = False,
    timestamps_file: str | None = None,
):
    """
    Run inference on a dataset and compute aggregate metrics.

    Args:
        num_examples: Number of examples to process
        output_dir: Directory to save results
        num_workers: Number of parallel worker threads
        use_dummy: Use DummyVLMInterface for testing (no real VLM calls)
        dataset_type: "cgbench" or "lvhaystack"
        eval_only: Skip inference; only compute metrics from existing results
        resume: Skip examples that are already processed (default: False, regenerate all)
    """
    if num_examples == -1:
        num_examples = None

    if video_native and not direct:
        raise ValueError("--video-native is only supported with --direct mode")

    fixed_timestamps_map = None
    if timestamps_file is not None:
        if not direct:
            raise ValueError("--timestamps-file is only supported with --direct mode")
        with open(timestamps_file) as f:
            fixed_timestamps_map = json.load(f)
        print(f"\nLoaded fixed timestamps for {len(fixed_timestamps_map)} videos "
              f"from {timestamps_file}")

    # When filtering by qids, load all samples first so qid filtering works
    if qids is not None:
        num_examples = None

    seed = 42
    random.seed(seed)

    print("=" * 80)
    print("Video Inference Engine")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    samples_output_path = output_path / "samples"
    samples_output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {output_path.absolute()}")

    if eval_only:
        results = []
        for i, sample_path in enumerate(sorted(samples_output_path.glob("*.json"))):
            with open(sample_path) as f:
                data = json.load(f)
            results.append({
                'index': i,
                'video_id': data.get('video_id', ''),
                'qid': data.get('qid', i),
                'success': True,
                'time': data.get('generation_time', 0.0),
                'error': None,
                'answer_correct': data.get('answer_correct'),
                'evidence_iou': data.get('evidence_iou'),
                'num_turns': data.get('num_turns'),
                'total_frames': data.get('total_frames'),
                'num_keyframes': len(data['keyframe_timestamps']) if data.get('keyframe_timestamps') else None,
                'forced_termination_reason': data.get('metadata', {}).get('forced_termination_reason'),
            })
        failed_path = samples_output_path / "failed"
        if failed_path.exists():
            for sample_path in sorted(failed_path.glob("*.json")):
                with open(sample_path) as f:
                    data = json.load(f)
                results.append({
                    'index': len(results),
                    'video_id': data.get('video_id', ''),
                    'qid': data.get('qid', len(results)),
                    'success': False,
                    'time': data.get('generation_time', 0.0),
                    'error': None,
                    'answer_correct': data.get('answer_correct'),
                    'evidence_iou': data.get('evidence_iou'),
                    'num_turns': data.get('num_turns'),
                    'total_frames': data.get('total_frames'),
                    'num_keyframes': len(data['keyframe_timestamps']) if data.get('keyframe_timestamps') else None,
                    'forced_termination_reason': data.get('metadata', {}).get('forced_termination_reason'),
                })
        wall_elapsed = 0.0
        print(f"Loaded {len(results)} results")
    else:
        # Resolve paths: use explicit args or fall back to dataset defaults
        defaults = DATASET_DEFAULTS[dataset_type]
        if anno_path is None:
            anno_path = defaults["anno_path"]
        if video_base_path is None:
            video_base_path = defaults["video_base_path"]
        if tree_cache_dir is None:
            tree_cache_dir = defaults.get("tree_cache_dir")

        # Load dataset
        print(f"\nLoading {dataset_type} data...")
        print(f"  Annotations: {anno_path}")
        print(f"  Videos: {video_base_path}")
        try:
            if dataset_type in ["cgbench", "cgbench_mini"]:
                dataset = load_cgbench_data(
                    anno_path, video_base_path,
                    num_samples=num_examples, shuffle=False,
                    multi_interval=True
                )
            elif dataset_type == "lvhaystack_ego4d":
                dataset = load_longvideohaystack_data(
                    anno_path, video_base_path,
                    num_samples=num_examples, shuffle=False
                )
            elif dataset_type == "lvhaystack_longvideobench":
                dataset = load_tstar_data(
                    anno_path, video_base_path,
                    num_samples=num_examples, shuffle=False
                )
            elif dataset_type == "momentseeker":
                dataset = load_momentseeker_data(
                    anno_path, video_base_path,
                    num_samples=num_examples, shuffle=False,
                    multi_interval=True  # Enable for R@1 evaluation with multi-interval support
                )
            elif dataset_type == "lvbench":
                dataset = load_lvbench_data(
                    anno_path, video_base_path,
                    num_samples=num_examples, shuffle=False,
                )
            elif dataset_type in ["mlvu", "mlvu_dev"]:
                dataset = load_mlvu_data(
                    anno_path, video_base_path,
                    num_samples=num_examples, shuffle=False,
                )
            elif dataset_type == "videomme":
                dataset = load_videomme_data(
                    anno_path, video_base_path,
                    num_samples=num_examples, shuffle=False,
                )
            elif dataset_type == "longvideobench":
                dataset = load_longvideobench_data(
                    anno_path, video_base_path,
                    num_samples=num_examples, shuffle=False,
                )
            else:
                raise ValueError(f"Invalid dataset type: {dataset_type}")
            print(f"Loaded {len(dataset)} samples")
        except FileNotFoundError as e:
            print(f"Error: Could not find file - {e}")
            print("Please check the paths and try again.")
            return
        except Exception as e:
            print(f"Error loading data: {e}")
            return

        # Filter by specific qids if requested
        if qids is not None:
            qid_set = set(qids)
            dataset = [item for item in dataset if item['qid'] in qid_set]
            print(f"Filtered to {len(dataset)} samples matching qids: {qids}")

        if not dataset:
            print("No valid samples found!")
            return

        # Filter out already-processed samples if in resume mode
        samples_to_process = []
        existing_results = []
        if resume:
            print(f"\nResume mode: checking for already-processed samples...")
            for i, item in enumerate(dataset):
                safe_id = item['video_id'].replace('/', '_')
                filename = f"{i+1:03d}_{safe_id}_inference.json"
                meta_path = samples_output_path / filename
                failed_meta_path = samples_output_path / "failed" / filename
                found_path = meta_path if meta_path.exists() else (failed_meta_path if failed_meta_path.exists() else None)
                if found_path:
                    # Load existing result
                    try:
                        with open(found_path) as f:
                            data = json.load(f)
                        existing_results.append({
                            'index': i,
                            'video_id': data.get('video_id', item['video_id']),
                            'qid': data.get('qid', i),
                            'success': not data.get('metadata', {}).get('forced_termination', False),
                            'time': data.get('generation_time', 0.0),
                            'error': None,
                            'answer_correct': data.get('answer_correct'),
                            'evidence_iou': data.get('evidence_iou'),
                            'num_turns': data.get('num_turns'),
                            'total_frames': data.get('total_frames'),
                            'num_keyframes': len(data['keyframe_timestamps']) if data.get('keyframe_timestamps') else None,
                            'forced_termination_reason': data.get('metadata', {}).get('forced_termination_reason'),
                        })
                    except Exception as e:
                        print(f"Warning: Failed to load existing result for {item['video_id']}: {e}")
                        samples_to_process.append((i, item))
                else:
                    samples_to_process.append((i, item))

            print(f"  Found {len(existing_results)} already-processed samples")
            print(f"  Will process {len(samples_to_process)} remaining samples")

            if not samples_to_process:
                print("\nAll samples already processed! Loading existing results...")
                results = existing_results
                wall_elapsed = 0.0
        else:
            # Process all samples
            samples_to_process = [(i, item) for i, item in enumerate(dataset)]

        # Run inference
        if samples_to_process:
            if num_workers <= 1:
                print(f"\nRunning inference in single-process mode...")
            else:
                print(f"\nRunning inference with {num_workers} workers...")
            print("=" * 80)

            wall_start = time.time()
            results = []
            total_samples = len(samples_to_process)

            # Common kwargs for _process_sample
            sample_kwargs = dict(
                base_url=base_url, model_name=model_name, use_dummy=use_dummy,
                direct=direct, direct_prompt_style=direct_prompt_style,
                video_native=video_native,
                action_mode=action_mode, keyframe_mode=keyframe_mode, clue_mode=clue_mode,
                dataset_type=dataset_type,
                max_turns=max_turns, fps=fps,
                max_frames=max_frames, short_side=short_side,
                min_turns=min_turns,
                max_in_loop=max_in_loop,
                loop_detection_mode=loop_detection_mode,
                loop_detection_window=loop_detection_window,
                segment_mode=segment_mode,
                min_segment_duration=min_segment_duration,
                max_depth=max_depth,
                clip_url=clip_url,
                num_children=num_children,
                uf_child_duration=uf_child_duration,
                scene_min_segments=scene_min_segments, scene_max_segments=scene_max_segments,
                scene_k=scene_k, scene_fps=scene_fps, scene_max_frames=scene_max_frames,
                scene_min_duration=scene_min_duration,
                dn_num_segments=dn_num_segments, dn_clip_fps=dn_clip_fps,
                save_prompts=save_prompts,
                vlm_type=vlm_type,
                caption_base_url=caption_base_url,
                caption_model=caption_model,
                llm_base_url=llm_base_url,
                llm_model=llm_model,
                llm_api_key=llm_api_key,
                captioning_mode=captioning_mode,
                memory_frames_per_node=memory_frames_per_node,
                image_tokens=image_tokens,
                separate_caption_generation=separate_caption_generation,
                caption_frames_from_parent=caption_frames_from_parent,
                skip_reasoning=skip_reasoning,
                pregenerated_caption_path=pregenerated_caption_path,
                use_frame_captions=use_frame_captions,
                reasoning_video_representation=reasoning_video_representation,
                tree_cache_dir=tree_cache_dir,
                decide_action_type=decide_action_type,
                enforce_valid_segments=enforce_valid_segments,
                allowed_actions=allowed_actions,
                keyframe_sampling=keyframe_sampling,
                keyframe_uniform_num=keyframe_uniform_num,
                keyframe_clip_fps=keyframe_clip_fps,
                keyframe_clip_max_frames=keyframe_clip_max_frames,
                fixed_timestamps_map=fixed_timestamps_map,
            )

            if num_workers <= 1:
                for idx, item in tqdm(samples_to_process, desc="Inference"):
                    results.append(
                        _process_sample(
                            item, idx, total_samples, samples_output_path,
                            **sample_kwargs
                        )
                    )
            else:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(
                            _process_sample,
                            item, idx, total_samples, samples_output_path,
                            **sample_kwargs
                        ): idx
                        for idx, item in samples_to_process # if idx > 700 # change here to do muiltple split
                    }
                    for future in tqdm(
                        as_completed(futures), total=len(futures), desc="Inference"
                    ):
                        results.append(future.result())

            wall_elapsed = time.time() - wall_start

            # Combine with existing results if in resume mode
            if resume and existing_results:
                results.extend(existing_results)

            results.sort(key=lambda r: r['index'])

        # Backfill stats from saved JSON files for any results missing them
        # (e.g., samples that hit exceptions in _process_sample never set
        # num_turns/total_frames). This mirrors what --eval-only does so the
        # summary's all_turn_stats / frame_stats come out the same either way.
        for r in results:
            needs_backfill = (
                r.get('num_turns') is None
                or r.get('total_frames') is None
                or (not r.get('success', True) and r.get('forced_termination_reason') is None)
            )
            if not needs_backfill:
                continue
            safe_id = r['video_id'].replace('/', '_')
            filename = f"{r['index']+1:03d}_{safe_id}_inference.json"
            for path in (samples_output_path / filename,
                         samples_output_path / "failed" / filename):
                if not path.exists():
                    continue
                try:
                    with open(path) as f:
                        data = json.load(f)
                except Exception:
                    continue
                if r.get('num_turns') is None:
                    r['num_turns'] = data.get('num_turns')
                if r.get('total_frames') is None:
                    r['total_frames'] = data.get('total_frames')
                if r.get('num_keyframes') is None and data.get('keyframe_timestamps'):
                    r['num_keyframes'] = len(data['keyframe_timestamps'])
                if r.get('forced_termination_reason') is None:
                    r['forced_termination_reason'] = (
                        data.get('metadata', {}).get('forced_termination_reason')
                    )
                break

    # ===== Aggregate Metrics =====
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    failed_max_turns = [r for r in failed if r.get('forced_termination_reason') == 'max_turns_reached']
    failed_decide_action = [r for r in failed if r.get('forced_termination_reason') == 'decide_action_error']
    failed_loop = [r for r in failed if r.get('forced_termination_reason') == 'loop_detected']
    failed_unknown_action = [r for r in failed if r.get('forced_termination_reason') == 'unknown_action']

    # Answer accuracy (successful only)
    correct_answers = [r for r in successful if r['answer_correct'] is True]
    total_answered = [r for r in successful if r['answer_correct'] is not None]
    accuracy = len(correct_answers) / len(total_answered) if total_answered else 0.0

    # Evidence IoU (successful only)
    ious = [r['evidence_iou'] for r in successful
            if r['evidence_iou'] is not None]
    mean_iou = float(np.mean(ious)) if ious else 0.0

    # Failed samples metrics
    failed_correct = [r for r in failed if r['answer_correct'] is True]
    failed_answered = [r for r in failed if r['answer_correct'] is not None]
    failed_accuracy = len(failed_correct) / len(failed_answered) if failed_answered else 0.0
    failed_ious = [r['evidence_iou'] for r in failed if r['evidence_iou'] is not None]
    failed_mean_iou = float(np.mean(failed_ious)) if failed_ious else 0.0

    # All samples metrics (union of successful + failed)
    all_correct = [r for r in results if r['answer_correct'] is True]
    all_answered = [r for r in results if r['answer_correct'] is not None]
    all_accuracy = len(all_correct) / len(all_answered) if all_answered else 0.0
    all_ious = [r['evidence_iou'] for r in results if r['evidence_iou'] is not None]
    all_mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    # Turn statistics
    turn_counts = [r['num_turns'] for r in successful if r.get('num_turns') is not None]
    turn_stats = {}
    if turn_counts:
        turn_stats = {
            'mean': float(np.mean(turn_counts)),
            'median': float(np.median(turn_counts)),
            'min': int(np.min(turn_counts)),
            'max': int(np.max(turn_counts)),
            'std': float(np.std(turn_counts)),
        }

    all_turn_counts = [r['num_turns'] for r in results if r.get('num_turns') is not None]
    all_turn_stats = {}
    if all_turn_counts:
        all_turn_stats = {
            'mean': float(np.mean(all_turn_counts)),
            'median': float(np.median(all_turn_counts)),
            'min': int(np.min(all_turn_counts)),
            'max': int(np.max(all_turn_counts)),
            'std': float(np.std(all_turn_counts)),
        }

    # Keyframe statistics
    keyframe_counts = [r['num_keyframes'] for r in successful if r.get('num_keyframes') is not None]
    keyframe_stats = {}
    if keyframe_counts:
        keyframe_stats = {
            'mean': float(np.mean(keyframe_counts)),
            'median': float(np.median(keyframe_counts)),
            'min': int(np.min(keyframe_counts)),
            'max': int(np.max(keyframe_counts)),
            'std': float(np.std(keyframe_counts)),
        }

    # Frame statistics
    frame_counts = [r['total_frames'] for r in successful if r.get('total_frames') is not None]
    frame_stats = {}
    if frame_counts:
        frame_stats = {
            'mean': float(np.mean(frame_counts)),
            'median': float(np.median(frame_counts)),
            'min': int(np.min(frame_counts)),
            'max': int(np.max(frame_counts)),
            'std': float(np.std(frame_counts)),
        }

    all_frame_counts = [r['total_frames'] for r in results if r.get('total_frames') is not None]
    all_frame_stats = {}
    if all_frame_counts:
        all_frame_stats = {
            'mean': float(np.mean(all_frame_counts)),
            'median': float(np.median(all_frame_counts)),
            'min': int(np.min(all_frame_counts)),
            'max': int(np.max(all_frame_counts)),
            'std': float(np.std(all_frame_counts)),
        }

    # Timing
    total_cpu_time = sum(r['time'] for r in successful)

    # ===== Longvideohaystack-specific Metrics =====
    lvh_metrics = None
    if "lvhaystack" in dataset_type:
        print("\n" + "=" * 80)
        print("COMPUTING LONGVIDEOHAYSTACK METRICS")
        print("=" * 80)

        # Load per-sample JSONs (success + forced-termination) and split.
        success_data: list = []
        failed_data: list = []

        def _collect(sample_paths, target):
            for sample_path in sample_paths:
                try:
                    with open(sample_path) as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load {sample_path}: {e}")
                    continue
                if (data.get('keyframe_timestamps') is not None and
                        data.get('gt_frame_index') is not None and
                        data.get('video_path') is not None):
                    target.append(data)

        _collect(sorted(samples_output_path.glob("*.json")), success_data)
        failed_path = samples_output_path / "failed"
        if failed_path.is_dir():
            _collect(sorted(failed_path.glob("*.json")), failed_data)

        overall_data = success_data + failed_data

        if overall_data:
            from utils.eval import evaluate_longvideohaystack
            if lvh_breakdown:
                splits = [
                    ("success", success_data),
                    ("failed", failed_data),
                    ("overall", overall_data),
                ]
            else:
                splits = [("overall", overall_data)]
            lvh_metrics = {}
            for split_name, split_data in splits:
                if not split_data:
                    print(f"\n[{split_name}] no samples — skipping eval")
                    lvh_metrics[split_name] = None
                    continue
                print(f"\n[{split_name}] evaluating {len(split_data)} samples ...")
                try:
                    metrics = evaluate_longvideohaystack(
                        result_data=split_data,
                        threshold=5,
                        max_workers=min(num_workers, 4)
                    )
                    lvh_metrics[split_name] = {
                        "num_evaluated": len(split_data),
                        **metrics,
                    }
                    print(f"  Temporal P/R/F1: "
                          f"{metrics['Average Temporal Precision']:.4f} / "
                          f"{metrics['Average Temporal Recall']:.4f} / "
                          f"{metrics['Average Temporal F1 Score']:.4f}")
                    print(f"  SSIM     P/R/F1: "
                          f"{metrics['Average SSIM Precision']:.4f} / "
                          f"{metrics['Average SSIM Recall']:.4f} / "
                          f"{metrics['Average SSIM F1 Score']:.4f}")
                except Exception as e:
                    print(f"Error computing Longvideohaystack metrics for {split_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    lvh_metrics[split_name] = None

            # Compact side-by-side table.
            print("\n" + "=" * 80)
            if lvh_breakdown:
                print("LONGVIDEOHAYSTACK METRICS (success / failed / overall)")
            else:
                print("LONGVIDEOHAYSTACK METRICS (overall)")
            print("=" * 80)
            metric_keys = [
                "Average Temporal Precision",
                "Average Temporal Recall",
                "Average Temporal F1 Score",
                "Average SSIM Precision",
                "Average SSIM Recall",
                "Average SSIM F1 Score",
            ]
            split_cols = [s for s, _ in splits]
            header = f"{'metric':<32}" + "".join(f" {c:>10}" for c in split_cols)
            print(header)
            print("-" * len(header))
            counts = "  ".join(
                f"n={lvh_metrics[s]['num_evaluated'] if lvh_metrics.get(s) else 0}"
                for s in split_cols
            )
            print(f"{'num_evaluated':<32} {counts}")
            for k in metric_keys:
                row = f"{k:<32}"
                for s in split_cols:
                    v = lvh_metrics.get(s)
                    cell = f"{v[k]:.4f}" if v is not None else "   -  "
                    row += f" {cell:>10}"
                print(row)
        else:
            print("Warning: No valid samples found for Longvideohaystack evaluation")

    # ===== MomentSeeker-specific Metrics =====
    ms_metrics = None
    if dataset_type == "momentseeker":
        print("\n" + "=" * 80)
        print("COMPUTING MOMENTSEEKER R@1 METRICS")
        print("=" * 80)

        # Load per-sample JSONs and prepare data for evaluation
        eval_data = []
        for sample_path in sorted(samples_output_path.glob("*.json")):
            try:
                with open(sample_path) as f:
                    data = json.load(f)
                    # Include samples with required fields
                    if data.get('gt_timestamps') and data.get('turns'):
                        eval_data.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {sample_path}: {e}")

        if eval_data:
            from utils.eval import evaluate_momentseeker
            try:
                ms_metrics = evaluate_momentseeker(
                    result_data=eval_data,
                    iou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]
                )

                print("\n" + "=" * 80)
                print("MOMENTSEEKER R@1 METRICS")
                print("=" * 80)
                print(f"R@1 @ IoU=0.1: {ms_metrics['R@1_IoU=0.1']:.4f}")
                print(f"R@1 @ IoU=0.2: {ms_metrics['R@1_IoU=0.2']:.4f}")
                print(f"R@1 @ IoU=0.3: {ms_metrics['R@1_IoU=0.3']:.4f}  (main metric)")
                print(f"R@1 @ IoU=0.4: {ms_metrics['R@1_IoU=0.4']:.4f}")
                print(f"R@1 @ IoU=0.5: {ms_metrics['R@1_IoU=0.5']:.4f}")
                print(f"Evaluated samples: {ms_metrics['num_samples']}")
            except Exception as e:
                print(f"Error computing MomentSeeker metrics: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Warning: No valid samples found for MomentSeeker evaluation")

    # ===== Video-MME-specific Metrics =====
    vmme_metrics = None
    if dataset_type == "videomme":
        print("\n" + "=" * 80)
        print("COMPUTING VIDEO-MME METRICS")
        print("=" * 80)

        # Load per-sample JSONs for evaluation
        eval_data = []
        for sample_path in sorted(samples_output_path.glob("*.json")):
            try:
                with open(sample_path) as f:
                    data = json.load(f)
                    if data.get('answer_correct') is not None:
                        eval_data.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {sample_path}: {e}")

        # Also include failed samples
        failed_path = samples_output_path / "failed"
        if failed_path.exists():
            for sample_path in sorted(failed_path.glob("*.json")):
                try:
                    with open(sample_path) as f:
                        data = json.load(f)
                        if data.get('answer_correct') is not None:
                            eval_data.append(data)
                except Exception as e:
                    print(f"Warning: Failed to load {sample_path}: {e}")

        if eval_data:
            from utils.eval import evaluate_videomme
            try:
                vmme_metrics = evaluate_videomme(result_data=eval_data)

                print("\n" + "=" * 80)
                print("VIDEO-MME ACCURACY")
                print("=" * 80)
                print(f"Overall:  {vmme_metrics['overall_accuracy']:.4f} ({vmme_metrics['num_samples']} samples)")
                for cat in ['short', 'medium', 'long']:
                    print(f"  {cat.capitalize():8s}: {vmme_metrics[f'{cat}_accuracy']:.4f} ({vmme_metrics[f'{cat}_count']} samples)")
            except Exception as e:
                print(f"Error computing Video-MME metrics: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Warning: No valid samples found for Video-MME evaluation")

    # ===== MLVU-specific Metrics =====
    mlvu_metrics = None
    if dataset_type in ["mlvu", "mlvu_dev"]:
        print("\n" + "=" * 80)
        print("COMPUTING MLVU METRICS")
        print("=" * 80)

        eval_data = []
        for sample_path in sorted(samples_output_path.glob("*.json")):
            try:
                with open(sample_path) as f:
                    data = json.load(f)
                    if data.get('answer_correct') is not None:
                        eval_data.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {sample_path}: {e}")

        failed_path = samples_output_path / "failed"
        if failed_path.exists():
            for sample_path in sorted(failed_path.glob("*.json")):
                try:
                    with open(sample_path) as f:
                        data = json.load(f)
                        if data.get('answer_correct') is not None:
                            eval_data.append(data)
                except Exception as e:
                    print(f"Warning: Failed to load {sample_path}: {e}")

        if eval_data:
            from utils.eval import evaluate_mlvu
            try:
                mlvu_metrics = evaluate_mlvu(result_data=eval_data)

                print("\n" + "=" * 80)
                print("MLVU ACCURACY")
                print("=" * 80)
                print(f"Overall:  {mlvu_metrics['overall_accuracy']:.4f} ({mlvu_metrics['num_samples']} samples)")
                for key in sorted(mlvu_metrics.keys()):
                    if key.endswith('_accuracy') and key != 'overall_accuracy':
                        qtype = key.removesuffix('_accuracy')
                        print(f"  {qtype:20s}: {mlvu_metrics[key]:.4f} ({mlvu_metrics[f'{qtype}_count']} samples)")
            except Exception as e:
                print(f"Error computing MLVU metrics: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Warning: No valid samples found for MLVU evaluation")

    # ===== LongVideoBench-specific Metrics =====
    lvb_metrics = None
    if dataset_type == "longvideobench":
        print("\n" + "=" * 80)
        print("COMPUTING LONGVIDEOBENCH METRICS")
        print("=" * 80)

        eval_data = []
        for sample_path in sorted(samples_output_path.glob("*.json")):
            try:
                with open(sample_path) as f:
                    data = json.load(f)
                    if data.get('answer_correct') is not None:
                        eval_data.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {sample_path}: {e}")

        failed_path = samples_output_path / "failed"
        if failed_path.exists():
            for sample_path in sorted(failed_path.glob("*.json")):
                try:
                    with open(sample_path) as f:
                        data = json.load(f)
                        if data.get('answer_correct') is not None:
                            eval_data.append(data)
                except Exception as e:
                    print(f"Warning: Failed to load {sample_path}: {e}")

        if eval_data:
            from utils.eval import evaluate_longvideobench
            try:
                lvb_metrics = evaluate_longvideobench(result_data=eval_data)

                print("\n" + "=" * 80)
                print("LONGVIDEOBENCH ACCURACY")
                print("=" * 80)
                print(f"Overall:  {lvb_metrics['overall_accuracy']:.4f} ({lvb_metrics['num_samples']} samples)")
                for key in sorted(lvb_metrics.keys()):
                    if key.endswith('_accuracy') and key != 'overall_accuracy':
                        category = key.removesuffix('_accuracy')
                        print(f"  {category:20s}: {lvb_metrics[key]:.4f} ({lvb_metrics[f'{category}_count']} samples)")
            except Exception as e:
                print(f"Error computing LongVideoBench metrics: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Warning: No valid samples found for LongVideoBench evaluation")

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Wall-clock time: {wall_elapsed:.2f}s ({wall_elapsed/60:.2f} min)")
    print(f"Total CPU time:  {total_cpu_time:.2f}s ({total_cpu_time/60:.2f} min)")
    print(f"Workers: {num_workers}")
    print(f"Successful: {len(successful)}/{len(results)}")
    if failed:
        print(f"Failed: {len(failed)}")
        print(f"  - max_turns_reached: {len(failed_max_turns)}")
        print(f"  - decide_action_error: {len(failed_decide_action)}")
        print(f"  - loop_detected: {len(failed_loop)}")
        print(f"  - unknown_action: {len(failed_unknown_action)}")
    if successful:
        print(f"Average per example: {total_cpu_time/len(successful):.2f}s")

    print(f"\nSuccessful Samples:")
    print(f"  Answer Accuracy: {accuracy:.4f} "
          f"({len(correct_answers)}/{len(total_answered)})")
    print(f"  Mean Evidence IoU: {mean_iou:.4f} (over {len(ious)} samples)")
    if failed:
        print(f"\nFailed (forced termination) Samples:")
        print(f"  Answer Accuracy: {failed_accuracy:.4f} "
              f"({len(failed_correct)}/{len(failed_answered)})")
        print(f"  Mean Evidence IoU: {failed_mean_iou:.4f} (over {len(failed_ious)} samples)")
    print(f"\nAll Samples:")
    print(f"  Answer Accuracy: {all_accuracy:.4f} "
          f"({len(all_correct)}/{len(all_answered)})")
    print(f"  Mean Evidence IoU: {all_mean_iou:.4f} (over {len(all_ious)} samples)")
    if turn_stats:
        print(f"\nTurn Statistics (over {len(turn_counts)} samples):")
        print(f"  Mean: {turn_stats['mean']:.2f} | Median: {turn_stats['median']:.1f}")
        print(f"  Min: {turn_stats['min']} | Max: {turn_stats['max']}")
        print(f"  Std: {turn_stats['std']:.2f}")
    if all_turn_stats:
        print(f"\nTurn Statistics (over {len(all_turn_counts)} all samples):")
        print(f"  Mean: {all_turn_stats['mean']:.2f} | Median: {all_turn_stats['median']:.1f}")
        print(f"  Min: {all_turn_stats['min']} | Max: {all_turn_stats['max']}")
        print(f"  Std: {all_turn_stats['std']:.2f}")
    if keyframe_stats:
        print(f"\nKeyframe Statistics (over {len(keyframe_counts)} samples):")
        print(f"  Mean: {keyframe_stats['mean']:.2f} | Median: {keyframe_stats['median']:.1f}")
        print(f"  Min: {keyframe_stats['min']} | Max: {keyframe_stats['max']}")
        print(f"  Std: {keyframe_stats['std']:.2f}")
    if frame_stats:
        print(f"\nFrame Statistics (over {len(frame_counts)} successful samples):")
        print(f"  Mean: {frame_stats['mean']:.2f} | Median: {frame_stats['median']:.1f}")
        print(f"  Min: {frame_stats['min']} | Max: {frame_stats['max']}")
        print(f"  Std: {frame_stats['std']:.2f}")
    if all_frame_stats:
        print(f"\nFrame Statistics (over {len(all_frame_counts)} all samples):")
        print(f"  Mean: {all_frame_stats['mean']:.2f} | Median: {all_frame_stats['median']:.1f}")
        print(f"  Min: {all_frame_stats['min']} | Max: {all_frame_stats['max']}")
        print(f"  Std: {all_frame_stats['std']:.2f}")

    # Save aggregate summary
    summary_path = output_path / "summary.json"
    uses_clip = segment_mode in ("query-agnostic-clip", "query-aware-clip", "direct-nonuniform")
    summary_dict = {
        'config': {
            'num_examples': num_examples,
            'num_workers': num_workers,
            'use_dummy': use_dummy,
            'dataset_type': dataset_type,
            'direct_baseline': direct,
            'action_mode': action_mode,
            'max_turns': max_turns,
            'fps': fps,
            'max_frames_per_segment': max_frames,
            'short_side': short_side,
            'min_segment_duration': min_segment_duration,
            'max_depth': max_depth,
            'keyframe_mode': keyframe_mode,
            'clue_mode': clue_mode,
            'resume': resume,
            'segment_mode': segment_mode,
            'clip_url': clip_url if uses_clip else None,
            'num_children': num_children if segment_mode == "uniform" else None,
            'uf_child_duration': uf_child_duration if segment_mode == "uniform-fixed" else None,
            'scene_min_segments': scene_min_segments if segment_mode == "query-agnostic-clip" else None,
            'scene_max_segments': scene_max_segments if segment_mode == "query-agnostic-clip" else None,
            'scene_k': scene_k if segment_mode == "query-agnostic-clip" else None,
            'scene_fps': scene_fps if segment_mode == "query-agnostic-clip" else None,
            'scene_max_frames': scene_max_frames if segment_mode == "query-agnostic-clip" else None,
            'scene_min_duration': scene_min_duration if segment_mode == "query-agnostic-clip" else None,
            'dn_num_segments': dn_num_segments if segment_mode == "direct-nonuniform" else None,
            'dn_clip_fps': dn_clip_fps if segment_mode == "direct-nonuniform" else None,
        },
        'total_samples': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'failed_breakdown': {
            'max_turns_reached': len(failed_max_turns),
            'decide_action_error': len(failed_decide_action),
            'loop_detected': len(failed_loop),
        },
        'accuracy': accuracy,
        'mean_evidence_iou': mean_iou,
        'failed_accuracy': failed_accuracy,
        'failed_mean_evidence_iou': failed_mean_iou,
        'all_accuracy': all_accuracy,
        'all_mean_evidence_iou': all_mean_iou,
        'turn_stats': turn_stats,
        'all_turn_stats': all_turn_stats,
        'keyframe_stats': keyframe_stats,
        'frame_stats': frame_stats,
        'all_frame_stats': all_frame_stats,
        'wall_time': wall_elapsed,
        'total_cpu_time': total_cpu_time,
    }

    # Add Longvideohaystack metrics if available
    if lvh_metrics is not None:
        summary_dict['lvhaystack_metrics'] = lvh_metrics

    # Add MomentSeeker metrics if available
    if ms_metrics is not None:
        summary_dict['momentseeker_metrics'] = ms_metrics

    if vmme_metrics is not None:
        summary_dict['videomme_metrics'] = vmme_metrics

    if lvb_metrics is not None:
        summary_dict['longvideobench_metrics'] = lvb_metrics

    with open(summary_path, 'w') as f:
        json.dump(summary_dict, f, indent=4)

    # Save per-sample results to separate file
    per_sample_path = output_path / "per_sample.json"
    with open(per_sample_path, 'w') as f:
        json.dump([
            {
                'index': r['index'],
                'video_id': r['video_id'],
                'qid': r['qid'],
                'success': r['success'],
                'forced_termination_reason': r.get('forced_termination_reason'),
                'time': r['time'],
                'num_turns': r.get('num_turns'),
                'total_frames': r.get('total_frames'),
                'num_keyframes': r.get('num_keyframes'),
                'answer_correct': r['answer_correct'],
                'evidence_iou': r['evidence_iou'],
                'error': r['error'],
            }
            for r in results
        ], f, indent=4)

    print(f"\nResults saved to: {output_path.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on video temporal reasoning datasets."
    )

    # General / I/O arguments
    parser.add_argument("-n", "--num-examples", type=int, default=10,
                        help="Number of examples to process, -1 for all (default: 10)")
    parser.add_argument("-o", "--output-dir", type=str,
                        default="./output/inference/cgbench_test",
                        help="Directory to save results")
    parser.add_argument("-w", "--num-workers", type=int, default=16,
                        help="Number of parallel worker threads (default: 16)")
    parser.add_argument("-d", "--dataset", type=str, default="cgbench",
                        choices=["cgbench", "cgbench_mini", "lvhaystack_ego4d", "lvhaystack_longvideobench", "momentseeker", "lvbench", "videomme", "mlvu", "mlvu_dev", "longvideobench"],
                        help="Dataset type (default: cgbench)")
    parser.add_argument("--anno-path", type=str, default=None,
                        help="Path to annotation file (default: dataset-specific)")
    parser.add_argument("--video-base-path", type=str, default=None,
                        help="Base path to video files (default: dataset-specific)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results: skip already-processed examples. "
                             "By default (without --resume), all examples are regenerated from scratch.")
    parser.add_argument("--qids", type=int, nargs="+", default=None,
                        help="Only process these specific question IDs (e.g. --qids 18 25)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip inference; only compute metrics from existing results in output-dir")
    parser.add_argument("--dummy", action="store_true",
                        help="Use DummyVLMInterface for testing (no real VLM calls)")
    parser.add_argument("--save-prompts", action="store_true",
                        help="Save all VLM prompts to disk (images replaced with <image> placeholder)")

    # VLM selection
    parser.add_argument("--vlm-type", type=str, default="gpt",
                        choices=["gpt", "caption_llm"],
                        help="VLM interface type: 'gpt' for GPTVLMInterface, "
                             "'caption_llm' for CaptionLLMInterface (default: gpt)")

    # GPT VLM arguments (--vlm-type gpt)
    parser.add_argument("--base-url", type=str, default="http://bumblebee:1234/v1",
                        help="Base URL for vLLM server")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Model name for VLM server (default: Qwen/Qwen3-VL-8B-Instruct)")

    # CaptionLLM arguments (--vlm-type caption_llm)
    parser.add_argument("--caption-base-url", type=str, default=None,
                        help="Base URL for captioning VLM server (caption_llm mode, defaults to --base-url)")
    parser.add_argument("--caption-model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Model name for captioning VLM (caption_llm mode)")
    parser.add_argument("--llm-base-url", type=str, default="https://api.deepseek.com",
                        help="Base URL for reasoning LLM server (caption_llm mode)")
    parser.add_argument("--llm-model", type=str, default="deepseek-reasoner",
                        help="Model name for reasoning LLM (caption_llm mode)")
    parser.add_argument("--llm-api-key", type=str, default="EMPTY",
                        help="API key for reasoning LLM (caption_llm mode)")
    parser.add_argument("--captioning-mode", type=str, default="frame",
                        choices=["segment", "frame"],
                        help="Captioning mode for caption_llm: 'segment' captions all frames "
                             "together, 'frame' captions each frame individually (default: segment)")
    parser.add_argument("--pregenerated-caption-path", type=str, default="/mnt/arc/cezhang/projects/Qwen2.5-VL/captions/cg_captions/qwen3_8b_frames_1fps",
                        help="Path to directory of pre-generated caption JSON files. "
                             "Each file is named {video_id}.json with timestamp keys.")
    parser.add_argument("--use-frame-captions", action="store_true",
                        help="Include pregenerated frame captions for child segments in the prompt "
                             "(requires --pregenerated-caption-path)")

    # Inference config arguments
    parser.add_argument("--max-turns", type=int, default=15,
                        help="Maximum number of action turns before forced termination (default: 20)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second for sampling within each segment (default: 1.0)")
    parser.add_argument("--max-frames", type=int, default=64,
                        help="Maximum frames per segment (default: 64)")
    parser.add_argument("--short-side", type=int, default=256,
                        help="Short side for frame resizing (default: 256)")
    parser.add_argument("--direct", action="store_true",
                        help="Use direct baseline (single-turn ANSWER at root, no search)")
    parser.add_argument("--direct-prompt-style", type=str, default="json",
                        choices=["json", "lmm"],
                        help="Prompt format used by --direct / --clue-mode / "
                             "--segment-mode direct-nonuniform: 'json' (codebase default; "
                             "asks for answer + evidence_start/end as JSON) or 'lmm' "
                             "(lmms-eval-style letter-only prompt with Qwen3-VL "
                             "<{ts:.1f} seconds> labels, temperature=0, max_tokens=32). "
                             "Use 'lmm' to match lmms-eval longvideobench evaluation.")
    parser.add_argument("--video-native", action="store_true",
                        help="--direct only: send the full video as a single video_url "
                             "attachment instead of interleaving sampled frames+timestamps")
    parser.add_argument("--action-mode", type=str, default="segmented",
                        choices=["segmented", "continuous"],
                        help="Action representation mode: 'segmented' uses segment_id, "
                             "'continuous' uses start_sec/end_sec (default: segmented)")
    parser.add_argument("--reasoning-video-representation", type=str, default="segmented",
                        choices=["segmented", "continuous", "segmented-frames-first"],
                        help="Video representation mode for reasoning prompts. "
                             "Defaults to 'segmented' if --direct or --clue-mode, else 'continuous'")
    parser.add_argument("--keyframe-mode", action="store_true",
                        help="Use direct keyframe prediction: predict specific timestamps "
                             "in seconds instead of evidence intervals (for lvhaystack/tstar)")
    parser.add_argument("--clue-mode", action="store_true",
                        help="Oracle clue baseline: feed GT clue interval to VLM "
                             "instead of full video (requires GT timestamps)")
    parser.add_argument("--separate-caption-generation", action="store_true",
                        help="Generate captions in separate per-segment VLM calls before decide_action")
    parser.add_argument("--caption-frames-from-parent", action="store_true",
                        help="Caption/decide each child using parent frames cropped to the child's "
                             "time range, instead of frames newly sampled within the child")
    parser.add_argument("--skip-reasoning", action="store_true",
                        help="Omit the reasoning field from VLM JSON schema (no chain-of-thought)")
    parser.add_argument("--decide-action-type", type=str, default="default",
                        choices=["default", "multiple_choice"],
                        help="Action decision formulation: 'default' (JSON with action type) or "
                             "'multiple_choice' (MCQ where each action is a numbered option)")
    parser.add_argument("--no-enforce-valid-segments", dest="enforce_valid_segments",
                        action="store_false", default=True,
                        help="Disable enforcement of valid (unvisited) segments in the action space")
    parser.add_argument("--allowed-actions", type=str, nargs="+",
                        default=["ZOOM_IN", "ZOOM_OUT", "SHIFT", "ANSWER"],
                        choices=["ZOOM_IN", "ZOOM_OUT", "SHIFT", "ANSWER"],
                        help="Whitelist of action types allowed during inference. Level-based "
                             "and validity filters are intersected with this list "
                             "(default: all four actions)")
    parser.add_argument("--min-turns", type=int, default=0,
                        help="Minimum turns before ANSWER action is allowed (default: 0)")
    parser.add_argument("--max-in-loop", type=int, default=1,
                        help="Force answer after this many consecutive loop detections (default: 1 = immediate)")
    parser.add_argument("--loop-detection-mode", type=str, default="visited",
                        choices=["visited", "window", "none"],
                        help="Loop detection mode: 'visited' (revisit any visited node) or 'window' (detect repeating sequences of this length)")
    parser.add_argument("--loop-detection-window", type=int, default=2,
                        help="Window size for 'window' loop detection mode (detects repeating sequences of this length, default: 2)")

    # Memory arguments
    parser.add_argument("--memory-frames-per-node", type=int, default=0,
                        help="Memory mode: -1=no memory, 0=caption-only (default), "
                             ">0=N uniformly sampled frames per visited node")
    parser.add_argument("--image-tokens", type=int, default=None,
                        help="Pin every image to this many vision tokens "
                             "(Qwen-VL only; sets min_pixels=max_pixels=N*784 via "
                             "mm_processor_kwargs). Default: server-decided.")

    # === Segment splitting (common, applies to all modes) ===
    seg_group = parser.add_argument_group("Segment splitting (common)")
    seg_group.add_argument("--segment-mode", type=str, default="query-agnostic-clip",
                           choices=["uniform", "uniform-fixed", "query-agnostic-clip",
                                    "query-aware-clip", "direct-nonuniform", "no-split"],
                           help="Strategy for splitting a node into children. "
                                "'uniform' = equal non-overlapping splits "
                                "(--num-children chunks); "
                                "'uniform-fixed' = non-overlapping chunks of fixed duration "
                                "(--uf-child-duration seconds each, last chunk may be shorter); "
                                "'query-agnostic-clip' = CLIP scene-boundary segmentation "
                                "(no query); "
                                "'query-aware-clip' = CLIP segmentation conditioned on the "
                                "question text embedding (NOT YET IMPLEMENTED); "
                                "'direct-nonuniform' = single-pass non-uniform baseline "
                                "(uses engine.infer_direct_nonuniform); "
                                "'no-split' = never split; the tree is just the root node "
                                "(equivalent to --max-depth 0).")
    seg_group.add_argument("--min-segment-duration", type=float, default=60.0,
                           help="If a node's duration is below this, treat it as a leaf "
                                "(no further children produced). Applies to all modes "
                                "(default: 60.0).")
    seg_group.add_argument("--max-depth", type=int, default=-1,
                           help="Maximum tree depth (root=0). 0 = no segmentation at all "
                                "(root is never split); -1 = no restriction (default: -1). "
                                "Applies to all modes.")
    seg_group.add_argument("--clip-url", type=str, default="grpc://localhost:51000",
                           help="gRPC URL for CLIP server (used by query-agnostic-clip, "
                                "query-aware-clip, and direct-nonuniform modes; "
                                "also used by --keyframe-sampling=clip).")

    # === Uniform segmentation (--segment-mode uniform) ===
    uni_group = parser.add_argument_group("Uniform segmentation (--segment-mode uniform)")
    uni_group.add_argument("--num-children", type=int, default=4,
                           help="Number of equal children per split (default: 4)")

    # === Uniform fixed-duration segmentation (--segment-mode uniform-fixed) ===
    uf_group = parser.add_argument_group(
        "Uniform fixed-duration segmentation (--segment-mode uniform-fixed)"
    )
    uf_group.add_argument("--uf-child-duration", type=float, default=60.0,
                          help="Target child duration in seconds. Each child is exactly "
                               "this long; the last child takes the remainder and may be "
                               "shorter. If parent duration <= this value, the node is "
                               "treated as a leaf (default: 60.0).")

    # === Query-agnostic CLIP segmentation (--segment-mode query-agnostic-clip) ===
    qa_group = parser.add_argument_group(
        "Query-agnostic CLIP segmentation (--segment-mode query-agnostic-clip)"
    )
    qa_group.add_argument("--scene-min-segments", type=int, default=3,
                          help="Min segments per split for scene detection (default: 3)")
    qa_group.add_argument("--scene-max-segments", type=int, default=8,
                          help="Max segments per split for scene detection (default: 8)")
    qa_group.add_argument("--scene-k", type=float, default=1.5,
                          help="Z-score multiplier for threshold strategy (default: 1.5)")
    qa_group.add_argument("--scene-fps", type=float, default=1.0,
                          help="FPS for frame sampling in scene detection (default: 1.0)")
    qa_group.add_argument("--scene-max-frames", type=int, default=64,
                          help="Max frames for scene detection per node (default: 64)")
    qa_group.add_argument("--scene-min-duration", type=float, default=4.0,
                          help="Min segment duration in seconds; shorter segments are merged "
                               "(default: 4.0)")

    # === Query-aware CLIP segmentation (--segment-mode query-aware-clip) ===
    # TODO: implement query-aware CLIP segmentation. The intent is to use the
    # question's text embedding to bias boundary placement (e.g., favor
    # boundaries near frames whose CLIP similarity to the query changes
    # sharply). No mode-specific args yet.

    # === Direct-nonuniform segmentation (--segment-mode direct-nonuniform) ===
    dn_group = parser.add_argument_group(
        "Direct-nonuniform segmentation (--segment-mode direct-nonuniform)"
    )
    dn_group.add_argument("--dn-num-segments", type=int, default=64,
                          help="Number of non-uniform segments produced by single-pass "
                               "scene segmentation (top-k mode, default: 64).")
    dn_group.add_argument("--dn-clip-fps", type=float, default=1.0,
                          help="FPS for sampling frames for boundary detection "
                               "(default: 1.0).")

    # Keyframe sampling (legacy lvhaystack eval: derive keyframes from the
    # predicted evidence interval)
    parser.add_argument("--keyframe-sampling", type=str, default="uniform",
                        choices=["uniform", "clip"],
                        help="Strategy for deriving keyframe timestamps from the "
                             "predicted evidence interval (lvhaystack eval only). "
                             "'uniform': linspace; 'clip': CLIP query-frame similarity "
                             "at --keyframe-clip-fps, capped by --keyframe-clip-max-frames.")
    parser.add_argument("--keyframe-uniform-num", type=int, default=8,
                        help="Number of frames for --keyframe-sampling=uniform (default: 8)")
    parser.add_argument("--keyframe-clip-fps", type=float, default=0.5,
                        help="Candidate extraction FPS for --keyframe-sampling=clip (default: 0.5)")
    parser.add_argument("--keyframe-clip-max-frames", type=int, default=64,
                        help="Max frames returned for --keyframe-sampling=clip (default: 64)")
    parser.add_argument("--lvh-breakdown", action="store_true",
                        help="LVHaystack eval: also report success / failed splits "
                             "in addition to the overall metrics.")
    parser.add_argument("--timestamps-file", type=str, default=None,
                        help="JSON file mapping {video_id: {qid: [t1, t2, ...]}} "
                             "(or {video_id: [t1, ...]}). When set, the precomputed "
                             "timestamps replace fps-based frame sampling. "
                             "Only supported with --direct.")

    # Tree cache
    parser.add_argument("--tree-cache-dir", type=str, default=None,
                        help="Path to directory of pre-built tree cache JSONs "
                             "(from build_tree_cache.py). When set, uses cached segment "
                             "boundaries and captions instead of runtime splitting/captioning.")

    args = parser.parse_args()

    run_inference(
        base_url=args.base_url,
        model_name=args.model,
        num_examples=args.num_examples,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        use_dummy=args.dummy,
        dataset_type=args.dataset,
        anno_path=args.anno_path,
        video_base_path=args.video_base_path,
        eval_only=args.eval_only,
        resume=args.resume,
        save_prompts=args.save_prompts,
        max_turns=args.max_turns,
        fps=args.fps,
        max_frames=args.max_frames,
        short_side=args.short_side,
        min_turns=args.min_turns,
        max_in_loop=args.max_in_loop,
        loop_detection_mode=args.loop_detection_mode,
        loop_detection_window=args.loop_detection_window,
        direct=args.direct,
        direct_prompt_style=args.direct_prompt_style,
        video_native=args.video_native,
        action_mode=args.action_mode,
        keyframe_mode=args.keyframe_mode,
        clue_mode=args.clue_mode,
        # Segment splitting
        segment_mode=args.segment_mode,
        min_segment_duration=args.min_segment_duration,
        max_depth=args.max_depth,
        clip_url=args.clip_url,
        num_children=args.num_children,
        uf_child_duration=args.uf_child_duration,
        scene_min_segments=args.scene_min_segments,
        scene_max_segments=args.scene_max_segments,
        scene_k=args.scene_k,
        scene_fps=args.scene_fps,
        scene_max_frames=args.scene_max_frames,
        scene_min_duration=args.scene_min_duration,
        dn_num_segments=args.dn_num_segments,
        dn_clip_fps=args.dn_clip_fps,
        vlm_type=args.vlm_type,
        caption_base_url=args.caption_base_url,
        caption_model=args.caption_model,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        captioning_mode=args.captioning_mode,
        memory_frames_per_node=args.memory_frames_per_node,
        image_tokens=args.image_tokens,
        separate_caption_generation=args.separate_caption_generation,
        caption_frames_from_parent=args.caption_frames_from_parent,
        skip_reasoning=args.skip_reasoning,
        pregenerated_caption_path=args.pregenerated_caption_path,
        use_frame_captions=args.use_frame_captions,
        reasoning_video_representation=args.reasoning_video_representation,
        tree_cache_dir=args.tree_cache_dir,
        decide_action_type=args.decide_action_type,
        enforce_valid_segments=args.enforce_valid_segments,
        allowed_actions=args.allowed_actions,
        qids=args.qids,
        keyframe_sampling=args.keyframe_sampling,
        keyframe_uniform_num=args.keyframe_uniform_num,
        keyframe_clip_fps=args.keyframe_clip_fps,
        keyframe_clip_max_frames=args.keyframe_clip_max_frames,
        lvh_breakdown=args.lvh_breakdown,
        timestamps_file=args.timestamps_file,
    )


# Example commands:
'''
# CGBench mini
## Baseline 1: Direct (single-turn, no search)
python example_inference.py -n -1 -w 16 -d cgbench_mini -o ./output/inference/cgbench_mini/direct_qwen8b_f64 --direct --max-frames 64 --anno-path /mnt/arc/cezhang/datasets/CG-Bench/cgbench_mini.json 
python example_inference.py -n -1 -w 16 -d cgbench_mini -o ./output/inference/cgbench_mini/direct_qwen8b_f256 --direct --max-frames 256 --anno-path /mnt/arc/cezhang/datasets/CG-Bench/cgbench_mini.json 
python example_inference.py -n -1 -w 4 -d cgbench_mini -o ./output/inference/cgbench_mini/direct_qwen8b_f786 --direct --max-frames 768 --anno-path /mnt/arc/cezhang/datasets/CG-Bench/cgbench_mini.json 

## Baseline 1b: Direct non-uniform (scene segmentation → center frames, single-turn)
python example_inference.py -n -1 -w 8 -d cgbench_mini -o ./output/inference/cgbench_mini/direct_nonuniform_qwen8b_f64 --segment-mode direct-nonuniform --dn-num-segments 64 --max-frames 64 --clip-url grpc://localhost:51000 --anno-path /mnt/arc/cezhang/datasets/CG-Bench/cgbench_mini.json

## Baseline 2: Segmented (multi-turn agentic with segment_id actions)
python example_inference.py -n -1 -w 8 -d cgbench_mini -o ./output/inference/cgbench_mini/seg_qwen8b_f64 --action-mode segmented --max-frames 64 --anno-path /mnt/arc/cezhang/datasets/CG-Bench/cgbench_mini.json



# LVHaystack Ego4D:
## Baseline 1: Direct (single-turn, no search)
python example_inference.py -n -1 -w 8 -d lvhaystack_ego4d -o ./output/inference/lvhaystack_ego4d_val/direct_qwen8b_f64 --direct --max-frames 64 --anno-path /mnt/arc/cezhang/datasets/LongVideoHaystack/data/val-00000-of-00001.parquet --keyframe-mode
python example_inference.py -n -1 -w 8 -d lvhaystack_ego4d -o ./output/inference/lvhaystack_ego4d_test_tiny/direct_qwen8b_f64 --direct --max-frames 64 --anno-path /mnt/arc/cezhang/datasets/LongVideoHaystack/data/test_tiny-00000-of-00001.parquet --keyframe-mode

python example_inference.py -n -1 -w 8 -d lvhaystack_ego4d -o ./output/inference/lvhaystack_ego4d_val/direct_qwen8b_f256 --direct --max-frames 256 --anno-path /mnt/arc/cezhang/datasets/LongVideoHaystack/data/val-00000-of-00001.parquet --keyframe-mode
python example_inference.py -n -1 -w 8 -d lvhaystack_ego4d -o ./output/inference/lvhaystack_ego4d_test_tiny/direct_qwen8b_f256 --direct --max-frames 256 --anno-path /mnt/arc/cezhang/datasets/LongVideoHaystack/data/test_tiny-00000-of-00001.parquet --keyframe-mode

## Baseline 2: Segmented (multi-turn agentic with segment_id actions)
python example_inference.py -n -1 -w 8 -d lvhaystack_ego4d -o ./output/inference/lvhaystack_ego4d_val/seg_qwen8b_f64 --action-mode segmented --max-frames 64 --anno-path /mnt/arc/cezhang/datasets/LongVideoHaystack/data/val-00000-of-00001.parquet
python example_inference.py -n -1 -w 8 -d lvhaystack_ego4d -o ./output/inference/lvhaystack_ego4d_test_tiny/seg_qwen8b_f64 --action-mode segmented --max-frames 64 --anno-path /mnt/arc/cezhang/datasets/LongVideoHaystack/data/test_tiny-00000-of-00001.parquet




# LVHaystack LongVideoBench:
## Baseline 1: Direct (single-turn, no search)
python example_inference.py -n -1 -w 8 -d lvhaystack_longvideobench -o ./output/inference/lvhaystack_longvideobench/direct_qwen8b_f64 --direct --max-frames 64 --anno-path /mnt/arc/cezhang/projects/TStar/lvb_val_TStarFormat_with_metadata.json --keyframe-mode
python example_inference.py -n -1 -w 8 -d lvhaystack_longvideobench -o ./output/inference/lvhaystack_longvideobench/direct_qwen8b_f256 --direct --max-frames 256 --anno-path /mnt/arc/cezhang/projects/TStar/lvb_val_TStarFormat_with_metadata.json --keyframe-mode

## Baseline 2: Segmented (multi-turn agentic with segment_id actions)
python example_inference.py -n -1 -w 8 -d lvhaystack_longvideobench -o ./output/inference/lvhaystack_longvideobench/seg_qwen8b_f64 --action-mode segmented --max-frames 64 --anno-path /mnt/arc/cezhang/projects/TStar/lvb_val_TStarFormat_with_metadata.json


# MomentSeeker
python example_inference.py -n -1 -w 8 -d momentseeker -o ./output/inference/momentseeker/direct_qwen8b_f64 --direct --max-frames 64 --anno-path /mnt/arc/cezhang/datasets/MomentSeeker/t2v.json --base-url http://oprime:1234/v1
'''





# scene segmentation
'''
python example_inference.py -n -1 -w 16 -d cgbench_mini \
    --base-url http://localhost:4321/v1 \
    --clip-url grpc://localhost:51000 \
    --save-prompts \
    --segment-mode query-agnostic-clip \
    -o ./output/inference/cgbench_mini/agentic_qwen8b_f64 \
    --action-mode segmented \
    --max-frames 64


python example_inference.py -n -1 -w 8 -d cgbench_mini \
    --base-url http://localhost:4321/v1 \
    --clip-url grpc://localhost:51000 \
    --save-prompts \
    --segment-mode query-agnostic-clip \
    -o ./output/inference/cgbench_mini/agentic_qwen8b_f256 \
    --action-mode segmented \
    --max-frames 256
'''


# caption llm
'''
python example_inference.py -n 1000 -w 32 -d cgbench_mini \
    --vlm-type caption_llm \
    --base-url http://localhost:4321/v1 \
    --llm-api-key sk-3990209d16164af1acdf345e9cd301ed \
    --clip-url grpc://localhost:51000 \
    --save-prompts \
    --segment-mode query-agnostic-clip \
    -o ./output/inference/cgbench_mini/deepseek_r1_f64 \
    --action-mode segmented \
    --captioning-mode frame \
    --max-frames 64 \
    --resume
'''
'''
python example_inference.py -n 1000 -w 16 -d cgbench_mini \
    --vlm-type caption_llm \
    --base-url http://bumblebee:1234/v1 \
    --llm-api-key sk-3990209d16164af1acdf345e9cd301ed \
    --clip-url grpc://localhost:51000 \
    --save-prompts \
    --segment-mode query-agnostic-clip \
    -o ./output/inference/cgbench_mini/deepseek_r1_f128 \
    --action-mode segmented \
    --captioning-mode frame \
    --max-frames 128 \
    --resume
'''