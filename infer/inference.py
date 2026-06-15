"""
Inference engine for multi-round tool-augmented long-video reasoning.

Unlike the TrajectoryGenerator which uses ground truth to guide exploration,
the InferenceEngine relies entirely on the VLM to autonomously decide which
segments to explore, when to backtrack, and when to answer.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from .trajectory import Trajectory, Node, Action, Observation, Reasoning, ActionType
    from .vlm_interface import VLMInterface
    from .utils.temporal import compute_interval_iou, compute_interval_coverage, compute_multi_interval_iou, compute_multi_interval_coverage
except ImportError:
    from trajectory import Trajectory, Node, Action, Observation, Reasoning, ActionType
    from vlm_interface import VLMInterface
    from utils.temporal import compute_interval_iou, compute_interval_coverage, compute_multi_interval_iou, compute_multi_interval_coverage


@dataclass
class InferenceConfig:
    """
    Configuration for the inference engine.

    Attributes:
        num_children: Number of uniform non-overlapping children to split into
        max_turns: Maximum number of action turns before forced termination
        fps: Frames per second for sampling within each segment
        max_frames_per_segment: Maximum frames per child segment
        short_side: Resize frames so shorter side has this size (-1 for no resize)
        seed: Random seed for reproducibility
        force_answer_on_max_turns: Force an ANSWER when max_turns is reached
        min_turns: Minimum number of turns before ANSWER action is allowed
    """
    num_children: int = 4
    max_turns: int = 20
    fps: float = 1.0
    max_frames_per_segment: int = 64
    short_side: int = 256
    seed: Optional[int] = None
    force_answer_on_max_turns: bool = True
    min_turns: int = 0  # Minimum turns before ANSWER is allowed
    min_segment_duration: float = 60.0
    max_depth: int = -1  # Max tree depth (root=0). 0 = never split, -1 = unrestricted.
    max_in_loop: int = 1  # Force answer after this many consecutive loop detections (1 = immediate)
    loop_detection_mode: str = 'visited'  # 'visited' = revisit any visited node; 'window' = detect repeating action cycles
    loop_detection_window: int = 2  # Window size for 'window' mode; -1 = dynamic (try all sizes from 1..len//2)
    separate_caption_generation: bool = False  # If True, generate captions in a separate VLM call per segment before decide_action
    caption_frames_from_parent: bool = False  # If True, caption each child using parent frames cropped to the child's time range (instead of child.frames)
    skip_reasoning: bool = False  # If True, omit the reasoning field from the VLM JSON schema
    pregenerated_caption_path: Optional[str] = None  # Path to pregenerated caption JSON files for frame captions
    use_frame_captions: bool = False  # If True, include pregenerated frame captions in the prompt
    tree_cache_dir: Optional[str] = None  # Path to directory of pre-built tree cache JSONs (from build_tree_cache.py)
    decide_action_type: str = 'default'  # 'default' or 'multiple_choice' (uses MCQ formulation for action decisions)
    enforce_valid_segments: bool = True  # If True, enforce valid segments in the action space
    direct_prompt_style: str = 'json'  # 'json' (codebase default) or 'lmm' (lmms-eval-style letter-only prompt for direct/clue/direct-nonuniform paths)
    allowed_actions: Tuple[str, ...] = ("ZOOM_IN", "ZOOM_OUT", "SHIFT", "ANSWER")  # Global whitelist of action types; level/state filters intersect with this


class InferenceEngine:
    """
    Inference engine for autonomous video temporal reasoning.

    Algorithm:
    1. Initialize: S = [0, video_duration], create root observation
    2. Repeat until termination or max_turns:
       a. Split S into num_children uniform non-overlapping child segments
       b. Compress interaction history (actions + time ranges only)
       c. Call VLM.decide_action() with child segments + frames + history
       d. Execute action:
          - ZOOM_IN: move to selected child segment
          - ZOOM_OUT: move to parent segment
          - ANSWER: return evidence interval + final answer
    """

    def __init__(
        self,
        vlm_interface: VLMInterface,
        config: Optional[InferenceConfig] = None,
        frame_sampler: Optional[Callable] = None,
        segment_fn: Optional[Callable] = None
    ):
        """
        Initialize the inference engine.

        Args:
            vlm_interface: VLM interface for action decisions
            config: Inference configuration
            frame_sampler: Function to sample frames from video
                          Signature: (video_path, start_sec, end_sec) -> List[float]
            segment_fn: Optional function for content-aware segmentation (e.g. scene detection).
                       Signature: (video_path, start_sec, end_sec) -> List[Tuple[float, float]]
                       When provided, replaces uniform splitting in _split_into_children.
        """
        self.vlm = vlm_interface
        self.config = config if config is not None else InferenceConfig()
        self.frame_sampler = frame_sampler or self._default_frame_sampler
        self.segment_fn = segment_fn

        # Tree cache for pre-built segment trees and captions
        self.tree_cache = None
        if self.config.tree_cache_dir:
            try:
                from .utils.tree_cache import TreeCache
            except ImportError:
                from utils.tree_cache import TreeCache
            self.tree_cache = TreeCache(self.config.tree_cache_dir)

        if self.config.seed is not None:
            np.random.seed(self.config.seed)

    def _default_frame_sampler(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
    ) -> List[float]:
        """
        Default uniform frame sampler.

        Returns timestamps (in seconds) uniformly sampled based on fps,
        capped at max_frames_per_segment.
        """
        duration = end_sec - start_sec
        num_frames = min(
            self.config.max_frames_per_segment,
            max(1, int(duration * self.config.fps) + 1)
        )
        if num_frames == 1:
            return [(start_sec + end_sec) / 2.0]
        return np.linspace(start_sec, end_sec, num_frames).tolist()

    def infer(
        self,
        video_path: str,
        video_id: str,
        question: str,
        choices: List[str],
        video_duration: float,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> Trajectory:
        """
        Run inference on a single video + question.

        Args:
            video_path: Path to video file
            video_id: Video identifier
            question: The question to answer
            choices: Multiple choice options
            video_duration: Video duration in seconds
            ground_truth: Optional GT for post-hoc evaluation (not used for decisions)

        Returns:
            Trajectory with all turns recorded and evaluation metrics in metadata
        """
        self._current_video_id = video_id

        # Build GT node for evaluation only (not used during search)
        gt_node = None
        if ground_truth:
            gt_timestamps = ground_truth.get('timestamps')
            if gt_timestamps:
                gt_start, gt_end = gt_timestamps
                gt_node = Node(
                    start_sec=gt_start, end_sec=gt_end,
                    frames=[], video_path=video_path, level=-1
                )

        trajectory = Trajectory(
            video_id=video_id,
            question=question,
            ground_truth=ground_truth or {},
            gt_node=gt_node
        )

        # Step 1: Create root node covering entire video
        root_frames = self.frame_sampler(video_path, 0, video_duration)
        root_node = Node(
            start_sec=0, end_sec=video_duration,
            frames=root_frames, video_path=video_path,
            level=0, parent=None, node_id=0
        )
        root_node.visited = True

        # Populate root caption from tree cache if available
        if self.tree_cache is not None:
            root_caption = self.tree_cache.get_caption(video_id, 0, video_duration)
            if root_caption:
                root_node.text_description = root_caption

        # Set initial observation
        initial_obs = Observation(
            frames=root_frames,
            text_description=root_node.text_description or f"Root segment [0.0s - {video_duration:.1f}s]",
            node=root_node
        )
        trajectory.set_initial_observation(initial_obs)

        # State tracking
        current_node = root_node
        frames_per_turn = []
        loop_count = 0  # Consecutive loop detections
        action_history = []  # For window-mode loop detection: list of (action_type, start_sec, end_sec)

        # Step 2: Main loop
        for turn_idx in range(self.config.max_turns):
            if trajectory.is_terminated():
                break

            # (a) Split current segment into children
            children = self._split_into_children(current_node, video_path)

            # (b) Build trajectory context
            trajectory_context = {
                'video_id': trajectory.video_id,
                'question': question,
                'choices': choices,
                'tree': root_node.to_tree_dict(),
                'history': [
                    {
                        # 'reasoning': str(turn.reasoning) if turn.reasoning else None,
                        'action': str(turn.action) if turn.action else None,
                        'observation': {
                            'node_id': turn.observation.node.node_id,
                            'start_sec': turn.observation.node.start_sec,
                            'end_sec': turn.observation.node.end_sec,
                            'description': turn.observation.text_description if hasattr(turn.observation, 'text_description') else ''
                        } if turn.observation else None
                    }
                    for turn in trajectory.turns
                ]
            }

            # (c) Build child segment info for VLM
            def _frames_for_child(child):
                if not self.config.caption_frames_from_parent:
                    return child.frames
                cropped = [
                    f for f in current_node.frames
                    if child.start_sec <= f <= child.end_sec
                ]
                return cropped if cropped else child.frames

            child_segments_info = [
                {
                    'segment_id': child.node_id,
                    'start_sec': child.start_sec,
                    'end_sec': child.end_sec,
                    'frames': _frames_for_child(child),
                }
                for child in children
            ]

            # (c2) Build navigation context for VLM
            nav_context = {
                'start_sec': current_node.start_sec,
                'end_sec': current_node.end_sec,
                'children_info': [
                    {
                        'segment_id': child.node_id,
                        'start_sec': child.start_sec,
                        'end_sec': child.end_sec,
                        'visited': child.visited,
                    }
                    for child in children
                ],
            }
            if current_node.parent is not None:
                nav_context['parent'] = {
                    'start_sec': current_node.parent.start_sec,
                    'end_sec': current_node.parent.end_sec,
                    'node_id': current_node.parent.node_id,
                }
                nav_context['siblings'] = [
                    {
                        'segment_id': sibling.node_id,
                        'start_sec': sibling.start_sec,
                        'end_sec': sibling.end_sec,
                        'visited': sibling.visited,
                    }
                    for sibling in current_node.parent.children
                    if sibling is not current_node
                ]

            # Track frames used this turn
            frames_per_turn.append(len(current_node.frames))

            # (c3) Separate caption generation: caption each child before decide_action
            if self.config.separate_caption_generation:
                for child in children:
                    if child.text_description:
                        continue
                    # Try tree cache first to avoid unnecessary VLM calls
                    if self.tree_cache is not None:
                        cached_caption = self.tree_cache.get_caption(
                            self._current_video_id, child.start_sec, child.end_sec
                        )
                        if cached_caption:
                            child.text_description = cached_caption
                            continue
                    try:
                        caption = self.vlm.generate_description(
                            question=question,
                            frames=_frames_for_child(child),
                            start_sec=child.start_sec,
                            end_sec=child.end_sec,
                            video_path=video_path,
                            short_side=self.config.short_side,
                            detailed=True,
                        )
                        child.text_description = caption
                    except Exception as e:
                        print(f"Warning: generate_description failed for segment {child.node_id}: {e}")

            # (d) Call VLM to decide action
            try:
                if current_node.level == 0:
                    allowed_actions = ["ZOOM_IN", "ANSWER"]
                else:
                    allowed_actions = ["ZOOM_IN", "ZOOM_OUT", "SHIFT", "ANSWER"]
                config_allowed = set(self.config.allowed_actions)
                allowed_actions = [a for a in allowed_actions if a in config_allowed]

                if self.config.enforce_valid_segments:
                    # Forbid ZOOM_IN if no unvisited children (or no children at all)
                    if not any(not child.visited for child in children):
                        allowed_actions = [a for a in allowed_actions if a != "ZOOM_IN"]
                    # Forbid SHIFT if no unvisited siblings (or no siblings at all)
                    if current_node.parent is not None:
                        siblings = [s for s in current_node.parent.children if s is not current_node]
                        has_unvisited_siblings = any(not s.visited for s in siblings)
                    else:
                        has_unvisited_siblings = False
                    if not has_unvisited_siblings:
                        allowed_actions = [a for a in allowed_actions if a != "SHIFT"]
                    if turn_idx < self.config.min_turns:
                        allowed_actions = [a for a in allowed_actions if a != "ANSWER"]

                decide_fn = (self.vlm.decide_action_mcq
                             if self.config.decide_action_type == 'multiple_choice'
                             else self.vlm.decide_action)
                decision, raw_response = decide_fn(
                    question=question,
                    choices=choices,
                    child_segments=child_segments_info,
                    trajectory_context=trajectory_context,
                    current_start_sec=current_node.start_sec,
                    current_end_sec=current_node.end_sec,
                    current_frames=current_node.frames,
                    video_path=video_path,
                    short_side=self.config.short_side,
                    allowed_action_list=allowed_actions,
                    navigation_context=nav_context,
                    generate_captions=not self.config.separate_caption_generation,
                    skip_reasoning=self.config.skip_reasoning,
                    pregenerated_caption_path=self.config.pregenerated_caption_path,
                    use_frame_captions=self.config.use_frame_captions,
                    enforce_valid_segments=self.config.enforce_valid_segments,
                )
            except Exception as e:
                print(f"Warning: decide_action failed at turn {turn_idx}: {e}. "
                      f"Forcing ANSWER.")
                decision = self._force_answer_decision(
                    current_node, choices,
                    question=question, video_path=video_path,
                    trajectory_context=trajectory_context,
                )
                raw_response = ""
                trajectory.metadata['forced_termination'] = True
                trajectory.metadata['forced_termination_reason'] = 'decide_action_error'

            # (d2) Update child nodes with VLM-generated captions
            if not self.config.separate_caption_generation:
                captions = decision.get('captions', {})
                for child in children:
                    caption = captions.get(child.node_id)
                    if caption:
                        child.text_description = caption

            # (e) Execute action
            action_type = decision['action_type']
            reasoning = Reasoning(
                content=decision['reasoning'],
                context={'trajectory_context': trajectory_context},
                raw_response=raw_response,
            )

            target_node_visited_before = False
            if action_type in allowed_actions and action_type == 'ZOOM_IN':
                target_node = self._execute_zoom_in(
                    trajectory, current_node, children, decision, reasoning
                )
                target_node_visited_before = target_node.visited
                target_node.visited = True
            elif action_type in allowed_actions and action_type == 'ZOOM_OUT':
                target_node = self._execute_zoom_out(
                    trajectory, current_node, root_node, reasoning
                )
                target_node_visited_before = target_node.visited
                target_node.visited = True
            elif action_type in allowed_actions and action_type == 'SHIFT':
                target_node = self._execute_shift(
                    trajectory, current_node, decision, reasoning
                )
                target_node_visited_before = target_node.visited
                target_node.visited = True
            elif action_type in allowed_actions and action_type == 'ANSWER':
                target_node = self._execute_answer(
                    trajectory, current_node, decision, reasoning
                )
                target_node_visited_before = target_node.visited
                target_node.visited = True
            else:
                # Unknown action: force ANSWER
                print(f"Warning: Unknown action '{action_type}', forcing ANSWER.")
                decision = self._force_answer_decision(
                    current_node, choices,
                    question=question, video_path=video_path,
                    trajectory_context=trajectory_context,
                )
                reasoning = Reasoning(
                    content=decision.get('reasoning') or f'Unknown action {action_type} — forcing answer.',
                    context={'trajectory_context': trajectory_context},
                    raw_response='',
                )
                target_node = self._execute_answer(trajectory, current_node, decision, reasoning)
                trajectory.metadata['forced_termination'] = True
                trajectory.metadata['forced_termination_reason'] = 'unknown_action'
                break

            # --- Loop detection ---
            # Build action_history entry for window-mode detection
            if self.config.loop_detection_mode == 'window':
                state = (action_type, current_node.start_sec, current_node.end_sec)
                action_history.append(state)

            if self._detect_loop(action_type, target_node_visited_before, action_history):
                loop_count += 1
                if loop_count >= self.config.max_in_loop:
                    print(f"Loop detected at turn {turn_idx + 1} ({loop_count} consecutive). Forcing ANSWER.")
                    trajectory_context = {
                        'video_id': trajectory.video_id,
                        'question': question,
                        'choices': choices,
                        'tree': root_node.to_tree_dict(),
                        'history': [
                            {
                                'action': str(turn.action) if turn.action else None,
                                'observation': {
                                    'node_id': turn.observation.node.node_id,
                                    'start_sec': turn.observation.node.start_sec,
                                    'end_sec': turn.observation.node.end_sec,
                                    'description': turn.observation.text_description if hasattr(turn.observation, 'text_description') else ''
                                } if turn.observation else None
                            }
                            for turn in trajectory.turns
                        ]
                    }
                    decision = self._force_answer_decision(
                        target_node, choices,
                        question=question, video_path=video_path,
                        trajectory_context=trajectory_context,
                    )
                    reasoning = Reasoning(
                        content=decision.get('reasoning') or 'Loop detected — forcing answer.',
                        context={'trajectory_context': trajectory_context},
                        raw_response='',
                    )
                    self._execute_answer(trajectory, target_node, decision, reasoning)  # TODO note that target_node has not called split_into_children yet
                    trajectory.metadata['forced_termination'] = True
                    trajectory.metadata['forced_termination_reason'] = 'loop_detected'
                    break
                else:
                    print(f"Loop detected at turn {turn_idx + 1} ({loop_count}/{self.config.max_in_loop}), tolerating.")
            else:
                loop_count = 0
            # --- End loop detection ---
            
            # (f) Update current node
            current_node = target_node

        # Forced termination if not yet answered
        if not trajectory.is_terminated() and self.config.force_answer_on_max_turns:
            # Build trajectory context for forced answer on last visited node
            forced_trajectory_context = {
                'video_id': trajectory.video_id,
                'question': question,
                'choices': choices,
                'tree': root_node.to_tree_dict(),
                'history': [
                    {
                        'action': str(turn.action) if turn.action else None,
                        'observation': {
                            'node_id': turn.observation.node.node_id,
                            'start_sec': turn.observation.node.start_sec,
                            'end_sec': turn.observation.node.end_sec,
                            'description': turn.observation.text_description if hasattr(turn.observation, 'text_description') else ''
                        } if turn.observation else None
                    }
                    for turn in trajectory.turns
                ]
            }
            forced_decision = self._force_answer_decision(
                current_node, choices,
                question=question, video_path=video_path,
                trajectory_context=forced_trajectory_context,
            )
            reasoning = Reasoning(
                content=forced_decision.get('reasoning') or 'Max turns reached. Forced answer on last visited node.',
                context={'trajectory_context': forced_trajectory_context},
                raw_response='',
            )
            self._execute_answer(trajectory, current_node, forced_decision, reasoning)
            trajectory.metadata['forced_termination'] = True
            trajectory.metadata['forced_termination_reason'] = 'max_turns_reached'

        # Add metadata
        trajectory.metadata['num_turns'] = len(trajectory.turns)
        trajectory.metadata['engine'] = 'InferenceEngine'
        trajectory.metadata['frames_per_turn'] = frames_per_turn
        trajectory.metadata['total_frames'] = sum(frames_per_turn)

        # Compute evaluation metrics if GT available
        if ground_truth:
            self._compute_eval_metrics(trajectory, ground_truth)

        return trajectory

    def infer_direct(
        self,
        video_path: str,
        video_id: str,
        question: str,
        choices: List[str],
        video_duration: float,
        ground_truth: Optional[Dict[str, Any]] = None,
        video_native: bool = False,
    ) -> Trajectory:
        """
        Direct (single-turn) baseline: sample frames from the full video,
        split into child segments, and force the VLM to ANSWER immediately.

        Same visual input as the first turn of agentic inference, but only
        the ANSWER action is permitted.

        If `video_native` is True, the VLM is sent the entire video as a
        single attachment instead of interleaved sampled frames+timestamps.
        """
        # Build GT node for evaluation only
        gt_node = None
        if ground_truth:
            gt_timestamps = ground_truth.get('timestamps')
            if gt_timestamps:
                gt_start, gt_end = gt_timestamps
                gt_node = Node(
                    start_sec=gt_start, end_sec=gt_end,
                    frames=[], video_path=video_path, level=-1
                )

        trajectory = Trajectory(
            video_id=video_id,
            question=question,
            ground_truth=ground_truth or {},
            gt_node=gt_node
        )

        # Create root node covering entire video
        root_frames = self.frame_sampler(video_path, 0, video_duration)
        root_node = Node(
            start_sec=0, end_sec=video_duration,
            frames=root_frames, video_path=video_path,
            level=0, parent=None, node_id=0
        )
        root_node.visited = True

        initial_obs = Observation(
            frames=root_frames,
            text_description=f"Root segment [0.0s - {video_duration:.1f}s]",
            node=root_node
        )
        trajectory.set_initial_observation(initial_obs)

        # Call VLM to directly answer
        direct_fn = (self.vlm.direct_answer_lmm
                     if self.config.direct_prompt_style == 'lmm'
                     else self.vlm.direct_answer)
        try:
            result, raw_response = direct_fn(
                question=question,
                choices=choices,
                frames=root_frames,
                start_sec=0,
                end_sec=video_duration,
                video_path=video_path,
                short_side=self.config.short_side,
                skip_reasoning=self.config.skip_reasoning,
                video_native=video_native,
            )
        except Exception as e:
            print(f"Warning: direct_answer failed in direct mode: {e}. "
                  f"Forcing ANSWER.")
            result = {
                'reasoning': f'Forced answer due to error: {e}',
                'answer': 'A',
                'evidence_start': root_node.start_sec,
                'evidence_end': root_node.end_sec,
            }
            raw_response = ""
            trajectory.metadata['forced_termination'] = True
            trajectory.metadata['forced_termination_reason'] = 'decide_action_error'

        decision = {
            'answer': result.get('answer') or 'A',
            'evidence_start': result.get('evidence_start') or root_node.start_sec,
            'evidence_end': result.get('evidence_end') or root_node.end_sec,
        }

        reasoning = Reasoning(
            content=result.get('reasoning') or 'Direct baseline answer.',
            context={'compressed_history': ''},
            raw_response=raw_response,
        )
        self._execute_answer(trajectory, root_node, decision, reasoning)

        # Metadata
        trajectory.metadata['num_turns'] = len(trajectory.turns)
        trajectory.metadata['engine'] = 'InferenceEngine_direct'
        trajectory.metadata['frames_per_turn'] = [len(root_frames)]
        trajectory.metadata['total_frames'] = len(root_frames)

        if ground_truth:
            self._compute_eval_metrics(trajectory, ground_truth)

        return trajectory


    def infer_clue(
        self,
        video_path: str,
        video_id: str,
        question: str,
        choices: List[str],
        video_duration: float,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> Trajectory:
        """
        Oracle clue baseline: sample frames from the ground-truth clue interval
        and force the VLM to ANSWER immediately.

        Same as infer_direct but the input segment is the GT clue interval
        instead of the full video. This serves as an upper bound on search.
        """
        if not ground_truth or 'timestamps' not in ground_truth or not ground_truth['timestamps']:
            raise ValueError("infer_clue requires ground_truth with 'timestamps'")

        gt_start, gt_end = ground_truth['timestamps']
        if gt_start >= gt_end:
            raise ValueError(f"Invalid clue interval: [{gt_start}, {gt_end}]")

        # Build GT node for evaluation
        gt_node = Node(
            start_sec=gt_start, end_sec=gt_end,
            frames=[], video_path=video_path, level=-1
        )

        trajectory = Trajectory(
            video_id=video_id,
            question=question,
            ground_truth=ground_truth,
            gt_node=gt_node
        )

        # Create root node covering the GT clue interval
        root_frames = self.frame_sampler(video_path, gt_start, gt_end)
        root_node = Node(
            start_sec=gt_start, end_sec=gt_end,
            frames=root_frames, video_path=video_path,
            level=0, parent=None, node_id=0
        )
        root_node.visited = True

        initial_obs = Observation(
            frames=root_frames,
            text_description=f"Root segment [{gt_start:.1f}s - {gt_end:.1f}s]",
            node=root_node
        )
        trajectory.set_initial_observation(initial_obs)

        # Call VLM to directly answer
        direct_fn = (self.vlm.direct_answer_lmm
                     if self.config.direct_prompt_style == 'lmm'
                     else self.vlm.direct_answer)
        try:
            result, raw_response = direct_fn(
                question=question,
                choices=choices,
                frames=root_frames,
                start_sec=gt_start,
                end_sec=gt_end,
                video_path=video_path,
                short_side=self.config.short_side,
                skip_reasoning=self.config.skip_reasoning,
            )
        except Exception as e:
            print(f"Warning: direct_answer failed in clue mode: {e}. "
                  f"Forcing ANSWER.")
            result = {
                'reasoning': f'Forced answer due to error: {e}',
                'answer': 'A',
                'evidence_start': gt_start,
                'evidence_end': gt_end,
            }
            raw_response = ""
            trajectory.metadata['forced_termination'] = True
            trajectory.metadata['forced_termination_reason'] = 'decide_action_error'

        decision = {
            'answer': result.get('answer') or 'A',
            'evidence_start': result.get('evidence_start') or float(gt_start),
            'evidence_end': result.get('evidence_end') or float(gt_end),
        }

        reasoning = Reasoning(
            content=result.get('reasoning') or 'Clue baseline answer.',
            context={'compressed_history': ''},
            raw_response=raw_response,
        )
        self._execute_answer(trajectory, root_node, decision, reasoning)

        # Metadata
        trajectory.metadata['num_turns'] = len(trajectory.turns)
        trajectory.metadata['engine'] = 'InferenceEngine_clue'
        trajectory.metadata['clue_interval'] = [gt_start, gt_end]
        trajectory.metadata['frames_per_turn'] = [len(root_frames)]
        trajectory.metadata['total_frames'] = len(root_frames)

        if ground_truth:
            self._compute_eval_metrics(trajectory, ground_truth)

        return trajectory


    def infer_direct_nonuniform(
        self,
        video_path: str,
        video_id: str,
        question: str,
        choices: List[str],
        video_duration: float,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> Trajectory:
        """
        Non-uniform direct baseline: use scene-based segmentation to split
        the video into segments, sample 1 center frame per segment, then
        force the VLM to ANSWER immediately.

        Requires self.segment_fn to be set (e.g., CLIP-based scene segmenter).
        """
        if self.segment_fn is None:
            raise ValueError(
                "infer_direct_nonuniform requires a segment_fn "
                "(e.g., scene-based segmenter). Pass segment_fn to InferenceEngine."
            )

        # Build GT node for evaluation only
        gt_node = None
        if ground_truth:
            gt_timestamps = ground_truth.get('timestamps')
            if gt_timestamps:
                gt_start, gt_end = gt_timestamps
                gt_node = Node(
                    start_sec=gt_start, end_sec=gt_end,
                    frames=[], video_path=video_path, level=-1
                )

        trajectory = Trajectory(
            video_id=video_id,
            question=question,
            ground_truth=ground_truth or {},
            gt_node=gt_node
        )

        # Use scene segmentation to get non-uniform segments
        segments = self.segment_fn(video_path, 0, video_duration)

        # Sample 1 center frame per segment
        center_frames = [(s + e) / 2.0 for s, e in segments]

        # Create root node covering entire video with center frames
        root_node = Node(
            start_sec=0, end_sec=video_duration,
            frames=center_frames, video_path=video_path,
            level=0, parent=None, node_id=0
        )
        root_node.visited = True

        initial_obs = Observation(
            frames=center_frames,
            text_description=f"Root segment [0.0s - {video_duration:.1f}s] "
                             f"({len(segments)} non-uniform segments)",
            node=root_node
        )
        trajectory.set_initial_observation(initial_obs)

        # Call VLM to directly answer
        direct_fn = (self.vlm.direct_answer_lmm
                     if self.config.direct_prompt_style == 'lmm'
                     else self.vlm.direct_answer)
        try:
            result, raw_response = direct_fn(
                question=question,
                choices=choices,
                frames=center_frames,
                start_sec=0,
                end_sec=video_duration,
                video_path=video_path,
                short_side=self.config.short_side,
                skip_reasoning=self.config.skip_reasoning,
            )
        except Exception as e:
            print(f"Warning: direct_answer failed in direct_nonuniform mode: {e}. "
                  f"Forcing ANSWER.")
            result = {
                'reasoning': f'Forced answer due to error: {e}',
                'answer': 'A',
                'evidence_start': root_node.start_sec,
                'evidence_end': root_node.end_sec,
            }
            raw_response = ""
            trajectory.metadata['forced_termination'] = True
            trajectory.metadata['forced_termination_reason'] = 'decide_action_error'

        decision = {
            'answer': result.get('answer') or 'A',
            'evidence_start': result.get('evidence_start') or float(root_node.start_sec),
            'evidence_end': result.get('evidence_end') or float(root_node.end_sec),
        }

        reasoning = Reasoning(
            content=result.get('reasoning') or 'Direct non-uniform baseline answer.',
            context={'compressed_history': ''},
            raw_response=raw_response,
        )
        self._execute_answer(trajectory, root_node, decision, reasoning)

        # Metadata
        trajectory.metadata['num_turns'] = len(trajectory.turns)
        trajectory.metadata['engine'] = 'InferenceEngine_direct_nonuniform'
        trajectory.metadata['frames_per_turn'] = [len(center_frames)]
        trajectory.metadata['total_frames'] = len(center_frames)
        trajectory.metadata['num_segments'] = len(segments)
        trajectory.metadata['segments'] = [(s, e) for s, e in segments]

        if ground_truth:
            self._compute_eval_metrics(trajectory, ground_truth)

        return trajectory

    def infer_direct_keyframes(
        self,
        video_path: str,
        video_id: str,
        question: str,
        choices: List[str],
        video_duration: float,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> Trajectory:
        """
        Direct keyframe prediction baseline: sample frames from the full video
        and ask the VLM to predict specific keyframe timestamps (in seconds)
        plus the MCQ answer.

        Unlike infer_direct (which predicts an evidence interval), this method
        asks the model to output precise timestamps of key evidence frames.
        The predicted seconds can later be converted to frame IDs via fps.
        """
        # Build GT node for evaluation only
        gt_node = None
        if ground_truth:
            gt_timestamps = ground_truth.get('timestamps')
            if gt_timestamps:
                gt_start, gt_end = gt_timestamps
                gt_node = Node(
                    start_sec=gt_start, end_sec=gt_end,
                    frames=[], video_path=video_path, level=-1
                )

        trajectory = Trajectory(
            video_id=video_id,
            question=question,
            ground_truth=ground_truth or {},
            gt_node=gt_node
        )

        # Create root node covering entire video
        root_frames = self.frame_sampler(video_path, 0, video_duration)
        root_node = Node(
            start_sec=0, end_sec=video_duration,
            frames=root_frames, video_path=video_path,
            level=0, parent=None, node_id=0
        )
        root_node.visited = True

        initial_obs = Observation(
            frames=root_frames,
            text_description=f"Root segment [0.0s - {video_duration:.1f}s]",
            node=root_node
        )
        trajectory.set_initial_observation(initial_obs)

        # Call VLM predict_keyframes
        try:
            result, raw_response = self.vlm.predict_keyframes(
                question=question,
                choices=choices,
                frames=root_frames,
                start_sec=0,
                end_sec=video_duration,
                video_path=video_path,
                video_duration=video_duration,
                short_side=self.config.short_side,
                skip_reasoning=self.config.skip_reasoning,
            )
        except Exception as e:
            print(f"Warning: predict_keyframes failed: {e}. Using fallback.")
            result = {
                'reasoning': f'Fallback due to error: {e}',
                'answer': 'A',
                'keyframe_timestamps': [video_duration / 2.0],
            }
            raw_response = ""

        # Extract results
        keyframe_ts = result.get('keyframe_timestamps') or [video_duration / 2.0]

        # Build decision dict compatible with _execute_answer
        decision = {
            'answer': result.get('answer') or 'A',
            'evidence_start': min(keyframe_ts),
            'evidence_end': max(keyframe_ts),
        }

        reasoning = Reasoning(
            content=result.get('reasoning') or 'Direct keyframe prediction.',
            context={'compressed_history': ''},
            raw_response=raw_response,
        )
        self._execute_answer(trajectory, root_node, decision, reasoning)

        # Store keyframe timestamps in metadata for direct access
        trajectory.metadata['keyframe_timestamps'] = keyframe_ts
        trajectory.metadata['num_turns'] = len(trajectory.turns)
        trajectory.metadata['engine'] = 'InferenceEngine_direct_keyframes'
        trajectory.metadata['frames_per_turn'] = [len(root_frames)]
        trajectory.metadata['total_frames'] = len(root_frames)

        if ground_truth:
            self._compute_eval_metrics(trajectory, ground_truth)

        return trajectory

    # ------------------------------------------------------------------
    # Loop detection
    # ------------------------------------------------------------------

    def _detect_loop(
        self,
        action_type: str,
        target_node_visited_before: bool,
        action_history: Optional[list] = None,
    ) -> bool:
        """Detect loop based on configured mode.

        Modes:
            'none': no loop detection
            'visited': navigating to an already-visited node (except ZOOM_OUT).
            'window': detect repeating action cycles of length loop_detection_window.
                      If loop_detection_window == -1, dynamically try all window sizes.
        """
        if action_type == 'ANSWER':
            return False

        mode = self.config.loop_detection_mode

        if mode == 'none':
            return False

        if mode == 'visited':
            if action_type == 'ZOOM_OUT':
                return False
            return target_node_visited_before

        if mode == 'window':
            if action_history is None:
                return False
            w = self.config.loop_detection_window
            if w == -1:
                n = len(action_history)
                for cand in range(1, n // 2 + 1):
                    if action_history[-cand:] == action_history[-2 * cand:-cand]:
                        return True
                return False
            if w <= 0 or len(action_history) < 2 * w:
                return False
            return action_history[-w:] == action_history[-2 * w:-w]

        return False


    # ------------------------------------------------------------------
    # Segment splitting
    # ------------------------------------------------------------------

    def _split_into_children(
        self, parent: Node, video_path: str
    ) -> List[Node]:
        """Split parent into non-overlapping children.

        Uses tree cache if available, then segment_fn for content-aware
        splitting, otherwise falls back to uniform splitting.
        """
        if parent.children:
            return parent.children

        if self.config.min_segment_duration > 0 and parent.duration() < self.config.min_segment_duration:
            return []

        if self.config.max_depth != -1 and parent.level >= self.config.max_depth:
            return []

        # Try tree cache first
        if self.tree_cache is not None:
            cached_children = self.tree_cache.get_children(
                self._current_video_id, parent.start_sec, parent.end_sec
            )
            if cached_children is not None:
                children = []
                for i, cc in enumerate(cached_children):
                    frames = self.frame_sampler(video_path, cc["start_sec"], cc["end_sec"])
                    child = Node(
                        start_sec=cc["start_sec"], end_sec=cc["end_sec"],
                        frames=frames, video_path=video_path,
                        level=parent.level + 1, parent=parent,
                        node_id=i
                    )
                    children.append(child)
                parent.children = children
                return children

        if self.segment_fn is not None:
            child_segments = self.segment_fn(video_path, parent.start_sec, parent.end_sec)
        else:
            # Uniform splitting
            num = self.config.num_children
            child_duration = (parent.end_sec - parent.start_sec) / num
            child_segments = []
            for i in range(num):
                start = parent.start_sec + i * child_duration
                end = parent.end_sec if i == num - 1 else parent.start_sec + (i + 1) * child_duration
                child_segments.append((start, end))

        children = []
        for i, (start, end) in enumerate(child_segments):
            frames = self.frame_sampler(video_path, start, end)
            child = Node(
                start_sec=start, end_sec=end,
                frames=frames, video_path=video_path,
                level=parent.level + 1, parent=parent,
                node_id=i
            )
            children.append(child)
        if len(children) > 0:
            parent.children = children
        return children

    # ------------------------------------------------------------------
    # History compression
    # ------------------------------------------------------------------

    def _compress_history(self, trajectory: Trajectory) -> str:
        """
        DEPRECATED: Use trajectory_context dict instead.

        Compress trajectory history: keep only actions and time ranges,
        removing reasoning content and visual frames.
        """
        import warnings
        warnings.warn(
            "_compress_history is deprecated. Use trajectory_context dict instead.",
            DeprecationWarning,
            stacklevel=2
        )
        lines = []
        for i, turn in enumerate(trajectory.turns):
            if turn.action is None and turn.observation is not None:
                # Initial observation
                obs = turn.observation
                lines.append(
                    f"Turn {i}: Observed [{obs.node.start_sec:.1f}s - "
                    f"{obs.node.end_sec:.1f}s]"
                )
            elif turn.action is not None:
                action = turn.action
                if action.action_type == ActionType.ZOOM_IN:
                    lines.append(
                        f"Turn {i}: ZOOM_IN Segment {action.segment_id} "
                        f"[{action.start_sec:.1f}s - {action.end_sec:.1f}s]"
                    )
                elif action.action_type == ActionType.ZOOM_OUT:
                    lines.append(f"Turn {i}: ZOOM_OUT")
                elif action.action_type == ActionType.SHIFT:
                    lines.append(
                        f"Turn {i}: SHIFT Segment {action.segment_id} "
                        f"[{action.start_sec:.1f}s - {action.end_sec:.1f}s]"
                    )
                elif action.action_type == ActionType.ANSWER:
                    lines.append(f"Turn {i}: ANSWER {action.answer}")
        return "\n".join(lines) if lines else ""

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_zoom_in(
        self,
        trajectory: Trajectory,
        current_node: Node,
        children: List[Node],
        decision: Dict[str, Any],
        reasoning: Reasoning
    ) -> Node:
        """Execute ZOOM_IN action. Returns the new current node."""
        # Find matching child based on decision type
        target = None

        if decision.get('segment_id') is not None:
            # Structured-tree mode: use segment_id
            seg_id = decision['segment_id']
            for child in children:
                if child.node_id == seg_id:
                    target = child
                    break

        elif decision.get('start_sec') is not None and decision.get('end_sec') is not None:
            # Free-form mode: create a new child node with specified timestamps
            zoom_start = decision['start_sec']
            zoom_end = decision['end_sec']

            # Clamp to current_node boundaries
            zoom_start = max(current_node.start_sec, min(zoom_start, current_node.end_sec))
            zoom_end = max(current_node.start_sec, min(zoom_end, current_node.end_sec))

            # Sample frames for the new child
            frames = self.frame_sampler(current_node.video_path, zoom_start, zoom_end)

            # Create new child node with next available node_id
            next_node_id = len(children)
            target = Node(
                start_sec=zoom_start,
                end_sec=zoom_end,
                frames=frames,
                video_path=current_node.video_path,
                level=current_node.level + 1,
                parent=current_node,
                node_id=next_node_id
            )
            # Add to parent's children list
            current_node.children.append(target)

        # Fallback to first child if no match found
        if target is None and children:
            target = children[0]

        if target is None:
            raise ValueError("No children available for ZOOM_IN action")

        action = Action.zoom_in(target.start_sec, target.end_sec, target.node_id)
        observation = Observation(
            frames=target.frames,
            text_description=(
                f"Segment {target.node_id} "
                f"[{target.start_sec:.1f}s - {target.end_sec:.1f}s]"
            ),
            node=target
        )
        trajectory.add_turn(reasoning, action, observation)
        return target

    def _execute_zoom_out(
        self,
        trajectory: Trajectory,
        current_node: Node,
        root_node: Node,
        reasoning: Reasoning
    ) -> Node:
        """Execute ZOOM_OUT action. Returns the new current node (parent)."""
        if current_node.parent is not None:
            parent = current_node.parent
        else:
            # Already at root -- cannot zoom out further
            print("Warning: ZOOM_OUT from root node. Staying at root.")
            parent = root_node

        action = Action.zoom_out()
        observation = Observation(
            frames=parent.frames,
            text_description=(
                f"Segment {parent.node_id} "
                f"[{parent.start_sec:.1f}s - {parent.end_sec:.1f}s]"
            ),
            node=parent
        )
        trajectory.add_turn(reasoning, action, observation)
        return parent

    def _execute_shift(
        self,
        trajectory: Trajectory,
        current_node: Node,
        decision: Dict[str, Any],
        reasoning: Reasoning
    ) -> Node:
        """Execute SHIFT action. Returns the new current node (sibling)."""
        parent = current_node.parent
        if parent is None:
            print("Warning: SHIFT from root node. Staying at root.")
            action = Action.zoom_out()
            trajectory.add_turn(reasoning, action, None)
            return current_node

        # Find sibling by segment_id
        siblings = parent.children
        target = None
        seg_id = decision.get('segment_id')
        if seg_id is not None:
            for sibling in siblings:
                if sibling.node_id == seg_id:
                    target = sibling
                    break

        # Fallback: pick first unvisited sibling
        if target is None:
            unvisited = [s for s in siblings if not s.visited and s is not current_node]
            target = unvisited[0] if unvisited else siblings[0]

        action = Action.shift(target.start_sec, target.end_sec, target.node_id)
        observation = Observation(
            frames=target.frames,
            text_description=(
                f"Segment {target.node_id} "
                f"[{target.start_sec:.1f}s - {target.end_sec:.1f}s]"
            ),
            node=target
        )
        trajectory.add_turn(reasoning, action, observation)
        return target

    def _execute_answer(
        self,
        trajectory: Trajectory,
        current_node: Node,
        decision: Dict[str, Any],
        reasoning: Reasoning
    ) -> None:
        """Execute ANSWER action. Terminates the trajectory."""
        answer_letter = decision.get('answer')
        if answer_letter is None:
            answer_letter = 'A'
        ev_start = decision.get('evidence_start')
        if ev_start is None:
            ev_start = float(current_node.start_sec)
        ev_end = decision.get('evidence_end')
        if ev_end is None:
            ev_end = float(current_node.end_sec)

        evidence = {
            'timestamps': (ev_start, ev_end),
            'node_interval': (current_node.start_sec, current_node.end_sec),
        }
        action = Action.answer(answer_letter, evidence)
        observation = Observation(
            frames=current_node.frames,
            text_description=(
                f"Segment {current_node.node_id} "
                f"[{current_node.start_sec:.1f}s - {current_node.end_sec:.1f}s]"
            ),
            node=current_node
        )
        trajectory.add_turn(reasoning, action, observation)
        return current_node

    # ------------------------------------------------------------------
    # Fallback / forced answer
    # ------------------------------------------------------------------

    def _force_answer_decision(
        self,
        current_node: Node,
        choices: List[str],
        question: str = "",
        video_path: str = "",
        trajectory_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a forced ANSWER decision, trying the VLM first."""
        # Try VLM call with forced ANSWER action
        try:
            decide_fn = (self.vlm.decide_action_mcq
                         if self.config.decide_action_type == 'multiple_choice'
                         else self.vlm.decide_action)
            decision, _ = decide_fn(
                question=question,
                choices=choices,
                child_segments=[],
                trajectory_context=trajectory_context or {},
                current_start_sec=current_node.start_sec,
                current_end_sec=current_node.end_sec,
                current_frames=current_node.frames,
                video_path=video_path,
                short_side=self.config.short_side,
                allowed_action_list=["ANSWER"],
                navigation_context=None,
                generate_captions=False,
                skip_reasoning=self.config.skip_reasoning,
                pregenerated_caption_path=self.config.pregenerated_caption_path,
                use_frame_captions=self.config.use_frame_captions,
                enforce_valid_segments=self.config.enforce_valid_segments,
            )
            decision['action_type'] = 'ANSWER'
            return decision
        except Exception as e:
            print(f"Warning: VLM call in _force_answer_decision failed: {e}. "
                  f"Using synthetic fallback.")
            return {
                'reasoning': 'Forced answer due to max turns or error.',
                'action_type': 'ANSWER',
                'segment_id': None,
                'answer': 'A',
                'evidence_start': current_node.start_sec,
                'evidence_end': current_node.end_sec,
            }

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------

    def _compute_eval_metrics(
        self,
        trajectory: Trajectory,
        ground_truth: Dict[str, Any]
    ) -> None:
        """Compute evaluation metrics and store in trajectory.metadata."""
        # Answer accuracy
        final_answer = trajectory.get_final_answer()
        gt_answer = ground_truth.get('right_answer')

        trajectory.metadata['predicted_answer'] = final_answer
        trajectory.metadata['gt_answer'] = gt_answer
        trajectory.metadata['answer_correct'] = (
            (final_answer == gt_answer)
            if (final_answer is not None and gt_answer is not None)
            else None
        )

        # Evidence interval IoU
        gt_timestamps = ground_truth.get('timestamps')
        gt_all_timestamps = ground_truth.get('all_timestamps')
        if gt_timestamps and trajectory.is_terminated():
            last_action = trajectory.turns[-1].action
            if last_action and last_action.evidence:
                pred_timestamps = last_action.evidence.get('timestamps')
                if pred_timestamps and None not in pred_timestamps:
                    if gt_all_timestamps and len(gt_all_timestamps) > 1:
                        iou = compute_multi_interval_iou(pred_timestamps, gt_all_timestamps)
                        coverage = compute_multi_interval_coverage(pred_timestamps, gt_all_timestamps)
                    else:
                        iou = compute_interval_iou(pred_timestamps, gt_timestamps)
                        coverage = compute_interval_coverage(pred_timestamps, gt_timestamps)

                    trajectory.metadata['evidence_iou'] = iou
                    trajectory.metadata['evidence_coverage'] = coverage

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_video_duration(video_path: str) -> float:
        """Infer video duration from video file using cv2."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps > 0:
            return frame_count / fps
        raise ValueError(f"Invalid FPS ({fps}) for video: {video_path}")
