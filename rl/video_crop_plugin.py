"""
RL plugin for tree-based hierarchical video agent training.

Implements:
- TreeSearchScheduler: Multi-turn scheduler with context compression using
  tree-based navigation (ZOOM_IN, ZOOM_OUT, SHIFT, ANSWER).
- TreeFormatReward, TreeAccuracyReward, TreeIOUReward: Reward functions.

Usage:
  swift rollout --external_plugins rl/video_crop_plugin.py \
      --multi_turn_scheduler tree_search_scheduler ...
  swift rlhf --external_plugins rl/video_crop_plugin.py \
      --reward_funcs tree_acc_reward tree_iou_reward tree_format_reward ...
"""

import json
import os
import re
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

try:
    from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
    from swift.plugin import ORM, orms
except:
    from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns
    from swift.rewards import ORM, orms

from swift.utils import get_logger


logger = get_logger()


# Import local utilities (relative to rl/ directory via sys.path or PYTHONPATH)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.tree_utils import (
    TreeNode, TreeState, load_tree_from_cache, create_uniform_tree,
    get_frame_paths, populate_gt_coverage, compute_tree_distance,
    find_best_node, bfs_distance,
)
from utils.prompt_utils import (
    build_system_prompt, build_user_content, build_memory_text,
    build_navigation_context, parse_action_response,
    parse_action_response_sft,
    VALID_ACTIONS,
)


# =========================================================================== #
# Scheduler
# =========================================================================== #
class TreeSearchScheduler(MultiTurnScheduler):
    """
    Multi-turn scheduler for tree-based hierarchical video search.

    Overrides ``run()`` to implement context compression: each turn is
    rebuilt from scratch (system + compressed memory + current frames),
    and returned as a separate RolloutOutput for per-turn training.

    Per-sample paths (from dataset JSONL):
      tree_cache_dir: Path to pre-built tree cache JSONs.
      frames_base_dir: Path to pre-extracted frames (1fps).

    Config via environment variables:
      NUM_FRAMES_PER_SEGMENT: Frames to sample for the current segment
        (default 64). Uses 1fps if the segment is shorter than this.
      NUM_FALLBACK_CHILDREN: Children count for uniform fallback (default 4).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_frames_per_segment = int(
            os.environ.get('NUM_FRAMES_PER_SEGMENT', '64')
        )
        self.num_fallback_children = int(
            os.environ.get('NUM_FALLBACK_CHILDREN', '4')
        )
        self.max_turns = int(
            os.environ.get('MAX_TURNS', '10')
        )
        # "constrained" (default): only structurally valid actions allowed
        # "free": all actions (ZOOM_IN, ZOOM_OUT, SHIFT, ANSWER) every turn
        self.action_mode = os.environ.get('ACTION_MODE', 'free')
        # self.action_mode = os.environ.get('ACTION_MODE', 'constrained')

        # "tagged": <think>...</think><action>{JSON}</action>
        # "sft": <think>...</think>\n```json\n{JSON}\n```
        self.output_format = os.environ.get('OUTPUT_FORMAT', 'sft')

        # Global action-space restriction (variant ablations). Comma-separated
        # subset of VALID_ACTIONS; e.g. "ZOOM_IN,ANSWER" trains a variant that
        # can only descend or answer (no ZOOM_OUT/SHIFT). Filters every turn's
        # allowed set, which cascades to the system prompt, JSON spec, rules,
        # and navigation context. Default preserves the original four actions.
        raw = os.environ.get('ALLOWED_ACTIONS', 'ZOOM_IN,ZOOM_OUT,SHIFT,ANSWER')
        self.allowed_actions = {a.strip().upper() for a in raw.split(',') if a.strip()}
        invalid = self.allowed_actions - VALID_ACTIONS
        if invalid:
            raise ValueError(f'ALLOWED_ACTIONS contains invalid actions: {invalid}')
        if 'ANSWER' not in self.allowed_actions:
            raise ValueError('ALLOWED_ACTIONS must include ANSWER')

    async def run(self, infer_request, request_config, **kwargs):
        """
        Execute multi-turn tree search with context compression.

        Each turn is returned as a separate RolloutOutput. The model sees
        only the current segment's frames + compressed text memory of past
        turns, not the full conversation history.

        Returns:
            List[RolloutOutput], one per turn.
        """
        from swift.llm.infer.protocol import RolloutOutput

        # --- Extract metadata from infer_request ---
        data = infer_request.data_dict
        video_id = data.get('video_id', '')
        question = data.get('question', '')
        choices = data.get('choices', [])
        duration = float(data.get('duration', 0.0))

        # Per-sample paths from dataset JSONL
        tree_cache_dir = data.get('tree_cache_dir', '')
        frames_base_dir = data.get('frames_base_dir', '')

        # --- Load or create tree ---
        root = load_tree_from_cache(tree_cache_dir, video_id)
        if root is None:
            logger.warning(
                f'No tree cache for {video_id}, using uniform fallback'
            )
            root = create_uniform_tree(
                duration, self.num_fallback_children, depth=4
            )

        tree_state = TreeState(root, frames_base_dir, video_id)
        history: List[Dict] = []
        node_visit_count: Dict[Tuple[float, float], int] = {}
        rollout_outputs: List[RolloutOutput] = []

        max_turns = self.max_turns
        termination_reason = 'answer'  # default; updated at each break point

        for turn_idx in range(1, max_turns + 1):

            current_node = tree_state.current_node
            children = tree_state.get_children()
            siblings = tree_state.get_siblings()

            # --- Loop detection: track node visits ---
            node_key = (current_node.start_sec, current_node.end_sec)
            # ZOOM_OUT is a backtrack — returning to a parent doesn't count
            prev_action = history[-1]['action'] if history else None
            if prev_action is None or not prev_action.startswith('ZOOM_OUT'):
                node_visit_count[node_key] = node_visit_count.get(node_key, 0) + 1

            # --- Determine allowed actions ---
            allowed = self._get_allowed_actions(
                current_node, children, siblings, turn_idx, max_turns
            )

            # Force ANSWER on loop detection or final turn
            force_answer = node_visit_count.get(node_key, 0) >= 3 or turn_idx >= max_turns
            if force_answer:
                allowed = ['ANSWER']
                if node_visit_count.get(node_key, 0) >= 3:
                    termination_reason = 'forced_answer_loop'
                else:
                    termination_reason = 'forced_answer_max_turns'
            if node_visit_count.get(node_key, 0) >= 3:
                logger.info(
                    f'[{video_id}] Turn {turn_idx}: Loop detected at node '
                    f'({node_key[0]:.1f}s, {node_key[1]:.1f}s), forcing ANSWER.'
                )

            # --- Compute valid segment IDs for constraints ---
            if self.action_mode == 'free':
                valid_zoom_in = [c for c in children] if children else []
                valid_shift = [s for s in siblings if s is not current_node] if siblings else []
            else:
                valid_zoom_in = [
                    c for c in children if not c.visited
                ] if 'ZOOM_IN' in allowed else []
                valid_shift = [
                    s for s in siblings if not s.visited and s is not current_node
                ] if 'SHIFT' in allowed else []
            valid_zoom_in_ids = [n.node_id for n in valid_zoom_in] or None
            valid_shift_ids = [n.node_id for n in valid_shift] or None

            # --- Sample frames for the current segment, then bin by child ---
            frame_paths = tree_state.get_node_frame_paths(
                current_node, max_frames=self.num_frames_per_segment
            )
            child_segments = []
            for child in children:
                child_segments.append({
                    'segment_id': child.node_id,
                    'start_sec': child.start_sec,
                    'end_sec': child.end_sec,
                    'frame_paths': [],
                })
            for p in frame_paths:
                fname = p.rsplit('/', 1)[-1].replace('.jpg', '')
                ts = float(fname)
                for seg in child_segments:
                    if seg['start_sec'] <= ts <= seg['end_sec']:
                        seg['frame_paths'].append(p)
                        break

            # --- Mark current node as visited ---
            current_node.visited = True

            # --- Build prompt components ---
            tree_dict = tree_state.to_tree_dict()
            current_key = (current_node.start_sec, current_node.end_sec)
            memory_text = build_memory_text(tree_dict, history, current_key)

            if force_answer:
                nav_text = ''
            else:
                nav_text = build_navigation_context(
                    current_node, valid_zoom_in, valid_shift,
                )

            has_children = bool(children)
            system_prompt = build_system_prompt(
                allowed, has_children=has_children,
                output_format=self.output_format,
            )
            user_content, image_paths = build_user_content(
                current_node.start_sec, current_node.end_sec,
                child_segments, frame_paths,
                memory_text, nav_text, question, choices, allowed,
                valid_zoom_in_ids, valid_shift_ids,
                output_format=self.output_format,
            )

            # --- Rebuild infer_request from scratch (context compression) ---
            infer_request.messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_content},
            ]
            infer_request.images = image_paths
            infer_request.videos = []

            infer_request_print = deepcopy(infer_request)

            # --- Run inference ---
            try:
                response = await self.infer_engine.infer_async(
                    infer_request, request_config, **kwargs
                )
            except Exception as e:
                logger.warning(
                    f'[{video_id}] Turn {turn_idx}: infer_async failed '
                    f'({type(e).__name__}: {e}); ending rollout.'
                )
                termination_reason = 'infer_failure'
                break
            response_choice = response.choices[0]
            completion = response_choice.message.content

            # print(f'turn_idx: {turn_idx}')
            # print(f'infer_request: {infer_request_print}')
            # print(f'completion: {completion}')
            # print()

            # --- Build messages snapshot for this turn ---
            turn_messages = list(infer_request.messages) + [
                {'role': 'assistant', 'content': completion},
            ]

            response_token_ids = response_choice.token_ids
            response_loss_mask = [1] * len(response_token_ids)

            if not response_token_ids:
                logger.warning(
                    f'[{video_id}] Turn {turn_idx}: empty response_token_ids '
                    f'(finish_reason={response_choice.finish_reason!r}, '
                    f'completion={completion!r})'
                )

            turn_output = RolloutOutput(
                response=response,
                messages=turn_messages,
                response_token_ids=[response_token_ids],
                response_loss_mask=[response_loss_mask],
                rollout_infos={
                    'num_turns': turn_idx,
                    'images': image_paths,
                    'node_start': current_node.start_sec,
                    'node_end': current_node.end_sec,
                },
            )
            rollout_outputs.append(turn_output)

            # --- Parse action ---
            if self.output_format == 'sft':
                parsed = parse_action_response_sft(completion)
            else:
                parsed = parse_action_response(completion)

            if parsed is None:
                logger.warning(
                    f'[{video_id}] Turn {turn_idx}: Failed to parse response'
                )
                termination_reason = 'parse_failure'
                break

            action_type = parsed['action']
            if action_type not in VALID_ACTIONS:
                logger.warning(
                    f'[{video_id}] Turn {turn_idx}: '
                    f'Invalid action {action_type}'
                )
                termination_reason = 'invalid_action'
                break
            if action_type not in allowed:
                logger.warning(
                    f'[{video_id}] Turn {turn_idx}: '
                    f'Action {action_type} not in allowed={allowed}'
                )
                termination_reason = 'disallowed_action'
                break

            # --- Update history for memory ---
            action_str = self._format_action_string(
                action_type, parsed, children
            )
            history.append({
                'action': action_str,
                'reasoning': parsed.get('reasoning', ''),
                'observation': {
                    'start_sec': current_node.start_sec,
                    'end_sec': current_node.end_sec,
                },
            })

            # --- Execute action ---
            if action_type == 'ANSWER':
                if not force_answer:
                    termination_reason = 'answer'
                break

            if action_type == 'ZOOM_IN':
                sid = parsed.get('segment_id')
                if (
                    isinstance(sid, int) and not isinstance(sid, bool)
                    and 0 <= sid < len(children)
                ):
                    tree_state.zoom_in(sid)
                else:
                    logger.warning(
                        f'[{video_id}] Turn {turn_idx}: '
                        f'Invalid ZOOM_IN segment_id={sid!r}'
                    )
                    termination_reason = 'invalid_zoom_in'
                    break

            elif action_type == 'ZOOM_OUT':
                result = tree_state.zoom_out()
                if result is None:
                    logger.warning(
                        f'[{video_id}] Turn {turn_idx}: '
                        'Cannot ZOOM_OUT from root'
                    )
                    termination_reason = 'invalid_zoom_out'
                    break

            elif action_type == 'SHIFT':
                sid = parsed.get('segment_id')
                if (
                    isinstance(sid, int) and not isinstance(sid, bool)
                    and current_node.parent is not None
                ):
                    try:
                        tree_state.shift(sid)
                    except (IndexError, ValueError) as e:
                        logger.warning(
                            f'[{video_id}] Turn {turn_idx}: '
                            f'Invalid SHIFT segment_id={sid!r}: {e}'
                        )
                        termination_reason = 'invalid_shift'
                        break
                else:
                    logger.warning(
                        f'[{video_id}] Turn {turn_idx}: '
                        f'Invalid SHIFT: sid={sid!r}, has_parent={current_node.parent is not None}'
                    )
                    termination_reason = 'invalid_shift'
                    break

        # Compute tree distance from the final node to nearest GT-covering node
        gt_intervals = []
        for ts in data.get('timestamp', []):
            if len(ts) >= 2:
                gt_intervals.append((float(ts[0]), float(ts[1])))

        if gt_intervals:
            populate_gt_coverage(root, gt_intervals)
            tree_distance = compute_tree_distance(tree_state.current_node)
            best_node, best_node_score = find_best_node(root, gt_intervals)
            if best_node is not None:
                gt_distance_to_best = bfs_distance(tree_state.current_node, best_node)
            else:
                gt_distance_to_best = None
        else:
            tree_distance = None
            best_node_score = None
            gt_distance_to_best = None

        # Store termination reason, tree distance, and tree structure in the last output
        if rollout_outputs:
            rollout_outputs[-1].rollout_infos['termination_reason'] = termination_reason
            rollout_outputs[-1].rollout_infos['tree_distance'] = tree_distance
            rollout_outputs[-1].rollout_infos['gt_distance_to_best'] = gt_distance_to_best
            rollout_outputs[-1].rollout_infos['best_node_score'] = best_node_score
            rollout_outputs[-1].rollout_infos['tree_dict'] = tree_state.to_tree_dict()

        # Ensure at least one output
        if not rollout_outputs:
            logger.error(f'[{video_id}] No rollout outputs produced')
            try:
                response = await self.infer_engine.infer_async(
                    infer_request, request_config, **kwargs
                )
                fallback_choice = response.choices[0]
                fallback_token_ids = fallback_choice.token_ids or []
                fallback_completion = fallback_choice.message.content or ''
                if not fallback_token_ids:
                    logger.warning(
                        f'[{video_id}] Turn fallback: empty response_token_ids '
                        f'(finish_reason={fallback_choice.finish_reason!r}, '
                        f'completion={fallback_completion!r})'
                    )
            except Exception as e:
                logger.error(
                    f'[{video_id}] Fallback infer_async also failed '
                    f'({type(e).__name__}: {e}); synthesizing empty output.'
                )
                from swift.llm.infer.protocol import (
                    ChatCompletionResponse, ChatCompletionResponseChoice,
                    ChatMessage, UsageInfo,
                )
                response = ChatCompletionResponse(
                    model='',
                    choices=[ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role='assistant', content=''),
                        finish_reason='stop',
                        token_ids=[],
                    )],
                    usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                )
                fallback_token_ids = []
                fallback_completion = ''
                termination_reason = 'infer_failure'
            rollout_outputs.append(RolloutOutput(
                response=response,
                messages=infer_request.messages + [
                    {'role': 'assistant', 'content': fallback_completion}
                ],
                response_token_ids=[fallback_token_ids],
                response_loss_mask=[[1] * len(fallback_token_ids)],
                rollout_infos={
                    'num_turns': 1,
                    'termination_reason': termination_reason,
                },
            ))

        # print(f'total turns: {turn_idx}')

        return rollout_outputs

    def _get_allowed_actions(
        self,
        current_node: TreeNode,
        children: List[TreeNode],
        siblings: List[TreeNode],
        turn_idx: int,
        max_turns: int,
    ) -> List[str]:
        """Determine allowed actions based on tree position.

        Modes (controlled by self.action_mode / ACTION_MODE env var):
          - "constrained": only unvisited children/siblings qualify.
          - "free": any existing children/siblings qualify.
        Structural restrictions apply to both modes:
          - Root node: SHIFT and ZOOM_OUT are never allowed.
          - No qualifying children: ZOOM_IN not allowed.
          - No qualifying siblings: SHIFT not allowed.
        Last turn always forces ANSWER only.
        """
        if turn_idx >= max_turns:
            return ['ANSWER']

        if self.action_mode == 'free':
            can_zoom_in = bool(children)
            can_shift = bool(siblings)
        else:
            can_zoom_in = any(not c.visited for c in children)
            can_shift = any(not s.visited for s in siblings)

        allowed = ['ANSWER']
        if can_zoom_in:
            allowed.append('ZOOM_IN')
        if current_node.parent is not None:
            allowed.append('ZOOM_OUT')
            if can_shift:
                allowed.append('SHIFT')

        # Apply the global variant restriction (e.g., zoom-in/answer-only).
        allowed = [a for a in allowed if a in self.allowed_actions]
        return allowed

    @staticmethod
    def _format_action_string(
        action_type: str,
        parsed: Dict,
        children: List[TreeNode],
    ) -> str:
        """Format an action for the interaction history."""
        if action_type == 'ZOOM_IN':
            sid = parsed.get('segment_id')
            if (
                isinstance(sid, int) and not isinstance(sid, bool)
                and 0 <= sid < len(children)
            ):
                c = children[sid]
                return (
                    f'ZOOM_IN(segment={sid}, '
                    f'[{c.start_sec:.1f}s, {c.end_sec:.1f}s])'
                )
            return f'ZOOM_IN(segment={sid})'
        elif action_type == 'ZOOM_OUT':
            return 'ZOOM_OUT()'
        elif action_type == 'SHIFT':
            sid = parsed.get('segment_id')
            return f'SHIFT(segment={sid})'
        elif action_type == 'ANSWER':
            ans = parsed.get('answer', '?')
            ev_s = parsed.get('evidence_start', '?')
            ev_e = parsed.get('evidence_end', '?')
            return f'ANSWER({ans}, [{ev_s}, {ev_e}])'
        return action_type


multi_turns['tree_search_scheduler'] = TreeSearchScheduler


# =========================================================================== #
# Reward functions
# =========================================================================== #

def _extract_final_answer(final_turn: Dict) -> Optional[Dict]:
    """
    Parse the action from a turn's assistant message.

    The turn's ``messages`` list is always ``[system, user, assistant]``
    (see ``TreeSearchScheduler.run``), so the assistant response is the
    last entry. Returns the parsed action dict (any action, not
    necessarily ANSWER) or ``None`` if parsing fails.
    """
    messages = final_turn.get('messages', [])
    if not messages:
        return None
    content = messages[-1].get('content', '')
    if os.environ.get('OUTPUT_FORMAT', 'sft') == 'sft':
        return parse_action_response_sft(content)
    return parse_action_response(content)


def _get_final_turn(global_trajectories: Dict, tra_id: str) -> Optional[Dict]:
    """
    Return the terminal turn entry for a trajectory, or None if absent.

    Swift's ``_get_trajectory_inputs`` builds the per-request_id list by
    appending in ``gather_object`` order (rank-major), so when a trajectory's
    turns are distributed across ranks the list is NOT ordered by turn index.
    Select the terminal turn explicitly via ``rollout_infos['num_turns']``
    rather than relying on list position.
    """
    turns = global_trajectories.get(tra_id, [])
    if not turns:
        return None
    return max(turns, key=lambda e: e.get('rollout_infos', {}).get('num_turns', 0))


def _first_letter(s: str) -> str:
    """Extract the first alphabetic character from a string."""
    if not s:
        return ''
    for ch in s.strip():
        if ch.isalpha():
            return ch.upper()
    return ''


def _calculate_iou(
    pred: Tuple[float, float],
    true: Tuple[float, float],
) -> float:
    """Calculate IoU of two temporal intervals."""
    pred_start, pred_end = pred
    true_start, true_end = true

    overlap_start = max(pred_start, true_start)
    overlap_end = min(pred_end, true_end)
    overlap = max(0.0, overlap_end - overlap_start)

    union = (pred_end - pred_start) + (true_end - true_start) - overlap
    return overlap / union if union > 0 else 0.0


class TreeFormatReward(ORM):
    """
    Reward for checking response format correctness.

    Only the terminal turn's assistant message is validated, which is
    sufficient: ``TreeSearchScheduler`` breaks the rollout on any parse
    or action-validation failure, so a malformed intermediate turn would
    itself become the terminal turn (and fail this check). Surviving
    intermediate turns are provably valid — the scheduler's checks on
    ``segment_id`` are stricter than this reward's. The terminal turn
    also carries the ANSWER-only fields (answer letter, evidence span)
    that the scheduler never validates.
    """

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        trajectory_ids: List[str] = kwargs.get('request_id', [])
        global_trajectories: Dict = kwargs.get('trajectory_inputs', {})

        rewards = []
        for tra_id in trajectory_ids:
            final_turn = _get_final_turn(global_trajectories, tra_id)
            if final_turn is None:
                rewards.append(0.0)
                continue
            messages = final_turn.get('messages', [])

            reward = 1.0
            last_action = ''

            for msg in messages:
                if msg.get('role') != 'assistant':
                    continue
                content = msg.get('content', '')

                output_format = os.environ.get('OUTPUT_FORMAT', 'sft')
                if output_format == 'sft':
                    parsed = parse_action_response_sft(content)
                else:
                    parsed = parse_action_response(content)

                if parsed is None:
                    reward = 0.0
                    break

                action = parsed['action']
                if action not in VALID_ACTIONS:
                    reward = 0.0
                    break

                # Validate required fields per action
                if action in ('ZOOM_IN', 'SHIFT'):
                    if parsed.get('segment_id') is None:
                        reward = 0.0
                        break
                    try:
                        int(parsed['segment_id'])
                    except (TypeError, ValueError):
                        reward = 0.0
                        break

                if action == 'ANSWER':
                    if not parsed.get('answer'):
                        reward = 0.0
                        break
                    if parsed.get('evidence_start') is None or parsed.get('evidence_end') is None:
                        reward = 0.0
                        break
                    try:
                        evidence_start = float(parsed['evidence_start'])
                        evidence_end = float(parsed['evidence_end'])
                        if evidence_start >= evidence_end:
                            reward = 0.0
                            break
                    except (TypeError, ValueError):
                        reward = 0.0
                        break
                
                last_action = action
 
            # ----- final action must be ANSWER -----
            if last_action != 'ANSWER':
                reward = 0.0
 
            rewards.append(reward)

        return rewards


orms['tree_format_reward'] = TreeFormatReward


class TreeAccuracyReward(ORM):
    """
    Reward for answer accuracy.

    Extracts the answer letter from the final ANSWER action and compares
    to the ground truth solution.
    """

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        trajectory_ids: List[str] = kwargs.get('request_id', [])
        global_trajectories: Dict = kwargs.get('trajectory_inputs', {})

        rewards = []
        for tra_id in trajectory_ids:
            final_turn = _get_final_turn(global_trajectories, tra_id)
            if final_turn is None:
                rewards.append(0.0)
                continue

            final = _extract_final_answer(final_turn)
            if final is None or 'action' not in final or final['action'] != 'ANSWER':
                rewards.append(0.0)
                continue

            model_answer = _first_letter(str(final.get('answer', '')))
            ref_answer = _first_letter(str(final_turn.get('solution', '')))

            if model_answer and ref_answer and model_answer == ref_answer:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        return rewards


orms['tree_acc_reward'] = TreeAccuracyReward


class TreeConstantAnswerReward(ORM):
    """
    Debug reward: 1.0 iff the final ANSWER letter equals a fixed constant.

    Purpose: sanity-check the GRPO pipeline. Baseline ~1/N over N choices; if
    training works, the policy will bias its answer token toward the constant
    and the reward will climb toward 1.0 within a few hundred optimizer steps.
    The constant is read from the DEBUG_CONSTANT_ANSWER env var (default "A").
    """

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        trajectory_ids: List[str] = kwargs.get('request_id', [])
        global_trajectories: Dict = kwargs.get('trajectory_inputs', {})
        target = _first_letter(os.environ.get('DEBUG_CONSTANT_ANSWER', 'A'))

        rewards = []
        for tra_id in trajectory_ids:
            final_turn = _get_final_turn(global_trajectories, tra_id)
            if final_turn is None:
                rewards.append(0.0)
                continue
            final = _extract_final_answer(final_turn)
            if final is None or 'action' not in final or final['action'] != 'ANSWER':
                rewards.append(0.0)
                continue
            model_letter = _first_letter(str(final.get('answer', '')))
            rewards.append(1.0 if model_letter == target else 0.0)
        return rewards


orms['tree_constant_answer_reward'] = TreeConstantAnswerReward


def _compute_gt_coverage(
    node_interval: Tuple[float, float],
    gt_interval: Tuple[float, float],
) -> float:
    """
    Compute what fraction of the GT clue falls inside the node interval.
 
    coverage = overlap(node, gt) / length(gt)
    """
    node_start, node_end = node_interval
    gt_start, gt_end = gt_interval
 
    gt_length = gt_end - gt_start
    if gt_length <= 0:
        return 0.0
 
    overlap_start = max(node_start, gt_start)
    overlap_end = min(node_end, gt_end)
    overlap = max(0.0, overlap_end - overlap_start)
 
    return overlap / gt_length


class TreeGatedAccuracyReward(ORM):
    """
    Gated accuracy reward.
 
    Returns 1.0 iff:
      (1) The final action is ANSWER with the correct answer letter.
      (2) The node where ANSWER was produced covers >50% of the GT clue.
    Otherwise returns 0.0.
 
    The coverage gate prevents the model from being rewarded for lucky
    guesses made from a node that doesn't actually contain the evidence.
    """
 
    COVERAGE_THRESHOLD = 0.5
 
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        trajectory_ids: List[str] = kwargs.get('request_id', [])
        global_trajectories: Dict = kwargs.get('trajectory_inputs', {})
 
        rewards = []
        for tra_id in trajectory_ids:
            final_turn = _get_final_turn(global_trajectories, tra_id)
            if final_turn is None:
                rewards.append(0.0)
                continue

            # --- Check answer correctness ---
            final = _extract_final_answer(final_turn)
            if final is None or 'action' not in final or final['action'] != 'ANSWER':
                rewards.append(0.0)
                continue

            model_answer = _first_letter(str(final.get('answer', '')))
            ref_answer = _first_letter(str(final_turn.get('solution', '')))

            if not (model_answer and ref_answer and model_answer == ref_answer):
                rewards.append(0.0)
                continue

            # --- Check node coverage of GT clue ---
            rollout_infos = final_turn.get('rollout_infos', {})
            node_start = rollout_infos.get('node_start')
            node_end = rollout_infos.get('node_end')
            if node_start is None or node_end is None:
                rewards.append(0.0)
                continue
            node_interval = (node_start, node_end)

            true_timestamps = final_turn.get('timestamp', [])
            if not true_timestamps:
                rewards.append(0.0)
                continue
 
            # Gate passes if the node covers >threshold of ANY GT interval
            max_coverage = 0.0
            for ts in true_timestamps:
                if len(ts) >= 2:
                    coverage = _compute_gt_coverage(
                        node_interval, (float(ts[0]), float(ts[1]))
                    )
                    max_coverage = max(max_coverage, coverage)
 
            if max_coverage > self.COVERAGE_THRESHOLD:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
 
        return rewards
 
orms['tree_gated_acc_reward'] = TreeGatedAccuracyReward


class TreeIOUReward(ORM):
    """Reward based on mean temporal IoU of the evidence interval."""

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        trajectory_ids: List[str] = kwargs.get('request_id', [])
        global_trajectories: Dict = kwargs.get('trajectory_inputs', {})

        rewards = []
        for tra_id in trajectory_ids:
            final_turn = _get_final_turn(global_trajectories, tra_id)
            if final_turn is None:
                rewards.append(0.0)
                continue

            final = _extract_final_answer(final_turn)
            if final is None or final['action'] != 'ANSWER':
                rewards.append(0.0)
                continue

            try:
                pred_start = float(final['evidence_start'])
                pred_end = float(final['evidence_end'])
            except (TypeError, ValueError):
                rewards.append(0.0)
                continue

            true_timestamps = final_turn.get('timestamp', [])
            if not true_timestamps:
                rewards.append(0.0)
                continue

            ious = []
            for ts in true_timestamps:
                if len(ts) >= 2:
                    ious.append(_calculate_iou(
                        (pred_start, pred_end),
                        (float(ts[0]), float(ts[1])),
                    ))

            rewards.append(sum(ious) / len(ious) if ious else 0.0)

        return rewards


orms['tree_iou_reward'] = TreeIOUReward


class TreeDistanceReward(ORM):
    """
    Reward based on tree distance from the ANSWER node to the nearest
    node covering >50% of the ground-truth clue.

    Reward = max(0, 1 - distance * DISTANCE_DECAY).
    Distance 0 (already at a good node) gives 1.0; distance 5+ gives 0.0.
    """

    DISTANCE_DECAY = 0.2

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        trajectory_ids: List[str] = kwargs.get('request_id', [])
        global_trajectories: Dict = kwargs.get('trajectory_inputs', {})

        rewards = []
        for tra_id in trajectory_ids:
            final_turn = _get_final_turn(global_trajectories, tra_id)
            if final_turn is None:
                rewards.append(0.0)
                continue

            rollout_infos = final_turn.get('rollout_infos', {})
            tree_distance = rollout_infos.get('tree_distance')

            if tree_distance is None:
                rewards.append(0.0)
                continue

            reward = max(0.0, 1.0 - tree_distance * self.DISTANCE_DECAY)
            rewards.append(reward)

        return rewards


orms['tree_distance_reward'] = TreeDistanceReward


class TreeGTDistanceReward(ORM):
    """
    Reward based on tree distance from the ANSWER node to the unique "best"
    node for this sample.

    The best node is argmax over all tree nodes of
        max over gt in gt_intervals of [IoU(node, gt) + IoG(node, gt)]
    with ties broken by deeper node, then smaller span (see
    utils.tree_utils.find_best_node).

    Reward = max(0, 1 - gt_distance_to_best * DISTANCE_DECAY).
    Distance 0 (at the ideal node) gives 1.0; distance 5+ gives 0.0.

    Unlike TreeDistanceReward, this is not susceptible to root collapse: the
    root's score is dominated by its oversized span (low IoU), so the root is
    almost never the best node for a short GT interval in a long video.
    """

    DISTANCE_DECAY = 0.15

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        trajectory_ids: List[str] = kwargs.get('request_id', [])
        global_trajectories: Dict = kwargs.get('trajectory_inputs', {})

        rewards = []
        for tra_id in trajectory_ids:
            final_turn = _get_final_turn(global_trajectories, tra_id)
            if final_turn is None:
                rewards.append(0.0)
                continue

            rollout_infos = final_turn.get('rollout_infos', {})
            gt_distance = rollout_infos.get('gt_distance_to_best')

            if gt_distance is None:
                rewards.append(0.0)
                continue

            reward = max(0.0, 1.0 - gt_distance * self.DISTANCE_DECAY)
            rewards.append(reward)

        return rewards


orms['tree_gt_distance_reward'] = TreeGTDistanceReward


# =========================================================================== #
# Debug trajectory logger
# =========================================================================== #

def _render_tree_text(tree_dict: Dict) -> str:
    """Render tree structure as ASCII text (no captions)."""
    lines: List[str] = []
    _render_tree_node(tree_dict, lines, prefix='', is_root=True, is_last=True)
    return '\n'.join(lines)


def _render_tree_node(
    node: Dict, lines: List[str],
    prefix: str, is_root: bool, is_last: bool,
) -> None:
    start = node['start_sec']
    end = node['end_sec']
    visited = node.get('visited', False)
    marker = ' *' if visited else ''
    label = f'[{start:.0f}s-{end:.0f}s]{marker}'

    if is_root:
        lines.append(label)
        child_prefix = ''
    else:
        connector = '└── ' if is_last else '├── '
        lines.append(f'{prefix}{connector}{label}')
        child_prefix = prefix + ('    ' if is_last else '│   ')

    children = node.get('children', [])
    for i, child in enumerate(children):
        _render_tree_node(child, lines, child_prefix, False, i == len(children) - 1)


def _compute_all_rewards(final_turn: Dict) -> Dict:
    """
    Compute all reward types for a single trajectory's terminal turn.

    Mirrors the production reward classes so debug output matches the
    rewards that actually feed GRPO. Pass in the terminal turn entry
    (as returned by ``_get_final_turn``), not a merged multi-turn blob.
    """
    # --- Parse the terminal assistant message ---
    final = _extract_final_answer(final_turn)

    # --- Format reward: ANSWER action with all required fields valid ---
    format_reward = 0.0
    if final is not None and final.get('action') == 'ANSWER':
        ev_s = final.get('evidence_start')
        ev_e = final.get('evidence_end')
        if final.get('answer') and ev_s is not None and ev_e is not None:
            try:
                if float(ev_s) < float(ev_e):
                    format_reward = 1.0
            except (TypeError, ValueError):
                pass

    # --- Accuracy reward ---
    model_answer = ''
    acc_reward = 0.0
    if final is not None and final.get('action') == 'ANSWER':
        model_answer = _first_letter(str(final.get('answer', '')))
        ref_answer = _first_letter(str(final_turn.get('solution', '')))
        if model_answer and ref_answer and model_answer == ref_answer:
            acc_reward = 1.0

    # --- Constant-answer reward (sanity check for the GRPO pipeline) ---
    constant_target = _first_letter(os.environ.get('DEBUG_CONSTANT_ANSWER', 'A'))
    constant_answer_reward = 1.0 if model_answer and model_answer == constant_target else 0.0

    # --- IoU reward ---
    iou_reward = 0.0
    if final is not None and final.get('action') == 'ANSWER':
        try:
            pred_start = float(final['evidence_start'])
            pred_end = float(final['evidence_end'])
            true_timestamps = final_turn.get('timestamp', [])
            ious = [
                _calculate_iou((pred_start, pred_end), (float(ts[0]), float(ts[1])))
                for ts in true_timestamps if len(ts) >= 2
            ]
            if ious:
                iou_reward = sum(ious) / len(ious)
        except (TypeError, ValueError, KeyError):
            pass

    # --- Gated accuracy reward ---
    gated_acc_reward = 0.0
    if acc_reward == 1.0:
        rollout_infos = final_turn.get('rollout_infos', {})
        node_start = rollout_infos.get('node_start')
        node_end = rollout_infos.get('node_end')
        true_timestamps = final_turn.get('timestamp', [])
        if node_start is not None and node_end is not None and true_timestamps:
            max_coverage = max(
                (_compute_gt_coverage((node_start, node_end), (float(ts[0]), float(ts[1])))
                 for ts in true_timestamps if len(ts) >= 2),
                default=0.0,
            )
            if max_coverage > 0.5:
                gated_acc_reward = 1.0

    # --- Tree distance reward ---
    rollout_infos = final_turn.get('rollout_infos', {})
    tree_distance_raw = rollout_infos.get('tree_distance')
    tree_distance_reward = (
        max(0.0, 1.0 - tree_distance_raw * 0.2)
        if tree_distance_raw is not None else 0.0
    )

    # --- GT-distance reward (distance to the unique best node) ---
    gt_distance_raw = rollout_infos.get('gt_distance_to_best')
    gt_distance_reward = (
        max(0.0, 1.0 - gt_distance_raw * 0.2)
        if gt_distance_raw is not None else 0.0
    )
    best_node_score = rollout_infos.get('best_node_score')

    return {
        'format': format_reward,
        'accuracy': acc_reward,
        'iou': iou_reward,
        'gated_accuracy': gated_acc_reward,
        'tree_distance': tree_distance_raw,
        'tree_distance_reward': tree_distance_reward,
        'gt_distance_to_best': gt_distance_raw,
        'gt_distance_reward': gt_distance_reward,
        'best_node_score': best_node_score,
        'constant_answer': constant_answer_reward,
        'constant_answer_target': constant_target,
        'final_answer': model_answer,
    }


class TreeDebugLogger(ORM):
    """
    Debug-only reward function that dumps trajectory data to organized files.

    Controlled by environment variables:
      DEBUG_TRAJECTORIES=1        Enable logging (default: disabled)
      DEBUG_OUTPUT_DIR=path       Output directory (default: rl/debug_output)
      DEBUG_LOG_INTERVAL=N        Log every N steps (default: 10)

    Returns 0.0 for all trajectories (no effect on GRPO training).
    """

    def __init__(self):
        self.enabled = os.environ.get('DEBUG_TRAJECTORIES', '0') == '1'
        base_dir = os.environ.get('DEBUG_OUTPUT_DIR', 'rl/debug_output')
        from datetime import datetime
        run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(base_dir, run_time)
        self._step_counter = 0
        self.save_interval = int(os.environ.get('DEBUG_LOG_INTERVAL', '15'))

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = [0.0] * len(completions)
        if not self.enabled:
            return rewards

        # Get global step early so we can skip expensive work on non-save steps
        trainer_state = kwargs.get('trainer_state')
        if trainer_state and hasattr(trainer_state, 'global_step'):
            global_step = trainer_state.global_step
        else:
            self._step_counter += 1
            global_step = self._step_counter

        if global_step % self.save_interval != 0:
            return rewards

        rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
        local_trajectories: Dict = kwargs.get('trajectory_inputs', {})

        # Gather trajectory data from ALL ranks so rank 0 sees all generations.
        # All ranks must participate in the collective op to avoid mismatched
        # distributed calls (which would corrupt subsequent gather_object calls).
        try:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_world_size() > 1:
                all_traj_dicts = [None] * dist.get_world_size()
                dist.all_gather_object(all_traj_dicts, local_trajectories)
                global_trajectories: Dict = {}
                for rank_dict in all_traj_dicts:
                    if not rank_dict:
                        continue
                    for tra_id, entries in rank_dict.items():
                        if tra_id not in global_trajectories:
                            global_trajectories[tra_id] = []
                        # Deduplicate turns by turn index
                        existing_turn_nums = {
                            e.get('rollout_infos', {}).get('num_turns')
                            for e in global_trajectories[tra_id]
                        }
                        for entry in entries:
                            turn_num = entry.get('rollout_infos', {}).get('num_turns')
                            if turn_num not in existing_turn_nums:
                                global_trajectories[tra_id].append(entry)
                                existing_turn_nums.add(turn_num)
            else:
                global_trajectories = local_trajectories
        except Exception:
            global_trajectories = local_trajectories

        # Only rank 0 writes files
        if rank != 0:
            return rewards

        # Group trajectories by example using (video_id, qid)
        from collections import defaultdict
        example_groups = defaultdict(list)
        for tra_id, turns in global_trajectories.items():
            if not turns:
                continue
            traj = turns[-1]
            video_id = traj.get('video_id', 'unknown')
            qid = traj.get('qid', 'unknown')
            example_groups[(video_id, qid)].append(tra_id)

        for (video_id, qid), tra_ids in example_groups.items():
            self._write_example(global_step, video_id, qid, tra_ids, global_trajectories)

        return rewards

    def _write_example(
        self,
        global_step: int,
        video_id: str,
        qid,
        tra_ids: List[str],
        global_trajectories: Dict,
    ) -> None:
        """Write all debug files for one example."""
        step_dir = os.path.join(
            self.output_dir, f'step_{global_step:06d}',
            f'example_{qid}_{video_id}',
        )
        os.makedirs(step_dir, exist_ok=True)

        # Use first trajectory for example metadata
        first_traj = global_trajectories[tra_ids[0]][-1]

        # --- example_info.json ---
        example_info = {
            'qid': qid,
            'video_id': video_id,
            'question': first_traj.get('question', ''),
            'choices': first_traj.get('choices', []),
            'solution': first_traj.get('solution', ''),
            'timestamp': first_traj.get('timestamp', []),
            'duration': first_traj.get('duration', 0),
        }
        with open(os.path.join(step_dir, 'example_info.json'), 'w') as f:
            json.dump(example_info, f, indent=2, ensure_ascii=False)

        # --- tree.txt: tree structure (no captions) with visited markers ---
        # Extract tree_dict from any trajectory's last turn (structure is shared)
        tree_dict = None
        for tra_id in tra_ids:
            turns = global_trajectories.get(tra_id, [])
            if turns:
                tree_dict = turns[-1].get('rollout_infos', {}).get('tree_dict')
                if tree_dict:
                    break
        if tree_dict:
            tree_text = _render_tree_text(tree_dict)
            with open(os.path.join(step_dir, 'tree.txt'), 'w') as f:
                f.write(tree_text + '\n')

        # --- Process each trajectory ---
        rewards_summary = []
        output_format = os.environ.get('OUTPUT_FORMAT', 'sft')

        for traj_idx, tra_id in enumerate(tra_ids):
            turn_entries = global_trajectories.get(tra_id, [])
            if not turn_entries:
                continue
            # Sort by turn index to ensure correct order after cross-rank merging
            turn_entries.sort(key=lambda e: e.get('rollout_infos', {}).get('num_turns', 0))

            # Compute rewards from the terminal turn (mirrors production reward classes)
            last_entry = turn_entries[-1]
            reward_info = _compute_all_rewards(last_entry)

            # Get termination reason from last turn's rollout_infos
            last_infos = last_entry.get('rollout_infos', {})
            num_turns = len(turn_entries)
            expected_turns = last_infos.get('num_turns', num_turns)
            if num_turns < expected_turns:
                logger.warning(
                    f'[{video_id}] Trajectory {tra_id}: only {num_turns} turn '
                    f'entries available but expected {expected_turns}'
                )
            termination_reason = last_infos.get('termination_reason', 'unknown')

            # Determine final action
            final = _extract_final_answer(last_entry)
            final_action = final['action'] if final else 'none'

            # --- Build navigation trace from all turns ---
            nav_trace = []
            for entry in turn_entries:
                infos = entry.get('rollout_infos', {})
                ns = infos.get('node_start')
                ne = infos.get('node_end')
                seg_label = f'[{ns:.0f}s-{ne:.0f}s]' if ns is not None else '[?]'

                # Parse this turn's action
                msgs = entry.get('messages', [])
                action_label = '?'
                for msg in reversed(msgs):
                    if msg.get('role') == 'assistant':
                        if output_format == 'sft':
                            parsed = parse_action_response_sft(msg['content'])
                        else:
                            parsed = parse_action_response(msg['content'])
                        if parsed:
                            a = parsed['action']
                            if a in ('ZOOM_IN', 'SHIFT'):
                                action_label = f'{a}({parsed.get("segment_id")})'
                            elif a == 'ANSWER':
                                ans = parsed.get('answer', '?')
                                ev_s = parsed.get('evidence_start', '?')
                                ev_e = parsed.get('evidence_end', '?')
                                action_label = f'ANSWER({ans}, [{ev_s}-{ev_e}])'
                            else:
                                action_label = a
                        else:
                            action_label = 'PARSE_FAIL'
                        break
                nav_trace.append(f'{seg_label} {action_label}')

            # Build rewards summary entry
            summary_entry = {
                'trajectory_idx': traj_idx,
                'format': reward_info['format'],
                'accuracy': reward_info['accuracy'],
                'iou': round(reward_info['iou'], 4),
                'gated_accuracy': reward_info['gated_accuracy'],
                'tree_distance': reward_info.get('tree_distance'),
                'tree_distance_reward': round(reward_info.get('tree_distance_reward', 0.0), 4),
                'gt_distance_to_best': reward_info.get('gt_distance_to_best'),
                'gt_distance_reward': round(reward_info.get('gt_distance_reward', 0.0), 4),
                'best_node_score': (
                    round(reward_info['best_node_score'], 4)
                    if reward_info.get('best_node_score') is not None else None
                ),
                'constant_answer': reward_info['constant_answer'],
                'num_turns': num_turns,
                'final_action': final_action,
                'final_answer': reward_info['final_answer'],
                'termination_reason': termination_reason,
                'nav_trace': ' → '.join(nav_trace),
            }
            rewards_summary.append(summary_entry)

            # --- Write trajectory_N.md ---
            self._write_trajectory_md(
                step_dir, traj_idx, turn_entries, reward_info,
                num_turns, final_action, termination_reason, nav_trace,
            )

        # --- rewards_summary.json ---
        with open(os.path.join(step_dir, 'rewards_summary.json'), 'w') as f:
            json.dump(rewards_summary, f, indent=2, ensure_ascii=False)

    def _write_trajectory_md(
        self,
        step_dir: str,
        traj_idx: int,
        turn_entries: List[Dict],
        reward_info: Dict,
        num_turns: int,
        final_action: str,
        termination_reason: str,
        nav_trace: List[str],
    ) -> None:
        """Write a single trajectory as a folder with per-turn files."""
        output_format = os.environ.get('OUTPUT_FORMAT', 'sft')

        traj_dir = os.path.join(step_dir, f'trajectory_{traj_idx}')
        os.makedirs(traj_dir, exist_ok=True)

        # --- summary.md: rewards + nav trace (one file per trajectory) ---
        lines = []
        lines.append(f'# Trajectory {traj_idx}\n')

        # Rewards table
        lines.append('| Reward | Value |')
        lines.append('|--------|-------|')
        lines.append(f'| Format | {reward_info["format"]} |')
        lines.append(f'| Accuracy | {reward_info["accuracy"]} |')
        lines.append(f'| IoU | {reward_info["iou"]:.4f} |')
        lines.append(f'| Gated Accuracy | {reward_info["gated_accuracy"]} |')
        td_raw = reward_info.get('tree_distance')
        td_reward = reward_info.get('tree_distance_reward', 0.0)
        lines.append(f'| Tree Distance | {td_raw if td_raw is not None else "N/A"} |')
        lines.append(f'| Tree Distance Reward | {td_reward:.4f} |')
        gtd_raw = reward_info.get('gt_distance_to_best')
        gtd_reward = reward_info.get('gt_distance_reward', 0.0)
        best_score = reward_info.get('best_node_score')
        best_score_str = f'{best_score:.4f}' if best_score is not None else 'N/A'
        lines.append(f'| GT Distance (to best node) | {gtd_raw if gtd_raw is not None else "N/A"} |')
        lines.append(f'| GT Distance Reward | {gtd_reward:.4f} |')
        lines.append(f'| Best Node Score (IoU+IoG) | {best_score_str} |')
        const_target = reward_info.get('constant_answer_target', '')
        lines.append(f'| Constant Answer (target={const_target}) | {reward_info["constant_answer"]} |')
        lines.append('')
        lines.append(
            f'**Num Turns**: {num_turns} | '
            f'**Final Action**: {final_action} | '
            f'**Termination**: {termination_reason}'
        )
        lines.append('')

        # Navigation trace
        lines.append('### Navigation Trace')
        lines.append('')
        lines.append(' \u2192 '.join(nav_trace))
        lines.append('')

        with open(os.path.join(traj_dir, 'summary.md'), 'w') as f:
            f.write('\n'.join(lines))

        # --- Per-turn files: turn_01.md, turn_02.md, ... ---
        for turn_num, entry in enumerate(turn_entries, start=1):
            msgs = entry.get('messages', [])
            infos = entry.get('rollout_infos', {})
            node_start = infos.get('node_start')
            node_end = infos.get('node_end')

            # Extract prompt and response from this turn's messages
            prompt_text = ''
            response = ''
            for msg in msgs:
                role = msg.get('role', '')
                if role in ('system', 'user'):
                    prompt_text += msg.get('content', '') + '\n\n'
                elif role == 'assistant':
                    response = msg.get('content', '')
            prompt_text = prompt_text.strip()

            # Count <image> tokens
            num_images = prompt_text.count('<image>')

            turn_lines = []
            turn_lines.append(f'# Turn {turn_num}\n')

            # Segment info
            if node_start is not None:
                turn_lines.append(f'**Segment**: [{node_start:.0f}s - {node_end:.0f}s] | '
                                  f'**Images**: {num_images} frames')
            else:
                turn_lines.append(f'**Images**: {num_images} frames')
            turn_lines.append('')

            # Parsed action
            if output_format == 'sft':
                parsed = parse_action_response_sft(response)
            else:
                parsed = parse_action_response(response)

            turn_lines.append('## Action')
            if parsed:
                action = parsed['action']
                parts = [f'**{action}**']
                if parsed.get('segment_id') is not None:
                    parts.append(f'segment={parsed["segment_id"]}')
                if parsed.get('answer'):
                    parts.append(f'answer={parsed["answer"]}')
                if parsed.get('evidence_start') is not None:
                    parts.append(f'evidence=[{parsed["evidence_start"]}, {parsed["evidence_end"]}]')
                turn_lines.append(' | '.join(parts))
            else:
                turn_lines.append('**PARSE FAILED**')
            turn_lines.append('')

            # Reasoning
            if parsed and parsed.get('reasoning'):
                turn_lines.append('## Reasoning')
                turn_lines.append(parsed['reasoning'])
                turn_lines.append('')

            # Response
            turn_lines.append('## Response')
            turn_lines.append(f'```\n{response}\n```')
            turn_lines.append('')

            # Prompt
            turn_lines.append(f'## Prompt ({num_images} images)')
            turn_lines.append(f'```\n{prompt_text}\n```')
            turn_lines.append('')

            turn_path = os.path.join(traj_dir, f'turn_{turn_num:02d}.md')
            with open(turn_path, 'w') as f:
                f.write('\n'.join(turn_lines))


orms['tree_debug_logger'] = TreeDebugLogger
