"""
Prompt construction utilities for the tree-based video agent.

Builds system prompts, memory text, navigation context, user content,
and parses model action responses. Adapted from the datagen vlm_interface.
"""

import json
import re
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Response format tags
# --------------------------------------------------------------------------- #
ACTION_TAG_PATTERN = re.compile(r'<action>(.*?)</action>', re.DOTALL)
THINK_TAG_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)
THINK_ACTION_PATTERN = re.compile(
    r'^<think>.*?</think>\s*<action>.*?</action>$', re.DOTALL
)
# SFT format: <think>...</think> followed by ```json...```
SFT_JSON_PATTERN = re.compile(r'```json\s*(\{.*?\})\s*```', re.DOTALL)
THINK_SFT_PATTERN = re.compile(
    r'^<think>.*?</think>\s*```json\s*\{.*?\}\s*```$', re.DOTALL
)

VALID_ACTIONS = {'ZOOM_IN', 'ZOOM_OUT', 'SHIFT', 'ANSWER'}


def _normalize_action(action: str) -> Optional[str]:
    """Match action by prefix (case-insensitive) against VALID_ACTIONS.

    Returns the canonical action name if `action` starts with one of the
    valid actions, else None. Checked longest-first so 'ZOOM_IN'/'ZOOM_OUT'
    take precedence over a hypothetical 'ZOOM' prefix.
    """
    if not isinstance(action, str):
        return None
    upper = action.upper()
    for valid in sorted(VALID_ACTIONS, key=len, reverse=True):
        if upper.startswith(valid):
            return valid
    return None


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #
def _build_action_list(allowed_actions: List[str]) -> Tuple[str, str]:
    """Build numbered action descriptions and count phrase.

    Returns:
        (action_list_text, count_word) tuple.
    """
    action_descs = []
    idx = 1
    if 'ZOOM_IN' in allowed_actions:
        action_descs.append(
            f'{idx}. ZOOM_IN <segment_id>: Examine a child segment at '
            'finer granularity.'
        )
        idx += 1
    if 'ZOOM_OUT' in allowed_actions:
        action_descs.append(
            f'{idx}. ZOOM_OUT: Backtrack to the parent segment to explore '
            'a different region.'
        )
        idx += 1
    if 'SHIFT' in allowed_actions:
        action_descs.append(
            f'{idx}. SHIFT <segment_id>: Move to a sibling segment under '
            'the same parent.'
        )
        idx += 1
    if 'ANSWER' in allowed_actions:
        action_descs.append(
            f'{idx}. ANSWER <letter> <evidence_start> <evidence_end>: '
            'Provide the answer to the question along with the time interval '
            '(in seconds) that contains the supporting evidence.'
        )
        idx += 1

    n = len(action_descs)
    count_word = {
        1: 'the following action',
        2: 'one of two actions',
        3: 'one of three actions',
        4: 'one of four actions',
    }.get(n, f'one of {n} actions')

    return '\n'.join(action_descs) + '\n', count_word


def build_system_prompt(
    allowed_actions: List[str],
    has_children: bool = True,
    output_format: str = 'tagged',
) -> str:
    """
    Build the system prompt describing the agent's role and available actions.

    Args:
        allowed_actions: List of allowed action names.
        has_children: Whether the current node has children segments.
        output_format: 'tagged' (think/action tags) or 'sft' (markdown JSON).

    Returns:
        System prompt string.
    """
    action_list, count_word = _build_action_list(allowed_actions)

    if has_children:
        intro = (
            'You are a video question-answering agent that navigates a long video '
            'through hierarchical temporal search. At each step, the current video '
            'segment is divided into non-overlapping child segments. You observe '
            f'frames from each child segment and decide {count_word}:\n\n'
        )
    else:
        intro = (
            'You are a video question-answering agent that navigates a long video '
            'through hierarchical temporal search. You observe frames from the '
            f'current segment and decide {count_word}:\n\n'
        )

    system = intro + action_list

    if output_format == 'tagged':
        system += (
            '\nEnclose your reasoning within <think> and </think> tags.\n'
            'Enclose your action JSON within <action> and </action> tags.\n'
        )

    return system


# --------------------------------------------------------------------------- #
# Memory text (tree structure + interaction history)
# --------------------------------------------------------------------------- #
def _render_tree_node(
    node_dict: Dict,
    lines: List[str],
    indent_level: int,
    node_path: str,
    current_node_key: Optional[Tuple[float, float]] = None,
    parent_is_current: bool = False,
) -> None:
    """Recursively render a tree node as indented text lines."""
    start = node_dict['start_sec']
    end = node_dict['end_sec']
    visited = node_dict.get('visited', False)

    is_current = False
    if current_node_key:
        if abs(start - current_node_key[0]) < 0.1 and abs(end - current_node_key[1]) < 0.1:
            is_current = True

    indent = '  ' * indent_level
    visited_tag = '[visited]' if visited else '[not visited]'
    current_tag = ' [current]' if is_current else ''

    if not node_path:
        line = f'{indent}[{start:.0f}s-{end:.0f}s] {visited_tag}{current_tag}'
    else:
        line = f'{indent}Child {node_path}: [{start:.0f}s-{end:.0f}s] {visited_tag}{current_tag}'

    # Suppress captions on the current node and its direct children
    # (those segments are being observed via frames right now)
    if not is_current and not parent_is_current:
        caption = node_dict.get('caption', '')
        if caption:
            line += f', Description: "{caption}"'

    lines.append(line)

    if visited:
        for child in node_dict.get('children', []):
            child_id = child['node_id']
            child_path = str(child_id) if not node_path else f'{node_path}.{child_id}'
            _render_tree_node(child, lines, indent_level + 1, child_path,
                              current_node_key, parent_is_current=is_current)


def build_memory_text(
    tree_dict: Dict,
    history: List[Dict],
    current_node_key: Tuple[float, float],
    max_history_turns: int = -1,
) -> str:
    """
    Build memory text combining tree structure and interaction history.

    Args:
        tree_dict: Serialized tree from TreeState.to_tree_dict().
        history: List of {action, reasoning, observation} dicts.
        current_node_key: (start_sec, end_sec) of current node.
        max_history_turns: Max history entries to include (-1 = all).

    Returns:
        Formatted memory string, or empty string if no history.
    """
    parts = []

    # Part 1: Tree Structure (always shown)
    if tree_dict:
        tree_lines: List[str] = []
        _render_tree_node(tree_dict, tree_lines, indent_level=0, node_path='',
                          current_node_key=current_node_key)
        parts.append('Tree Structure:\n' + '\n'.join(tree_lines))

    # Part 2: Interaction History
    history_lines: List[str] = []
    if not history:
        obs_start, obs_end = current_node_key
    else:
        obs_start = history[0]['observation']['start_sec']
        obs_end = history[0]['observation']['end_sec']
    history_lines.append(
        f'- Turn 1: Observed Segment [{obs_start:.0f}s-{obs_end:.0f}s]'
    )

    if history:
        display_history = history
        if max_history_turns > 0:
            display_history = history[-max_history_turns:]
        offset = len(history) - len(display_history)
        for idx, turn in enumerate(display_history):
            turn_num = offset + idx + 2
            action = turn.get('action', '')
            reasoning = turn.get('reasoning', '')
            line = f'- Turn {turn_num}: {action}'
            if reasoning:
                short_reasoning = reasoning.replace('\n', ' ')
                line += f'. Reasoning: {short_reasoning}'
            history_lines.append(line)

    if history_lines:
        parts.append('Interaction History:\n' + '\n'.join(history_lines))

    if not parts:
        return ''

    return 'Memory:\n\n' + '\n\n'.join(parts) + '\n'


# --------------------------------------------------------------------------- #
# Navigation context
# --------------------------------------------------------------------------- #
def build_navigation_context(
    current_node,
    valid_zoom_in: Optional[List] = None,
    valid_shift: Optional[List] = None,
) -> str:
    """
    Build navigation context showing current position and available targets.

    Args:
        current_node: The current TreeNode.
        valid_zoom_in: Pre-filtered list of valid child TreeNodes for ZOOM_IN.
        valid_shift: Pre-filtered list of valid sibling TreeNodes for SHIFT.

    Returns:
        Navigation context string.
    """
    nav_parts = [f'Current Position: [{current_node.start_sec:.1f}s-{current_node.end_sec:.1f}s]']

    if current_node.parent is not None:
        p = current_node.parent
        nav_parts.append(f'Parent: [{p.start_sec:.1f}s-{p.end_sec:.1f}s]')

    if valid_zoom_in:
        items = ', '.join(
            f"Segment {c.node_id} [{c.start_sec:.1f}s-{c.end_sec:.1f}s]"
            for c in valid_zoom_in
        )
        nav_parts.append(f'Valid ZOOM_IN targets: {items}')

    if valid_shift:
        items = ', '.join(
            f"Segment {s.node_id} [{s.start_sec:.1f}s-{s.end_sec:.1f}s]"
            for s in valid_shift
        )
        nav_parts.append(f'Valid SHIFT targets: {items}')

    return 'Navigation Context:\n' + '\n'.join(nav_parts) + '\n'


# --------------------------------------------------------------------------- #
# Important rules
# --------------------------------------------------------------------------- #
def _build_important_rules(
    allowed_actions: List[str],
    valid_zoom_in_ids: Optional[List[int]] = None,
    valid_shift_ids: Optional[List[int]] = None,
) -> str:
    """Build the IMPORTANT RULES section."""
    rule_parts = []

    if 'ZOOM_IN' in allowed_actions:
        rule = (
            '- Only use ZOOM_IN when you believe a specific child segment is '
            'likely to contain the answer and needs closer examination.'
        )
        if valid_zoom_in_ids:
            rule += (
                f' You MUST select one of the following child '
                f'segment IDs: {valid_zoom_in_ids}.'
            )
        rule_parts.append(rule)

    if 'ZOOM_OUT' in allowed_actions:
        rule_parts.append(
            '- Use ZOOM_OUT when none of the current child segments appear '
            'relevant.'
        )

    if 'SHIFT' in allowed_actions:
        rule = (
            '- Use SHIFT to move to a sibling segment at the same level when '
            'the current segment is not relevant but a sibling might be.'
        )
        if valid_shift_ids:
            rule += (
                f' You MUST select one of the followings sibling '
                f'segment IDs: {valid_shift_ids}.'
            )
        rule_parts.append(rule)

    if 'ANSWER' in allowed_actions:
        rule_parts.append(
            '- Use ANSWER when you are confident you have found sufficient '
            'evidence to answer the question. You must specify the answer '
            'letter AND the evidence time interval.'
        )

    return 'IMPORTANT RULES:\n' + '\n'.join(rule_parts) + '\n'


# --------------------------------------------------------------------------- #
# JSON format spec
# --------------------------------------------------------------------------- #
def _build_json_format(
    allowed_actions: List[str],
    output_format: str = 'tagged',
) -> str:
    """Build the JSON response format spec and field rules.

    Args:
        allowed_actions: List of allowed action names.
        output_format: 'tagged' or 'sft'.
    """
    action_values = ' | '.join(f'"{a}"' for a in allowed_actions)

    has_nav = 'ZOOM_IN' in allowed_actions or 'SHIFT' in allowed_actions
    has_answer = 'ANSWER' in allowed_actions

    reasoning_desc = (
        '<your step-by-step reasoning>'
        if output_format == 'tagged'
        else '<your step-by-step reasoning>'
    )
    field_parts = [
        f'  "reasoning": "{reasoning_desc}",\n',
        f'  "action": {action_values},\n',
    ]
    if has_nav:
        field_parts.append('  "segment_id": <int or null>,\n')
    if has_answer:
        null_suffix = ' or null' if has_nav else ''
        field_parts.append(f'  "answer": "<letter{null_suffix}>",\n')
        field_parts.append(f'  "evidence_start": <float{null_suffix}>,\n')
        field_parts.append(f'  "evidence_end": <float{null_suffix}>\n')
    else:
        if field_parts:
            field_parts[-1] = field_parts[-1].rstrip(',\n') + '\n'

    json_fields = ''.join(field_parts)

    # Rules per action
    rule_label = 'Rules:' if output_format == 'tagged' else 'Rules for the JSON:'
    rule_parts = [rule_label + '\n']
    if 'ZOOM_IN' in allowed_actions:
        null_note = ' Set answer/evidence to null.' if has_answer else ''
        rule_parts.append(
            f'- For ZOOM_IN: set segment_id to the chosen child\'s ID.{null_note}\n'
        )
    if 'ZOOM_OUT' in allowed_actions:
        rule_parts.append(
            '- For ZOOM_OUT: set all fields except action and reasoning to null.\n'
        )
    if 'SHIFT' in allowed_actions:
        null_note = ' Set answer/evidence to null.' if has_answer else ''
        rule_parts.append(
            f'- For SHIFT: set segment_id to the sibling segment\'s ID.{null_note}\n'
        )
    if 'ANSWER' in allowed_actions:
        seg_note = ' Set segment_id to null.' if has_nav else ''
        rule_parts.append(
            '- For ANSWER: set answer to the letter (A, B, C, etc.), and '
            f'evidence_start/evidence_end to the time interval.{seg_note}\n'
        )
    action_rules = ''.join(rule_parts)

    if output_format == 'sft':
        return (
            'Respond in EXACTLY this JSON format (no other text before or after):\n'
            '```json\n'
            '{\n'
            f'{json_fields}'
            '}\n'
            '```\n\n'
            f'{action_rules}'
        )
    else:
        return (
            'Respond in EXACTLY this format:\n'
            '<think>your reasoning</think>\n'
            '<action>\n'
            '{\n'
            f'{json_fields}'
            '}\n'
            '</action>\n\n'
            f'{action_rules}'
        )


# --------------------------------------------------------------------------- #
# User content builder
# --------------------------------------------------------------------------- #
def _ts_from_path(p: str, fallback: float) -> float:
    """Extract timestamp from a frame path filename, or return fallback."""
    fname = p.rsplit('/', 1)[-1].replace('.jpg', '')
    try:
        return float(fname)
    except ValueError:
        return fallback


def build_user_content(
    current_start: float,
    current_end: float,
    child_segments: List[Dict],
    frame_paths: List[str],
    memory_text: str,
    nav_text: str,
    question: str,
    choices: List[str],
    allowed_actions: List[str],
    valid_zoom_in_ids: Optional[List[int]] = None,
    valid_shift_ids: Optional[List[int]] = None,
    output_format: str = 'tagged',
) -> Tuple[str, List[str]]:
    """
    Build the user message content with <image> tokens and return ordered
    image paths.

    Args:
        current_start, current_end: Current segment time range.
        child_segments: [{segment_id, start_sec, end_sec, frame_paths}, ...].
            Empty when the current node has no children (leaf).
        frame_paths: All frame paths for the current segment (used for flat
            layout when child_segments is empty).
        memory_text: Pre-built memory string.
        nav_text: Pre-built navigation context string.
        question: The question text.
        choices: List of choice strings.
        allowed_actions: List of action names.
        valid_zoom_in_ids: Unvisited child IDs for ZOOM_IN constraint.
        valid_shift_ids: Sibling segment IDs for SHIFT constraint.
        output_format: 'tagged' or 'sft'.

    Returns:
        (content_text, ordered_image_paths) where <image> tokens in
        content_text correspond 1:1 to ordered_image_paths.
    """
    content_parts = []
    ordered_paths: List[str] = []

    # --- Frames ---
    content_parts.append(
        f'Current Segment [{current_start:.1f}s-{current_end:.1f}s]:'
    )

    if child_segments:
        # Segment-based layout: group frames by child segment
        content_parts.append('\n')
        for seg in child_segments:
            sid = seg['segment_id']
            s = seg['start_sec']
            e = seg['end_sec']
            paths = seg.get('frame_paths', [])

            content_parts.append(f'Segment {sid} [{s:.1f}s-{e:.1f}s]:')

            if paths:
                duration = e - s
                for i, p in enumerate(paths):
                    ts = _ts_from_path(p, s + i * (duration / max(len(paths), 1)))
                    content_parts.append(f'{ts:.1f}s: <image>\n')
                    ordered_paths.append(p)
            else:
                content_parts.append('(no frames available)\n')

            content_parts.append('\n')
    else:
        # Flat layout: no children, just list frames with timestamps
        for i, p in enumerate(frame_paths):
            duration = current_end - current_start
            ts = _ts_from_path(p, current_start + i * (duration / max(len(frame_paths), 1)))
            content_parts.append(f'{ts:.1f}s: <image>\n')
            ordered_paths.append(p)

        content_parts.append('\n')

    # --- Memory ---
    if memory_text:
        content_parts.append(memory_text + '\n')

    # --- Navigation context ---
    if nav_text:
        content_parts.append(nav_text + '\n')

    # --- Question & choices ---
    formatted_choices = '\n'.join(
        f'{chr(65 + i)}. {choice}' for i, choice in enumerate(choices)
    )
    content_parts.append(f'Question: {question}\n\nChoices:\n{formatted_choices}\n\n')

    # --- Analyze instruction + important rules ---
    content_parts.append(
        'Analyze the visual content, memory, navigation context, '
        'the question, and the choices above, then decide your next action.\n\n'
    )

    important_rules = _build_important_rules(
        allowed_actions, valid_zoom_in_ids, valid_shift_ids,
    )
    content_parts.append(important_rules + '\n')

    # --- JSON format spec ---
    json_format = _build_json_format(allowed_actions, output_format)
    content_parts.append(json_format)

    return ''.join(content_parts), ordered_paths


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #
def parse_action_response(text: str) -> Optional[Dict]:
    """
    Parse the model's response to extract the action JSON.

    Looks for JSON inside <action>...</action> tags.
    Returns dict with keys: action, segment_id, answer, evidence_start,
    evidence_end, reasoning. Returns None on parse failure.
    """
    # Extract reasoning
    reasoning = ''
    think_match = THINK_TAG_PATTERN.search(text)
    if think_match:
        reasoning = think_match.group(1).strip()

    # Extract action JSON
    action_match = ACTION_TAG_PATTERN.search(text)
    if not action_match:
        return None

    action_content = action_match.group(1).strip()

    # Try to parse as JSON (might be wrapped in ```json...```)
    json_str = action_content
    # Strip markdown code fence if present
    if json_str.startswith('```'):
        json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    action = _normalize_action(data.get('action', ''))
    if action is None:
        return None

    return {
        'action': action,
        'segment_id': data.get('segment_id'),
        'answer': data.get('answer'),
        'evidence_start': data.get('evidence_start'),
        'evidence_end': data.get('evidence_end'),
        'reasoning': data.get('reasoning', reasoning),
    }


def parse_action_response_sft(text: str) -> Optional[Dict]:
    """
    Parse an SFT-format response to extract the action JSON.

    Looks for JSON inside ```json...``` markdown code blocks.
    Returns dict with keys: action, segment_id, answer, evidence_start,
    evidence_end, reasoning. Returns None on parse failure.
    """
    # Extract JSON from markdown code block
    json_match = SFT_JSON_PATTERN.search(text)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group(1))
    except json.JSONDecodeError:
        return None

    action = _normalize_action(data.get('action', ''))
    if action is None:
        return None

    return {
        'action': action,
        'segment_id': data.get('segment_id'),
        'answer': data.get('answer'),
        'evidence_start': data.get('evidence_start'),
        'evidence_end': data.get('evidence_end'),
        'reasoning': data.get('reasoning', ''),
    }
