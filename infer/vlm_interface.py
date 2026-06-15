"""
VLM/LLM interface for generating descriptions, relevance scores, and reasoning.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
import base64
import math
import os
import tempfile
import cv2
from pathlib import Path
import time


class VLMInterface(ABC):
    """
    Abstract interface for Vision-Language Model interactions.
    
    This interface defines the methods needed for:
    1. Generating textual descriptions from video frames
    2. Computing relevance scores for segments
    3. Generating reasoning content
    """

    @staticmethod
    def _maybe_add_unanswerable(choices: List[str], include_unanswerable: bool) -> List[str]:
        if include_unanswerable:
            return choices + ["Given the current information, the question cannot be answered"]
        return choices

    @abstractmethod
    def generate_description(
        self,
        question: str,
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        additional_context: Optional[Dict[str, Any]] = None,
        short_side: int = -1,
        detailed: bool = False
    ) -> str:
        """
        Generate a textual description of a video segment.

        Args:
            question: Question to answer about the video
            frames: List of timestamps (in seconds) from the segment
            start_sec: Start time of the segment
            end_sec: End time of the segment
            video_path: Path to the video file
            additional_context: Optional additional context
            short_side: Resize frames so shorter side has this size (-1 for no resize)
            detailed: If True, generate a detailed description; if False (default), generate a brief one

        Returns:
            Textual description of the segment
        """
        pass
    
    @abstractmethod
    def compute_relevance_score(
        self,
        question: str,
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        text_description: Optional[str] = None,
        short_side: int = -1
    ) -> float:
        """
        Compute relevance score for a segment given a question.
        
        Args:
            question: The question to answer
            frames: List of timestamps (in seconds) from the segment
            start_sec: Start time of the segment
            end_sec: End time of the segment
            video_path: Path to the video file
            text_description: Optional pre-computed text description
            short_side: Resize frames so shorter side has this size (-1 for no resize)
        
        Returns:
            Relevance score (higher = more relevant)
        """
        pass

    def rank_segments(
        self,
        question: str,
        segments: List[Dict[str, Any]],
        short_side: int = -1
    ) -> List[int]:
        """
        Rank segments by relevance to the question, returning only promising ones.

        Shows all segments to the VLM at once and asks it to return the most
        relevant segment indices ranked from most to least relevant.

        Args:
            question: The question to evaluate relevance for
            segments: List of dicts, each with keys:
                - frames: List of timestamps (in seconds)
                - start_sec: Start time of the segment
                - end_sec: End time of the segment
                - video_path: Path to the video file
                - text_description: Optional text description
            short_side: Resize frames so shorter side has this size (-1 for no resize)

        Returns:
            List of segment indices ranked by relevance (most relevant first).
            Segments not included are considered irrelevant.
        """
        # Default: fall back to per-segment scoring, return indices with score > 0.5
        scores = []
        for i, seg in enumerate(segments):
            score = self.compute_relevance_score(
                question,
                seg["frames"],
                seg["start_sec"],
                seg["end_sec"],
                seg["video_path"],
                seg.get("text_description"),
                short_side=short_side,
            )
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [i for i, s in scores if s > 0.5]

    @abstractmethod
    def generate_reasoning(
        self,
        question: str,
        trajectory_context: Dict[str, Any],
        action_type: str,
        frames: List[float],
        current_observation: Optional[Dict[str, Any]] = None,
        target_observation: Optional[Dict[str, Any]] = None,
        short_side: int = -1,
        include_unanswerable: bool = False
    ) -> str:
        """
        Generate reasoning content explaining why an action is taken.

        Args:
            question: The question being answered
            trajectory_context: Context from the entire trajectory
            action_type: Type of action (ZOOM_IN, ZOOM_OUT, ANSWER)
            frames: List of frame timestamps for current observation
            current_observation: Current segment observation being viewed
            target_observation: Target segment for ZOOM_IN action
            short_side: Resize frames so shorter side has this size (-1 for no resize)

        Returns:
            Reasoning text
        """
        pass

    @abstractmethod
    def predict_clue_interval(
        self,
        question: str,
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        short_side: int = -1
    ) -> Tuple[float, float]:
        """
        Predict the time interval containing the answer to the question.

        Args:
            question: The question to answer
            frames: List of timestamps (in seconds) from the segment
            start_sec: Start time of the current segment (prediction must be >= this)
            end_sec: End time of the current segment (prediction must be <= this)
            video_path: Path to the video file
            short_side: Resize frames so shorter side has this size

        Returns:
            Tuple of (predicted_start_sec, predicted_end_sec) constrained to [start_sec, end_sec]
        """
        pass

    @abstractmethod
    def detect_query_timestamps(self, question: str) -> Optional[List[float]]:
        """
        Detect if the query contains timestamp references.

        Args:
            question: The question text

        Returns:
            List of timestamps in seconds if found, None if no timestamps detected.
        """
        pass

    @abstractmethod
    def predict_answer(
        self,
        question: str,
        choices: List[str],
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        short_side: int = -1,
        include_unanswerable: bool = False
    ) -> Dict[str, Any]:
        """
        Predict the answer to a multiple choice question and return probabilities for each option.

        Args:
            question: The question to answer
            choices: List of answer options (e.g., ["Option A", "Option B", "Option C", "Option D"])
            frames: List of timestamps (in seconds) from the segment
            start_sec: Start time of the current segment
            end_sec: End time of the current segment
            video_path: Path to the video file
            short_side: Resize frames so shorter side has this size

        Returns:
            Dict containing:
                - 'predicted_answer_char': str - character of the predicted answer (A, B, C, D, etc.)
                - 'predicted_answer': str - the predicted answer text
                - 'answer_probs': Dict[str, float] - probability for each choice keyed by character (e.g., {"A": 0.95, "B": 0.03, ...})
        """
        pass

    def decide_action(
        self,
        question: str,
        choices: List[str],
        child_segments: List[Dict[str, Any]],
        trajectory_context: Dict[str, Any],
        current_start_sec: float,
        current_end_sec: float,
        current_frames: List[float],
        video_path: str,
        short_side: int = -1,
        allowed_action_list: Optional[List[str]] = None,
        navigation_context: Optional[Dict[str, Any]] = None,
        generate_captions: Optional[bool] = None,
        skip_reasoning: bool = False,
        include_unanswerable: bool = False,
        enforce_valid_segments: bool = True
    ) -> Dict[str, Any]:
        """
        Autonomously decide the next action given child segments with frames.

        Used during inference (no GT access). The VLM sees frames from each child
        segment, trajectory history, and the question/choices, then
        produces reasoning + an action decision.

        Args:
            question: The question to answer
            choices: List of answer choices
            child_segments: List of dicts, each with:
                - 'segment_id': int
                - 'start_sec': float
                - 'end_sec': float
                - 'frames': List[float] (timestamps in seconds)
            trajectory_context: Dict containing history, video_id, question, choices
            current_start_sec: Start of the current segment
            current_end_sec: End of the current segment
            current_frames: Frame timestamps for the current segment
            video_path: Path to the video file
            short_side: Resize frames so shorter side has this size
            allowed_action_list: List of allowed action types (e.g. ["ZOOM_IN", "ZOOM_OUT", "SHIFT", "ANSWER"]).
                Callers must explicitly specify which actions are available.
            navigation_context: Optional dict with children_info, siblings, parent for nav context

        Returns:
            Dict with keys:
                - 'reasoning': str
                - 'action_type': str ('ZOOM_IN', 'ZOOM_OUT', 'SHIFT', or 'ANSWER')
                - 'segment_id': int | None (for ZOOM_IN in segmented mode)
                - 'start_sec': float | None (for ZOOM_IN, actual zoom range)
                - 'end_sec': float | None (for ZOOM_IN, actual zoom range)
                - 'answer': str | None (for ANSWER, e.g. 'A')
                - 'evidence_start': float | None (for ANSWER)
                - 'evidence_end': float | None (for ANSWER)
        """
        raise NotImplementedError("decide_action() not implemented for this VLM interface")

    def predict_keyframes(
        self,
        question: str,
        choices: List[str],
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        video_duration: float,
        short_side: int = -1,
        include_unanswerable: bool = False,
        skip_reasoning: bool = False,
        salvage_truncated_json: bool = False,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Predict specific keyframe timestamps (in seconds) and MCQ answer.

        Single-turn method: shows the model sampled frames from the video and
        asks it to identify specific timestamps where key evidence is visible,
        plus the answer to the multiple-choice question.

        Args:
            question: The question to answer
            choices: List of multiple choice options
            frames: List of frame timestamps (seconds) sampled from the video
            start_sec: Start of the video segment (typically 0)
            end_sec: End of the video segment (typically video_duration)
            video_path: Path to video file
            video_duration: Total video duration in seconds
            short_side: Frame resize dimension
            skip_reasoning: If True, omit the reasoning field from the JSON schema
            salvage_truncated_json: If True, when the response JSON is incomplete
                (e.g. truncated by max_tokens), best-effort extract any timestamps
                from the partial array instead of falling back to a single midpoint.

        Returns:
            Tuple of (parsed_result_dict, raw_response_text)
            parsed_result_dict keys:
                - 'reasoning': str
                - 'answer': str (letter, e.g. 'A')
                - 'keyframe_timestamps': List[float] (timestamps in seconds)
        """
        raise NotImplementedError("predict_keyframes() not implemented for this VLM interface")

    def direct_answer(
        self,
        question: str,
        choices: List[str],
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        short_side: int = -1,
        predict_caption: bool = False,
        skip_reasoning: bool = False,
        include_unanswerable: bool = False,
        video_native: bool = False,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Directly answer a question about a video segment and provide an evidence interval.

        Single-turn method: shows the model sampled frames and asks it to answer
        the multiple-choice question and identify the time interval containing
        the supporting evidence. No action selection, no trajectory history.

        Args:
            question: The question to answer
            choices: List of multiple choice options
            frames: List of frame timestamps (seconds) sampled from the video
            start_sec: Start of the video segment (typically 0)
            end_sec: End of the video segment (typically video_duration)
            video_path: Path to video file
            short_side: Frame resize dimension
            predict_caption: If True, also ask the model to generate a caption
                describing the video segment (for ablation; ignored by parser)
            skip_reasoning: If True, omit the reasoning field from the JSON schema
            video_native: If True, send the entire video as a single video_url
                attachment instead of interleaving sampled frames with timestamps

        Returns:
            Tuple of (parsed_result_dict, raw_response_text)
            parsed_result_dict keys:
                - 'reasoning': str
                - 'answer': str (letter, e.g. 'A')
                - 'evidence_start': float
                - 'evidence_end': float
        """
        raise NotImplementedError("direct_answer() not implemented for this VLM interface")


class DummyVLMInterface(VLMInterface):
    """
    Dummy implementation for testing and development.
    Uses simple heuristics instead of actual VLM calls.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize dummy VLM.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def generate_description(
        self,
        question: str,
        frames: List[Any],
        start_sec: float,
        end_sec: float,
        video_path: str,
        additional_context: Optional[Dict[str, Any]] = None,
        short_side: int = -1,
        detailed: bool = False
    ) -> str:
        """Generate a dummy description."""
        duration = end_sec - start_sec
        num_frames = len(frames) if frames else 0
        
        descriptions = [
            f"This segment shows {duration:.1f} seconds of video content.",
            f"The video segment contains {num_frames} frames.",
            f"Visual content spans from {start_sec:.1f}s to {end_sec:.1f}s.",
            f"A {duration:.1f}-second segment with various activities.",
        ]
        
        return np.random.choice(descriptions)
    
    def compute_relevance_score(
        self,
        question: str,
        frames: List[Any],
        start_sec: float,
        end_sec: float,
        video_path: str,
        text_description: Optional[str] = None,
        short_side: int = -1
    ) -> float:
        """
        Compute a dummy relevance score.
        In a real implementation, this would use a strong model like GPT-4V or Gemini.
        """
        # Random score with some bias
        base_score = np.random.uniform(0.3, 0.9)
        
        # Add small bias based on segment position (for testing)
        position_bias = 0.05 * np.sin(start_sec / 60)
        
        score = np.clip(base_score + position_bias, 0.0, 1.0)
        return float(score)

    def rank_segments(
        self,
        question: str,
        segments: List[Dict[str, Any]],
        short_side: int = -1
    ) -> List[int]:
        """Return a random subset of segment indices in random order."""
        n = len(segments)
        k = np.random.randint(1, max(2, n))  # pick 1 to n-1 segments
        indices = list(range(n))
        np.random.shuffle(indices)
        return indices[:k]

    def generate_reasoning(
        self,
        question: str,
        trajectory_context: Dict[str, Any],
        action_type: str,
        frames: List[float],
        current_observation: Optional[Dict[str, Any]] = None,
        target_observation: Optional[Dict[str, Any]] = None,
        short_side: int = -1,
        include_unanswerable: bool = False
    ) -> str:
        """Generate dummy reasoning."""
        if action_type == "ZOOM_IN":
            reasoning_templates = [
                "The current segment appears relevant to the question. I will zoom in to inspect it more closely.",
                "Based on the visual content, this time range might contain the answer. Let me examine it in finer detail.",
                "This segment shows promising content related to the query. I'll narrow down the search.",
            ]
        elif action_type == "ZOOM_OUT":
            reasoning_templates = [
                "The content in this segment doesn't match what I'm looking for. I should zoom out and explore a different time range.",
                "This is not the right segment. Let me return to the parent level and try a different branch.",
                "The visual content here is irrelevant to the question. I'll backtrack and search elsewhere.",
            ]
        elif action_type == "SHIFT":
            reasoning_templates = [
                "This segment doesn't contain the relevant information, but a sibling segment under the same parent might. Let me shift to explore it.",
                "The current segment is not relevant. I'll shift to a different sibling segment at the same level to continue searching.",
                "None of the content here matches the query. Let me move to an adjacent sibling segment to explore.",
            ]
        elif action_type == "ANSWER":
            reasoning_templates = [
                "I've found the relevant segment that answers the question. The visual evidence clearly shows the requested information.",
                "This segment contains exactly what the question asks for. I can provide the answer now.",
                "The search is complete. This time range contains the answer to the query.",
            ]
        else:
            reasoning_templates = ["Reasoning for action."]

        return np.random.choice(reasoning_templates)

    def predict_clue_interval(
        self,
        question: str,
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        short_side: int = -1
    ) -> Tuple[float, float]:
        """Predict random interval within bounds for testing."""
        duration = end_sec - start_sec
        # Random sub-interval (at least 10% of duration)
        min_len = duration * 0.1
        pred_len = np.random.uniform(min_len, duration)
        pred_start = np.random.uniform(start_sec, end_sec - pred_len)
        pred_end = pred_start + pred_len
        return (pred_start, pred_end)

    def detect_query_timestamps(self, question: str) -> Optional[List[float]]:
        """Return None for testing (no timestamp detection)."""
        return None

    def predict_answer(
        self,
        question: str,
        choices: List[str],
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        short_side: int = -1,
        include_unanswerable: bool = False
    ) -> Dict[str, Any]:
        """Predict a random answer for testing."""
        choices = self._maybe_add_unanswerable(choices, include_unanswerable)
        num_choices = len(choices)
        if num_choices == 0:
            return {
                'predicted_answer_char': "",
                'predicted_answer': "",
                'answer_probs': {}
            }

        # Random prediction with Dirichlet distribution (sums to 1)
        probs = np.random.dirichlet(np.ones(num_choices))
        predicted_idx = int(np.argmax(probs))
        predicted_char = chr(65 + predicted_idx)  # A, B, C, D...

        # Build probs dict with character keys and rounded values
        answer_probs = {
            chr(65 + i): round(float(p), 2)
            for i, p in enumerate(probs)
        }

        return {
            'predicted_answer_char': predicted_char,
            'predicted_answer': choices[predicted_idx],
            'answer_probs': answer_probs
        }

    def decide_action(
        self,
        question: str,
        choices: List[str],
        child_segments: List[Dict[str, Any]],
        trajectory_context: Dict[str, Any],
        current_start_sec: float,
        current_end_sec: float,
        current_frames: List[float],
        video_path: str,
        short_side: int = -1,
        allowed_action_list: Optional[List[str]] = None,
        navigation_context: Optional[Dict[str, Any]] = None,
        generate_captions: Optional[bool] = None,
        skip_reasoning: bool = False,
        include_unanswerable: bool = False,
        enforce_valid_segments: bool = True
    ) -> Dict[str, Any]:
        """Dummy decide_action for testing -- picks random action."""
        choices = self._maybe_add_unanswerable(choices, include_unanswerable)
        all_actions = ["ZOOM_IN", "ZOOM_OUT", "SHIFT", "ANSWER"]
        # Base weights: bias toward ZOOM_IN early, ANSWER later
        history_length = len(trajectory_context.get('history', []))
        if history_length >= 5:
            base_weights = {"ZOOM_IN": 0.15, "ZOOM_OUT": 0.1, "SHIFT": 0.05, "ANSWER": 0.7}
        elif history_length >= 2:
            base_weights = {"ZOOM_IN": 0.45, "ZOOM_OUT": 0.15, "SHIFT": 0.1, "ANSWER": 0.3}
        else:
            base_weights = {"ZOOM_IN": 0.75, "ZOOM_OUT": 0.1, "SHIFT": 0.05, "ANSWER": 0.1}

        allowed = allowed_action_list if allowed_action_list is not None else all_actions
        actions = [a for a in all_actions if a in set(allowed)]
        weights = np.array([base_weights[a] for a in actions])
        weights = weights / weights.sum()

        action_type = str(np.random.choice(actions, p=weights))

        result = {
            'reasoning': f"Dummy reasoning for {action_type}.",
            'action_type': action_type,
            'segment_id': None,
            'start_sec': None,
            'end_sec': None,
            'answer': None,
            'evidence_start': None,
            'evidence_end': None,
            'captions': {
                seg['segment_id']: f"Dummy caption for segment {seg['segment_id']}"
                for seg in child_segments
            },
        }

        if action_type == "ZOOM_IN" and child_segments:
            chosen = np.random.choice(len(child_segments))
            result['segment_id'] = child_segments[chosen]['segment_id']
            result['start_sec'] = child_segments[chosen]['start_sec']
            result['end_sec'] = child_segments[chosen]['end_sec']
        elif action_type == "SHIFT" and child_segments:
            chosen = np.random.choice(len(child_segments))
            result['segment_id'] = child_segments[chosen]['segment_id']
            result['start_sec'] = child_segments[chosen]['start_sec']
            result['end_sec'] = child_segments[chosen]['end_sec']
        elif action_type == "ANSWER":
            if choices:
                idx = np.random.randint(len(choices))
                result['answer'] = chr(65 + idx)
            duration = current_end_sec - current_start_sec
            result['evidence_start'] = current_start_sec + duration * 0.2
            result['evidence_end'] = current_end_sec - duration * 0.2

        # Simulate raw JSON response
        import json
        raw_response = json.dumps(result, indent=2)

        return result, raw_response

    def predict_keyframes(
        self,
        question: str,
        choices: List[str],
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        video_duration: float,
        short_side: int = -1,
        include_unanswerable: bool = False,
        skip_reasoning: bool = False,
        salvage_truncated_json: bool = False,
    ) -> Tuple[Dict[str, Any], str]:
        """Dummy predict_keyframes for testing."""
        choices = self._maybe_add_unanswerable(choices, include_unanswerable)
        import json

        idx = np.random.randint(len(choices)) if choices else 0
        answer = chr(65 + idx)

        num_kf = np.random.randint(1, 4)
        timestamps = sorted(np.random.uniform(start_sec, end_sec, num_kf).tolist())

        result = {
            'reasoning': 'Dummy keyframe prediction.',
            'answer': answer,
            'keyframe_timestamps': timestamps,
        }
        raw_response = json.dumps(result, indent=2)
        return result, raw_response

    def direct_answer(
        self,
        question: str,
        choices: List[str],
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        short_side: int = -1,
        predict_caption: bool = False,
        skip_reasoning: bool = False,
        include_unanswerable: bool = False,
        video_native: bool = False,
    ) -> Tuple[Dict[str, Any], str]:
        """Dummy direct_answer for testing -- picks random answer and evidence interval."""
        choices = self._maybe_add_unanswerable(choices, include_unanswerable)
        import json

        if choices:
            idx = np.random.randint(len(choices))
            answer = chr(65 + idx)
        else:
            answer = 'A'

        duration = end_sec - start_sec
        result = {
            'reasoning': 'Dummy direct answer.',
            'answer': answer,
            'evidence_start': start_sec + duration * 0.2,
            'evidence_end': end_sec - duration * 0.2,
        }
        raw_response = json.dumps(result, indent=2)
        return result, raw_response


class GPTVLMInterface(VLMInterface):
    """
    Interface for OpenAI GPT-4V or similar models via vLLM.
    Uses Qwen3-VL-8B-Instruct or similar vision-language models.
    """
    
    def __init__(
        self,
        base_url: str = "http://arcee:1234/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        timeout: int = 3600,
        # Reasoning generation control parameters
        reasoning_use_node_text: bool = False,
        reasoning_use_node_frames: bool = True,
        reasoning_history_turns: int = 0,
        reasoning_video_representation: str = "segmented",
        reasoning_action_representation: str = "segmented",
        prompt_log_dir: Optional[str] = None,
        memory_frames_per_node: int = 0,
        image_tokens: Optional[int] = None,
    ):
        """
        Initialize GPT VLM interface.

        Args:
            base_url: vLLM server base URL
            api_key: API key (use "EMPTY" for vLLM)
            model: Model name
            timeout: Request timeout in seconds
            reasoning_use_node_text: Include text descriptions for nodes in reasoning
            reasoning_use_node_frames: Include visual frames for nodes in reasoning
            reasoning_history_turns: Number of history turns to include (-1=all, 0=none, N=last N)
            reasoning_video_representation: Video representation mode ("continuous", "segmented", or "segmented-frames-first")
            reasoning_action_representation: Action representation mode ("segmented" for segment_id, "continuous" for timestamps)
            prompt_log_dir: Directory to save prompts to (None = disabled)
            memory_frames_per_node: Memory mode (-1=no memory, 0=caption-only, >0=N frames per visited node)
            image_tokens: Pin every image to this many vision tokens (Qwen-VL only;
                sets min_pixels=max_pixels=image_tokens*28*28 via mm_processor_kwargs).
                None = let server decide.
        """
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )

        # Store reasoning control parameters
        self.reasoning_use_node_text = reasoning_use_node_text
        self.reasoning_use_node_frames = reasoning_use_node_frames
        self.reasoning_history_turns = reasoning_history_turns
        self.reasoning_video_representation = reasoning_video_representation
        self.reasoning_action_representation = reasoning_action_representation
        self.memory_frames_per_node = memory_frames_per_node
        self.image_tokens = image_tokens

        # Prompt logging
        self.prompt_log_dir = prompt_log_dir
        self._prompt_counter = 0

        # Pre-generated captions cache
        self._caption_cache: Dict[str, Dict[float, str]] = {}

    def _load_pregenerated_captions(self, video_path: str, caption_path: str) -> Optional[Dict[float, str]]:
        """Load pre-generated captions for a video, with caching."""
        video_id = Path(video_path).stem
        if video_id in self._caption_cache:
            return self._caption_cache[video_id]
        caption_file = Path(caption_path) / f"{video_id}.json"
        if not caption_file.exists():
            self._caption_cache[video_id] = None
            return None
        with open(caption_file) as f:
            raw = json.load(f)
        captions = {float(k): v for k, v in raw.items()}
        self._caption_cache[video_id] = captions
        return captions

    def _lookup_pregenerated_caption(self, captions: Dict[float, str], timestamp: float) -> tuple:
        """Find the closest timestamp in pre-generated captions."""
        closest_ts = min(captions.keys(), key=lambda k: abs(k - timestamp))
        return closest_ts, captions[closest_ts]

    @staticmethod
    def _messages_to_text(messages: List[Dict[str, Any]]) -> Tuple[str, int]:
        """Flatten messages into plain text, replacing images with <image>.

        Returns:
            Tuple of (text, num_images).
        """
        output = []
        num_images = 0
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, str):
                output.append(content)
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        num_images += 1
                        output.append(" <image>\n")
                    elif item.get("type") == "text":
                        output.append(item["text"])
        return "".join(output), num_images

    def _create_completion(self, *, method_name: str, messages: List[Dict[str, Any]], **kwargs):
        """
        Wrapper around self.client.chat.completions.create() that optionally
        logs the prompt to disk before making the API call.
        """
        if self.prompt_log_dir is not None:
            self._prompt_counter += 1
            log_dir = Path(self.prompt_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            prompt_text, num_images = self._messages_to_text(messages)

            # Build compact header with stats
            params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            header = f"[{method_name}] images={num_images}, model={self.model}, {params}\n---\n"

            filename = f"{self._prompt_counter:06d}_{method_name}.txt"
            log_path = log_dir / filename
            with open(log_path, "w") as f:
                f.write(header + prompt_text)

        if self.image_tokens is not None:
            pixels = self.image_tokens * 28 * 28
            extra = kwargs.setdefault("extra_body", {})
            extra.setdefault("mm_processor_kwargs", {}).update(
                {"min_pixels": pixels, "max_pixels": pixels}
            )

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, **kwargs
                )
                break
            except Exception as e:
                if attempt < max_retries:
                    sleep_time = 5 * (2 ** attempt)
                    print(f"[_create_completion] Attempt {attempt + 1} failed: {e}. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise

        if self.prompt_log_dir is not None:
            with open(log_path, "a") as f:
                response_text = response.choices[0].message.content if response.choices else ""
                f.write(f"\n\n===== RESPONSE =====\n{response_text}")

        return response

    @staticmethod
    def _encode_image_array(image_array: np.ndarray) -> str:
        """
        Encode numpy image array to base64 string.
        
        Args:
            image_array: RGB image as numpy array
        
        Returns:
            Base64 encoded JPEG string
        """
        # Convert RGB to BGR for cv2
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Encode to JPEG
        success, buffer = cv2.imencode('.jpg', image_bgr)
        if not success:
            raise ValueError("Failed to encode image")
        
        # Convert to base64
        return base64.b64encode(buffer).decode('utf-8')
    
    @staticmethod
    def _extract_frames_at_timestamps(
        video_path: str, timestamps: List[float], short_side: int = -1
    ) -> Dict[float, np.ndarray]:
        """
        Extract multiple frames from video, opening the file only once.

        Args:
            video_path: Path to the video file
            timestamps: List of timestamps in seconds
            short_side: Resize frames so shorter side has this size (-1 for no resize)

        Returns:
            Dict mapping each timestamp to its RGB frame as numpy array
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video duration to clamp timestamps
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = (frame_count / fps) if fps > 0 else float('inf')

        # Process in sorted order for efficient forward seeking
        sorted_timestamps = sorted(set(timestamps))
        frames = {}

        for ts in sorted_timestamps:
            clamped_ts = min(ts, video_duration - 0.1) if fps > 0 else ts

            cap.set(cv2.CAP_PROP_POS_MSEC, clamped_ts * 1000)
            ret, frame = cap.read()

            if not ret and clamped_ts > 0.5:
                cap.set(cv2.CAP_PROP_POS_MSEC, (clamped_ts - 0.5) * 1000)
                ret, frame = cap.read()

            if not ret:
                cap.release()
                raise ValueError(f"Could not read frame at {ts}s from {video_path}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if short_side > 0:
                h, w = frame_rgb.shape[:2]
                if h < w:
                    new_h = short_side
                    new_w = int(w * (new_h / h))
                else:
                    new_w = short_side
                    new_h = int(h * (new_w / w))
                interpolation = cv2.INTER_AREA if new_h < h else cv2.INTER_LINEAR
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=interpolation)

            frames[ts] = frame_rgb

        cap.release()
        return frames
    
    def _build_video_content(
        self,
        frames: List[float],
        prompt_text: str,
        video_path: str,
        short_side: int = -1
    ) -> List[Dict[str, Any]]:
        """
        Build message content with frames and text.
        
        Args:
            frames: List of frame timestamps (in seconds)
            prompt_text: The prompt text to add at the end
            video_path: Path to the video file to extract frames from
            short_side: Resize frames so shorter side has this size (-1 for no resize)
        
        Returns:
            List of content items for the API
        """
        content = []

        # Extract all frames in one pass (single file open)1
        frame_map = self._extract_frames_at_timestamps(video_path, frames, short_side)

        # Build content in original order
        for timestamp in frames:
            content.append({
                "type": "text",
                "text": f"{timestamp:.1f}s:"
            })

            base64_image = self._encode_image_array(frame_map[timestamp])
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        # Add the main prompt text
        content.append({
            "type": "text",
            "text": prompt_text
        })

        return content

    @staticmethod
    def _description_prompt(
        timestamps: List[float], start_sec: float, end_sec: float, detailed: bool
    ) -> str:
        if len(timestamps) == 1:
            return (
                f"This is a single frame from a video at {timestamps[0]:.1f}s. "
                "Briefly describe what you see in this frame within 30 words.\n\nDescription:"
            )
        if detailed:
            return (
                f"The segment is cropped from a long video (from {start_sec:.1f}s to "
                f"{end_sec:.1f}s). Please provide a detailed description of this video "
                "segment within 50 words.\n\nDescription:"
            )
        return (
            f"The segment is cropped from a long video (from {start_sec:.1f}s to "
            f"{end_sec:.1f}s). Please provide a brief description of this video "
            "segment within 30 words.\n\nDescription:"
        )

    def _build_video_content_from_frames(
        self,
        frames: List[np.ndarray],
        timestamps: List[float],
        prompt_text: str,
    ) -> List[Dict[str, Any]]:
        """Like _build_video_content but with already-decoded RGB frames."""
        content = []
        for ts, frame in zip(timestamps, frames):
            content.append({"type": "text", "text": f"{ts:.1f}s:"})
            b64 = self._encode_image_array(frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt_text})
        return content

    def generate_description(
        self,
        question: str,
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        additional_context: Optional[Dict[str, Any]] = None,
        short_side: int = -1,
        detailed: bool = False
    ) -> str:
        """
        Generate description using VLM.

        Args:
            question: Question to answer about the video
            frames: List of timestamps (in seconds) to sample from the video
            start_sec: Start time of the segment
            end_sec: End time of the segment
            video_path: Path to the video file
            additional_context: Optional additional context
            short_side: Resize frames so shorter side has this size (-1 for no resize)
            detailed: If True, generate a detailed description; if False (default), generate a brief one

        Prompt Design:
        - Present frames with timestamps
        - Ask for detailed description of events
        - Focus on visual content and actions
        """
        if len(frames) == 0:
            return "Caption not available."
        prompt_text = self._description_prompt(frames, start_sec, end_sec, detailed)

        # Build content with frames
        content = self._build_video_content(frames, prompt_text, video_path, short_side)

        # Make API call
        response = self._create_completion(
            method_name="generate_description",
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    def generate_description_from_frames(
        self,
        question: str,
        frames: List[np.ndarray],
        timestamps: List[float],
        start_sec: float,
        end_sec: float,
        detailed: bool = False,
    ) -> str:
        """Same as generate_description but using pre-decoded RGB frames.

        Lets a caller decode each segment's frames once and reuse them for both
        the VLM caption and downstream work (e.g. CLIP scene segmentation),
        skipping a redundant cv2.VideoCapture open + seek + decode pass.
        """
        prompt_text = self._description_prompt(timestamps, start_sec, end_sec, detailed)
        content = self._build_video_content_from_frames(frames, timestamps, prompt_text)
        response = self._create_completion(
            method_name="generate_description",
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    def compute_relevance_score(
        self,
        question: str,
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        text_description: Optional[str] = None,
        short_side: int = -1
    ) -> float:
        """
        Compute relevance score using VLM logprobs.
        
        Args:
            question: The question to evaluate relevance for
            frames: List of timestamps (in seconds) to sample from the video
            start_sec: Start time of the segment
            end_sec: End time of the segment
            video_path: Path to the video file
            text_description: Optional text description of the segment
            short_side: Resize frames so shorter side has this size (-1 for no resize)
        
        Prompt Design:
        - Present the question clearly
        - Show frames with timestamps
        - Ask binary yes/no question about relevance
        - Extract probability from logprobs
        """
        # Build organized prompt for relevance scoring
        prompt_text = f"""Question: {question}

Based on the visual content shown in the frames above (from {start_sec:.1f}s to {end_sec:.1f}s), does this video segment contain information relevant to answering the question?

Consider:
- Does the segment show objects, people, or events mentioned in the question?
- Does it provide visual evidence that could help answer the question?
- Is the content directly or indirectly related to the question?

IMPORTANT: Do NOT answer the question itself. Only respond with "Yes" or "No" - nothing else."""

        print(f"DEBUG: Computing relevance score: Building content with frames starts at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Build content with frames
        content = self._build_video_content(frames, prompt_text, video_path, short_side)

        print(f"DEBUG: Computing relevance score: Building content with frames ends at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"DEBUG: Computing relevance score: API call starts at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Make API call with logprobs
        response = self._create_completion(
            method_name="compute_relevance_score",
            messages=[{"role": "user", "content": content}],
            max_tokens=10,
            temperature=0.0,
            logprobs=True,
            top_logprobs=5
        )
        print(f"DEBUG: Computing relevance score: API call ends at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        # Extract relevance probability from logprobs
        try:
            # Get the first token's logprobs
            logprobs_content = response.choices[0].logprobs.content
            if not logprobs_content:
                return 0.5  # Default neutral score
            
            top_logprobs = logprobs_content[0].top_logprobs
            
            # Find "yes" and "no" tokens and their probabilities
            # Prioritize in order: Yes/No > yes/no > y/n
            yes_logprob = None
            no_logprob = None
            
            # Prioritized token lists
            yes_tokens = ["Yes", "yes", "y"]
            no_tokens = ["No", "no", "n"]
            
            # Look for yes tokens in priority order
            for yes_variant in yes_tokens:
                for logprob_item in top_logprobs:
                    if logprob_item.token.strip() == yes_variant:
                        yes_logprob = logprob_item.logprob
                        break
                if yes_logprob is not None:
                    break
            
            # Look for no tokens in priority order
            for no_variant in no_tokens:
                for logprob_item in top_logprobs:
                    if logprob_item.token.strip() == no_variant:
                        no_logprob = logprob_item.logprob
                        break
                if no_logprob is not None:
                    break
            
            # Compute probability using softmax
            if yes_logprob is not None and no_logprob is not None:
                # Use temperature for calibration
                T = 1.0
                yes_logprob_T = yes_logprob / T
                no_logprob_T = no_logprob / T
                
                # Stable softmax
                max_logprob = max(yes_logprob_T, no_logprob_T)
                yes_exp = math.exp(yes_logprob_T - max_logprob)
                no_exp = math.exp(no_logprob_T - max_logprob)
                
                yes_prob = yes_exp / (yes_exp + no_exp)
                return float(yes_prob)
            elif yes_logprob is not None:
                return 0.9  # High confidence for yes
            elif no_logprob is not None:
                return 0.1  # Low confidence (no is dominant)
            else:
                return 0.5  # Fallback neutral score
                
        except Exception as e:
            print(f"Warning: Failed to parse relevance score: {e}")
            return 0.5  # Fallback neutral score

    def rank_segments(
        self,
        question: str,
        segments: List[Dict[str, Any]],
        short_side: int = -1
    ) -> List[int]:
        """
        Show all segments to the VLM and ask it to rank the most relevant ones.

        Args:
            question: The question to evaluate relevance for
            segments: List of segment dicts with keys: frames, start_sec, end_sec,
                      video_path, text_description
            short_side: Resize frames so shorter side has this size (-1 for no resize)

        Returns:
            List of segment indices ranked by relevance (most relevant first).
        """
        if not segments:
            return []

        n = len(segments)
        prompt_text = f"""Question: {question}

You are shown {n} video segments above, each labeled with a segment number. Your task is to identify which segments are relevant to answering the question, and rank them from most relevant to least relevant.

Rules:
- Do NOT answer the question itself.
- Respond with the segment numbers as a comma-separated list, ranked from most relevant to least relevant.

Example response: 0, 5, 3"""

        # Build content by reusing _build_video_content per segment
        content = []
        for i, seg in enumerate(segments):
            # Segment header
            content.append({
                "type": "text",
                "text": f"Segment {i}: [{seg['start_sec']:.1f}s - {seg['end_sec']:.1f}s]"
            })
            # Reuse _build_video_content with empty prompt, then strip the trailing empty text
            seg_content = self._build_video_content(
                seg["frames"], "", seg["video_path"], short_side
            )
            # Remove the trailing empty-string text item appended by _build_video_content
            if seg_content and seg_content[-1].get("text") == "":
                seg_content = seg_content[:-1]
            content.extend(seg_content)

        # Append the final prompt
        content.append({
            "type": "text",
            "text": prompt_text
        })

        response = self._create_completion(
            method_name="rank_segments",
            messages=[{"role": "user", "content": content}],
            max_tokens=64,
            temperature=0.0,
        )

        # Parse response
        response_text = response.choices[0].message.content.strip()

        if response_text.lower() == "none":
            return []

        try:
            indices = []
            for token in response_text.split(","):
                token = token.strip()
                if token.isdigit():
                    idx = int(token)
                    if 0 <= idx < n and idx not in indices:
                        indices.append(idx)
            return indices
        except Exception as e:
            print(f"Warning: Failed to parse rank_segments response '{response_text}': {e}")
            return []

    def _render_tree_node(
        self,
        node_dict: Dict[str, Any],
        lines: List[str],
        indent_level: int,
        node_path: str,
        current_node_key: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Recursively render a single tree node and its children as text.

        Args:
            node_dict: Dict for this node (from Node.to_tree_dict()).
            lines: Accumulator list of output lines.
            indent_level: Current indentation depth.
            node_path: Dot-separated path label (e.g., "0.3.1"). Empty for root.
            current_node_key: (start_sec, end_sec) tuple to mark the current node with [current].
        """
        start = node_dict['start_sec']
        end = node_dict['end_sec']
        visited = node_dict.get('visited', False)
        description = node_dict.get('description', '')

        indent = "  " * indent_level
        visited_tag = "[visited]" if visited else "[not visited]"

        # Mark current node
        current_tag = ""
        if current_node_key and abs(start - current_node_key[0]) < 0.1 and abs(end - current_node_key[1]) < 0.1:
            current_tag = " [current]"

        # Build the line
        if not node_path:
            # Root node: no "Child" prefix
            line = f"{indent}[{start:.0f}s-{end:.0f}s] {visited_tag}{current_tag}"
        else:
            line = f"{indent}Child {node_path}: [{start:.0f}s-{end:.0f}s] {visited_tag}{current_tag}"

        # Append description if non-empty
        if description:
            line += f', Description: "{description}"'

        lines.append(line)

        # Recurse into children
        for child in node_dict.get('children', []):
            child_id = child['node_id']
            if not node_path:
                child_path = str(child_id)
            else:
                child_path = f"{node_path}.{child_id}"
            self._render_tree_node(child, lines, indent_level + 1, child_path, current_node_key)

    def _build_tree_structure_text(self, tree_dict: Dict[str, Any], current_node_key: Optional[Tuple[float, float]] = None) -> str:
        """
        Render a tree dict as indented text.

        Args:
            tree_dict: Tree dictionary from trajectory_context['tree'].
            current_node_key: (start_sec, end_sec) tuple to mark the current node with [current].

        Returns:
            Formatted tree structure string.
        """
        lines = []
        self._render_tree_node(tree_dict, lines, indent_level=0, node_path="", current_node_key=current_node_key)
        return "\n".join(lines)

    def _build_memory_text(self, trajectory_context: Dict[str, Any], current_node_key: Optional[Tuple[float, float]] = None) -> str:
        """
        Build memory text combining tree structure and interaction history.

        Replaces the old _build_history_text. When reasoning_history_turns == 0,
        returns empty string. Otherwise returns a formatted block with the tree
        visualization and sequential interaction history.

        Args:
            trajectory_context: Context containing 'tree' dict and 'history' list.
            current_node_key: (start_sec, end_sec) tuple to mark the current node with [current].

        Returns:
            Formatted memory string, or empty string if disabled.
        """
        if self.reasoning_history_turns == 0:
            return ""

        parts = []

        # Part 1: Tree Structure
        tree_dict = trajectory_context.get('tree')
        if tree_dict:
            tree_text = self._build_tree_structure_text(tree_dict, current_node_key=current_node_key)
            parts.append(f"Tree Structure:\n{tree_text}")

        # Part 2: Interaction History
        history = trajectory_context.get('history', [])
        if history:
            if self.reasoning_history_turns > 0:
                history = history[-self.reasoning_history_turns:]

            history_lines = []
            for idx, turn in enumerate(history, start=1):
                action = turn.get('action')
                obs = turn.get('observation')

                if obs:
                    start = obs.get('start_sec', 0)
                    end = obs.get('end_sec', 0)

                    if action:
                        line = f"- Turn {idx}: Action: {action}"
                    else:
                        line = f"- Turn {idx}: Observed Segment [{start:.0f}s-{end:.0f}s]"
                elif action:
                    line = f"- Turn {idx}: Action: {action}"
                else:
                    continue

                reasoning = turn.get('reasoning')
                if reasoning and action:
                    reasoning = str(reasoning).replace('\n', '')
                    line += f". Reasoning: {reasoning}"

                history_lines.append(line)

            if history_lines:
                parts.append("Interaction History:\n" + "\n".join(history_lines))

        if not parts:
            return ""

        return "Memory:\n\n" + "\n\n".join(parts) + "\n\n"

    @staticmethod
    def _count_visited_nodes(node_dict: Dict[str, Any]) -> int:
        """Count the number of visited nodes in a tree dict."""
        count = 1 if node_dict.get('visited', False) else 0
        for child in node_dict.get('children', []):
            count += GPTVLMInterface._count_visited_nodes(child)
        return count

    @staticmethod
    def _subsample_frames(frames: List[float], n: int) -> List[float]:
        """Uniformly subsample n frames from a list of frame timestamps."""
        if n <= 0 or not frames:
            return []
        if n >= len(frames):
            return list(frames)
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        return [frames[i] for i in indices]

    def _render_tree_node_with_frames(
        self,
        node_dict: Dict[str, Any],
        content: List[Dict[str, Any]],
        indent_level: int,
        node_path: str,
        short_side: int = -1,
        current_node_key: Optional[Tuple[float, float]] = None,
        frames_per_node: Optional[int] = None
    ) -> None:
        """
        Recursively render a tree node as content items (text + images for visited nodes).

        Similar to _render_tree_node but produces content items with frame images
        for visited nodes instead of text descriptions.
        """
        start = node_dict['start_sec']
        end = node_dict['end_sec']
        visited = node_dict.get('visited', False)

        indent = "  " * indent_level
        visited_tag = "[visited]" if visited else "[not visited]"

        current_tag = ""
        if current_node_key and abs(start - current_node_key[0]) < 0.1 and abs(end - current_node_key[1]) < 0.1:
            current_tag = " [current]"

        if not node_path:
            line = f"{indent}[{start:.0f}s-{end:.0f}s] {visited_tag}{current_tag}"
        else:
            line = f"{indent}Child {node_path}: [{start:.0f}s-{end:.0f}s] {visited_tag}{current_tag}"

        content.append({"type": "text", "text": line + "\n"})

        # Add subsampled frames for visited nodes
        if visited:
            frames = node_dict.get('frames', [])
            video_path = node_dict.get('video_path')
            if frames and video_path:
                effective_n = frames_per_node if frames_per_node is not None else self.memory_frames_per_node
                subsampled = self._subsample_frames(frames, effective_n)
                if subsampled:
                    frame_content = self._build_video_content(subsampled, "", video_path, short_side)
                    # Remove trailing empty text item
                    if frame_content and frame_content[-1].get("type") == "text" and not frame_content[-1].get("text", "").strip():
                        frame_content.pop()
                    content.extend(frame_content)

        # Recurse into children
        for child in node_dict.get('children', []):
            child_id = child['node_id']
            if not node_path:
                child_path = str(child_id)
            else:
                child_path = f"{node_path}.{child_id}"
            self._render_tree_node_with_frames(child, content, indent_level + 1, child_path, short_side, current_node_key, frames_per_node)

    def _build_memory_content(
        self,
        trajectory_context: Dict[str, Any],
        current_node_key: Optional[Tuple[float, float]] = None,
        short_side: int = -1
    ) -> List[Dict[str, Any]]:
        """
        Build memory as content items (text + images) for frame-based memory mode.

        Used when memory_frames_per_node > 0. Renders the tree structure with
        subsampled frame images for visited nodes instead of text captions.

        Returns:
            List of content items (text and image dicts).
        """
        content = []
        content.append({"type": "text", "text": "Memory:\n\n"})

        # Part 1: Tree Structure with frames
        tree_dict = trajectory_context.get('tree')
        if tree_dict:
            # Cap total history frames at 512 by reducing per-node frames if needed
            max_memory_frames = 512
            num_visited = self._count_visited_nodes(tree_dict)
            if num_visited > 0 and num_visited * self.memory_frames_per_node > max_memory_frames:
                effective_frames_per_node = max(1, max_memory_frames // num_visited)
            else:
                effective_frames_per_node = self.memory_frames_per_node

            content.append({"type": "text", "text": "Tree Structure:\n"})
            self._render_tree_node_with_frames(
                tree_dict, content, indent_level=0, node_path="",
                short_side=short_side, current_node_key=current_node_key,
                frames_per_node=effective_frames_per_node
            )

        # Part 2: Interaction History (same as text-based memory)
        history = trajectory_context.get('history', [])
        if history:
            if self.reasoning_history_turns > 0:
                history = history[-self.reasoning_history_turns:]
            elif self.reasoning_history_turns == 0:
                history = []

            history_lines = []
            for idx, turn in enumerate(history, start=1):
                action = turn.get('action')
                obs = turn.get('observation')

                if obs:
                    start = obs.get('start_sec', 0)
                    end = obs.get('end_sec', 0)
                    if action:
                        line = f"- Turn {idx}: Action: {action}"
                    else:
                        line = f"- Turn {idx}: Observed Segment [{start:.0f}s-{end:.0f}s]"
                elif action:
                    line = f"- Turn {idx}: Action: {action}"
                else:
                    continue

                reasoning = turn.get('reasoning')
                if reasoning and action:
                    line += f". Reasoning: {reasoning}"

                history_lines.append(line)

            if history_lines:
                content.append({
                    "type": "text",
                    "text": "\n\nInteraction History:\n" + "\n".join(history_lines) + "\n\n"
                })

        return content

    def _build_navigation_context(
        self,
        current_observation: Dict[str, Any],
        action_mode: str = "free",
    ) -> str:
        """
        Build a concise navigation context showing the agent's position and candidates.

        Always shows both children and siblings for consistency across all action types.

        Args:
            current_observation: Current node dict (may contain 'parent', 'children_info', 'siblings').
            action_mode: "restricted" (default) lists only unvisited children/siblings as valid
                targets; "free" lists all children/siblings (including visited) as valid targets.

        Returns:
            Formatted navigation context string, or empty string if insufficient data.
        """
        parts = []
        start = current_observation.get('start_sec', 0)
        end = current_observation.get('end_sec', 0)

        parts.append(f"Current Position: [{start:.1f}s-{end:.1f}s]")

        parent_info = current_observation.get('parent')
        if parent_info:
            parts.append(f"Parent: [{parent_info['start_sec']:.1f}s-{parent_info['end_sec']:.1f}s]")

        def _format(nodes):
            return ", ".join([
                f"Segment {s['segment_id']} [{s['start_sec']:.1f}s-{s['end_sec']:.1f}s]"
                for s in nodes
            ])

        # Children info (always shown if available)
        children = current_observation.get('children_info', [])
        if children:
            if action_mode == "free":
                parts.append(f"Children (valid ZOOM_IN targets): {_format(children)}")
            else:
                unvisited = [s for s in children if not s.get('visited', False)]
                if unvisited:
                    parts.append(f"Unvisited children (valid ZOOM_IN targets): {_format(unvisited)}")

        # Siblings info (always shown if available)
        siblings = current_observation.get('siblings', [])
        if siblings:
            if action_mode == "free":
                parts.append(f"Siblings (valid SHIFT targets): {_format(siblings)}")
            else:
                unvisited = [s for s in siblings if not s.get('visited', False)]
                if unvisited:
                    parts.append(f"Unvisited siblings (valid SHIFT targets): {_format(unvisited)}")

        if len(parts) <= 1:
            return ""

        return "Navigation Context:\n" + "\n".join(f"  {p}" for p in parts) + "\n\n"

    def _add_observation_to_content(
        self,
        content: List[Dict[str, Any]],
        observation: Dict[str, Any],
        video_path: str,
        label: str = "Observation",
        frames: Optional[List[float]] = None,
        short_side: int = -1
    ) -> None:
        """
        Add observation frames and/or text to content list.

        Supports two modes based on self.reasoning_video_representation:
        - "continuous": Original behavior - sample frames uniformly with absolute timestamps
        - "segmented": Divide frames into segments based on children, label as "Segment 0", etc.

        Args:
            content: Content list to append to
            observation: Observation dict with start_sec, end_sec, description, and optionally 'segments'
            video_path: Path to video file
            label: Label for this observation (e.g., "Parent Observation")
            frames: Optional pre-sampled frame timestamps to use instead of sampling new ones
            short_side: Resize frames so shorter side has this size (-1 for no resize)
        """
        start_sec = observation.get('start_sec', 0)
        end_sec = observation.get('end_sec', 0)
        description = observation.get('description', '')
        segments = observation.get('segments', None)

        # Add observation header
        content.append({
            "type": "text",
            "text": f"{label} [{start_sec:.1f}s-{end_sec:.1f}s]:"
        })

        # Determine representation mode
        use_segmented = (
            self.reasoning_video_representation in ("segmented", "segmented-frames-first")
            and segments is not None
            and len(segments) > 0
        )

        # Add frames if enabled
        if self.reasoning_use_node_frames:
            if use_segmented:
                if self.reasoning_video_representation == "segmented-frames-first":
                    # Frames-first representation: all frames then segment boundaries
                    self._add_frames_then_segments(content, frames, segments, video_path, short_side)
                else:
                    # Segmented representation: group frames by segments
                    self._add_frames_by_segments(content, frames, segments, video_path, short_side)
            else:
                # Continuous representation: use _build_video_content
                frame_content = self._build_video_content(frames, "", video_path, short_side)
                # Remove the empty prompt text at the end
                if frame_content and frame_content[-1]["type"] == "text":
                    frame_content.pop()
                content.extend(frame_content)

        # Add text description if enabled
        if self.reasoning_use_node_text and description:
            content.append({
                "type": "text",
                "text": f"Description: {description}"
            })

        content.append({
            "type": "text",
            "text": "\n"
        })

    def _add_frames_by_segments(
        self,
        content: List[Dict[str, Any]],
        frames: List[float],
        segments: List[Dict[str, Any]],
        video_path: str,
        short_side: int = -1
    ) -> None:
        """
        Add provided frames grouped by segments with segment labels.

        Args:
            content: Content list to append to
            frames: Pre-provided list of frame timestamps
            segments: List of segment dicts with segment_id, start_sec, end_sec
            video_path: Path to video file
            short_side: Resize frames so shorter side has this size (-1 for no resize)
        """
        # Group frames by which segment they belong to
        segment_frames = {seg['segment_id']: [] for seg in segments}

        for timestamp in frames:
            # Find which segment this frame belongs to
            for segment in segments:
                if segment['start_sec'] <= timestamp <= segment['end_sec']:
                    segment_frames[segment['segment_id']].append(timestamp)
                    break

        # Extract all frames in one pass (single file open)
        frame_map = self._extract_frames_at_timestamps(video_path, frames, short_side)

        # Add frames for each segment
        for segment in segments:
            segment_id = segment['segment_id']
            seg_start = segment['start_sec']
            seg_end = segment['end_sec']
            seg_frames = segment_frames[segment_id]

            if not seg_frames:
                continue  # Skip segments with no frames

            # Add segment header
            content.append({
                "type": "text",
                "text": f"Segment {segment_id} [{seg_start:.1f}s-{seg_end:.1f}s]:"
            })

            # Add each frame's timestamp label and encoded image
            for timestamp in seg_frames:
                content.append({
                    "type": "text",
                    "text": f"{timestamp:.1f}s:"
                })
                base64_image = self._encode_image_array(frame_map[timestamp])
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

    def _add_frames_then_segments(
        self,
        content: List[Dict[str, Any]],
        frames: List[float],
        segments: List[Dict[str, Any]],
        video_path: str,
        short_side: int = -1
    ) -> None:
        """
        Add all frames first with timestamp labels, then append segment boundaries as text.

        Unlike _add_frames_by_segments which interleaves segment headers with frames,
        this method presents all frames sequentially first, followed by a separate
        text block listing child segment boundaries.

        Args:
            content: Content list to append to
            frames: Pre-provided list of frame timestamps
            segments: List of segment dicts with segment_id, start_sec, end_sec
            video_path: Path to video file
            short_side: Resize frames so shorter side has this size (-1 for no resize)
        """
        # Extract all frames in one pass (single file open)
        frame_map = self._extract_frames_at_timestamps(video_path, frames, short_side)

        # Add all frames with timestamp labels (no segment grouping)
        for timestamp in frames:
            content.append({
                "type": "text",
                "text": f"{timestamp:.1f}s:"
            })
            base64_image = self._encode_image_array(frame_map[timestamp])
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        # Add segment boundaries as text
        seg_lines = [
            f"- Segment {s['segment_id']}: [{s['start_sec']:.1f}s-{s['end_sec']:.1f}s]"
            for s in segments
        ]
        content.append({
            "type": "text",
            "text": "Child segments:\n" + "\n".join(seg_lines)
        })

    def generate_reasoning(
        self,
        question: str,
        trajectory_context: Dict[str, Any],
        action_type: str,
        frames: List[float],
        current_observation: Optional[Dict[str, Any]] = None,
        target_observation: Optional[Dict[str, Any]] = None,
        short_side: int = -1,
        include_unanswerable: bool = False
    ) -> str:
        """
        Generate reasoning using VLM with action-specific prompts.

        This method generates reasoning that respects the "no privileged information" principle:
        - ZOOM_IN: Only uses current observation (no target segment frames/text)
        - ZOOM_OUT: Only uses current observation
        - ANSWER: Only uses current (evidence) observation

        Args:
            question: The question being answered
            trajectory_context: Context from trajectory (includes history, video_id, etc.)
            action_type: Type of action (ZOOM_IN, ZOOM_OUT, ANSWER)
            frames: List of frame timestamps to use for current observation
            current_observation: Current segment observation being viewed
            target_observation: Target segment for ZOOM_IN action
            short_side: Resize frames so shorter side has this size (-1 for no resize)

        Returns:
            Reasoning text explaining why the action is taken  
        """
        # Determine video path from observations
        video_path = None
        if current_observation:
            video_path = current_observation.get('video_path')
        
        if not video_path:
            # Fallback to simple template if no video path
            return self._generate_fallback_reasoning(action_type)
        
        try:
            # Build content based on action type
            content = []

            # System prompt header
            system_header = (
                "You are a video question-answering agent that navigates a long video through "
                "multi-turn hierarchical interaction. At each turn you observe a video segment "
                "(which may be divided into sub-segments) and take one of the following actions:\n"
                "- ZOOM_IN <Segment ID>: Examine a sub-segment at finer granularity.\n"
                "- ZOOM_OUT: Backtrack to the parent segment to explore elsewhere.\n"
                "- SHIFT <Segment ID>: Shift to a sibling segment under the same parent.\n"
                "- ANSWER <answer> <evidence_start> <evidence_end>: Provide the final answer along with the evidence time interval once sufficient evidence is found.\n\n"
                "The next action has already been decided. Your task is to analyze the visual "
                "content and memory, then provide reasoning that justifies the action.\n\n"
            )
            content.append({
                "type": "text",
                "text": system_header
            })

            # Build memory text with current node marked in tree
            current_node_key = None
            if current_observation:
                current_node_key = (current_observation.get('start_sec', 0), current_observation.get('end_sec', 0))
            memory_text = self._build_memory_text(trajectory_context, current_node_key=current_node_key)

            # Build navigation context for action-specific guidance
            nav_context = ""
            if current_observation:
                nav_context = self._build_navigation_context(current_observation)

            # Action-specific methods handle ordering:
            # frames → memory → navigation context → question → action instruction
            if action_type == "ZOOM_IN":
                return self._generate_zoom_in_reasoning(
                    content, current_observation, target_observation, video_path, frames, short_side,
                    question=question, memory_text=memory_text, nav_context=nav_context
                )
            elif action_type == "ZOOM_OUT":
                return self._generate_zoom_out_reasoning(
                    content, current_observation, video_path, frames, short_side,
                    question=question, memory_text=memory_text, nav_context=nav_context
                )
            elif action_type == "SHIFT":
                return self._generate_shift_reasoning(
                    content, current_observation, target_observation, video_path, frames, short_side,
                    question=question, memory_text=memory_text, nav_context=nav_context
                )
            elif action_type == "ANSWER":
                # Extract answer from trajectory context
                answer = trajectory_context.get('answer', 'Unknown')
                right_answer = trajectory_context.get('right_answer', 'Unknown')
                choices = trajectory_context.get('choices', [])
                gt_start, gt_end = trajectory_context.get('gt_timestamps', (0, 0))
                return self._generate_answer_reasoning(
                    content, current_observation, video_path, question, answer, right_answer, choices, frames, gt_start, gt_end, short_side,
                    memory_text=memory_text, nav_context=nav_context, include_unanswerable=include_unanswerable
                )
            else:
                return f"Taking action: {action_type}"
                
        except Exception as e:
            print(f"Error generating reasoning: {e}")
            return self._generate_fallback_reasoning(action_type)
    
    def _generate_fallback_reasoning(self, action_type: str) -> str:
        """Generate simple template-based reasoning as fallback."""
        if action_type == "ZOOM_IN":
            return "Based on the visual content, this segment appears relevant. Zooming in to examine more closely."
        elif action_type == "ZOOM_OUT":
            return "This segment does not contain the relevant information. Zooming out to explore other time ranges."
        elif action_type == "SHIFT":
            return "This segment is not relevant. Shifting to a sibling segment to continue the search."
        elif action_type == "ANSWER":
            return "Found the segment that answers the question. Providing the answer based on visual evidence."
        else:
            return f"Taking action: {action_type}"

    def _generate_zoom_in_reasoning(
        self,
        content: List[Dict[str, Any]],
        current_observation: Optional[Dict[str, Any]],
        target_observation: Optional[Dict[str, Any]],
        video_path: str,
        frames: List[float],
        short_side: int = -1,
        question: str = "",
        memory_text: str = "",
        nav_context: str = ""
    ) -> str:
        """
        Generate ZOOM_IN reasoning.

        Uses only current observation (no target segment frames/text - that's privileged!).
        Component order: frames → memory → navigation context → question → action instruction.
        """
        if not current_observation or not target_observation:
            return self._generate_fallback_reasoning("ZOOM_IN")

        print(f"DEBUG: Generating ZOOM_IN reasoning: Current observation starts at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. Add current observation frames first
        self._add_observation_to_content(
            content, current_observation, video_path, "Current Observation", frames, short_side
        )

        print(f"DEBUG: Generating ZOOM_IN reasoning: Current observation ends at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 2. Add memory (tree structure + interaction history)
        if memory_text:
            content.append({
                "type": "text",
                "text": memory_text
            })

        # 3. Add navigation context
        if nav_context:
            content.append({
                "type": "text",
                "text": nav_context
            })

        # Extract target segment info (just timestamps, no frames/text!)
        target_start = target_observation.get('start_sec', 0)
        target_end = target_observation.get('end_sec', 0)
        target_id = target_observation.get('node_id', 0)

        # Determine if we're using segmented representation
        segments = current_observation.get('segments', None)
        use_segmented = (
            self.reasoning_video_representation in ("segmented", "segmented-frames-first")
            and segments is not None
            and len(segments) > 0
        )

        # 4. Add question + action instruction
        if use_segmented:
            segment_list = ", ".join([
                f"Segment {s['segment_id']} [{s['start_sec']:.1f}s-{s['end_sec']:.1f}s]"
                for s in segments
            ])

            prompt = f"""Question: {question}

The current segment has been divided into: {segment_list}.

The next action has been decided: ZOOM_IN {target_id}. Based on the visual content of each segment, the memory, and the navigation context, explain why this segment is the most likely to contain information relevant to answering the question and should be examined at finer granularity.

Your response must end with the action: ZOOM_IN {target_id}

Write concise reasoning (2-4 sentences) that justifies zooming into this segment and concludes with the action.

Your reasoning:"""
        else:
            prompt = f"""Question: {question}

The next action has been decided: ZOOM_IN [{target_start:.1f}s-{target_end:.1f}s]. Based on the visual content, the memory, and the navigation context, explain why this time range is the most likely to contain information relevant to answering the question and should be examined at finer granularity.

Your response must end with the action: ZOOM_IN [{target_start:.1f}s-{target_end:.1f}s]

Write concise reasoning (2-4 sentences) that justifies zooming into this time range and concludes with the action.

Your reasoning:"""

        content.append({
            "type": "text",
            "text": prompt
        })

        print(f"DEBUG: Generating ZOOM_IN reasoning: API call starts at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Make API call
        response = self._create_completion(
            method_name="generate_zoom_in_reasoning",
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
            temperature=0.7
        )
        print(f"DEBUG: Generating ZOOM_IN reasoning: API call ends at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        return response.choices[0].message.content.strip()
    
    def _generate_zoom_out_reasoning(
        self,
        content: List[Dict[str, Any]],
        current_observation: Optional[Dict[str, Any]],
        video_path: str,
        frames: List[float],
        short_side: int = -1,
        question: str = "",
        memory_text: str = "",
        nav_context: str = ""
    ) -> str:
        """
        Generate ZOOM_OUT reasoning.

        Uses only current observation.
        Component order: frames → memory → navigation context → question → action instruction.
        """
        if not current_observation:
            return self._generate_fallback_reasoning("ZOOM_OUT")

        # 1. Add current observation frames first
        self._add_observation_to_content(
            content, current_observation, video_path, "Current Observation", frames, short_side
        )

        # 2. Add memory (tree structure + interaction history)
        if memory_text:
            content.append({
                "type": "text",
                "text": memory_text
            })

        # 3. Add navigation context
        if nav_context:
            content.append({
                "type": "text",
                "text": nav_context
            })

        # Determine if we're using segmented representation
        segments = current_observation.get('segments', None)
        use_segmented = (
            self.reasoning_video_representation in ("segmented", "segmented-frames-first")
            and segments is not None
            and len(segments) > 0
        )

        # 4. Add question + action instruction
        if use_segmented:
            segment_list = ", ".join([
                f"Segment {s['segment_id']} [{s['start_sec']:.1f}s-{s['end_sec']:.1f}s]"
                for s in segments
            ])
            prompt = f"""Question: {question}

The current segment contains the following segments: {segment_list}.

The next action has been decided: ZOOM_OUT (backtrack to the parent segment). Based on the visual content of each segment, the memory, and the navigation context, explain why none of these segments contain information relevant to answering the question, making it necessary to backtrack and explore a different region.

Your response must end with the action: ZOOM_OUT

Write concise reasoning (2-4 sentences) that justifies backtracking and concludes with the action.

Your reasoning:"""
        else:
            prompt = f"""Question: {question}

The next action has been decided: ZOOM_OUT (backtrack to the parent segment). Based on the visual content, the memory, and the navigation context, explain why this segment does not contain information relevant to answering the question, making it necessary to backtrack and explore a different region.

Your response must end with the action: ZOOM_OUT

Write concise reasoning (2-4 sentences) that justifies backtracking and concludes with the action.

Your reasoning:"""

        content.append({
            "type": "text",
            "text": prompt
        })

        # Make API call
        response = self._create_completion(
            method_name="generate_zoom_out_reasoning",
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    def _generate_shift_reasoning(
        self,
        content: List[Dict[str, Any]],
        current_observation: Optional[Dict[str, Any]],
        target_observation: Optional[Dict[str, Any]],
        video_path: str,
        frames: List[float],
        short_side: int = -1,
        question: str = "",
        memory_text: str = "",
        nav_context: str = ""
    ) -> str:
        """
        Generate SHIFT reasoning.

        Uses current observation to explain why shifting to a sibling segment.
        Component order: frames → memory → navigation context → question → action instruction.
        """
        if not current_observation or not target_observation:
            return self._generate_fallback_reasoning("SHIFT")

        # 1. Add current observation frames first
        self._add_observation_to_content(
            content, current_observation, video_path, "Current Observation", frames, short_side
        )

        # 2. Add memory (tree structure + interaction history)
        if memory_text:
            content.append({
                "type": "text",
                "text": memory_text
            })

        # 3. Add navigation context
        if nav_context:
            content.append({
                "type": "text",
                "text": nav_context
            })

        # Extract target segment info
        target_start = target_observation.get('start_sec', 0)
        target_end = target_observation.get('end_sec', 0)
        target_id = target_observation.get('node_id', 0)

        # 4. Add question + action instruction
        if self.reasoning_action_representation == "segmented":
            prompt = f"""Question: {question}

The next action has been decided: SHIFT {target_id} (move to a sibling segment). Based on the visual content, the memory, and the navigation context, explain why the current segment is not relevant and why shifting to this sibling segment is warranted.

Your response must end with the action: SHIFT {target_id}

Write concise reasoning (2-4 sentences) that justifies shifting to the sibling segment and concludes with the action.

Your reasoning:"""
        else:
            prompt = f"""Question: {question}

The next action has been decided: SHIFT [{target_start:.1f}s-{target_end:.1f}s] (move to a sibling segment). Based on the visual content, the memory, and the navigation context, explain why the current segment is not relevant and why shifting to this sibling segment is warranted.

Your response must end with the action: SHIFT [{target_start:.1f}s-{target_end:.1f}s]

Write concise reasoning (2-4 sentences) that justifies shifting to the sibling segment and concludes with the action.

Your reasoning:"""

        content.append({
            "type": "text",
            "text": prompt
        })

        # Make API call
        response = self._create_completion(
            method_name="generate_shift_reasoning",
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    def _generate_answer_reasoning(
        self,
        content: List[Dict[str, Any]],
        current_observation: Optional[Dict[str, Any]],
        video_path: str,
        question: str,
        answer: str,
        right_answer: int,
        choices: List[str],
        frames: List[float],
        gt_start: float,
        gt_end: float,
        short_side: int = -1,
        memory_text: str = "",
        nav_context: str = "",
        include_unanswerable: bool = False
    ) -> str:
        """
        Generate ANSWER reasoning.

        Uses only current (evidence) observation.
        Component order: frames → memory → navigation context → question → action instruction.
        """
        if not current_observation:
            return self._generate_fallback_reasoning("ANSWER")

        # 1. Add evidence observation frames first
        self._add_observation_to_content(
            content, current_observation, video_path, "Current Observation", frames, short_side
        )

        # 2. Add memory (tree structure + interaction history)
        if memory_text:
            content.append({
                "type": "text",
                "text": memory_text
            })

        # 3. Add navigation context
        if nav_context:
            content.append({
                "type": "text",
                "text": nav_context
            })

        # Determine if we're using segmented representation
        segments = current_observation.get('segments', None)
        use_segmented = (
            self.reasoning_video_representation in ("segmented", "segmented-frames-first")
            and segments is not None
            and len(segments) > 0
        )

        # Format multiple choice options
        choices = self._maybe_add_unanswerable(choices, include_unanswerable)
        formatted_choices = "\n".join([
            f"{chr(65 + i)}. {choice}"
            for i, choice in enumerate(choices)
        ])

        # 3. Add question + action instruction
        if use_segmented:
            segment_list = ", ".join([
                f"Segment {s['segment_id']} [{s['start_sec']:.1f}s-{s['end_sec']:.1f}s]"
                for s in segments
            ])
            prompt = f"""The current segment contains the following segments: {segment_list}.

Question: {question}
Choices:
{formatted_choices}

The correct answer is {right_answer}. {answer}, and the relevant evidence appears in the interval [{gt_start:.1f}s-{gt_end:.1f}s].

Based on the visual evidence from the segments shown above and the memory, explain what you observe that supports this answer and why the evidence interval contains the relevant information.

Your response must end with: ANSWER {right_answer} {gt_start:.1f} {gt_end:.1f}

Write concise reasoning (2-4 sentences) that describes the visual evidence and concludes with the action.

Your reasoning:"""
        else:
            prompt = f"""Question: {question}
Choices:
{formatted_choices}

The correct answer is {right_answer}. {answer}, and the relevant evidence appears in the interval [{gt_start:.1f}s-{gt_end:.1f}s].

Based on the visual evidence shown above and the memory, explain what you observe that supports this answer and why the evidence interval contains the relevant information.

Your response must end with: ANSWER {right_answer} {gt_start:.1f} {gt_end:.1f}

Write concise reasoning (2-4 sentences) that describes the visual evidence and concludes with the action.

Your reasoning:"""

        content.append({
            "type": "text",
            "text": prompt
        })

        # Make API call
        response = self._create_completion(
            method_name="generate_answer_reasoning",
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    def predict_clue_interval(
        self,
        question: str,
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        short_side: int = -1
    ) -> Tuple[float, float]:
        """
        Predict the time interval containing the answer to the question.

        Uses the VLM to analyze frames and predict which time range contains
        the answer. The prediction is constrained to be within [start_sec, end_sec].

        Args:
            question: The question to answer
            frames: List of timestamps (in seconds) from the segment
            start_sec: Start time of the current segment
            end_sec: End time of the current segment
            video_path: Path to the video file
            short_side: Resize frames so shorter side has this size

        Returns:
            Tuple of (predicted_start_sec, predicted_end_sec)
        """
        import json
        import re

        # Build prompt for clue interval prediction
        prompt_text = f"""Question: {question}

You are viewing a video segment from {start_sec:.1f}s to {end_sec:.1f}s. Based on the visual content in the frames above, predict the time interval within this segment that most likely contains the answer to the question.

IMPORTANT: Do not answer the question directly. You only need to predict the interval. Your predicted interval must be within [{start_sec:.1f}, {end_sec:.1f}].

Respond with ONLY a JSON object in this exact format (no other text):
{{"start": <start_time>, "end": <end_time>}}

Example response: {{"start": 10.5, "end": 25.0}}"""

        # Build content with frames
        content = self._build_video_content(frames, prompt_text, video_path, short_side)

        # Make API call
        response = self._create_completion(
            method_name="predict_clue_interval",
            messages=[{"role": "user", "content": content}],
            max_tokens=100,
            temperature=0.3  # Lower temperature for more deterministic output
        )

        response_text = response.choices[0].message.content.strip()

        # Parse the response
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                pred_start = float(result.get('start', start_sec))
                pred_end = float(result.get('end', end_sec))
            else:
                # Fallback: try to parse the entire response as JSON
                result = json.loads(response_text)
                pred_start = float(result.get('start', start_sec))
                pred_end = float(result.get('end', end_sec))
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Warning: Failed to parse clue prediction response: {response_text}. Error: {e}")
            # Fallback to full interval
            return (start_sec, end_sec)

        # Clamp to valid range
        pred_start = max(start_sec, min(pred_start, end_sec))
        pred_end = max(start_sec, min(pred_end, end_sec))

        # Ensure start < end
        if pred_start >= pred_end:
            # Swap if needed, or return full interval as fallback
            if pred_start > pred_end:
                pred_start, pred_end = pred_end, pred_start
            else:
                # They're equal, return full interval
                return (start_sec, end_sec)

        return (pred_start, pred_end)

    def detect_query_timestamps(self, question: str) -> Optional[List[float]]:
        """
        Use the VLM to detect timestamp references in the query.

        Handles formats like [[10:42]], "1 minute 22 seconds", "at 5 minutes", etc.

        Returns:
            List of timestamps in seconds if found, None if no timestamps detected.
        """
        import json
        import re

        prompt_text = f"""Analyze the following question and extract any explicit timestamp references that indicate a specific moment in the video.

Question: {question}

Timestamps can appear in formats like:
- [[10:42]], [[1:22:30]]
- "1 minute 22 seconds", "at 5 minutes"
- "10:42 in the video", "at the 30 second mark"

If there are timestamps, respond with ONLY a JSON object:
{{"timestamps": [<seconds>, ...]}}

If there are NO timestamps in the question, respond with:
{{"timestamps": null}}

Examples:
- "What happened at [[10:42]]?" -> {{"timestamps": [642.0]}}
- "What are the words at 1 minute 22 seconds?" -> {{"timestamps": [82.0]}}
- "What is the person doing?" -> {{"timestamps": null}}"""

        response = self._create_completion(
            method_name="detect_query_timestamps",
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=100,
            temperature=0.0
        )

        response_text = response.choices[0].message.content.strip()

        try:
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response_text)

            timestamps = result.get('timestamps', None)
            if timestamps is None:
                return None
            if isinstance(timestamps, (int, float)):
                timestamps = [float(timestamps)]
            return [float(t) for t in timestamps] if timestamps else None
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Warning: Failed to parse timestamp detection response: {response_text}. Error: {e}")
            return None

    def predict_answer(
        self,
        question: str,
        choices: List[str],
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        short_side: int = -1,
        include_unanswerable: bool = False
    ) -> Dict[str, Any]:
        """
        Predict the answer using VLM logprobs for multiple choice.

        Uses logprobs to extract probability for each option token (A, B, C, D, etc.)
        similar to compute_relevance_score but for multiple options.
        """
        choices = self._maybe_add_unanswerable(choices, include_unanswerable)
        num_choices = len(choices)
        if num_choices == 0:
            return {
                'predicted_answer_char': "",
                'predicted_answer': "",
                'answer_probs': {}
            }

        # Format choices with letter labels
        formatted_choices = "\n".join([
            f"{chr(65 + i)}. {choice}"
            for i, choice in enumerate(choices)
        ])

        prompt_text = f"""Question: {question}

Choices:
{formatted_choices}

Based on the visual content shown in the frames above (from {start_sec:.1f}s to {end_sec:.1f}s), which answer best answers the question?

IMPORTANT: Respond with ONLY the letter (A, B, C, D, etc.) of your answer - nothing else."""

        # Build content with frames
        content = self._build_video_content(frames, prompt_text, video_path, short_side)

        # Make API call with logprobs
        response = self._create_completion(
            method_name="predict_answer",
            messages=[{"role": "user", "content": content}],
            max_tokens=10,
            temperature=0.0,
            logprobs=True,
            top_logprobs=15  # Get more logprobs to capture all options
        )

        # Extract probabilities from logprobs
        try:
            logprobs_content = response.choices[0].logprobs.content
            if not logprobs_content:
                # Fallback: uniform distribution
                uniform_prob = round(1.0 / num_choices, 2)
                answer_probs = {chr(65 + i): uniform_prob for i in range(num_choices)}
                return {
                    'predicted_answer_char': 'A',
                    'predicted_answer': choices[0],
                    'answer_probs': answer_probs
                }

            top_logprobs = logprobs_content[0].top_logprobs

            # Build logprob lookup for each option letter
            # Tokens to search for: A, B, C, D... (uppercase and lowercase)
            option_logprobs = []
            for i in range(num_choices):
                letter_upper = chr(65 + i)  # A, B, C, D...
                letter_lower = chr(97 + i)  # a, b, c, d...

                logprob = None
                for logprob_item in top_logprobs:
                    token = logprob_item.token.strip()
                    if token == letter_upper or token == letter_lower:
                        logprob = logprob_item.logprob
                        break

                # If not found in top logprobs, assign very low probability
                if logprob is None:
                    logprob = -100.0  # Very low logprob

                option_logprobs.append(logprob)

            # Convert logprobs to probabilities using softmax
            max_logprob = max(option_logprobs)
            exp_logprobs = [math.exp(lp - max_logprob) for lp in option_logprobs]
            sum_exp = sum(exp_logprobs)
            probs = [e / sum_exp for e in exp_logprobs]

            # Find predicted answer (highest probability)
            predicted_idx = probs.index(max(probs))
            predicted_char = chr(65 + predicted_idx)  # A, B, C, D...

            # Build probs dict with character keys and rounded values
            answer_probs = {
                chr(65 + i): round(p, 2)
                for i, p in enumerate(probs)
            }

            return {
                'predicted_answer_char': predicted_char,
                'predicted_answer': choices[predicted_idx],
                'answer_probs': answer_probs
            }

        except Exception as e:
            print(f"Warning: Failed to parse answer prediction: {e}")
            uniform_prob = round(1.0 / num_choices, 2) if num_choices > 0 else 0.0
            answer_probs = {chr(65 + i): uniform_prob for i in range(num_choices)} if num_choices > 0 else {}
            return {
                'predicted_answer_char': 'A' if choices else "",
                'predicted_answer': choices[0] if choices else "",
                'answer_probs': answer_probs
            }

    def decide_action(
        self,
        question: str,
        choices: List[str],
        child_segments: List[Dict[str, Any]],
        trajectory_context: Dict[str, Any],
        current_start_sec: float,
        current_end_sec: float,
        current_frames: List[float],
        video_path: str,
        short_side: int = -1,
        allowed_action_list: Optional[List[str]] = None,
        navigation_context: Optional[Dict[str, Any]] = None,
        generate_captions: bool = True,
        skip_reasoning: bool = False,
        include_unanswerable: bool = False,
        pregenerated_caption_path: Optional[str] = None,
        use_frame_captions: bool = False,
        enforce_valid_segments: bool = True
    ) -> Dict[str, Any]:
        """
        Autonomously decide the next action given child segments with frames.

        Builds a multimodal prompt showing child segment frames, trajectory
        history, question/choices, and asks the VLM to choose an action.

        Args:
            question: Question to answer
            choices: List of multiple choice options
            child_segments: List of child segment dicts with frames
            trajectory_context: Dict containing history, video_id, question, choices
            current_start_sec: Start time of current segment
            current_end_sec: End time of current segment
            current_frames: Frame timestamps for the current segment
            video_path: Path to video file
            short_side: Frame resize dimension
            allowed_action_list: List of allowed action types (e.g. ["ZOOM_IN", "ANSWER"]).
            navigation_context: Optional dict with children_info, siblings, parent for nav context
            generate_captions: Whether to ask the VLM to generate captions for child segments
            skip_reasoning: If True, omit the reasoning field from the JSON schema
            enforce_valid_segments: If True, explicitly list valid (unvisited) segment IDs in the prompt

        Returns:
            Dict with reasoning, action_type, segment_id/start_sec/end_sec, answer, evidence, captions
        """
        import json
        import re

        # Reasoning field for JSON schema (empty when skip_reasoning is set)
        reasoning_field = '' if skip_reasoning else '  "reasoning": "<your concise reasoning>",\n'

        content = []

        # === STEP 1: System header (role + action list only, rules moved to end) ===
        allowed_actions = set(allowed_action_list) if allowed_action_list else {"ZOOM_IN", "ZOOM_OUT", "SHIFT", "ANSWER"}
        is_segmented = self.reasoning_action_representation == "segmented"
        has_navigation = "ZOOM_IN" in allowed_actions or "ZOOM_OUT" in allowed_actions or "SHIFT" in allowed_actions

        # Collect valid (unvisited) segment IDs when enforce_valid_segments is enabled
        valid_zoom_in_ids = []
        valid_shift_ids = []
        if enforce_valid_segments and is_segmented and navigation_context:
            valid_zoom_in_ids = [
                c['segment_id'] for c in navigation_context.get('children_info', [])
                if not c.get('visited', False)
            ]
            valid_shift_ids = [
                s['segment_id'] for s in navigation_context.get('siblings', [])
                if not s.get('visited', False)
            ]

        # Build action descriptions
        action_descs = []
        action_idx = 1
        if "ZOOM_IN" in allowed_actions:
            if is_segmented:
                action_descs.append(f"{action_idx}. ZOOM_IN <segment_id>: Examine a child segment at finer granularity.")
            else:
                action_descs.append(f"{action_idx}. ZOOM_IN <start_sec> <end_sec>: Examine a specific time range at finer granularity.")
            action_idx += 1
        if "ZOOM_OUT" in allowed_actions:
            if is_segmented:
                action_descs.append(f"{action_idx}. ZOOM_OUT: Backtrack to the parent segment to explore a different region.")
            else:
                action_descs.append(f"{action_idx}. ZOOM_OUT: Backtrack to the parent segment to explore a different time range.")
            action_idx += 1
        if "SHIFT" in allowed_actions:
            if is_segmented:
                action_descs.append(f"{action_idx}. SHIFT <segment_id>: Move to a sibling segment under the same parent.")
            else:
                action_descs.append(f"{action_idx}. SHIFT <start_sec> <end_sec>: Move to a sibling time range under the same parent.")
            action_idx += 1
        if "ANSWER" in allowed_actions:
            action_descs.append(
                f"{action_idx}. ANSWER <letter> <evidence_start> <evidence_end>: Provide the answer to the "
                "question along with the time interval (in seconds) that contains the supporting evidence."
            )
            action_idx += 1

        num_actions = len(action_descs)
        action_list_text = "\n".join(action_descs) + "\n\n"

        if has_navigation:
            count_word = {1: "the following action", 2: "one of two actions", 3: "one of three actions", 4: "one of four actions"}.get(num_actions, f"one of {num_actions} actions")
            if is_segmented:
                system_text = (
                    "You are a video question-answering agent that navigates a long video through "
                    "hierarchical temporal search. At each step, the current video segment is divided "
                    "into non-overlapping child segments. You observe frames from each child segment "
                    f"and decide {count_word}:\n\n"
                    + action_list_text
                )
            else:
                system_text = (
                    "You are a video question-answering agent that navigates a long video through "
                    "temporal search. You observe frames from the current video segment with timestamps "
                    f"and decide {count_word}:\n\n"
                    + action_list_text
                )
        else:
            # Answer-only mode
            system_text = (
                "You are a video question-answering agent. The video has been divided into "
                "non-overlapping segments. You observe frames from each segment and must "
                "directly provide the final answer along with the time interval containing "
                "the evidence.\n\n"
                "Your task:\n"
                + action_list_text
            )

        # Build important rules
        important_rule_parts = []
        if "ZOOM_IN" in allowed_actions:
            if is_segmented:
                zoom_in_rule = (
                    "- Only use ZOOM_IN when you believe a specific child segment is likely to "
                    "contain the answer and needs closer examination."
                )
                if valid_zoom_in_ids:
                    zoom_in_rule += f" You MUST select one of the following unvisited child segment IDs: {valid_zoom_in_ids}."
                important_rule_parts.append(zoom_in_rule)
            else:
                important_rule_parts.append(
                    "- Use ZOOM_IN when you believe a specific time range is likely to "
                    "contain the answer and needs closer examination. Specify the exact start and end times."
                )
        if "ZOOM_OUT" in allowed_actions:
            if is_segmented:
                important_rule_parts.append("- Use ZOOM_OUT when none of the current child segments appear relevant.")
            else:
                important_rule_parts.append("- Use ZOOM_OUT when the current time range does not appear relevant.")
        if "SHIFT" in allowed_actions:
            if is_segmented:
                shift_rule = (
                    "- Use SHIFT to move to a sibling segment at the same level when the current "
                    "segment is not relevant but a sibling might be."
                )
                if valid_shift_ids:
                    shift_rule += f" You MUST select one of the following unvisited sibling segment IDs: {valid_shift_ids}."
                important_rule_parts.append(shift_rule)
            else:
                important_rule_parts.append(
                    "- Use SHIFT to move to a sibling time range at the same level when the current "
                    "range is not relevant but a sibling might be."
                )
        if "ANSWER" in allowed_actions:
            if has_navigation:
                important_rule_parts.append(
                    "- Use ANSWER when you are confident you have found sufficient evidence to "
                    "answer the question. You must specify the answer letter AND the evidence "
                    "time interval."
                )
            else:
                important_rule_parts.append(
                    "You must provide your answer now based on the visual content shown."
                )

        if has_navigation:
            important_rules = "IMPORTANT RULES:\n" + "\n".join(important_rule_parts) + "\n"
        else:
            important_rules = "IMPORTANT: " + "\n".join(important_rule_parts) + "\n"

        content.append({"type": "text", "text": system_text})

        # === STEP 2: Current Observation frames/images ===
        current_observation = {
            'start_sec': current_start_sec,
            'end_sec': current_end_sec,
            'description': '',  # No description for current observation
            'segments': [
                {
                    'segment_id': seg['segment_id'],
                    'start_sec': seg['start_sec'],
                    'end_sec': seg['end_sec'],
                }
                for seg in child_segments
            ] if self.reasoning_video_representation in ("segmented", "segmented-frames-first") else None
        }

        self._add_observation_to_content(
            content, current_observation, video_path,
            label="Current Segment", frames=current_frames, short_side=short_side
        )

        # === STEP 2.5: Frame captions for child segments (pregenerated) ===
        if use_frame_captions and pregenerated_caption_path and child_segments:
            captions = self._load_pregenerated_captions(video_path, pregenerated_caption_path)
            if captions:
                caption_lines = [f"Frame Captions [{current_start_sec:.1f}s-{current_end_sec:.1f}s]:"]
                for seg in child_segments:
                    seg_id = seg['segment_id']
                    seg_start = seg['start_sec']
                    seg_end = seg['end_sec']
                    caption_lines.append(f"Segment {seg_id} [{seg_start:.1f}s-{seg_end:.1f}s]:")
                    seg_frames = [t for t in current_frames if seg_start <= t <= seg_end]
                    for ts in seg_frames:
                        matched_ts, caption_text = self._lookup_pregenerated_caption(captions, ts)
                        caption_lines.append(f"[{matched_ts:.1f}s]: {caption_text}")
                content.append({
                    "type": "text",
                    "text": "\n".join(caption_lines)
                })

        # === STEP 3: Memory (tree structure + interaction history) ===
        current_node_key = (current_start_sec, current_end_sec)
        if self.memory_frames_per_node == -1:
            pass  # no memory at all
        elif self.memory_frames_per_node > 0:
            memory_items = self._build_memory_content(
                trajectory_context, current_node_key=current_node_key, short_side=short_side
            )
            content.extend(memory_items)
        else:
            # caption-only path (memory_frames_per_node == 0)
            memory_text = self._build_memory_text(trajectory_context, current_node_key=current_node_key)
            if memory_text:
                content.append({
                    "type": "text",
                    "text": memory_text
                })

        # === STEP 4: Navigation context (available zoom_in targets and shift targets) ===
        if navigation_context:
            nav_text = self._build_navigation_context(navigation_context)
            if nav_text:
                content.append({
                    "type": "text",
                    "text": nav_text
                })

        # === STEP 5: Question and choices ===
        choices = self._maybe_add_unanswerable(choices, include_unanswerable)
        formatted_choices = "\n".join([
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
        ])
        content.append({
            "type": "text",
            "text": f"Question: {question}\n\nChoices:\n{formatted_choices}\n\n"
        })

        # === Build JSON schema and rules based on mode ===
        if generate_captions:
            # captions_field = '  "captions": {"<segment_id>": "<brief caption>", ...},\n'
            # captions_rule = (
            #     "- captions: For EACH child segment, provide a brief description.\n"
            # )
            captions_field = '  "captions": {"<segment_id>": "<detailed caption>", ...},\n'
            captions_rule = (
                "- captions: For EACH child segment, provide a detailed description.\n"
            )
        else:
            captions_field = ''
            captions_rule = ''

        # Build action_json_values, json_fields, and action_rules dynamically
        has_answer = "ANSWER" in allowed_actions
        has_zoom_in = "ZOOM_IN" in allowed_actions
        has_zoom_out = "ZOOM_OUT" in allowed_actions
        has_shift = "SHIFT" in allowed_actions
        needs_nav_fields = has_zoom_in or has_shift

        action_json_values = ' | '.join(f'"{a}"' for a in allowed_action_list or ["ZOOM_IN", "ZOOM_OUT", "SHIFT", "ANSWER"])

        # Build JSON fields
        json_field_parts = [captions_field, reasoning_field, f'  "action": {action_json_values},\n']
        if needs_nav_fields:
            if is_segmented:
                json_field_parts.append('  "segment_id": <int or null>,\n')
            else:
                json_field_parts.append('  "zoom_start": <float or null>,\n')
                json_field_parts.append('  "zoom_end": <float or null>,\n')
        if has_answer:
            null_suffix = " or null" if needs_nav_fields else ""
            json_field_parts.append(f'  "answer": "<letter{null_suffix}>",\n')
            json_field_parts.append(f'  "evidence_start": <float{null_suffix}>,\n')
            json_field_parts.append(f'  "evidence_end": <float{null_suffix}>\n')
        else:
            # Remove trailing comma from last field
            if json_field_parts:
                json_field_parts[-1] = json_field_parts[-1].rstrip(',\n') + '\n'

        json_fields = ''.join(json_field_parts)

        # Build action rules
        rule_parts = ["Rules for the JSON:\n"]
        if has_zoom_in:
            if is_segmented:
                null_note = " Set answer/evidence to null." if has_answer else ""
                if valid_zoom_in_ids:
                    rule_parts.append(f"- For ZOOM_IN: set segment_id to one of the unvisited children {valid_zoom_in_ids}.{null_note}\n")
                else:
                    rule_parts.append(f"- For ZOOM_IN: set segment_id to the chosen child's ID.{null_note}\n")
            else:
                null_note = " Set answer/evidence to null." if has_answer else ""
                rule_parts.append(
                    "- For ZOOM_IN: set zoom_start and zoom_end to the exact time range (in seconds) "
                    f"you want to explore within the current segment.{null_note}\n"
                )
        if has_zoom_out:
            if skip_reasoning:
                rule_parts.append("- For ZOOM_OUT: set all fields except action to null.\n")
            else:
                rule_parts.append("- For ZOOM_OUT: set all fields except action and reasoning to null.\n")
        if has_shift:
            if is_segmented:
                null_note = " Set answer/evidence to null." if has_answer else ""
                if valid_shift_ids:
                    rule_parts.append(f"- For SHIFT: set segment_id to one of the unvisited siblings {valid_shift_ids}.{null_note}\n")
                else:
                    rule_parts.append(f"- For SHIFT: set segment_id to the sibling segment's ID.{null_note}\n")
            else:
                null_note = " Set answer/evidence to null." if has_answer else ""
                rule_parts.append(f"- For SHIFT: set zoom_start and zoom_end to the sibling time range.{null_note}\n")
        if has_answer:
            if needs_nav_fields:
                if is_segmented:
                    rule_parts.append(
                        "- For ANSWER: set answer to the letter (A, B, C, etc.), and evidence_start/evidence_end "
                        "to the time interval. Set segment_id to null.\n"
                    )
                else:
                    rule_parts.append(
                        "- For ANSWER: set answer to the letter (A, B, C, etc.), and evidence_start/evidence_end "
                        "to the time interval. Set zoom_start/zoom_end to null.\n"
                    )
            else:
                rule_parts.append(
                    "- Set action to \"ANSWER\", answer to the letter (A, B, C, etc.), and "
                    "evidence_start/evidence_end to the time interval containing the evidence.\n"
                )
        rule_parts.append(captions_rule)
        action_rules = ''.join(rule_parts)

        # === STEP 6: Analysis instruction + IMPORTANT RULES + JSON format (emphasis at end) ===
        content.append({
            "type": "text",
            "text": (
                "Analyze the visual content, memory, navigation context, "
                "the question, and the choices above, then decide your next action.\n\n"
                + important_rules + "\n"
                "Respond in EXACTLY this JSON format (no other text before or after):\n"
                "```json\n"
                "{\n"
                f"{json_fields}"
                "}\n"
                "```\n\n"
                + action_rules
            )
        })

        # API call with retry on parse failure
        max_retries = 3
        for attempt in range(max_retries):
            response = self._create_completion(
                method_name="decide_action",
                messages=[{"role": "user", "content": content}],
                max_tokens=4096,
                temperature=0 if self.image_tokens is not None and attempt == 0 else 0.3
            )

            response_text = response.choices[0].message.content.strip()

            try:
                parsed_decision = self._parse_decide_action_response(
                    response_text, child_segments, current_start_sec, current_end_sec, choices
                )
                return parsed_decision, response_text
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    print(f"Warning: Parse attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                    continue
                raise

    def decide_action_mcq(
        self,
        question: str,
        choices: List[str],
        child_segments: List[Dict[str, Any]],
        trajectory_context: Dict[str, Any],
        current_start_sec: float,
        current_end_sec: float,
        current_frames: List[float],
        video_path: str,
        short_side: int = -1,
        allowed_action_list: Optional[List[str]] = None,
        navigation_context: Optional[Dict[str, Any]] = None,
        generate_captions: bool = True,
        skip_reasoning: bool = False,
        include_unanswerable: bool = False,
        pregenerated_caption_path: Optional[str] = None,
        use_frame_captions: bool = False,
        enforce_valid_segments: bool = True,
    ) -> Dict[str, Any]:
        """
        Decide the next action using a multiple-choice formulation.

        Each possible action (ZOOM_IN to a specific child, ZOOM_OUT, SHIFT to
        a specific sibling, ANSWER) becomes a numbered option. The VLM picks
        an option number. Only ANSWER requires additional output (answer letter
        and evidence timestamps).

        Args:
            Same as decide_action. enforce_valid_segments is ignored (MCQ
            inherently only shows valid/unvisited targets as options).

        Returns:
            Tuple of (parsed_decision dict, raw_response_text).
            The dict has the same keys as decide_action for compatibility.
        """
        import json
        import re

        reasoning_field = '' if skip_reasoning else '  "reasoning": "<your concise reasoning>",\n'

        allowed_actions = set(allowed_action_list) if allowed_action_list else {"ZOOM_IN", "ZOOM_OUT", "SHIFT", "ANSWER"}
        has_answer = "ANSWER" in allowed_actions

        # === Build MCQ options from navigation context ===
        options = []  # list of (opt_num, action_type, info_dict)
        option_texts = []
        opt_num = 1

        # ZOOM_IN: one option per unvisited child
        if "ZOOM_IN" in allowed_actions and navigation_context:
            for child in navigation_context.get('children_info', []):
                visited = child.get('visited', False)
                if enforce_valid_segments and visited:
                    continue
                else:
                    options.append((opt_num, "ZOOM_IN", {
                        'segment_id': child['segment_id'],
                        'start_sec': child['start_sec'],
                        'end_sec': child['end_sec'],
                    }))
                    option_texts.append(
                        f"Option {opt_num}: ZOOM_IN - Examine Child Segment {child['segment_id']} "
                        f"[{child['start_sec']:.1f}s-{child['end_sec']:.1f}s]"
                    )
                    opt_num += 1

        # ZOOM_OUT: single option
        if "ZOOM_OUT" in allowed_actions:
            parent_info = navigation_context.get('parent') if navigation_context else None
            if parent_info:
                options.append((opt_num, "ZOOM_OUT", {
                    'start_sec': parent_info['start_sec'],
                    'end_sec': parent_info['end_sec'],
                }))
                option_texts.append(
                    f"Option {opt_num}: ZOOM_OUT - Backtrack to parent segment "
                    f"[{parent_info['start_sec']:.1f}s-{parent_info['end_sec']:.1f}s]"
                )
            else:
                options.append((opt_num, "ZOOM_OUT", {}))
                option_texts.append(
                    f"Option {opt_num}: ZOOM_OUT - Backtrack to parent segment"
                )
            opt_num += 1

        # SHIFT: one option per unvisited sibling
        if "SHIFT" in allowed_actions and navigation_context:
            for sibling in navigation_context.get('siblings', []):
                visited = sibling.get('visited', False)
                if enforce_valid_segments and visited:
                    continue
                else:
                    options.append((opt_num, "SHIFT", {
                        'segment_id': sibling['segment_id'],
                        'start_sec': sibling['start_sec'],
                        'end_sec': sibling['end_sec'],
                    }))
                    option_texts.append(
                        f"Option {opt_num}: SHIFT - Move to Sibling Segment {sibling['segment_id']} "
                        f"[{sibling['start_sec']:.1f}s-{sibling['end_sec']:.1f}s]"
                    )
                    opt_num += 1

        # ANSWER: single option
        if has_answer:
            options.append((opt_num, "ANSWER", {}))
            option_texts.append(
                f"Option {opt_num}: ANSWER - Provide your final answer to the question"
            )
            opt_num += 1

        if not options:
            raise ValueError("No valid actions available for MCQ formulation")

        content = []

        # === STEP 1: System header with high-level action descriptions ===
        options_list_text = "\n".join(option_texts) + "\n\n"
        has_navigation = "ZOOM_IN" in allowed_actions or "ZOOM_OUT" in allowed_actions or "SHIFT" in allowed_actions

        # Build high-level action descriptions (like decide_action)
        action_descs = []
        action_idx = 1
        if "ZOOM_IN" in allowed_actions:
            action_descs.append(f"{action_idx}. ZOOM_IN: Examine a child segment at finer granularity.")
            action_idx += 1
        if "ZOOM_OUT" in allowed_actions:
            action_descs.append(f"{action_idx}. ZOOM_OUT: Backtrack to the parent segment to explore a different region.")
            action_idx += 1
        if "SHIFT" in allowed_actions:
            action_descs.append(f"{action_idx}. SHIFT: Move to a sibling segment under the same parent.")
            action_idx += 1
        if "ANSWER" in allowed_actions:
            action_descs.append(f"{action_idx}. ANSWER: Provide the answer to the question along with the time interval containing the supporting evidence.")
            action_idx += 1
        action_list_text = "\n".join(action_descs) + "\n\n"

        num_actions = len(action_descs)
        if has_navigation:
            count_word = {1: "the following action", 2: "one of two actions", 3: "one of three actions", 4: "one of four actions"}.get(num_actions, f"one of {num_actions} actions")
            system_text = (
                "You are a video question-answering agent that navigates a long video through "
                "hierarchical temporal search. At each step, the current video segment is divided "
                "into non-overlapping child segments. You observe frames from each child segment "
                f"and decide {count_word}:\n\n"
                + action_list_text
            )
        else:
            system_text = (
                "You are a video question-answering agent. You observe frames from the current "
                "video segment and must directly provide the final answer along with the time "
                "interval containing the evidence.\n\n"
                + action_list_text
            )

        content.append({"type": "text", "text": system_text})

        # === STEP 2: Current Observation frames/images ===
        current_observation = {
            'start_sec': current_start_sec,
            'end_sec': current_end_sec,
            'description': '',
            'segments': [
                {
                    'segment_id': seg['segment_id'],
                    'start_sec': seg['start_sec'],
                    'end_sec': seg['end_sec'],
                }
                for seg in child_segments
            ] if self.reasoning_video_representation in ("segmented", "segmented-frames-first") else None
        }

        self._add_observation_to_content(
            content, current_observation, video_path,
            label="Current Segment", frames=current_frames, short_side=short_side
        )

        # === STEP 2.5: Frame captions for child segments (pregenerated) ===
        if use_frame_captions and pregenerated_caption_path and child_segments:
            captions = self._load_pregenerated_captions(video_path, pregenerated_caption_path)
            if captions:
                caption_lines = [f"Frame Captions [{current_start_sec:.1f}s-{current_end_sec:.1f}s]:"]
                for seg in child_segments:
                    seg_id = seg['segment_id']
                    seg_start = seg['start_sec']
                    seg_end = seg['end_sec']
                    caption_lines.append(f"Segment {seg_id} [{seg_start:.1f}s-{seg_end:.1f}s]:")
                    seg_frames = [t for t in current_frames if seg_start <= t <= seg_end]
                    for ts in seg_frames:
                        matched_ts, caption_text = self._lookup_pregenerated_caption(captions, ts)
                        caption_lines.append(f"[{matched_ts:.1f}s]: {caption_text}")
                content.append({
                    "type": "text",
                    "text": "\n".join(caption_lines)
                })

        # === STEP 3: Memory (tree structure + interaction history) ===
        current_node_key = (current_start_sec, current_end_sec)
        if self.memory_frames_per_node == -1:
            pass
        elif self.memory_frames_per_node > 0:
            memory_items = self._build_memory_content(
                trajectory_context, current_node_key=current_node_key, short_side=short_side
            )
            content.extend(memory_items)
        else:
            memory_text = self._build_memory_text(trajectory_context, current_node_key=current_node_key)
            if memory_text:
                content.append({
                    "type": "text",
                    "text": memory_text
                })

        # === STEP 4: Navigation context ===
        if navigation_context:
            nav_text = self._build_navigation_context(navigation_context)
            if nav_text:
                content.append({
                    "type": "text",
                    "text": nav_text
                })

        # === STEP 5: Question and choices ===
        choices = self._maybe_add_unanswerable(choices, include_unanswerable)
        formatted_choices = "\n".join([
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
        ])
        content.append({
            "type": "text",
            "text": f"Question: {question}\n\nChoices:\n{formatted_choices}\n\n"
        })

        # === STEP 6: JSON format and rules ===
        if generate_captions:
            captions_field = '  "captions": {"<segment_id>": "<detailed caption>", ...},\n'
            captions_rule = "- captions: For EACH child segment, provide a detailed description.\n"
        else:
            captions_field = ''
            captions_rule = ''

        json_field_parts = [captions_field, reasoning_field, f'  "action_choice": <int>,\n']
        if has_answer:
            json_field_parts.append('  "answer": "<letter or null>",\n')
            json_field_parts.append('  "evidence_start": <float or null>,\n')
            json_field_parts.append('  "evidence_end": <float or null>\n')
        else:
            # Remove trailing comma from last field
            json_field_parts[-1] = json_field_parts[-1].rstrip(',\n') + '\n'

        json_fields = ''.join(json_field_parts)

        # Build rules
        rule_parts = ["Rules for the JSON:\n"]
        rule_parts.append(
            f"- action_choice: The option number (1 to {len(options)}) from the list above.\n"
        )
        if has_answer:
            answer_option_num = next(num for num, atype, _ in options if atype == "ANSWER")
            rule_parts.append(
                f"- For Option {answer_option_num} (ANSWER): set answer to the letter (A, B, C, etc.), "
                "and evidence_start/evidence_end to the time interval containing the evidence.\n"
            )
            rule_parts.append(
                "- For all other options: set answer, evidence_start, evidence_end to null.\n"
            )
        rule_parts.append(captions_rule)
        action_rules = ''.join(rule_parts)

        # Build important rules
        important_rule_parts = []
        has_zoom_in = "ZOOM_IN" in allowed_actions
        has_zoom_out = "ZOOM_OUT" in allowed_actions
        has_shift = "SHIFT" in allowed_actions
        if has_zoom_in:
            important_rule_parts.append(
                "- Choose ZOOM_IN when you believe a specific child segment is likely to "
                "contain the answer and needs closer examination."
            )
        if has_zoom_out:
            important_rule_parts.append(
                "- Choose ZOOM_OUT when none of the current child segments appear relevant."
            )
        if has_shift:
            important_rule_parts.append(
                "- Choose SHIFT to move to a sibling segment when the current segment is not "
                "relevant but a sibling might be."
            )
        if has_answer:
            if has_navigation:
                important_rule_parts.append(
                    "- Choose ANSWER only when you are confident you have found sufficient evidence. "
                    "You must also specify the answer letter and evidence time interval."
                )
            else:
                important_rule_parts.append(
                    "You must provide your answer now based on the visual content shown."
                )

        if has_navigation:
            important_rules = "IMPORTANT RULES:\n" + "\n".join(important_rule_parts) + "\n"
        else:
            important_rules = "IMPORTANT: " + "\n".join(important_rule_parts) + "\n"

        content.append({
            "type": "text",
            "text": (
                "Analyze the visual content, memory, navigation context, "
                "the question, and the choices above, then decide your next action.\n\n"
                "Choose one of the following specific options:\n"
                + options_list_text
                + important_rules + "\n"
                "Respond in EXACTLY this JSON format (no other text before or after):\n"
                "```json\n"
                "{\n"
                f"{json_fields}"
                "}\n"
                "```\n\n"
                + action_rules
            )
        })

        # API call with retry on parse failure
        max_retries = 3
        for attempt in range(max_retries):
            response = self._create_completion(
                method_name="decide_action_mcq",
                messages=[{"role": "user", "content": content}],
                max_tokens=4096,
                temperature=0.3
            )

            response_text = response.choices[0].message.content.strip()

            try:
                parsed_decision = self._parse_decide_action_mcq_response(
                    response_text, options, current_start_sec, current_end_sec, choices
                )
                return parsed_decision, response_text
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    print(f"Warning: MCQ parse attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                    continue
                raise

    def _parse_decide_action_response(
        self,
        response_text: str,
        child_segments: List[Dict[str, Any]],
        current_start_sec: float,
        current_end_sec: float,
        choices: List[str]
    ) -> Dict[str, Any]:
        """Parse VLM JSON response for decide_action. Raises ValueError on parse failure."""
        import json
        import re

        # Try JSON in markdown code block first
        json_match = re.search(
            r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL
        )
        if json_match:
            parsed = json.loads(json_match.group(1))
        else:
            # Try raw JSON object
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                raise ValueError(f"No JSON found in response: {response_text[:300]}")

        # Extract fields
        reasoning = parsed.get('reasoning', response_text)
        action_raw = str(parsed.get('action', '')).upper().strip()

        if 'ZOOM_IN' in action_raw:
            action_type = 'ZOOM_IN'
        elif 'ZOOM_OUT' in action_raw:
            action_type = 'ZOOM_OUT'
        elif 'SHIFT' in action_raw:
            action_type = 'SHIFT'
        elif 'ANSWER' in action_raw:
            action_type = 'ANSWER'
        else:
            raise ValueError(f"Unknown action '{action_raw}' in response")

        result = {
            'reasoning': reasoning,
            'action_type': action_type,
            'segment_id': None,
            'start_sec': None,
            'end_sec': None,
            'answer': None,
            'evidence_start': None,
            'evidence_end': None,
            'captions': {},
        }

        # Extract captions dict (keys may be int or str segment IDs)
        raw_captions = parsed.get('captions', {})
        if isinstance(raw_captions, dict):
            result['captions'] = {
                int(k) if str(k).isdigit() else k: str(v)
                for k, v in raw_captions.items()
            }

        if action_type == 'ZOOM_IN':
            if self.reasoning_action_representation == 'segmented':
                # Structured-tree mode: parse segment_id and look up timestamps
                seg_id = parsed.get('segment_id')
                # Normalize hierarchical IDs like "3.0.2" → 2
                if seg_id is not None:
                    seg_id_str = str(seg_id)
                    if '.' in seg_id_str:
                        seg_id_str = seg_id_str.split('.')[-1]
                    seg_id = int(seg_id_str)
                valid_ids = {s['segment_id'] for s in child_segments}
                if seg_id is not None and seg_id in valid_ids:
                    result['segment_id'] = seg_id
                    # Look up timestamps from child_segments
                    for seg in child_segments:
                        if seg['segment_id'] == seg_id:
                            result['start_sec'] = seg['start_sec']
                            result['end_sec'] = seg['end_sec']
                            break
                else:
                    raise ValueError(
                        f"Invalid segment_id {seg_id}, valid IDs: {valid_ids}"
                    )
            else:
                # Free-form mode: parse zoom_start and zoom_end
                zoom_start = float(parsed.get('zoom_start', current_start_sec))
                zoom_end = float(parsed.get('zoom_end', current_end_sec))
                zoom_start = max(current_start_sec, min(zoom_start, current_end_sec))
                zoom_end = max(current_start_sec, min(zoom_end, current_end_sec))

                # Ensure valid interval
                if zoom_start >= zoom_end:
                    raise ValueError(
                        f"Invalid zoom interval [{zoom_start}, {zoom_end}]"
                    )

                result['segment_id'] = None  # Not used in continuous mode
                result['start_sec'] = zoom_start
                result['end_sec'] = zoom_end

        elif action_type == 'SHIFT':
            if self.reasoning_action_representation == 'segmented':
                # Structured-tree mode: parse segment_id for sibling
                seg_id = parsed.get('segment_id')
                # Normalize hierarchical IDs like "3.0.2" → 2
                if seg_id is not None:
                    seg_id_str = str(seg_id)
                    if '.' in seg_id_str:
                        seg_id_str = seg_id_str.split('.')[-1]
                    seg_id = int(seg_id_str)
                if seg_id is not None:
                    result['segment_id'] = seg_id
                    for seg in child_segments:
                        if seg['segment_id'] == seg_id:
                            result['start_sec'] = seg['start_sec']
                            result['end_sec'] = seg['end_sec']
                            break
                else:
                    raise ValueError(
                        f"Invalid SHIFT segment_id {seg_id}"
                    )
            else:
                # Free-form mode: parse zoom_start and zoom_end for sibling
                zoom_start = float(parsed.get('zoom_start', current_start_sec))
                zoom_end = float(parsed.get('zoom_end', current_end_sec))
                result['segment_id'] = None
                result['start_sec'] = zoom_start
                result['end_sec'] = zoom_end

        elif action_type == 'ANSWER':
            # Parse answer letter
            answer_raw = parsed.get('answer', '')
            if answer_raw and isinstance(answer_raw, str):
                answer_letter = answer_raw.strip().upper()[0]
                valid_letters = {chr(65 + i) for i in range(len(choices))}
                if answer_letter in valid_letters:
                    result['answer'] = answer_letter
                else:
                    raise ValueError(
                        f"Invalid answer '{answer_raw}', valid: {valid_letters}"
                    )
            else:
                raise ValueError(f"Missing or invalid answer field: '{answer_raw}'")

            # Parse evidence interval
            ev_start = parsed.get('evidence_start')
            ev_end = parsed.get('evidence_end')
            if ev_start is not None and ev_end is not None:
                ev_start = float(ev_start)
                ev_end = float(ev_end)
                if ev_start == ev_end:
                    ev_start, ev_end = ev_start - 1, ev_end + 1
                ev_start = max(current_start_sec, min(ev_start, current_end_sec))
                ev_end = max(current_start_sec, min(ev_end, current_end_sec))
                if ev_start > ev_end:
                    ev_start, ev_end = current_start_sec, current_end_sec
                result['evidence_start'] = ev_start
                result['evidence_end'] = ev_end
            else:
                result['evidence_start'] = current_start_sec
                result['evidence_end'] = current_end_sec

        return result

    def _parse_decide_action_mcq_response(
        self,
        response_text: str,
        options: List,
        current_start_sec: float,
        current_end_sec: float,
        choices: List[str],
    ) -> Dict[str, Any]:
        """Parse VLM JSON response for decide_action_mcq.

        Maps the chosen option number back to action_type, segment_id, and
        timestamps from the pre-built options list. Only ANSWER requires
        parsing additional fields (answer letter + evidence timestamps).

        Args:
            response_text: Raw VLM response text
            options: List of (opt_num, action_type, info_dict) tuples
            current_start_sec: Start time of current segment
            current_end_sec: End time of current segment
            choices: List of multiple choice options (for answer validation)

        Returns:
            Dict with reasoning, action_type, segment_id, start_sec, end_sec,
            answer, evidence_start, evidence_end, captions.

        Raises:
            ValueError: On parse failure (invalid JSON, invalid option, etc.)
        """
        import json
        import re

        # Try JSON in markdown code block first
        json_match = re.search(
            r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL
        )
        if json_match:
            parsed = json.loads(json_match.group(1))
        else:
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                raise ValueError(f"No JSON found in response: {response_text[:300]}")

        # Extract and validate action_choice
        action_choice = parsed.get('action_choice')
        if action_choice is None:
            raise ValueError(f"Missing action_choice in response")
        action_choice = int(action_choice)
        if action_choice < 1 or action_choice > len(options):
            raise ValueError(
                f"Invalid action_choice {action_choice}, must be 1-{len(options)}"
            )

        # Look up the selected option
        opt_num, action_type, info = options[action_choice - 1]

        reasoning = parsed.get('reasoning', response_text)

        result = {
            'reasoning': reasoning,
            'action_type': action_type,
            'segment_id': info.get('segment_id'),
            'start_sec': info.get('start_sec'),
            'end_sec': info.get('end_sec'),
            'answer': None,
            'evidence_start': None,
            'evidence_end': None,
            'captions': {},
        }

        # Extract captions dict (keys may be int or str segment IDs)
        raw_captions = parsed.get('captions', {})
        if isinstance(raw_captions, dict):
            result['captions'] = {
                int(k) if str(k).isdigit() else k: str(v)
                for k, v in raw_captions.items()
            }

        if action_type == 'ANSWER':
            # Parse answer letter
            answer_raw = parsed.get('answer', '')
            if answer_raw and isinstance(answer_raw, str):
                answer_letter = answer_raw.strip().upper()[0]
                valid_letters = {chr(65 + i) for i in range(len(choices))}
                if answer_letter in valid_letters:
                    result['answer'] = answer_letter
                else:
                    raise ValueError(
                        f"Invalid answer '{answer_raw}', valid: {valid_letters}"
                    )
            else:
                raise ValueError(f"Missing or invalid answer field: '{answer_raw}'")

            # Parse evidence interval
            ev_start = parsed.get('evidence_start')
            ev_end = parsed.get('evidence_end')
            if ev_start is not None and ev_end is not None:
                ev_start = float(ev_start)
                ev_end = float(ev_end)
                if ev_start == ev_end:
                    ev_start, ev_end = ev_start - 1, ev_end + 1
                ev_start = max(current_start_sec, min(ev_start, current_end_sec))
                ev_end = max(current_start_sec, min(ev_end, current_end_sec))
                if ev_start > ev_end:
                    ev_start, ev_end = current_start_sec, current_end_sec
                result['evidence_start'] = ev_start
                result['evidence_end'] = ev_end
            else:
                result['evidence_start'] = current_start_sec
                result['evidence_end'] = current_end_sec

        return result

    def direct_answer(
        self,
        question: str,
        choices: List[str],
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        short_side: int = -1,
        predict_caption: bool = False,
        skip_reasoning: bool = False,
        include_unanswerable: bool = False,
        video_native: bool = False,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Directly answer a question and provide an evidence time interval.

        Builds a simple multimodal prompt showing frames with timestamps,
        the question and choices, and asks for answer + evidence interval.
        No action selection, no trajectory history, no child segments.

        If predict_caption is True, also asks the model to generate a caption
        describing the video segment (for ablation purposes; ignored by parser).

        If video_native is True, dispatch to `direct_answer_video_native`,
        which sends the entire video via a single `video_url` attachment.
        """
        if video_native:
            return self.direct_answer_video_native(
                question=question,
                choices=choices,
                frames=frames,
                start_sec=start_sec,
                end_sec=end_sec,
                video_path=video_path,
                predict_caption=predict_caption,
                skip_reasoning=skip_reasoning,
                include_unanswerable=include_unanswerable,
                nframes=len(frames)
            )

        import json

        # Reasoning field for JSON schema (empty when skip_reasoning is set)
        reasoning_field = '' if skip_reasoning else '  "reasoning": "<your concise reasoning>",\n'

        content = []

        # System header
        if predict_caption:
            system_text = (
                f"You are a video question-answering agent. You are shown frames sampled "
                f"from a video segment ({start_sec:.1f}s to {end_sec:.1f}s). Your task is to:\n"
                "1. Describe the visual content of the video segment.\n"
                "2. Answer the multiple-choice question based on the visual content.\n"
                "3. Identify the time interval (in seconds) that contains the visual "
                "evidence supporting your answer.\n\n"
            )
        else:
            system_text = (
                f"You are a video question-answering agent. You are shown frames sampled "
                f"from a video segment ({start_sec:.1f}s to {end_sec:.1f}s). Your task is to:\n"
                "1. Answer the multiple-choice question based on the visual content.\n"
                "2. Identify the time interval (in seconds) that contains the visual "
                "evidence supporting your answer.\n\n"
            )
        content.append({"type": "text", "text": system_text})

        # Question and choices
        choices = self._maybe_add_unanswerable(choices, include_unanswerable)
        formatted_choices = "\n".join([
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
        ])
        content.append({
            "type": "text",
            "text": f"Question: {question}\n\nChoices:\n{formatted_choices}\n\n"
        })

        # Visual input: sampled frames with timestamps
        content.append({
            "type": "text",
            "text": f"Video [{start_sec:.1f}s-{end_sec:.1f}s]:"
        })
        frame_content = self._build_video_content(frames, "", video_path, short_side)
        # Remove the trailing empty prompt text
        if frame_content and frame_content[-1].get("type") == "text" and not frame_content[-1].get("text", "").strip():
            frame_content.pop()
        content.extend(frame_content)

        # JSON output prompt
        if predict_caption:
            json_template = (
                "{\n"
                '  "caption": "<description of the video segment>",\n'
                + reasoning_field +
                '  "answer": "<letter>",\n'
                '  "evidence_start": <float>,\n'
                '  "evidence_end": <float>\n'
                "}"
            )
            caption_rule = "- caption: A concise description of the visual content in the video segment\n"
        else:
            json_template = (
                "{\n"
                + reasoning_field +
                '  "answer": "<letter>",\n'
                '  "evidence_start": <float>,\n'
                '  "evidence_end": <float>\n'
                "}"
            )
            caption_rule = ""

        content.append({
            "type": "text",
            "text": (
                "\n\nBased on the visual content above, answer the question and identify "
                "the time interval containing the supporting evidence.\n\n"
                "Respond in EXACTLY this JSON format (no other text before or after):\n"
                "```json\n"
                f"{json_template}\n"
                "```\n\n"
                "Rules:\n"
                f"{caption_rule}"
                "- answer: The letter of the correct choice (A, B, C, etc.)\n"
                f"- evidence_start/evidence_end: The time interval (in seconds, between "
                f"{start_sec:.1f} and {end_sec:.1f}) containing the visual evidence "
                "for your answer.\n"
            )
        })

        # API call
        response = self._create_completion(
            method_name="direct_answer",
            messages=[{"role": "user", "content": content}],
            max_tokens=2048,
            temperature=0.3
            # max_tokens=256,
            # temperature=0
        )

        response_text = response.choices[0].message.content.strip()

        parsed = self._parse_direct_answer_response(
            response_text, start_sec, end_sec, choices
        )

        return parsed, response_text

    def direct_answer_lmm(
        self,
        question: str,
        choices: List[str],
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        short_side: int = -1,
        predict_caption: bool = False,
        skip_reasoning: bool = False,
        include_unanswerable: bool = False,
        video_native: bool = False,
        json_output: bool = True,
        predict_evidence: bool = False,
        question_first: bool = False,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Variant of `direct_answer` that uses lmms-eval's
        `longvideobench_val_v` per-frame label format for Qwen3-VL:
        ``<{ts:.1f} seconds>`` instead of ``{ts:.1f}s:``.

        Two prompt modes via ``json_output``:

        - ``json_output=False`` (lmms-eval letter-only): asks "Answer with the
          option's letter from the given choices directly." Generation:
          ``temperature=0``, ``max_tokens=32``. Parser takes the first A-Z
          letter in the response. ``evidence_start``/``evidence_end`` in the
          returned dict are set to the full segment range.
        - ``json_output=True`` (default): asks for a JSON object with the
          answer (and ``reasoning`` unless ``skip_reasoning``). Parsed via
          ``_parse_direct_answer_response``.

        ``predict_evidence`` (only meaningful with ``json_output=True``):
        when True, the JSON schema also includes ``evidence_start`` and
        ``evidence_end`` and the model is asked to localize the visual
        evidence within ``[start_sec, end_sec]``. When False, the model is
        only asked for the answer (temporal-grounding ablation).

        ``question_first``: when True, the question/choices/instruction block
        is placed before the frames; when False (default), it is placed after
        the frames, matching the original layout. Instruction wording is
        position-neutral so the same text works in either order.

        ``predict_caption`` and ``video_native`` are accepted for signature
        parity with ``direct_answer`` but ignored.
        """
        import re

        choices = self._maybe_add_unanswerable(choices, include_unanswerable)

        # Extract frames once, then build content with lmms-eval label format.
        frame_map = self._extract_frames_at_timestamps(video_path, frames, short_side)

        content: List[Dict[str, Any]] = []

        if json_output and predict_evidence:
            system_text = (
                f"You are a video question-answering agent. You are shown frames sampled "
                f"from a video segment ({start_sec:.1f}s to {end_sec:.1f}s). Your task is to:\n"
                "1. Answer the multiple-choice question based on the visual content.\n"
                "2. Identify the time interval (in seconds) that contains the visual "
                "evidence supporting your answer.\n\n"
            )
        else:
            system_text = (
                f"You are a video question-answering agent. You are shown frames sampled "
                f"from a video segment ({start_sec:.1f}s to {end_sec:.1f}s). Your task is to:\n"
                "Answer the multiple-choice question based on the visual content.\n"
            )
        content.append({"type": "text", "text": system_text})

        frame_content: List[Dict[str, Any]] = []
        for ts in frames:
            frame_content.append({"type": "text", "text": f"<{ts:.1f} seconds>"})
            b64 = self._encode_image_array(frame_map[ts])
            frame_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        candidates_text = "\n".join([
            f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
        ])

        question_content: List[Dict[str, Any]] = []

        if json_output:
            reasoning_field = (
                '' if skip_reasoning
                else '  "reasoning": "<your concise reasoning>",\n'
            )

            if predict_evidence:
                json_template = (
                    "{\n"
                    + reasoning_field +
                    '  "answer": "<letter>",\n'
                    '  "evidence_start": <float>,\n'
                    '  "evidence_end": <float>\n'
                    "}"
                )
                evidence_rule = (
                    f"- evidence_start/evidence_end: The time interval (in seconds, "
                    f"between {start_sec:.1f} and {end_sec:.1f}) containing the visual "
                    "evidence for your answer.\n"
                )
                task_instruction = (
                    "Based on the visual content, answer the question and "
                    "identify the time interval containing the supporting evidence.\n\n"
                )
            else:
                json_template = (
                    "{\n"
                    + reasoning_field +
                    '  "answer": "<letter>"\n'
                    "}"
                )
                evidence_rule = ""
                task_instruction = (
                    "Based on the visual content, answer the question.\n\n"
                )

            question_content.append({
                "type": "text",
                "text": (
                    f"Question: {question}\n\nChoices:\n{candidates_text}\n\n"
                    f"{task_instruction}"
                    "Respond in EXACTLY this JSON format (no other text before or after):\n"
                    "```json\n"
                    f"{json_template}\n"
                    "```\n\n"
                    "Rules:\n"
                    "- answer: The letter of the correct choice (A, B, C, etc.)\n"
                    f"{evidence_rule}"
                ),
            })
        else:
            question_content.append({
                "type": "text",
                "text": (
                    f"{question}\n{candidates_text}\n"
                    "Answer with the option's letter from the given choices directly.\n"
                ),
            })

        if question_first:
            content.extend(question_content)
            content.extend(frame_content)
        else:
            content.extend(frame_content)
            content.extend(question_content)

        if json_output:
            response = self._create_completion(
                method_name="direct_answer_lmm",
                messages=[{"role": "user", "content": content}],
                max_tokens=2048,
                temperature=0.3,
            )
            response_text = (response.choices[0].message.content or "").strip()

            # Reuse the shared JSON parser; it falls back to [start_sec, end_sec]
            # for evidence when the fields are absent (predict_evidence=False).
            parsed = self._parse_direct_answer_response(
                response_text, start_sec, end_sec, choices
            )
        else:
            response = self._create_completion(
                method_name="direct_answer_lmm",
                messages=[{"role": "user", "content": content}],
                max_tokens=1024,
                temperature=0.0,
            )
            response_text = (response.choices[0].message.content or "").strip()

            # Parser: same approach as lmms-eval's parse_multi_choice_response.
            valid_letters = [chr(65 + i) for i in range(len(choices))]
            valid_set = set(valid_letters)
            s = response_text
            for prefix in ("The best answer is", "The correct answer is",
                        "The answer is", "The answer", "The best option is",
                        "The correct option is", "Best answer:", "Best option:"):
                s = s.replace(prefix, "")
            m = re.search(r"\b([A-Z])\b", s)
            if m and m.group(1) in valid_set:
                answer_letter = m.group(1)
            else:
                answer_letter = valid_letters[0]

            parsed: Dict[str, Any] = {
                "reasoning": response_text,
                "answer": answer_letter,
                "evidence_start": start_sec,
                "evidence_end": end_sec,
            }

        return parsed, response_text

    def direct_answer_video_native(
        self,
        question: str,
        choices: List[str],
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        predict_caption: bool = False,
        skip_reasoning: bool = False,
        include_unanswerable: bool = False,
        nframes: Optional[int] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Video-native variant of `direct_answer`.

        Sends the entire video as a single `video_url` attachment and lets the
        server handle frame sampling. The number of frames to sample and the
        per-frame pixel budget are passed through `mm_processor_kwargs`.

        Args:
            nframes: Number of frames the server should sample from the video.
                Defaults to `len(frames)` so callers can keep using the same
                frames-list contract as `direct_answer`.
            min_pixels / max_pixels: Per-frame pixel budget. When None and
                `self.image_tokens` is set, falls back to image_tokens * 28 * 28.
        """
        # Reasoning field for JSON schema (empty when skip_reasoning is set)
        reasoning_field = '' if skip_reasoning else '  "reasoning": "<your concise reasoning>",\n'

        if nframes is None:
            nframes = len(frames) if frames else 0
        if min_pixels is None and self.image_tokens is not None:
            min_pixels = self.image_tokens * 28 * 28
        if max_pixels is None and self.image_tokens is not None:
            max_pixels = self.image_tokens * 28 * 28

        content = []

        # System header
        if predict_caption:
            system_text = (
                f"You are a video question-answering agent. You are shown the video "
                f"({start_sec:.1f}s to {end_sec:.1f}s). Your task is to:\n"
                "1. Describe the visual content of the video segment.\n"
                "2. Answer the multiple-choice question based on the visual content.\n"
                "3. Identify the time interval (in seconds) that contains the visual "
                "evidence supporting your answer.\n\n"
            )
        else:
            system_text = (
                f"You are a video question-answering agent. You are shown the video "
                f"({start_sec:.1f}s to {end_sec:.1f}s). Your task is to:\n"
                "1. Answer the multiple-choice question based on the visual content.\n"
                "2. Identify the time interval (in seconds) that contains the visual "
                "evidence supporting your answer.\n\n"
            )
        content.append({"type": "text", "text": system_text})

        # Question and choices
        choices = self._maybe_add_unanswerable(choices, include_unanswerable)
        formatted_choices = "\n".join([
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
        ])
        content.append({
            "type": "text",
            "text": f"Question: {question}\n\nChoices:\n{formatted_choices}\n\n"
        })

        # Visual input: full video as a single video_url attachment
        content.append({
            "type": "text",
            "text": f"Video [{start_sec:.1f}s-{end_sec:.1f}s]:"
        })
        if "://" in video_path:
            video_url = video_path
        else:
            video_url = f"file://{os.path.abspath(video_path)}"
        content.append({
            "type": "video_url",
            "video_url": {"url": video_url},
        })

        # JSON output prompt
        if predict_caption:
            json_template = (
                "{\n"
                '  "caption": "<description of the video segment>",\n'
                + reasoning_field +
                '  "answer": "<letter>",\n'
                '  "evidence_start": <float>,\n'
                '  "evidence_end": <float>\n'
                "}"
            )
            caption_rule = "- caption: A concise description of the visual content in the video segment\n"
        else:
            json_template = (
                "{\n"
                + reasoning_field +
                '  "answer": "<letter>",\n'
                '  "evidence_start": <float>,\n'
                '  "evidence_end": <float>\n'
                "}"
            )
            caption_rule = ""

        content.append({
            "type": "text",
            "text": (
                "\n\nBased on the visual content above, answer the question and identify "
                "the time interval containing the supporting evidence.\n\n"
                "Respond in EXACTLY this JSON format (no other text before or after):\n"
                "```json\n"
                f"{json_template}\n"
                "```\n\n"
                "Rules:\n"
                f"{caption_rule}"
                "- answer: The letter of the correct choice (A, B, C, etc.)\n"
                f"- evidence_start/evidence_end: The time interval (in seconds, between "
                f"{start_sec:.1f} and {end_sec:.1f}) containing the visual evidence "
                "for your answer.\n"
            )
        })

        # Build mm_processor_kwargs with nframes / min_pixels / max_pixels
        mm_kwargs: Dict[str, Any] = {}
        if nframes:
            mm_kwargs["nframes"] = nframes
        if min_pixels is not None:
            mm_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            mm_kwargs["max_pixels"] = max_pixels

        extra_body: Dict[str, Any] = {}
        if mm_kwargs:
            extra_body["mm_processor_kwargs"] = mm_kwargs

        # API call
        response = self._create_completion(
            method_name="direct_answer_video_native",
            messages=[{"role": "user", "content": content}],
            max_tokens=2048,
            temperature=0.3,
            extra_body=extra_body,
        )

        response_text = response.choices[0].message.content.strip()

        parsed = self._parse_direct_answer_response(
            response_text, start_sec, end_sec, choices
        )

        return parsed, response_text

    def _parse_direct_answer_response(
        self,
        response_text: str,
        start_sec: float,
        end_sec: float,
        choices: List[str],
    ) -> Dict[str, Any]:
        """Parse VLM JSON response for direct_answer with fallback handling."""
        import json
        import re

        fallback = {
            'reasoning': response_text,
            'answer': 'A',
            'evidence_start': start_sec,
            'evidence_end': end_sec,
        }

        try:
            # Try JSON in markdown code block first
            json_match = re.search(
                r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL
            )
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    print(f"Warning: No JSON found in direct_answer response: "
                          f"{response_text[:200]}")
                    return fallback

            reasoning = parsed.get('reasoning', response_text)

            # Validate answer letter
            answer_raw = parsed.get('answer', '')
            if answer_raw and isinstance(answer_raw, str):
                answer_letter = answer_raw.strip().upper()[0]
                valid_letters = {chr(65 + i) for i in range(len(choices))}
                if answer_letter not in valid_letters:
                    answer_letter = 'A'
            else:
                answer_letter = 'A'

            # Parse and validate evidence interval
            ev_start = parsed.get('evidence_start')
            ev_end = parsed.get('evidence_end')
            if ev_start is not None and ev_end is not None:
                ev_start = float(ev_start)
                ev_end = float(ev_end)
                if ev_start == ev_end:
                    ev_start, ev_end = ev_start - 1, ev_end + 1
                ev_start = max(start_sec, min(ev_start, end_sec))
                ev_end = max(start_sec, min(ev_end, end_sec))
                if ev_start > ev_end:
                    ev_start, ev_end = start_sec, end_sec
            else:
                ev_start = start_sec
                ev_end = end_sec

            return {
                'reasoning': reasoning,
                'answer': answer_letter,
                'evidence_start': ev_start,
                'evidence_end': ev_end,
            }

        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            print(f"Warning: Failed to parse direct_answer response: {e}")
            print(f"Response: {response_text[:300]}")
            return fallback

    def predict_keyframes(
        self,
        question: str,
        choices: List[str],
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        video_duration: float,
        short_side: int = -1,
        include_unanswerable: bool = False,
        skip_reasoning: bool = False,
        salvage_truncated_json: bool = False,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Predict specific keyframe timestamps and MCQ answer for a video.

        Builds a multimodal prompt showing sampled frames with timestamps,
        and asks the VLM to identify precise keyframe timestamps where
        evidence is visible, along with the MCQ answer.

        When ``salvage_truncated_json`` is True, a response whose JSON is cut
        off (e.g. by max_tokens) will be best-effort parsed for whatever
        keyframe timestamps are present, instead of falling back to a single
        midpoint timestamp.

        Returns:
            Tuple of (parsed_result_dict, raw_response_text)
        """
        # Reasoning field for JSON schema (empty when skip_reasoning is set)
        reasoning_field = '' if skip_reasoning else '  "reasoning": "<your concise reasoning>",\n'

        content = []

        # System header
        system_text = (
            "You are a video question-answering agent. You are shown frames sampled "
            f"from a video that is {video_duration:.1f} seconds long. Your task is to:\n"
            "1. Answer the multiple-choice question based on the visual content.\n"
            "2. Identify the specific timestamps (in seconds) of the key frames that "
            "contain the visual evidence needed to answer the question.\n\n"
            "You may predict up to 30 keyframe timestamps (no more). "
            "Each timestamp should pinpoint a specific moment where critical evidence "
            "is visible. Be as precise as possible.\n\n"
        )
        content.append({"type": "text", "text": system_text})

        # Question and choices
        choices = self._maybe_add_unanswerable(choices, include_unanswerable)
        formatted_choices = "\n".join([
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
        ])
        content.append({
            "type": "text",
            "text": f"Question: {question}\n\nChoices:\n{formatted_choices}\n\n"
        })

        # Video frames with timestamps
        content.append({
            "type": "text",
            "text": f"Video [{start_sec:.1f}s-{end_sec:.1f}s]:"
        })
        frame_content = self._build_video_content(frames, "", video_path, short_side)
        # Remove the empty prompt text at the end
        if frame_content and frame_content[-1]["type"] == "text":
            frame_content.pop()
        content.extend(frame_content)

        # JSON output prompt
        content.append({
            "type": "text",
            "text": (
                "\n\nBased on the visual content above, answer the question and identify "
                "the specific timestamps of key evidence frames.\n\n"
                "Respond in EXACTLY this JSON format (no other text before or after):\n"
                "```json\n"
                "{\n"
                + reasoning_field +
                '  "answer": "<letter>",\n'
                '  "keyframe_timestamps": [<float>, <float>, ...]\n'
                "}\n"
                "```\n\n"
                "Rules:\n"
                "- answer: The letter of the correct choice (A, B, C, etc.)\n"
                f"- keyframe_timestamps: A list of timestamps (in seconds, between "
                f"0 and {video_duration:.1f}) identifying the exact moments that contain "
                "the visual evidence for your answer. Be precise. Output less than 20 timestamps.\n"
                + ("" if skip_reasoning else "- Make the reasoning concise and to the point.")
            )
        })

        # API call
        response = self._create_completion(
            method_name="predict_keyframes",
            messages=[{"role": "user", "content": content}],
            max_tokens=2048,
            temperature=0.3
        )

        response_text = response.choices[0].message.content.strip()

        parsed = self._parse_predict_keyframes_response(
            response_text, start_sec, end_sec, choices,
            salvage_truncated_json=salvage_truncated_json,
        )

        return parsed, response_text

    def _salvage_truncated_keyframes(
        self,
        response_text: str,
        start_sec: float,
        end_sec: float,
        choices: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Best-effort extraction of answer + timestamps from a truncated JSON response."""
        import re

        # Locate the keyframe_timestamps array (closing ']' may be missing)
        array_match = re.search(
            r'"keyframe_timestamps"\s*:\s*\[(.*)', response_text, re.DOTALL
        )
        if not array_match:
            return None

        array_body = array_match.group(1)
        # Trim at the closing bracket if present, otherwise consume to end-of-string
        end_bracket = array_body.find(']')
        if end_bracket != -1:
            array_body = array_body[:end_bracket]
        # The trailing partial token (e.g. "73.") may be incomplete — drop it
        # by only keeping numbers that are followed by a delimiter.
        numbers = re.findall(r'-?\d+(?:\.\d+)?(?=\s*[,\]\s])', array_body)

        validated = []
        for num in numbers:
            try:
                ts = max(start_sec, min(float(num), end_sec))
                validated.append(ts)
            except (ValueError, TypeError):
                continue
        if not validated:
            return None

        # Best-effort answer letter
        answer_letter = 'A'
        ans_match = re.search(r'"answer"\s*:\s*"([A-Za-z])', response_text)
        if ans_match:
            cand = ans_match.group(1).upper()
            if cand in {chr(65 + i) for i in range(len(choices))}:
                answer_letter = cand

        # Best-effort reasoning (may be cut off; that's fine)
        reasoning = response_text
        rsn_match = re.search(
            r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', response_text, re.DOTALL
        )
        if rsn_match:
            reasoning = rsn_match.group(1)

        return {
            'reasoning': reasoning,
            'answer': answer_letter,
            'keyframe_timestamps': sorted(set(validated)),
        }

    def _parse_predict_keyframes_response(
        self,
        response_text: str,
        start_sec: float,
        end_sec: float,
        choices: List[str],
        salvage_truncated_json: bool = False,
    ) -> Dict[str, Any]:
        """Parse VLM JSON response for predict_keyframes with fallback handling.

        When ``salvage_truncated_json`` is True, an incomplete JSON response
        (e.g. cut off mid-array due to max_tokens) falls back to a best-effort
        extraction of as many keyframe timestamps as possible instead of the
        single-midpoint fallback.
        """
        import json
        import re

        midpoint = (start_sec + end_sec) / 2.0
        fallback = {
            'reasoning': response_text,
            'answer': 'A',
            'keyframe_timestamps': [midpoint],
        }

        def _maybe_salvage() -> Dict[str, Any]:
            if not salvage_truncated_json:
                return fallback
            salvaged = self._salvage_truncated_keyframes(
                response_text, start_sec, end_sec, choices
            )
            if salvaged is None:
                return fallback
            print(f"Salvaged {len(salvaged['keyframe_timestamps'])} timestamps "
                  f"from truncated predict_keyframes response")
            return salvaged

        try:
            # Try JSON in markdown code block first
            json_match = re.search(
                r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL
            )
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    print(f"Warning: No JSON found in predict_keyframes response: "
                          f"{response_text[:200]}")
                    return _maybe_salvage()

            reasoning = parsed.get('reasoning', response_text)

            # Validate answer letter
            answer_raw = parsed.get('answer', '')
            if answer_raw and isinstance(answer_raw, str):
                answer_letter = answer_raw.strip().upper()[0]
                valid_letters = {chr(65 + i) for i in range(len(choices))}
                if answer_letter not in valid_letters:
                    answer_letter = 'A'
            else:
                answer_letter = 'A'

            # Validate keyframe timestamps
            raw_timestamps = parsed.get('keyframe_timestamps', [])
            if not isinstance(raw_timestamps, list) or len(raw_timestamps) == 0:
                print(f"Warning: No keyframe_timestamps in response, using midpoint")
                return {
                    'reasoning': reasoning,
                    'answer': answer_letter,
                    'keyframe_timestamps': [midpoint],
                }

            validated = []
            for ts in raw_timestamps:
                try:
                    ts = float(ts)
                    ts = max(start_sec, min(ts, end_sec))
                    validated.append(ts)
                except (ValueError, TypeError):
                    continue

            if not validated:
                validated = [midpoint]

            validated = sorted(set(validated))

            return {
                'reasoning': reasoning,
                'answer': answer_letter,
                'keyframe_timestamps': validated,
            }

        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            print(f"Warning: Failed to parse predict_keyframes response: {e}")
            print(f"Response: {response_text[:300]}")
            return _maybe_salvage()


class CaptionLLMInterface(VLMInterface):
    """
    Two-stage interface: a captioning VLM converts frames to per-segment text
    captions, then a text-only reasoning LLM makes decisions based on those captions.

    Methods that directly use the captioning VLM (delegated):
        generate_description, compute_relevance_score, predict_answer, predict_clue_interval

    Methods that use the caption-then-LLM pipeline:
        decide_action, generate_reasoning, direct_answer, predict_keyframes
    """

    def __init__(
        self,
        # Captioning VLM config
        caption_base_url: str,
        caption_model: str,
        caption_api_key: str = "EMPTY",
        caption_timeout: int = 3600,
        # Reasoning LLM config
        llm_base_url: str = "https://api.deepseek.com",
        llm_model: str = "deepseek-reasoner",
        llm_api_key: str = "EMPTY",
        llm_timeout: int = 3600,
        # Reasoning generation control (forwarded to caption VLM)
        reasoning_use_node_text: bool = False,
        reasoning_use_node_frames: bool = True,
        reasoning_history_turns: int = 0,
        reasoning_video_representation: str = "segmented",
        reasoning_action_representation: str = "segmented",
        prompt_log_dir: Optional[str] = None,
        captioning_mode: str = "segment",  # "segment" | "frame"
        pregenerated_caption_path: Optional[str] = None,
    ):
        from openai import OpenAI

        # Internal captioning VLM (reuses GPTVLMInterface for frame extraction, descriptions, etc.)
        self._caption_vlm = GPTVLMInterface(
            base_url=caption_base_url,
            api_key=caption_api_key,
            model=caption_model,
            timeout=caption_timeout,
            reasoning_use_node_text=reasoning_use_node_text,
            reasoning_use_node_frames=reasoning_use_node_frames,
            reasoning_history_turns=reasoning_history_turns,
            reasoning_video_representation=reasoning_video_representation,
            reasoning_action_representation=reasoning_action_representation,
            prompt_log_dir=prompt_log_dir,
        )

        # Text-only reasoning LLM client
        self._llm_client = OpenAI(
            api_key=llm_api_key,
            base_url=llm_base_url,
            timeout=llm_timeout,
        )
        self._llm_model = llm_model

        # Mirror config for local use
        self.reasoning_action_representation = reasoning_action_representation
        self.reasoning_video_representation = reasoning_video_representation
        self.captioning_mode = captioning_mode

        # Prompt logging for LLM calls
        self.prompt_log_dir = prompt_log_dir
        self._prompt_counter = 0

        # Pre-generated captions
        self.pregenerated_caption_path = pregenerated_caption_path
        self._caption_cache: Dict[str, Dict[float, str]] = {}

    def _load_pregenerated_captions(self, video_path: str) -> Optional[Dict[float, str]]:
        """Load pre-generated captions for a video, with caching."""
        if self.pregenerated_caption_path is None:
            return None
        video_id = Path(video_path).stem
        if video_id in self._caption_cache:
            return self._caption_cache[video_id]
        caption_file = Path(self.pregenerated_caption_path) / f"{video_id}.json"
        if not caption_file.exists():
            self._caption_cache[video_id] = None
            return None
        with open(caption_file) as f:
            raw = json.load(f)
        captions = {float(k): v for k, v in raw.items()}
        self._caption_cache[video_id] = captions
        return captions

    def _lookup_pregenerated_caption(self, captions: Dict[float, str], timestamp: float) -> tuple:
        """Find the closest timestamp in pre-generated captions.

        Returns (matched_timestamp, caption_text).
        """
        closest_ts = min(captions.keys(), key=lambda k: abs(k - timestamp))
        return closest_ts, captions[closest_ts]

    # ── VLM-direct methods (delegated) ──────────────────────────────────

    def generate_description(
        self, question, frames, start_sec, end_sec, video_path,
        additional_context=None, short_side=-1, detailed=False,
    ) -> str:
        pregen = self._load_pregenerated_captions(video_path)
        if pregen is not None and frames:
            frame_lines = []
            for ts in frames:
                matched_ts, caption = self._lookup_pregenerated_caption(pregen, ts)
                frame_lines.append(f"[{matched_ts:.1f}s]: {caption}")
            return "\n".join(frame_lines)
        return self._caption_vlm.generate_description(
            question, frames, start_sec, end_sec, video_path,
            additional_context, short_side, detailed,
        )

    def compute_relevance_score(
        self, question, frames, start_sec, end_sec, video_path,
        text_description=None, short_side=-1,
    ) -> float:
        return self._caption_vlm.compute_relevance_score(
            question, frames, start_sec, end_sec, video_path,
            text_description, short_side,
        )

    def rank_segments(
        self, question, segments, short_side=-1,
    ) -> List[int]:
        if not segments:
            return []

        n = len(segments)
        video_path = segments[0]["video_path"]

        # Step 1: Caption segments via VLM
        child_segments = [
            {
                "segment_id": i,
                "frames": seg["frames"],
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
            }
            for i, seg in enumerate(segments)
        ]
        captions = self._caption_segments(question, child_segments, video_path, short_side, captioning_mode='segment')

        # Step 2: Build text-only prompt
        segment_lines = []
        for i, seg in enumerate(segments):
            segment_lines.append(
                f"Segment {i} [{seg['start_sec']:.1f}s - {seg['end_sec']:.1f}s]: {captions[i]}"
            )
        segments_text = "\n\n".join(segment_lines)

        prompt = (
            f"Question: {question}\n\n"
            f"You are given text descriptions of {n} video segments. "
            "Your task is to identify which segments are relevant to answering the question, "
            "and rank them from most relevant to least relevant.\n\n"
            f"{segments_text}\n\n"
            "Rules:\n"
            "- Do NOT answer the question itself.\n"
            "- Respond with the segment numbers as a comma-separated list, ranked from most relevant to least relevant.\n\n"
            "- Rank all segments even if they seem not relevant to the question."
            "Example response: 0, 5, 3"
        )

        # Step 3: LLM completion
        response = self._llm_completion(
            method_name="rank_segments",
            messages=[{"role": "user", "content": prompt}],
            # max_tokens=2048,
            temperature=0.5,
        )

        # Step 4: Parse response
        response_text = response.choices[0].message.content.strip()
        try:
            indices = []
            for token in response_text.split(","):
                token = token.strip()
                if token.isdigit():
                    idx = int(token)
                    if 0 <= idx < n and idx not in indices:
                        indices.append(idx)
            return indices
        except Exception as e:
            print(f"Warning: Failed to parse rank_segments response '{response_text}': {e}")
            return []

    def predict_clue_interval(
        self, question, frames, start_sec, end_sec, video_path, short_side=-1,
    ) -> Tuple[float, float]:
        return self._caption_vlm.predict_clue_interval(
            question, frames, start_sec, end_sec, video_path, short_side,
        )

    def detect_query_timestamps(self, question: str) -> Optional[List[float]]:
        return self._caption_vlm.detect_query_timestamps(question)

    def predict_answer(
        self, question, choices, frames, start_sec, end_sec, video_path, short_side=-1,
        include_unanswerable=False,
    ) -> Dict[str, Any]:
        return self._caption_vlm.predict_answer(
            question, choices, frames, start_sec, end_sec, video_path, short_side,
            include_unanswerable=include_unanswerable,
        )

    # ── Caption helpers ─────────────────────────────────────────────────

    def _caption_frames_individually(
        self,
        question: str,
        frames: List[float],
        video_path: str,
        short_side: int = -1,
    ) -> str:
        """Caption each frame individually, returning timestamped lines joined by newlines.

        Tries pre-generated captions first. Falls back to per-frame VLM calls.
        """
        pregen = self._load_pregenerated_captions(video_path)
        if pregen is not None:
            frame_lines = []
            for ts in frames:
                matched_ts, caption = self._lookup_pregenerated_caption(pregen, ts)
                frame_lines.append(f"[{matched_ts:.1f}s]: {caption}")
            return "\n".join(frame_lines)

        frame_lines = []
        for ts in frames:
            caption = self._caption_vlm.generate_description(
                question=question,
                frames=[ts],
                start_sec=ts,
                end_sec=ts,
                video_path=video_path,
                short_side=short_side,
            )
            frame_lines.append(f"[{ts:.1f}s]: {caption}")
        return "\n".join(frame_lines)

    def _caption_segments(
        self,
        question: str,
        child_segments: List[Dict[str, Any]],
        video_path: str,
        short_side: int = -1,
        captioning_mode: str = None,
    ) -> Dict[int, str]:
        """Generate one text caption per child segment using the captioning VLM.

        In 'segment' mode (default), all frames of a segment are captioned together.
        In 'frame' mode, each frame is captioned individually and the per-frame
        captions are concatenated with timestamps to form the segment caption.
        """
        captioning_mode = captioning_mode or self.captioning_mode
        captions = {}
        for seg in child_segments:
            if captioning_mode == "frame":
                captions[seg['segment_id']] = self._caption_frames_individually(
                    question, seg['frames'], video_path, short_side,
                )
            else:
                captions[seg['segment_id']] = self._caption_vlm.generate_description(
                    question=question,
                    frames=seg['frames'],
                    start_sec=seg['start_sec'],
                    end_sec=seg['end_sec'],
                    video_path=video_path,
                    short_side=short_side,
                    detailed=True,
                )
        return captions

    def _caption_frames(
        self,
        question: str,
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        short_side: int = -1,
        captioning_mode: str = None,
    ) -> str:
        """Caption a single set of frames using the captioning VLM.

        In 'frame' mode, each frame is captioned individually and the per-frame
        captions are concatenated with timestamps.

        If pre-generated captions are available, looks up captions by closest
        timestamp instead of calling the VLM.
        """
        captioning_mode = captioning_mode or self.captioning_mode
        if captioning_mode == "frame":
            return self._caption_frames_individually(question, frames, video_path, short_side)
        return self._caption_vlm.generate_description(
            question=question,
            frames=frames,
            start_sec=start_sec,
            end_sec=end_sec,
            video_path=video_path,
            short_side=short_side,
        )

    def _caption_observation_segments(
        self,
        question: str,
        frames: List[float],
        observation: Dict[str, Any],
        video_path: str,
        short_side: int = -1,
        captioning_mode: str = None,
    ) -> str:
        """Caption an observation's frames grouped by segments, returning formatted text."""
        captioning_mode = captioning_mode or self.captioning_mode
        segments = observation.get('segments', None)
        use_segmented = (
            self.reasoning_video_representation in ("segmented", "segmented-frames-first")
            and segments is not None
            and len(segments) > 0
        )

        if use_segmented:
            # Group frames by segment, then delegate to _caption_segments
            segment_frames = {seg['segment_id']: [] for seg in segments}
            for ts in frames:
                for seg in segments:
                    if seg['start_sec'] <= ts <= seg['end_sec']:
                        segment_frames[seg['segment_id']].append(ts)
                        break

            child_segments = [
                {
                    'segment_id': seg['segment_id'],
                    'frames': segment_frames[seg['segment_id']],
                    'start_sec': seg['start_sec'],
                    'end_sec': seg['end_sec'],
                }
                for seg in segments
                if segment_frames[seg['segment_id']]
            ]
            captions = self._caption_segments(question, child_segments, video_path, short_side, captioning_mode=captioning_mode)
            parts = [
                f"Segment {seg['segment_id']} [{seg['start_sec']:.1f}s-{seg['end_sec']:.1f}s]: {captions[seg['segment_id']]}"
                for seg in child_segments
            ]
            return "\n".join(parts)
        else:
            start = observation.get('start_sec', 0)
            end = observation.get('end_sec', 0)
            caption = self._caption_frames(question, frames, start, end, video_path, short_side, captioning_mode=captioning_mode)
            # In frame mode, _caption_frames already includes per-frame timestamps
            if captioning_mode == "frame":
                return caption
            return f"[{start:.1f}s-{end:.1f}s]: {caption}"

    # ── LLM completion helper ───────────────────────────────────────────

    def _llm_completion(self, *, method_name: str, messages: List[Dict[str, Any]], **kwargs):
        """Make a text-only completion call to the reasoning LLM with optional logging."""
        if self.prompt_log_dir is not None:
            self._prompt_counter += 1
            log_dir = Path(self.prompt_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            text = "\n".join(
                m.get("content", "") if isinstance(m.get("content"), str)
                else str(m.get("content"))
                for m in messages
            )
            params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            header = f"[{method_name}] model={self._llm_model}, {params}\n---\n"
            filename = f"{self._prompt_counter:06d}_{method_name}_llm.txt"
            log_path = log_dir / filename
            with open(log_path, "w") as f:
                f.write(header + text)

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = self._llm_client.chat.completions.create(
                    model=self._llm_model, messages=messages, **kwargs
                )
                break
            except Exception as e:
                if attempt < max_retries:
                    sleep_time = 5 * (2 ** attempt)
                    print(f"[_llm_completion] Attempt {attempt + 1} failed: {e}. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise

        # DeepSeek-R1 models may return output in reasoning_content instead of content
        if response.choices:
            msg = response.choices[0].message
            if not msg.content:
                reasoning_content = getattr(msg, 'reasoning_content', None)
                if reasoning_content:
                    msg.content = reasoning_content

        if self.prompt_log_dir is not None:
            with open(log_path, "a") as f:
                response_text = response.choices[0].message.content if response.choices else ""
                f.write(f"\n\n===== RESPONSE =====\n{response_text}")

        return response

    # ── Caption-then-LLM methods ────────────────────────────────────────

    def decide_action(
        self,
        question: str,
        choices: List[str],
        child_segments: List[Dict[str, Any]],
        trajectory_context: Dict[str, Any],
        current_start_sec: float,
        current_end_sec: float,
        current_frames: List[float],
        video_path: str,
        short_side: int = -1,
        allowed_action_list: Optional[List[str]] = None,
        navigation_context: Optional[Dict[str, Any]] = None,
        generate_captions: Optional[bool] = None,
        skip_reasoning: bool = False,
        include_unanswerable: bool = False,
        pregenerated_caption_path: Optional[str] = None,
        use_frame_captions: bool = False,
    ) -> Tuple[Dict[str, Any], str]:
        """Caption child segments, then ask reasoning LLM to decide the next action."""
        import json
        import re

        # Reasoning field for JSON schema (empty when skip_reasoning is set)
        reasoning_field = '' if skip_reasoning else '  "reasoning": "<your concise reasoning>",\n'

        allowed_actions = set(allowed_action_list) if allowed_action_list else {"ZOOM_IN", "ZOOM_OUT", "SHIFT", "ANSWER"}
        captioning_mode = "frame" if "ZOOM_IN" not in allowed_actions else self.captioning_mode

        # Step 1: Caption all child segments
        current_observation = {
            'segments': child_segments,
            'start_sec': current_start_sec,
            'end_sec': current_end_sec,
        }
        segment_captions_text = self._caption_observation_segments(
            question, current_frames, current_observation, video_path, short_side, captioning_mode=captioning_mode
        )

        # Step 2: Build text-only prompt mirroring GPTVLMInterface structure

        # -- System header (dynamically built from allowed_actions) --
        is_segmented = self.reasoning_action_representation == "segmented"
        has_navigation = "ZOOM_IN" in allowed_actions or "ZOOM_OUT" in allowed_actions or "SHIFT" in allowed_actions
        has_answer = "ANSWER" in allowed_actions
        has_zoom_in = "ZOOM_IN" in allowed_actions
        has_zoom_out = "ZOOM_OUT" in allowed_actions
        has_shift = "SHIFT" in allowed_actions
        needs_nav_fields = has_zoom_in or has_shift

        # Build action descriptions
        action_descs = []
        action_idx = 1
        if has_zoom_in:
            if is_segmented:
                action_descs.append(f"{action_idx}. ZOOM_IN <segment_id>: Examine a child segment at finer granularity.")
            else:
                action_descs.append(f"{action_idx}. ZOOM_IN <start_sec> <end_sec>: Examine a specific time range at finer granularity.")
            action_idx += 1
        if has_zoom_out:
            if is_segmented:
                action_descs.append(f"{action_idx}. ZOOM_OUT: Backtrack to the parent segment to explore a different region.")
            else:
                action_descs.append(f"{action_idx}. ZOOM_OUT: Backtrack to the parent segment to explore a different time range.")
            action_idx += 1
        if has_shift:
            if is_segmented:
                action_descs.append(f"{action_idx}. SHIFT <segment_id>: Move to a sibling segment under the same parent.")
            else:
                action_descs.append(f"{action_idx}. SHIFT <start_sec> <end_sec>: Move to a sibling time range under the same parent.")
            action_idx += 1
        if has_answer:
            action_descs.append(
                f"{action_idx}. ANSWER <letter> <evidence_start> <evidence_end>: Provide the answer to the "
                "question along with the time interval (in seconds) that contains the supporting evidence."
            )
            action_idx += 1

        num_actions = len(action_descs)
        action_list_text = "\n".join(action_descs) + "\n\n"

        if has_navigation:
            count_word = {1: "the following action", 2: "one of two actions", 3: "one of three actions", 4: "one of four actions"}.get(num_actions, f"one of {num_actions} actions")
            if is_segmented:
                system_text = (
                    "You are a video question-answering agent that navigates a long video through "
                    "hierarchical temporal search. At each step, the current video segment is divided "
                    "into non-overlapping child segments. You are given text descriptions of each child "
                    f"segment and decide {count_word}:\n\n"
                    + action_list_text
                )
            else:
                system_text = (
                    "You are a video question-answering agent that navigates a long video through "
                    "temporal search. You are given text descriptions of video segments with timestamps "
                    f"and decide {count_word}:\n\n"
                    + action_list_text
                )
        else:
            system_text = (
                "You are a video question-answering agent. The video has been divided into "
                "non-overlapping segments. You are given text descriptions of each segment and must "
                "directly provide the final answer along with the time interval containing "
                "the evidence.\n\n"
                "Your task:\n"
                + action_list_text
            )

        # Build important rules
        important_rule_parts = []
        if has_zoom_in:
            if is_segmented:
                important_rule_parts.append(
                    "- Only use ZOOM_IN when you believe a specific child segment is likely to "
                    "contain the answer and needs closer examination."
                )
            else:
                important_rule_parts.append(
                    "- Use ZOOM_IN when you believe a specific time range is likely to "
                    "contain the answer and needs closer examination. Specify the exact start and end times."
                )
        if has_zoom_out:
            if is_segmented:
                important_rule_parts.append("- Use ZOOM_OUT when none of the current child segments appear relevant.")
            else:
                important_rule_parts.append("- Use ZOOM_OUT when the current time range does not appear relevant.")
        if has_shift:
            if is_segmented:
                important_rule_parts.append(
                    "- Use SHIFT to move to a sibling segment at the same level when the current "
                    "segment is not relevant but a sibling might be."
                )
            else:
                important_rule_parts.append(
                    "- Use SHIFT to move to a sibling time range at the same level when the current "
                    "range is not relevant but a sibling might be."
                )
        if has_answer:
            if has_navigation:
                important_rule_parts.append(
                    "- Use ANSWER when you are confident you have found sufficient evidence to "
                    "answer the question. You must specify the answer letter AND the evidence "
                    "time interval."
                )
            else:
                important_rule_parts.append(
                    "You must provide your answer now based on the segment descriptions."
                )

        if has_navigation:
            important_rules = "IMPORTANT RULES:\n" + "\n".join(important_rule_parts) + "\n"
        else:
            important_rules = "IMPORTANT: " + "\n".join(important_rule_parts) + "\n"

        # -- Segment captions --
        segment_text = (
            f"Current Segment [{current_start_sec:.1f}s-{current_end_sec:.1f}s]:\n"
            f"{segment_captions_text}\n\n"
        )

        # -- Memory --
        current_node_key = (current_start_sec, current_end_sec)
        memory_text = self._caption_vlm._build_memory_text(
            trajectory_context, current_node_key=current_node_key
        )

        # -- Navigation context --
        nav_text = ""
        if navigation_context:
            nav_text = self._caption_vlm._build_navigation_context(navigation_context)

        # -- Question and choices --
        choices = self._maybe_add_unanswerable(choices, include_unanswerable)
        formatted_choices = "\n".join([
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
        ])
        question_text = f"Question: {question}\n\nChoices:\n{formatted_choices}\n\n"

        # -- JSON schema and rules (dynamically built from allowed_actions) --
        action_json_values = ' | '.join(f'"{a}"' for a in (allowed_action_list or ["ZOOM_IN", "ZOOM_OUT", "SHIFT", "ANSWER"]))

        json_field_parts = [reasoning_field, f'  "action": {action_json_values},\n']
        if needs_nav_fields:
            if is_segmented:
                json_field_parts.append('  "segment_id": <int or null>,\n')
            else:
                json_field_parts.append('  "zoom_start": <float or null>,\n')
                json_field_parts.append('  "zoom_end": <float or null>,\n')
        if has_answer:
            null_suffix = " or null" if needs_nav_fields else ""
            json_field_parts.append(f'  "answer": "<letter{null_suffix}>",\n')
            json_field_parts.append(f'  "evidence_start": <float{null_suffix}>,\n')
            json_field_parts.append(f'  "evidence_end": <float{null_suffix}>\n')
        else:
            if json_field_parts:
                json_field_parts[-1] = json_field_parts[-1].rstrip(',\n') + '\n'
        json_fields = ''.join(json_field_parts)

        rule_parts = ["Rules for the JSON:\n"]
        if has_zoom_in:
            if is_segmented:
                null_note = " Set answer/evidence to null." if has_answer else ""
                rule_parts.append(f"- For ZOOM_IN: set segment_id to the chosen child's ID.{null_note}\n")
            else:
                null_note = " Set answer/evidence to null." if has_answer else ""
                rule_parts.append(
                    "- For ZOOM_IN: set zoom_start and zoom_end to the exact time range (in seconds) "
                    f"you want to explore within the current segment.{null_note}\n"
                )
        if has_zoom_out:
            if skip_reasoning:
                rule_parts.append("- For ZOOM_OUT: set all fields except action to null.\n")
            else:
                rule_parts.append("- For ZOOM_OUT: set all fields except action and reasoning to null.\n")
        if has_shift:
            if is_segmented:
                null_note = " Set answer/evidence to null." if has_answer else ""
                rule_parts.append(f"- For SHIFT: set segment_id to the sibling segment's ID.{null_note}\n")
            else:
                null_note = " Set answer/evidence to null." if has_answer else ""
                rule_parts.append(f"- For SHIFT: set zoom_start and zoom_end to the sibling time range.{null_note}\n")
        if has_answer:
            if needs_nav_fields:
                if is_segmented:
                    rule_parts.append(
                        "- For ANSWER: set answer to the letter (A, B, C, etc.), and evidence_start/evidence_end "
                        "to the time interval. Set segment_id to null.\n"
                    )
                else:
                    rule_parts.append(
                        "- For ANSWER: set answer to the letter (A, B, C, etc.), and evidence_start/evidence_end "
                        "to the time interval. Set zoom_start/zoom_end to null.\n"
                    )
            else:
                rule_parts.append(
                    "- Set action to \"ANSWER\", answer to the letter (A, B, C, etc.), and "
                    "evidence_start/evidence_end to the time interval containing the evidence.\n"
                )
        action_rules = ''.join(rule_parts)

        analysis_text = (
            "Analyze the segment descriptions, memory, navigation context, "
            "the question, and the choices above, then decide your next action.\n\n"
            + important_rules + "\n"
            "Respond in EXACTLY this JSON format (no other text before or after):\n"
            "```json\n"
            "{\n"
            f"{json_fields}"
            "}\n"
            "```\n\n"
            + action_rules
        )

        # Assemble full prompt
        full_prompt = system_text + segment_text
        if memory_text:
            full_prompt += memory_text
        if nav_text:
            full_prompt += nav_text
        full_prompt += question_text + analysis_text

        # LLM call (text-only) with retry on parse failure
        max_retries = 3
        for attempt in range(max_retries):
            response = self._llm_completion(
                method_name="decide_action",
                messages=[{"role": "user", "content": full_prompt}],
                # max_tokens=2048,
                temperature=0.3,
            )

            response_text = response.choices[0].message.content.strip()

            try:
                parsed_decision = self._caption_vlm._parse_decide_action_response(
                    response_text, child_segments, current_start_sec, current_end_sec, choices
                )
                break
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    print(f"Warning: Parse attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                    continue
                raise
        # Compute child_segments_with_frames for attaching captions to parsed decision
        # _child_segs = child_segments or []
        # _seg_frames = {seg['segment_id']: [] for seg in _child_segs}
        # for ts in current_frames:
        #     for seg in _child_segs:
        #         if seg['start_sec'] <= ts <= seg['end_sec']:
        #             _seg_frames[seg['segment_id']].append(ts)
        #             break
        # child_segments_with_frames = [
        #     {
        #         'segment_id': seg['segment_id'],
        #         'frames': _seg_frames[seg['segment_id']],
        #         'start_sec': seg['start_sec'],
        #         'end_sec': seg['end_sec'],
        #     }
        #     for seg in _child_segs
        #     if _seg_frames[seg['segment_id']]
        # ]
        # captions = self._caption_segments(question, child_segments_with_frames, video_path, short_side, captioning_mode="segment")
        # parsed_decision['captions'] = captions
        # parsed_decision['captions'] = ''

        return parsed_decision, response_text

    def generate_reasoning(
        self,
        question: str,
        trajectory_context: Dict[str, Any],
        action_type: str,
        frames: List[float],
        current_observation: Optional[Dict[str, Any]] = None,
        target_observation: Optional[Dict[str, Any]] = None,
        short_side: int = -1,
        include_unanswerable: bool = False,
    ) -> str:
        """Caption current observation, then ask reasoning LLM to generate reasoning."""
        video_path = None
        if current_observation:
            video_path = current_observation.get('video_path')

        if not video_path:
            return self._caption_vlm._generate_fallback_reasoning(action_type)

        try:
            # System header
            system_header = (
                "You are a video question-answering agent that navigates a long video through "
                "multi-turn hierarchical interaction. At each turn you observe video segments "
                "and take one of the following actions:\n"
                "- ZOOM_IN <Segment ID>: Examine a sub-segment at finer granularity.\n"
                "- ZOOM_OUT: Backtrack to the parent segment to explore elsewhere.\n"
                "- SHIFT <Segment ID>: Shift to a sibling segment under the same parent.\n"
                "- ANSWER <answer> <evidence_start> <evidence_end>: Provide the final answer "
                "along with the evidence time interval once sufficient evidence is found.\n\n"
                "The next action has already been decided. Your task is to analyze the segments "
                "and memory, then provide reasoning that justifies the action.\n"
                "Your reasoning should describe what you observe in the video directly. "
                "Do not reference descriptions, captions, or text summaries.\n\n"
            )

            # Caption the current observation
            obs_caption = self._caption_observation_segments(
                question, frames, current_observation, video_path, short_side, captioning_mode="frame"
            )
            start_sec = current_observation.get('start_sec', 0)
            end_sec = current_observation.get('end_sec', 0)
            obs_text = f"Current Observation [{start_sec:.1f}s-{end_sec:.1f}s]:\n{obs_caption}\n\n"

            # Memory
            current_node_key = (start_sec, end_sec)
            memory_text = self._caption_vlm._build_memory_text(
                trajectory_context, current_node_key=current_node_key
            )

            # Navigation context
            nav_context = ""
            if current_observation:
                nav_context = self._caption_vlm._build_navigation_context(current_observation)

            # Build action-specific prompt
            segments = current_observation.get('segments', None)
            use_segmented = (
                self.reasoning_video_representation in ("segmented", "segmented-frames-first")
                and segments is not None
                and len(segments) > 0
            )

            if action_type == "ZOOM_IN":
                if not target_observation:
                    return self._caption_vlm._generate_fallback_reasoning("ZOOM_IN")
                target_id = target_observation.get('node_id', 0)
                target_start = target_observation.get('start_sec', 0)
                target_end = target_observation.get('end_sec', 0)

                if use_segmented:
                    segment_list = ", ".join([
                        f"Segment {s['segment_id']} [{s['start_sec']:.1f}s-{s['end_sec']:.1f}s]"
                        for s in segments
                    ])
                    action_prompt = (
                        f"Question: {question}\n\n"
                        f"The current segment has been divided into: {segment_list}.\n\n"
                        f"The next action has been decided: ZOOM_IN {target_id}. "
                        "Based on the segments, the memory, and the navigation context, "
                        "explain why this segment is the most likely to contain information "
                        "relevant to answering the question.\n\n"
                        f"Your response must end with the action: ZOOM_IN {target_id}\n\n"
                        "Write concise reasoning (2-4 sentences).\n\nYour reasoning:"
                    )
                else:
                    action_prompt = (
                        f"Question: {question}\n\n"
                        f"The next action has been decided: ZOOM_IN [{target_start:.1f}s-{target_end:.1f}s]. "
                        "Based on the segments, the memory, and the navigation context, "
                        "explain why this time range should be examined at finer granularity.\n\n"
                        f"Your response must end with the action: ZOOM_IN [{target_start:.1f}s-{target_end:.1f}s]\n\n"
                        "Write concise reasoning (2-4 sentences).\n\nYour reasoning:"
                    )

            elif action_type == "ZOOM_OUT":
                if use_segmented:
                    segment_list = ", ".join([
                        f"Segment {s['segment_id']} [{s['start_sec']:.1f}s-{s['end_sec']:.1f}s]"
                        for s in segments
                    ])
                    action_prompt = (
                        f"Question: {question}\n\n"
                        f"The current segment contains: {segment_list}.\n\n"
                        "The next action has been decided: ZOOM_OUT (backtrack to the parent segment). "
                        "Based on the segments, the memory, and the navigation context, "
                        "explain why none of these segments are relevant, making it necessary to backtrack.\n\n"
                        "Your response must end with the action: ZOOM_OUT\n\n"
                        "Write concise reasoning (2-4 sentences).\n\nYour reasoning:"
                    )
                else:
                    action_prompt = (
                        f"Question: {question}\n\n"
                        "The next action has been decided: ZOOM_OUT (backtrack to the parent segment). "
                        "Based on the segments, the memory, and the navigation context, "
                        "explain why this segment is not relevant, making it necessary to backtrack.\n\n"
                        "Your response must end with the action: ZOOM_OUT\n\n"
                        "Write concise reasoning (2-4 sentences).\n\nYour reasoning:"
                    )

            elif action_type == "SHIFT":
                if not target_observation:
                    return self._caption_vlm._generate_fallback_reasoning("SHIFT")
                target_id = target_observation.get('node_id', 0)
                target_start = target_observation.get('start_sec', 0)
                target_end = target_observation.get('end_sec', 0)

                if self.reasoning_action_representation == "segmented":
                    action_prompt = (
                        f"Question: {question}\n\n"
                        f"The next action has been decided: SHIFT {target_id} (move to a sibling segment). "
                        "Based on the segments, the memory, and the navigation context, "
                        "explain why the current segment is not relevant and why shifting to this sibling is warranted.\n\n"
                        f"Your response must end with the action: SHIFT {target_id}\n\n"
                        "Write concise reasoning (2-4 sentences).\n\nYour reasoning:"
                    )
                else:
                    action_prompt = (
                        f"Question: {question}\n\n"
                        f"The next action has been decided: SHIFT [{target_start:.1f}s-{target_end:.1f}s] (move to a sibling segment). "
                        "Based on the segments, the memory, and the navigation context, "
                        "explain why the current segment is not relevant and why shifting to this sibling is warranted.\n\n"
                        f"Your response must end with the action: SHIFT [{target_start:.1f}s-{target_end:.1f}s]\n\n"
                        "Write concise reasoning (2-4 sentences).\n\nYour reasoning:"
                    )

            elif action_type == "ANSWER":
                answer = trajectory_context.get('answer', 'Unknown')
                right_answer = trajectory_context.get('right_answer', 'Unknown')
                gt_start, gt_end = trajectory_context.get('gt_timestamps', (0, 0))
                choices_list = trajectory_context.get('choices', [])
                choices_list = self._maybe_add_unanswerable(choices_list, include_unanswerable)
                formatted_choices = "\n".join([
                    f"{chr(65 + i)}. {c}" for i, c in enumerate(choices_list)
                ])

                action_prompt = (
                    f"Question: {question}\nChoices:\n{formatted_choices}\n\n"
                    f"The correct answer is {right_answer}. {answer}, and the relevant evidence "
                    f"appears in the interval [{gt_start:.1f}s-{gt_end:.1f}s].\n\n"
                    "Based on the segments and the memory, explain what supports "
                    "this answer and why the evidence interval is relevant.\n\n"
                    f"Your response must end with: ANSWER {right_answer} {gt_start:.1f} {gt_end:.1f}\n\n"
                    "Write concise reasoning (2-4 sentences).\n\nYour reasoning:"
                )
            else:
                return f"Taking action: {action_type}"

            # Assemble full prompt
            full_prompt = system_header + obs_text
            if memory_text:
                full_prompt += memory_text
            if nav_context:
                full_prompt += nav_context
            full_prompt += action_prompt

            response = self._llm_completion(
                method_name=f"generate_{action_type.lower()}_reasoning",
                messages=[{"role": "user", "content": full_prompt}],
                # max_tokens=2048,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating reasoning: {e}")
            return self._caption_vlm._generate_fallback_reasoning(action_type)

    def direct_answer(
        self,
        question: str,
        choices: List[str],
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        short_side: int = -1,
        predict_caption: bool = False,
        skip_reasoning: bool = False,
        include_unanswerable: bool = False,
        video_native: bool = False,
    ) -> Tuple[Dict[str, Any], str]:
        """Caption frames, then ask reasoning LLM for answer + evidence interval."""
        # Reasoning field for JSON schema (empty when skip_reasoning is set)
        reasoning_field = '' if skip_reasoning else '  "reasoning": "<your concise reasoning>",\n'

        # Caption the video segment
        caption = self._caption_frames(question, frames, start_sec, end_sec, video_path, short_side, captioning_mode="frame")

        choices = self._maybe_add_unanswerable(choices, include_unanswerable)
        formatted_choices = "\n".join([
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
        ])

        prompt = (
            "You are a video question-answering agent. You are given a text description of "
            f"a video segment ({start_sec:.1f}s to {end_sec:.1f}s). Your task is to:\n"
            "1. Answer the multiple-choice question based on the description.\n"
            "2. Identify the time interval (in seconds) that contains the supporting evidence.\n\n"
            f"Video [{start_sec:.1f}s-{end_sec:.1f}s]: {caption}\n\n"
            f"Question: {question}\n\nChoices:\n{formatted_choices}\n\n"
            "Based on the description above, answer the question and identify the evidence interval.\n\n"
            "Respond in EXACTLY this JSON format (no other text before or after):\n"
            "```json\n"
            "{\n"
            + reasoning_field +
            '  "answer": "<letter>",\n'
            '  "evidence_start": <float>,\n'
            '  "evidence_end": <float>\n'
            "}\n"
            "```\n\n"
            "Rules:\n"
            "- answer: The letter of the correct choice (A, B, C, etc.)\n"
            f"- evidence_start/evidence_end: The time interval (in seconds, between "
            f"{start_sec:.1f} and {end_sec:.1f}) containing the evidence.\n"
        )

        response = self._llm_completion(
            method_name="direct_answer",
            messages=[{"role": "user", "content": prompt}],
            # max_tokens=1024,
            temperature=0.3,
        )

        response_text = response.choices[0].message.content.strip()
        parsed = self._caption_vlm._parse_direct_answer_response(
            response_text, start_sec, end_sec, choices
        )
        return parsed, response_text

    def predict_keyframes(
        self,
        question: str,
        choices: List[str],
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        video_duration: float,
        short_side: int = -1,
        include_unanswerable: bool = False,
        skip_reasoning: bool = False,
        salvage_truncated_json: bool = False,
    ) -> Tuple[Dict[str, Any], str]:
        """Caption frames, then ask reasoning LLM to predict keyframe timestamps."""
        caption = self._caption_frames(question, frames, start_sec, end_sec, video_path, short_side)

        choices = self._maybe_add_unanswerable(choices, include_unanswerable)
        formatted_choices = "\n".join([
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
        ])

        # Reasoning field for JSON schema (empty when skip_reasoning is set)
        reasoning_field = '' if skip_reasoning else '  "reasoning": "<your concise reasoning>",\n'

        prompt = (
            "You are a video question-answering agent. You are given a text description of "
            f"a video that is {video_duration:.1f} seconds long. Your task is to:\n"
            "1. Answer the multiple-choice question based on the description.\n"
            "2. Identify the specific timestamps (in seconds) of the key moments that "
            "contain the evidence needed to answer the question.\n\n"
            f"Video [{start_sec:.1f}s-{end_sec:.1f}s]: {caption}\n\n"
            f"Question: {question}\n\nChoices:\n{formatted_choices}\n\n"
            "Based on the description, answer the question and identify keyframe timestamps.\n\n"
            "Respond in EXACTLY this JSON format (no other text before or after):\n"
            "```json\n"
            "{\n"
            + reasoning_field +
            '  "answer": "<letter>",\n'
            '  "keyframe_timestamps": [<float>, <float>, ...]\n'
            "}\n"
            "```\n\n"
            "Rules:\n"
            "- answer: The letter of the correct choice (A, B, C, etc.)\n"
            f"- keyframe_timestamps: A list of timestamps (in seconds, between "
            f"0 and {video_duration:.1f}) identifying the exact moments containing evidence.\n"
            + ("" if skip_reasoning else "- Make the reasoning concise and to the point.")
        )

        response = self._llm_completion(
            method_name="predict_keyframes",
            messages=[{"role": "user", "content": prompt}],
            # max_tokens=1024,
            temperature=0.3,
        )

        response_text = response.choices[0].message.content.strip()
        parsed = self._caption_vlm._parse_predict_keyframes_response(
            response_text, start_sec, end_sec, choices,
            salvage_truncated_json=salvage_truncated_json,
        )
        return parsed, response_text


class GeminiVLMInterface(VLMInterface):
    """
    Interface for Google Gemini models.
    This is a placeholder - actual implementation would require API integration.
    """
    
    def __init__(self, api_key: str, model: str = "gemini-pro-vision"):
        """
        Initialize Gemini VLM interface.
        
        Args:
            api_key: Google API key
            model: Model name
        """
        self.api_key = api_key
        self.model = model
        # TODO: Initialize API client
    
    def generate_description(
        self,
        question: str,
        frames: List[Any],
        start_sec: float,
        end_sec: float,
        video_path: str,
        additional_context: Optional[Dict[str, Any]] = None,
        short_side: int = -1
    ) -> str:
        """Generate description using Gemini."""
        # TODO: Implement actual API call
        raise NotImplementedError("Gemini integration not yet implemented")
    
    def compute_relevance_score(
        self,
        question: str,
        frames: List[Any],
        start_sec: float,
        end_sec: float,
        video_path: str,
        text_description: Optional[str] = None,
        short_side: int = -1
    ) -> float:
        """Compute relevance score using Gemini."""
        # TODO: Implement actual API call
        raise NotImplementedError("Gemini integration not yet implemented")
    
    def generate_reasoning(
        self,
        question: str,
        trajectory_context: Dict[str, Any],
        action_type: str,
        frames: List[float],
        current_observation: Optional[Dict[str, Any]] = None,
        target_observation: Optional[Dict[str, Any]] = None,
        short_side: int = -1,
        include_unanswerable: bool = False
    ) -> str:
        """Generate reasoning using Gemini."""
        # TODO: Implement actual API call
        raise NotImplementedError("Gemini integration not yet implemented")

    def predict_clue_interval(
        self,
        question: str,
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        short_side: int = -1
    ) -> Tuple[float, float]:
        """Predict clue interval using Gemini."""
        # TODO: Implement actual API call
        raise NotImplementedError("Gemini integration not yet implemented")

    def detect_query_timestamps(self, question: str) -> Optional[List[float]]:
        """Detect query timestamps using Gemini."""
        # TODO: Implement actual API call
        raise NotImplementedError("Gemini integration not yet implemented")

    def predict_answer(
        self,
        question: str,
        choices: List[str],
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        short_side: int = -1,
        include_unanswerable: bool = False
    ) -> Dict[str, Any]:
        """Predict answer using Gemini."""
        # TODO: Implement actual API call
        raise NotImplementedError("Gemini integration not yet implemented")

    def direct_answer(
        self,
        question: str,
        choices: List[str],
        frames: List[float],
        start_sec: float,
        end_sec: float,
        video_path: str,
        short_side: int = -1,
        predict_caption: bool = False,
        skip_reasoning: bool = False,
        include_unanswerable: bool = False,
        video_native: bool = False,
    ) -> Tuple[Dict[str, Any], str]:
        """Direct answer using Gemini."""
        # TODO: Implement actual API call
        raise NotImplementedError("Gemini integration not yet implemented")
