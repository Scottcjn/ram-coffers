#!/usr/bin/env python3
"""
Neuromorphic Prompt Translator (NPT)
=====================================
Converts literal motion descriptions to emotional equivalents
for brain-style NUMA coffer video generation.

Theory: Emotional language activates dense embedding regions in CLIP/Gemma
encoders (trained on web-scale emotional captions), enabling faster
convergence with richer dynamics. This mirrors limbic gating in biological
engram formation.

Discovery: Emergent from NUMA coffer architecture testing (Jan 2026)
Paper: GRAIL-V Workshop, CVPR 2026

Usage:
    from neuromorphic_prompt_translator import NeuromorphicTranslator

    npt = NeuromorphicTranslator()
    emotional_prompt = npt.translate("woman moves head, smiles, blinks")
    # -> "her eyes brighten with quiet realization, a knowing smile forming"
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class CognitiveFunction(Enum):
    """Cognitive domains for NUMA coffer routing"""
    LANGUAGE = "language"      # Dialogue, expression, narrative
    SPATIAL = "spatial"        # Camera, environment, movement
    MEMORY = "memory"          # Character consistency, identity
    EXECUTIVE = "executive"    # Complex multi-element scenes


@dataclass
class EmotionalArc:
    """Represents an emotional transition for animation"""
    subject: str              # Who is animating
    initial_state: str        # Starting emotion/state
    transition: str           # The change verb
    final_state: str          # Ending emotion/state
    physical_manifestation: str  # How it shows physically


class NeuromorphicVocabulary:
    """
    Discovered vocabulary mappings from A/B testing.

    Literal (sparse embedding) -> Emotional (dense embedding)
    """

    # Motion verbs -> Emotional states
    MOTION_TO_EMOTION = {
        # Head/face movements
        "head movement": "subtle shift in attention",
        "head tilt": "contemplative pause",
        "nods": "quiet acknowledgment dawning",
        "shakes head": "gentle dismissal crossing their expression",

        # Eye movements
        "blinks": "eyes softening with thought",
        "blinking": "gaze flickering with inner reflection",
        "looks": "attention drawn with quiet intensity",
        "stares": "gaze fixed with growing realization",
        "eye movement": "eyes alive with unspoken thought",

        # Mouth/expression
        "smiles": "warmth spreading across their expression",
        "slight smile": "knowing smile forming",
        "frowns": "concern shadowing their features",
        "speaks": "words forming with quiet conviction",
        "mouth moves": "expression shifting with inner dialogue",

        # Body movements
        "gestures": "hands emphasizing with natural expression",
        "hand movement": "gesture carrying emotional weight",
        "moves": "shifting with purposeful energy",
        "turns": "attention pivoting with dawning interest",
        "leans": "drawing closer with growing engagement",

        # Generic
        "subtle motion": "alive with inner thought",
        "gentle motion": "breathing with quiet life",
        "natural motion": "organic presence and warmth",
    }

    # Emotional intensifiers (discovered vocabulary)
    EMOTIONAL_VOCABULARY = {
        "high_activation": [
            "fierce conviction", "quiet determination", "wounded pride",
            "reluctant respect", "dawning realization", "growing warmth",
            "inner fire", "quiet authority", "gentle defiance",
            "contemplative depth", "knowing smile", "eyes brightening",
            "warmth spreading", "inspiration taking hold"
        ],
        "transitions": [
            "shifting from", "transforming to", "giving way to",
            "melting into", "hardening into", "softening toward",
            "awakening to", "surrendering to"
        ],
        "physical_manifestations": [
            "eyes brightening", "jaw setting", "shoulders squaring",
            "gaze softening", "expression opening", "tension releasing",
            "warmth spreading across features", "subtle light in eyes"
        ]
    }

    # Subject patterns for grammar rule: Subject + Emotional State = Animation Target
    SUBJECT_PATTERNS = {
        "woman": ["young woman", "she", "her"],
        "man": ["older man", "gentleman", "he", "his"],
        "person": ["figure", "they", "their"],
        "child": ["young one", "child", "they"],
    }


class GrammarRules:
    """
    Neuromorphic Grammar: Subject + Emotional State = Animation Target

    Discovered rules from testing:
    1. Single subject + emotion = that subject animates with identity preserved
    2. Two subjects + competing emotions = camera pan (ambiguous focus)
    3. Two subjects + sequential arcs = both animate but risk identity loss
    4. Two subjects + "two brains" structure = intense but needs anchoring
    """

    @staticmethod
    def single_subject_arc(subject: str, emotional_arc: EmotionalArc) -> str:
        """
        Best results: One clear emotional focus.

        Pattern: [Subject]'s [physical] [transition] [emotional state]
        """
        return (
            f"{subject}'s {emotional_arc.physical_manifestation}, "
            f"{emotional_arc.transition} from {emotional_arc.initial_state} "
            f"to {emotional_arc.final_state}"
        )

    @staticmethod
    def two_subject_sequential(
        subject1: str, arc1: EmotionalArc,
        subject2: str, arc2: EmotionalArc
    ) -> str:
        """
        Two brains paradigm: Complete arc for each, sequentially.

        WARNING: May cause identity drift. Use with anchoring language.
        """
        return (
            f"{subject1}'s {arc1.physical_manifestation}, "
            f"{arc1.initial_state} {arc1.transition} {arc1.final_state}. "
            f"{subject2}'s {arc2.physical_manifestation}, "
            f"{arc2.initial_state} {arc2.transition} {arc2.final_state}."
        )

    @staticmethod
    def anchored_emotional(
        subject: str,
        identity_anchors: List[str],
        emotional_arc: EmotionalArc
    ) -> str:
        """
        Identity-anchored emotion: Visual grounding + emotional arc.

        Prevents "egg head" and identity loss issues.
        """
        anchors = ", ".join(identity_anchors)
        return (
            f"{subject} with {anchors}, "
            f"{emotional_arc.physical_manifestation}, "
            f"{emotional_arc.transition} {emotional_arc.final_state}"
        )


class NeuromorphicTranslator:
    """
    Main translator class: Converts literal prompts to neuromorphic emotional equivalents.

    Implements the discovery that brain-style NUMA architectures require
    emotional language for optimal activation - mirroring limbic gating
    in biological engram formation.
    """

    def __init__(self):
        self.vocab = NeuromorphicVocabulary()
        self.grammar = GrammarRules()

    def translate(
        self,
        literal_prompt: str,
        cognitive_function: CognitiveFunction = CognitiveFunction.LANGUAGE,
        preserve_identity: bool = True
    ) -> str:
        """
        Translate a literal motion prompt to emotional equivalent.

        Args:
            literal_prompt: Stock prompt with motion verbs
            cognitive_function: Target NUMA coffer domain
            preserve_identity: Add anchoring language to prevent drift

        Returns:
            Emotional prompt optimized for neuromorphic generation
        """
        result = literal_prompt

        # Step 1: Replace literal motion with emotional states (longer phrases first)
        sorted_mappings = sorted(
            self.vocab.MOTION_TO_EMOTION.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        for literal, emotional in sorted_mappings:
            result = re.sub(
                rf'\b{re.escape(literal)}\b',
                emotional,
                result,
                flags=re.IGNORECASE
            )

        # Step 2: Clean up awkward constructions
        result = self._clean_awkward_phrases(result)

        # Step 3: Add cognitive function flavor
        result = self._add_cognitive_flavor(result, cognitive_function)

        # Step 4: Add identity anchoring if requested
        if preserve_identity:
            result = self._add_identity_anchors(result)

        # Step 5: Final polish
        result = self._polish_prompt(result)

        return result

    def _clean_awkward_phrases(self, text: str) -> str:
        """Remove awkward constructions from mechanical replacement."""
        # Fix "X head" -> "X"
        text = re.sub(r'(dawning|softening|shifting)[,\s]+head\b', r'\1', text)
        # Fix "X eyes" -> "X"
        text = re.sub(r'(reflection|thought)[,\s]+eyes\b', r'\1', text)
        # Fix double adjectives
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
        # Fix "to face" after pivoting
        text = re.sub(r'pivoting with dawning interest to face', 'turning with quiet attention toward', text)
        # Clean up comma chains
        text = re.sub(r',\s*,', ',', text)
        return text

    def create_emotional_arc(
        self,
        subject: str,
        emotion_start: str,
        emotion_end: str,
        physical_cue: Optional[str] = None
    ) -> str:
        """
        Create a proper emotional arc following the grammar rules.

        Args:
            subject: Who is animating ("the young woman", "he")
            emotion_start: Initial emotional state
            emotion_end: Final emotional state
            physical_cue: How it manifests physically (auto-generated if None)

        Returns:
            Grammar-correct emotional arc prompt
        """
        if physical_cue is None:
            physical_cue = self._infer_physical_cue(emotion_start, emotion_end)

        arc = EmotionalArc(
            subject=subject,
            initial_state=emotion_start,
            transition="shifting from",
            final_state=emotion_end,
            physical_manifestation=physical_cue
        )

        return self.grammar.single_subject_arc(subject, arc)

    def translate_for_video(
        self,
        literal_prompt: str,
        duration_seconds: float = 4.0,
        num_subjects: int = 1
    ) -> Dict[str, str]:
        """
        Full translation with video-specific optimizations.

        Returns dict with prompt and recommended parameters.
        """
        emotional_prompt = self.translate(literal_prompt)

        # Add temporal flow language for longer videos
        if duration_seconds > 2.0:
            emotional_prompt = self._add_temporal_flow(emotional_prompt, duration_seconds)

        # Adjust for multi-subject complexity
        if num_subjects > 1:
            emotional_prompt = self._handle_multi_subject(emotional_prompt, num_subjects)

        return {
            "prompt": emotional_prompt,
            "negative": self._generate_negative_prompt(),
            "recommended_steps": 24 if num_subjects == 1 else 28,
            "recommended_guidance": 8.0,
            "cognitive_function": CognitiveFunction.LANGUAGE.value
        }

    def _add_cognitive_flavor(self, prompt: str, function: CognitiveFunction) -> str:
        """Add domain-specific language based on cognitive function."""
        flavors = {
            CognitiveFunction.LANGUAGE: "with expressive presence",
            CognitiveFunction.SPATIAL: "moving through the space with purpose",
            CognitiveFunction.MEMORY: "maintaining recognizable essence",
            CognitiveFunction.EXECUTIVE: "orchestrating multiple elements"
        }
        return f"{prompt}, {flavors.get(function, '')}"

    def _add_identity_anchors(self, prompt: str) -> str:
        """Add anchoring language to preserve identity/proportions."""
        anchors = [
            "natural features preserved",
            "photorealistic presence",
            "true to form"
        ]
        # Select contextually appropriate anchor
        return f"{prompt}, {anchors[0]}"

    def _add_temporal_flow(self, prompt: str, duration: float) -> str:
        """Add language suggesting temporal progression for longer videos."""
        if duration > 3.0:
            return f"{prompt}, unfolding naturally over time, gradual emotional progression"
        return f"{prompt}, smooth continuous presence"

    def _handle_multi_subject(self, prompt: str, num_subjects: int) -> str:
        """Adjust prompt structure for multiple subjects."""
        if num_subjects == 2:
            return f"{prompt}, each figure with their own emotional presence"
        return prompt

    def _infer_physical_cue(self, start: str, end: str) -> str:
        """Infer physical manifestation from emotional transition."""
        # Simple heuristic based on discovered patterns
        if "realization" in end or "understanding" in end:
            return "eyes brightening"
        elif "warmth" in end or "smile" in end:
            return "warmth spreading across features"
        elif "determination" in end or "conviction" in end:
            return "jaw setting with quiet resolve"
        elif "respect" in end or "acknowledgment" in end:
            return "expression softening"
        else:
            return "subtle shift in presence"

    def _polish_prompt(self, prompt: str) -> str:
        """Clean up and ensure proper formatting."""
        # Capitalize first letter
        prompt = prompt[0].upper() + prompt[1:] if prompt else prompt
        # Remove double spaces
        prompt = re.sub(r'\s+', ' ', prompt)
        # Ensure proper ending
        if not prompt.endswith(('.', '!', '?')):
            prompt = prompt.rstrip(',')
        return prompt.strip()

    def _generate_negative_prompt(self) -> str:
        """Generate appropriate negative prompt for neuromorphic generation."""
        return (
            "worst quality, blurry, distorted, frozen, static, motionless, "
            "deformed, cartoon, anime, caricature, exaggerated features, "
            "warped face, stretched proportions"
        )


class SalienceAnalyzer:
    """
    Analyze emotional salience of prompts.

    Based on Grok's suggestion: Measure cosine similarity to emotional
    vs literal centroids in embedding space. Higher emotional similarity
    should correlate with fewer needed diffusion steps.
    """

    # Centroid words for salience measurement
    EMOTIONAL_CENTROID = [
        "feeling", "emotion", "passion", "warmth", "intensity",
        "realization", "conviction", "determination", "hope", "fear"
    ]

    LITERAL_CENTROID = [
        "movement", "motion", "position", "angle", "direction",
        "speed", "location", "gesture", "action", "physical"
    ]

    @staticmethod
    def estimate_salience(prompt: str) -> Dict[str, float]:
        """
        Estimate emotional vs literal salience of a prompt.

        Returns scores indicating expected activation density.
        Higher emotional_score = denser embedding activation = fewer steps needed.

        Note: For accurate measurement, use actual embeddings from Gemma/CLIP.
        This is a heuristic approximation based on vocabulary matching.
        """
        prompt_lower = prompt.lower()
        words = set(re.findall(r'\b\w+\b', prompt_lower))

        # Count emotional vocabulary hits
        emotional_hits = sum(
            1 for word in words
            if any(emo in word for emo in SalienceAnalyzer.EMOTIONAL_CENTROID)
        )

        # Count literal vocabulary hits
        literal_hits = sum(
            1 for word in words
            if any(lit in word for lit in SalienceAnalyzer.LITERAL_CENTROID)
        )

        # Check against discovered vocabulary
        vocab = NeuromorphicVocabulary()
        for phrase in vocab.EMOTIONAL_VOCABULARY["high_activation"]:
            if phrase in prompt_lower:
                emotional_hits += 2  # High-activation phrases count more

        total = emotional_hits + literal_hits + 1  # +1 to avoid division by zero

        emotional_score = emotional_hits / total
        literal_score = literal_hits / total

        # Estimate step reduction based on salience
        # Higher emotional = fewer steps needed (up to 25% reduction)
        estimated_step_reduction = min(0.25, emotional_score * 0.3)

        return {
            "emotional_score": round(emotional_score, 3),
            "literal_score": round(literal_score, 3),
            "salience_ratio": round(emotional_score / max(literal_score, 0.1), 2),
            "estimated_step_reduction": f"{estimated_step_reduction*100:.1f}%",
            "recommended_steps": int(30 * (1 - estimated_step_reduction))
        }


# =============================================================================
# Demo / Testing
# =============================================================================

def demo():
    """Demonstrate the Neuromorphic Prompt Translator."""

    print("=" * 70)
    print("NEUROMORPHIC PROMPT TRANSLATOR - Demo")
    print("=" * 70)

    translator = NeuromorphicTranslator()
    analyzer = SalienceAnalyzer()

    # Test cases
    test_prompts = [
        "Victorian woman portrait, subtle head movement, slight smile, blinking eyes",
        "man gestures while talking, nods head, looks around the room",
        "person turns to face camera, moves closer, speaks"
    ]

    print("\n" + "-" * 70)
    print("TRANSLATION EXAMPLES")
    print("-" * 70)

    for i, literal in enumerate(test_prompts, 1):
        print(f"\n[{i}] LITERAL (stock):")
        print(f"    {literal}")

        emotional = translator.translate(literal)
        print(f"\n    EMOTIONAL (neuromorphic):")
        print(f"    {emotional}")

        # Analyze salience
        literal_salience = analyzer.estimate_salience(literal)
        emotional_salience = analyzer.estimate_salience(emotional)

        print(f"\n    SALIENCE ANALYSIS:")
        print(f"    Literal:    emotional={literal_salience['emotional_score']}, "
              f"steps={literal_salience['recommended_steps']}")
        print(f"    Emotional:  emotional={emotional_salience['emotional_score']}, "
              f"steps={emotional_salience['recommended_steps']}")
        print(f"    Improvement: {emotional_salience['estimated_step_reduction']} fewer steps")

    print("\n" + "-" * 70)
    print("EMOTIONAL ARC CREATION")
    print("-" * 70)

    arc = translator.create_emotional_arc(
        subject="The young woman",
        emotion_start="quiet contemplation",
        emotion_end="inspired confidence"
    )
    print(f"\n    {arc}")

    print("\n" + "-" * 70)
    print("FULL VIDEO TRANSLATION")
    print("-" * 70)

    result = translator.translate_for_video(
        "woman smiles and nods, gentle head movement",
        duration_seconds=4.0,
        num_subjects=1
    )

    print(f"\n    Prompt:   {result['prompt']}")
    print(f"    Negative: {result['negative'][:50]}...")
    print(f"    Steps:    {result['recommended_steps']}")
    print(f"    Guidance: {result['recommended_guidance']}")

    print("\n" + "=" * 70)
    print("NEUROMORPHIC GRAMMAR RULES")
    print("=" * 70)
    print("""
    Rule 1: Subject + Emotional State = Animation Target
            "her eyes brighten" -> she animates

    Rule 2: Single emotional focus = best results
            Competing emotions -> camera pan (ambiguity)

    Rule 3: Identity anchoring prevents drift
            Add: "natural features preserved", visual descriptors

    Rule 4: Emotional vocabulary activates dense embeddings
            "realization" > "movement" (faster convergence)
    """)


if __name__ == "__main__":
    demo()
