"""
t2i.eval.semantic_check — Structured semantic consistency scoring.

Evaluates generated images against expected entities, attributes, spatial
relations, counts, and excluded objects from the prompt metadata in JSONL files.

Supported backends:
  - OWL-ViT (open-vocabulary object detection via transformers)
  - BLIP-VQA (visual question answering for attribute / relation checks)

Each prompt JSONL line must contain:
  id, prompt, expected_entities, expected_attributes, expected_relations,
  expected_counts, excluded_entities
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import (
    BlipForQuestionAnswering,
    BlipProcessor,
    OwlViTForObjectDetection,
    OwlViTProcessor,
)

logger = logging.getLogger(__name__)


@dataclass
class PromptSemanticSpec:
    """Semantic expectations for one prompt."""

    id: str
    prompt: str
    expected_entities: list[str] = field(default_factory=list)
    expected_attributes: list[str] = field(default_factory=list)
    expected_relations: list[str] = field(default_factory=list)
    expected_counts: dict[str, int] = field(default_factory=dict)
    excluded_entities: list[str] = field(default_factory=list)


@dataclass
class SemanticScore:
    """Per-prompt semantic accuracy scores."""

    prompt_id: str
    entity_score: float  # fraction of expected entities found
    attribute_score: float  # fraction of expected attributes verified
    relation_score: float  # fraction of spatial relations verified
    count_score: float  # fraction of count checks passed
    exclusion_score: float  # fraction of excluded entities absent
    overall_score: float  # mean of the above


def load_prompt_specs(jsonl_path: Path) -> list[PromptSemanticSpec]:
    """
    Load prompt semantic specs from a JSONL file.

    Parameters
    ----------
    jsonl_path:
        Path to data/prompts/*.jsonl.

    Returns
    -------
    list[PromptSemanticSpec]
    """
    specs = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            specs.append(
                PromptSemanticSpec(
                    id=obj["id"],
                    prompt=obj["prompt"],
                    expected_entities=obj.get("expected_entities", []),
                    expected_attributes=obj.get("expected_attributes", []),
                    expected_relations=obj.get("expected_relations", []),
                    expected_counts=obj.get("expected_counts", {}),
                    excluded_entities=obj.get("excluded_entities", []),
                )
            )
    return specs


class OWLViTChecker:
    """
    Uses OWL-ViT to check object presence and counts in an image.

    Parameters
    ----------
    model_id:
        Hugging Face model id for OWL-ViT.
    device:
        Target device.
    score_threshold:
        Minimum detection confidence to count as a found object.
    """

    def __init__(
        self,
        model_id: str = "google/owlvit-base-patch32",
        device: str | torch.device = "cpu",
        score_threshold: float = 0.1,
    ) -> None:
        self.device = torch.device(device)
        self.score_threshold = score_threshold
        logger.info("Loading OWL-ViT from %s…", model_id)
        self.processor = OwlViTProcessor.from_pretrained(model_id)
        self.model = OwlViTForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def detect_entities(
        self,
        image: np.ndarray,
        entities: list[str],
    ) -> dict[str, bool]:
        """
        Check which entities are present in the image.

        Returns
        -------
        dict[str, bool]
            ``{entity: found}``
        """
        if not entities:
            return {}

        pil_img = PILImage.fromarray(image)
        texts = [[e] for e in entities]  # OWL-ViT expects list-of-lists

        inputs = self.processor(text=texts, images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs, threshold=self.score_threshold, target_sizes=[(image.shape[0], image.shape[1])]
        )[0]

        # scores shape: [n_detections]
        # labels shape: [n_detections]  — index into the entities list
        found = {e: False for e in entities}
        for score, label in zip(results["scores"], results["labels"]):
            if score.item() >= self.score_threshold:
                entity = entities[label.item()]
                found[entity] = True

        return found

    def count_entities(
        self,
        image: np.ndarray,
        entity: str,
        expected_count: int,
    ) -> bool:
        """Return True if the detected count matches expected_count."""
        pil_img = PILImage.fromarray(image)
        texts = [[entity]]
        inputs = self.processor(text=texts, images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs, threshold=self.score_threshold, target_sizes=[(image.shape[0], image.shape[1])]
        )[0]

        detected_count = int((results["scores"] >= self.score_threshold).sum().item())
        return detected_count == expected_count


class BLIPVQAChecker:
    """
    Uses BLIP-VQA to verify attributes and spatial relations.

    Parameters
    ----------
    model_id:
        Hugging Face model id for BLIP-VQA.
    device:
        Target device.
    """

    def __init__(
        self,
        model_id: str = "Salesforce/blip-vqa-base",
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        logger.info("Loading BLIP-VQA from %s…", model_id)
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForQuestionAnswering.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def ask(self, image: np.ndarray, question: str) -> str:
        """Return the VQA model's answer to a yes/no question."""
        pil_img = PILImage.fromarray(image)
        inputs = self.processor(pil_img, question, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out = self.model.generate(**inputs)

        return self.processor.decode(out[0], skip_special_tokens=True).strip().lower()


def score_prompt(
    image: np.ndarray,
    spec: PromptSemanticSpec,
    owlvit: Optional[OWLViTChecker] = None,
    blip: Optional[BLIPVQAChecker] = None,
) -> SemanticScore:
    """
    Score one generated image against its semantic specification.

    Parameters
    ----------
    image:
        [H, W, 3] uint8 RGB image.
    spec:
        Semantic expectations for the prompt.
    owlvit:
        OWL-ViT checker (for entity and count checks).
    blip:
        BLIP-VQA checker (for attribute and relation checks).

    Returns
    -------
    SemanticScore
    """
    # --- Entity presence ---
    entity_score = 0.0
    if spec.expected_entities and owlvit is not None:
        found = owlvit.detect_entities(image, spec.expected_entities)
        entity_score = sum(found.values()) / len(found)

    # --- Attribute checks (VQA) ---
    attribute_score = 0.0
    if spec.expected_attributes and blip is not None:
        passed = 0
        for attr in spec.expected_attributes:
            answer = blip.ask(image, f"Is there {attr} in this image?")
            if "yes" in answer:
                passed += 1
        attribute_score = passed / len(spec.expected_attributes)

    # --- Spatial relation checks (VQA) ---
    relation_score = 0.0
    if spec.expected_relations and blip is not None:
        passed = 0
        for rel in spec.expected_relations:
            answer = blip.ask(image, f"Is it true that {rel}?")
            if "yes" in answer:
                passed += 1
        relation_score = passed / len(spec.expected_relations)

    # --- Count checks ---
    count_score = 0.0
    if spec.expected_counts and owlvit is not None:
        passed = 0
        for entity, count in spec.expected_counts.items():
            if owlvit.count_entities(image, entity, count):
                passed += 1
        count_score = passed / len(spec.expected_counts)

    # --- Exclusion checks (entities that must NOT appear) ---
    exclusion_score = 0.0
    if spec.excluded_entities and owlvit is not None:
        found = owlvit.detect_entities(image, spec.excluded_entities)
        # Score = fraction of excluded entities that are absent
        absent = sum(1 for v in found.values() if not v)
        exclusion_score = absent / len(found)

    # Gather non-trivial scores for the mean
    active_scores = []
    if spec.expected_entities:
        active_scores.append(entity_score)
    if spec.expected_attributes:
        active_scores.append(attribute_score)
    if spec.expected_relations:
        active_scores.append(relation_score)
    if spec.expected_counts:
        active_scores.append(count_score)
    if spec.excluded_entities:
        active_scores.append(exclusion_score)

    overall = float(np.mean(active_scores)) if active_scores else 0.0

    return SemanticScore(
        prompt_id=spec.id,
        entity_score=entity_score,
        attribute_score=attribute_score,
        relation_score=relation_score,
        count_score=count_score,
        exclusion_score=exclusion_score,
        overall_score=overall,
    )
