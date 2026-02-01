"""OpenAI API classifier backend using raw HTTP requests."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from bibtex_updater.organizer.backends.base import (
    AbstractClassifier,
    ClassificationResult,
)
from bibtex_updater.organizer.config import ClassifierConfig

logger = logging.getLogger(__name__)

# OpenAI API endpoint
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Token pricing (per 1M tokens, as of 2024)
OPENAI_PRICING = {
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
}

CLASSIFICATION_PROMPT = """You are an expert research librarian specializing in organizing academic papers.

Given a paper's title and abstract, classify it into the most appropriate topic(s) from the
existing collection list. You may also suggest new topics if the paper doesn't fit well.

## Existing Collections
{existing_topics}

## Taxonomy Guidelines
{taxonomy_section}

## Paper to Classify
Title: {title}
Abstract: {abstract}

## Instructions
1. Identify the PRIMARY topic that best describes this paper
2. Identify up to 2 SECONDARY topics if applicable
3. If the paper doesn't fit existing topics well, suggest new topics
4. Provide confidence scores (0.0-1.0) for each classification
5. For new topics, suggest a hierarchical ID (e.g., "ml/transformers")

Respond with a JSON object in this exact format:
{{
  "primary_topic": {{
    "topic_id": "existing_collection_key_or_new_id",
    "topic_name": "Topic Name",
    "confidence": 0.85,
    "is_new": false
  }},
  "secondary_topics": [
    {{
      "topic_id": "another_collection_key",
      "topic_name": "Another Topic",
      "confidence": 0.6,
      "is_new": false
    }}
  ],
  "suggested_new_topics": [
    {{
      "topic_id": "suggested/new/topic",
      "topic_name": "Suggested New Topic",
      "confidence": 0.7,
      "is_new": true,
      "parent_topic": "parent_collection_key_if_applicable"
    }}
  ],
  "reasoning": "Brief explanation of the classification"
}}

IMPORTANT:
- Use existing collection keys as topic_id when matching existing collections
- Only suggest new topics if confidence for existing topics is below 0.5
- Confidence should reflect how well the paper fits the topic
- Return ONLY the JSON object, no additional text
- CRITICAL: Keep topic names ATOMIC (single concepts). If a paper spans multiple concepts,
  return them as SEPARATE topics, not combined. For example, instead of
  "Causal Inference in Machine Learning", return "Causal Inference" as primary and
  "Machine Learning" as secondary. If you must combine concepts in a single topic name,
  use " - " as separator (e.g., "Reinforcement Learning - Robotics")."""


class OpenAIClassifier(AbstractClassifier):
    """Classifier using OpenAI API for paper topic classification."""

    def __init__(self, config: ClassifierConfig) -> None:
        """Initialize the OpenAI classifier.

        Args:
            config: ClassifierConfig with API key and model settings
        """
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key in config.")
        self.model = config.get_model()
        self.temperature = config.temperature
        self.client = httpx.Client(timeout=60.0)

    def classify(
        self,
        title: str,
        abstract: str | None,
        existing_topics: list[dict[str, Any]],
        taxonomy: dict[str, Any] | None = None,
    ) -> ClassificationResult:
        """Classify a paper using OpenAI API.

        Args:
            title: Paper title
            abstract: Paper abstract (may be None)
            existing_topics: List of existing Zotero collections
            taxonomy: Optional taxonomy for additional guidance

        Returns:
            ClassificationResult with predicted topics
        """
        # Build prompt
        truncated_abstract = self._truncate_abstract(abstract, max_chars=3000)
        formatted_topics = self._format_existing_topics(existing_topics)
        taxonomy_section = self._format_taxonomy(taxonomy) if taxonomy else "No specific taxonomy provided."

        prompt = CLASSIFICATION_PROMPT.format(
            existing_topics=formatted_topics,
            taxonomy_section=taxonomy_section,
            title=title,
            abstract=truncated_abstract or "(No abstract available)",
        )

        # Make API request
        try:
            response = self._call_api(prompt)
            return self._parse_response(response, existing_topics)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ClassificationResult(reasoning=f"Error: {e}")

    def _call_api(self, prompt: str) -> dict[str, Any]:
        """Call the OpenAI API.

        Args:
            prompt: The classification prompt

        Returns:
            API response as dict
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert research librarian. Respond only with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        response = self.client.post(OPENAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    def _parse_response(
        self,
        response: dict[str, Any],
        existing_topics: list[dict[str, Any]],
    ) -> ClassificationResult:
        """Parse the OpenAI API response.

        Args:
            response: Raw API response
            existing_topics: List of existing collections for mapping

        Returns:
            ClassificationResult
        """
        # Extract text content
        choices = response.get("choices", [])
        if not choices:
            return ClassificationResult(reasoning="Empty response from API", raw_response=response)

        message = choices[0].get("message", {})
        text = message.get("content", "")

        # Parse JSON
        try:
            # Handle potential markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from response: {e}")
            return ClassificationResult(
                reasoning=f"Failed to parse response: {e}",
                raw_response=response,
            )

        # Parse topics
        primary = None
        if data.get("primary_topic"):
            primary = self._parse_topic_from_response(data["primary_topic"], existing_topics)

        secondary = []
        for t in data.get("secondary_topics", []):
            secondary.append(self._parse_topic_from_response(t, existing_topics))

        suggested_new = []
        for t in data.get("suggested_new_topics", []):
            topic = self._parse_topic_from_response(t, existing_topics)
            topic.is_new = True
            suggested_new.append(topic)

        # Get token usage
        usage = response.get("usage", {})
        tokens_used = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

        return ClassificationResult(
            primary_topic=primary,
            secondary_topics=secondary,
            suggested_new_topics=suggested_new,
            reasoning=data.get("reasoning", ""),
            raw_response=response,
            tokens_used=tokens_used,
        )

    def _format_taxonomy(self, taxonomy: dict[str, Any]) -> str:
        """Format taxonomy for the prompt.

        Args:
            taxonomy: Taxonomy dict with topic hierarchy

        Returns:
            Formatted taxonomy string
        """
        if not taxonomy:
            return "No specific taxonomy provided."

        lines = []
        topics = taxonomy.get("topics", {})

        def format_topic(topic_id: str, topic_data: dict, indent: int = 0) -> None:
            prefix = "  " * indent
            name = topic_data.get("name", topic_id)
            keywords = topic_data.get("keywords", [])
            lines.append(f"{prefix}- {name}")
            if keywords:
                lines.append(f"{prefix}  Keywords: {', '.join(keywords)}")
            for child_id, child_data in topic_data.get("children", {}).items():
                format_topic(child_id, child_data, indent + 1)

        for topic_id, topic_data in topics.items():
            format_topic(topic_id, topic_data)

        return "\n".join(lines) if lines else "No specific taxonomy provided."

    def estimate_cost(self, num_papers: int, avg_abstract_length: int = 500) -> float:
        """Estimate the cost of classifying papers.

        Args:
            num_papers: Number of papers to classify
            avg_abstract_length: Average abstract length in characters

        Returns:
            Estimated cost in USD
        """
        # Estimate tokens: ~4 chars per token
        # Prompt template: ~800 tokens
        # Abstract: avg_length / 4 tokens
        # Response: ~200 tokens
        input_tokens_per_paper = 800 + (avg_abstract_length / 4)
        output_tokens_per_paper = 200

        pricing = OPENAI_PRICING.get(self.model, OPENAI_PRICING["gpt-4o-mini"])

        input_cost = (input_tokens_per_paper * num_papers / 1_000_000) * pricing["input"]
        output_cost = (output_tokens_per_paper * num_papers / 1_000_000) * pricing["output"]

        return input_cost + output_cost
