"""Choose a real embedding provider (API or local) and implement this contract."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI

# Automatically load starter/.env if present
_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV_PATH)

logger = logging.getLogger("embeddings")

# Pricing per 1,000,000 tokens (USD)
MODEL_PRICING: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}

MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """OpenAI embedding provider adapter with token tracking, cost logging, and no mock fallbacks."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self.model_name = model_name

        # Resolve dimensions
        if dimensions is not None:
            self.dimensions = dimensions
        else:
            self.dimensions = MODEL_DIMENSIONS.get(model_name, 1536)

        # Resolve credentials - fail fast with explicit ValueError if missing or empty
        resolved_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        if not resolved_key or not resolved_key.strip():
            raise ValueError(
                "OPENAI_API_KEY is not configured or empty. "
                "Real embeddings are required; silent mock fallbacks are prohibited."
            )

        self.client = OpenAI(api_key=resolved_key)
        self.price_per_million = MODEL_PRICING.get(model_name, 0.02)

        # Cumulative usage metrics
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.total_calls: int = 0

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        if not texts:
            return []

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model_name,
            )

            prompt_tokens = (
                response.usage.prompt_tokens
                if hasattr(response, "usage") and response.usage
                else 0
            )
            cost = (prompt_tokens / 1_000_000.0) * self.price_per_million

            self.total_tokens += prompt_tokens
            self.total_cost += cost
            self.total_calls += 1

            logger.info(
                f"[Embedder] Batch {i // batch_size + 1}: {len(batch)} items | "
                f"tokens={prompt_tokens} | cost=${cost:.6f} | "
                f"cumulative_tokens={self.total_tokens} | cumulative_cost=${self.total_cost:.6f}"
            )
            print(
                f"[Embedder] Embedded {len(batch)} texts ({prompt_tokens} tokens, ${cost:.6f} USD) | "
                f"Total: {self.total_tokens} tokens, ${self.total_cost:.6f} USD"
            )

            # Preserve exact index order
            batch_vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            for vec in batch_vectors:
                if len(vec) != self.dimensions:
                    raise ValueError(
                        f"Expected {self.dimensions} dimensions, but received {len(vec)} from provider."
                    )
            all_vectors.extend(batch_vectors)

        return all_vectors

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        if not question or not question.strip():
            raise ValueError("question cannot be empty.")

        vectors = self.embed_documents([question])
        return vectors[0]
