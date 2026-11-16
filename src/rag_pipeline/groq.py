
"""Groq chat-completion adapter for RAG answer generation."""

from typing import Any

from rag_pipeline.errors import GenerationError


class GroqProvider:
    """Generate context-grounded answers using Groq."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from groq import Groq

                client = Groq(api_key=api_key)

            except Exception as exc:
                raise GenerationError(
                    f"Could not initialize Groq client: {exc}"
                ) from exc

        self.client = client
        self.model = model

    def generate(self, prompt: str) -> str:
        """Generate an answer using the constructed RAG prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise assistant. "
                            "Answer using only the supplied retrieved context. "
                            "If the context does not contain the answer, "
                            "say that the available documents do not provide "
                            "enough information. Never invent facts."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,
                max_completion_tokens=1024,
            )

            answer = (
                response.choices[0].message.content or ""
            ).strip()

            if not answer:
                raise GenerationError(
                    "Groq returned an empty response."
                )

            return answer

        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(
                f"Groq answer generation failed: {exc}"
            ) from exc

    def close(self) -> None:
        """Close the Groq client's HTTP connection."""
        close = getattr(self.client, "close", None)

        if callable(close):
            close()
