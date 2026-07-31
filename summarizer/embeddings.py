"""Embedding helpers built on the OpenAI Embedding API."""

import openai

EMBEDDING_MODEL = "text-embedding-ada-002"


def embed_text(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """Return the embedding vector for ``text``.

    Expects ``OPENAI_API_KEY`` to already be set in the environment by the caller.
    """
    client = openai.OpenAI()
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding
