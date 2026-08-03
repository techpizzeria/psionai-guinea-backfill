"""Embedding helpers built on the OpenAI Embedding API."""

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-ada-002"


def embed_text(client: OpenAI, text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """Return the embedding vector for ``text``.

    Expects an ``OpenAI`` client to be passed in by the caller.
    """
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding
