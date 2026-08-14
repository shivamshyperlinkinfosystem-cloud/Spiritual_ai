"""FastEmbeddings — fastembed ONNX wrapper implementing LangChain Embeddings."""

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings


class FastEmbeddings(Embeddings):
    """
    Wraps fastembed's TextEmbedding so it works as a LangChain Embeddings object.
    Uses ONNX Runtime — no PyTorch or torchvision required.
    """

    def __init__(self, model_name: str):
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self._model.embed([text])).tolist()
