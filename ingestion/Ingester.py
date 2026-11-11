from pathlib import Path
from app.schemas import Chunk
from embeddings import EmbeddingClient


class Ingester:
    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,):
        pass

    def ingest(self, source: Path) -> list[Chunk]:
        pass