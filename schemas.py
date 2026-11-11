from dataclasses import dataclass


@dataclass
class Document:
    id: int
    filename : str
    text : str
    
@dataclass
class Chunk:
    id : int
    document_id : int
    text : str
    embedding: list[float] | None = None
    
@dataclass
class SearchResult:
    chunk : Chunk
    score : float