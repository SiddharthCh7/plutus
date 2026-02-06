"""Qdrant vector store for long-term memory.

Portable design:
- Local Qdrant by default
- Cloud Qdrant ready (just set QDRANT_API_KEY)

Stores:
- News embeddings for semantic retrieval
- Analysis history for pattern matching
- Trend patterns for future reference
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from plutus.config import get_settings
from plutus.logging import get_logger

logger = get_logger(__name__)

# Embedding dimension (adjust based on model)
EMBEDDING_DIM = 768


class MemoryItem(BaseModel):
    """Item stored in vector memory."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(..., description="Text content")
    category: str = Field(..., description="news, analysis, trend, etc.")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Related entities
    tickers: list[str] = Field(default_factory=list)
    
    def to_payload(self) -> dict[str, Any]:
        """Convert to Qdrant payload."""
        return {
            "content": self.content,
            "category": self.category,
            "tickers": self.tickers,
            "created_at": self.created_at.isoformat(),
            **self.metadata,
        }


class QdrantMemory:
    """Vector memory store using Qdrant.
    
    Provides semantic search for:
    - Relevant past news
    - Historical analysis
    - Trend patterns
    
    Design:
    - Local by default (localhost:6333)
    - Cloud-ready (set QDRANT_API_KEY for cloud)
    """
    
    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: QdrantClient | None = None
        self._collection_name = self._settings.qdrant.collection_name
    
    @property
    def client(self) -> QdrantClient:
        """Lazy initialization of Qdrant client."""
        if self._client is None:
            qdrant_settings = self._settings.qdrant
            
            if qdrant_settings.is_cloud:
                # Cloud deployment
                self._client = QdrantClient(
                    host=qdrant_settings.host,
                    port=qdrant_settings.port,
                    api_key=qdrant_settings.api_key,
                )
                logger.info("Connected to Qdrant Cloud", host=qdrant_settings.host)
            else:
                # Local deployment
                self._client = QdrantClient(
                    host=qdrant_settings.host,
                    port=qdrant_settings.port,
                )
                logger.info(
                    "Connected to local Qdrant",
                    host=qdrant_settings.host,
                    port=qdrant_settings.port,
                )
            
            self._ensure_collection()
        
        return self._client
    
    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self._collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection", name=self._collection_name)
    
    async def store(
        self,
        content: str,
        category: str,
        embedding: list[float],
        tickers: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store an item in vector memory.
        
        Args:
            content: Text content (already summarized)
            category: Category (news, analysis, trend)
            embedding: Vector embedding of content
            tickers: Related stock tickers
            metadata: Additional metadata
            
        Returns:
            ID of stored item
        """
        item = MemoryItem(
            content=content,
            category=category,
            tickers=tickers or [],
            metadata=metadata or {},
        )
        
        self.client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=item.id,
                    vector=embedding,
                    payload=item.to_payload(),
                )
            ],
        )
        
        logger.debug(
            "Stored memory item",
            id=item.id,
            category=category,
            tickers=tickers,
        )
        
        return item.id
    
    async def search(
        self,
        query_embedding: list[float],
        category: str | None = None,
        tickers: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for relevant memories.
        
        Args:
            query_embedding: Query vector
            category: Filter by category
            tickers: Filter by related tickers
            limit: Max results
            
        Returns:
            List of matching items with scores
        """
        # Build filter
        filter_conditions = []
        
        if category:
            filter_conditions.append(
                FieldCondition(
                    key="category",
                    match=MatchValue(value=category),
                )
            )
        
        if tickers:
            # Match any of the tickers
            for ticker in tickers:
                filter_conditions.append(
                    FieldCondition(
                        key="tickers",
                        match=MatchValue(value=ticker),
                    )
                )
        
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        results = self.client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=limit,
        )
        
        return [
            {
                "id": r.id,
                "score": r.score,
                **r.payload,
            }
            for r in results
        ]
    
    async def get_recent(
        self,
        category: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent items by category (no embedding needed).
        
        Useful for getting latest news/analysis.
        """
        results = self.client.scroll(
            collection_name=self._collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category),
                    )
                ]
            ),
            limit=limit,
            order_by="created_at",
        )
        
        return [
            {
                "id": r.id,
                **r.payload,
            }
            for r in results[0]
        ]
    
    async def delete_old(
        self,
        category: str,
        days_old: int = 30,
    ) -> int:
        """Delete items older than specified days.
        
        Useful for cleaning up old news.
        """
        cutoff = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        
        # This is a simplified version - full implementation would use
        # datetime filtering which requires indexed payload fields
        logger.info(
            "Cleanup requested",
            category=category,
            days_old=days_old,
        )
        return 0  # Placeholder


# Singleton instance
_memory: QdrantMemory | None = None


def get_memory() -> QdrantMemory:
    """Get the Qdrant memory singleton."""
    global _memory
    if _memory is None:
        _memory = QdrantMemory()
    return _memory
