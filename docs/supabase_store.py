"""Supabase store for long-term memory."""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from supabase import Client, create_client
from langgraph.store.base import BaseStore

LOGGER = logging.getLogger(__name__)

class SupabaseStore(BaseStore):
    """Store that uses Supabase for persistence."""

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        table_name: str,
        index: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the store.

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase service role key
            table_name: Name of the table to store memories
            index: Optional configuration for vector search
        """
        super().__init__(index)
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.table_name = table_name
        self._ensure_table_exists()

    def _ensure_table_exists(self) -> None:
        """Create the memories table if it doesn't exist."""
        # Note: This assumes you've already created the table in Supabase with:
        # id: uuid (primary key)
        # key: text (not null)
        # value: jsonb (not null)
        # embedding: vector(512) (for semantic search)
        # created_at: timestamptz (default: now())
        pass

    async def aget(self, key: str) -> Optional[Any]:
        """Get a value from the store."""
        try:
            response = self.supabase.table(self.table_name)\
                .select("value")\
                .eq("key", key)\
                .execute()

            if not response.data:
                return None

            return json.loads(response.data[0]["value"])
        except Exception as e:
            LOGGER.error("Error getting value from Supabase: %s", e)
            return None

    async def aset(self, key: str, value: Any) -> None:
        """Set a value in the store."""
        try:
            # Convert value to JSON string
            value_json = json.dumps(value)

            # Generate embedding if index is configured
            embedding = None
            if self._index and "embed" in self._index:
                text = value.get("content", "") if isinstance(value, dict) else str(value)
                embedding = self._index["embed"](text)

            # Upsert the record
            self.supabase.table(self.table_name)\
                .upsert({
                    "key": key,
                    "value": value_json,
                    "embedding": embedding
                })\
                .execute()
        except Exception as e:
            LOGGER.error("Error setting value in Supabase: %s", e)

    async def adelete(self, key: str) -> None:
        """Delete a value from the store."""
        try:
            self.supabase.table(self.table_name)\
                .delete()\
                .eq("key", key)\
                .execute()
        except Exception as e:
            LOGGER.error("Error deleting value from Supabase: %s", e)

    async def akeys(self) -> Sequence[str]:
        """Get all keys in the store."""
        try:
            response = self.supabase.table(self.table_name)\
                .select("key")\
                .execute()
            return [record["key"] for record in response.data]
        except Exception as e:
            LOGGER.error("Error getting keys from Supabase: %s", e)
            return []

    async def asearch(
        self, query: str, config: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Search for similar items using vector similarity."""
        if not self._index or "embed" not in self._index:
            raise ValueError("Store not configured for vector search")

        try:
            # Generate query embedding
            query_embedding = self._index["embed"](query)

            # Perform vector similarity search
            # Note: This assumes you've set up vector similarity search in Supabase
            # using pgvector extension
            response = self.supabase.rpc(
                'match_memories',  # You need to create this stored procedure
                {
                    'query_embedding': query_embedding,
                    'match_threshold': config.get('threshold', 0.8) if config else 0.8,
                    'match_count': config.get('k', 4) if config else 4
                }
            ).execute()

            return [record["key"] for record in response.data]
        except Exception as e:
            LOGGER.error("Error performing vector search in Supabase: %s", e)
            return []
