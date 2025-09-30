#!/usr/bin/env python3
"""
AIVA Vector Store
Provides semantic search and long-term memory capabilities using ChromaDB
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class AIVAVectorStore:
    """
    Vector database for AIVA's long-term memory and knowledge retrieval
    """

    def __init__(self,
                 collection_name: str = "aiva_memory",
                 persist_directory: str = "./data/chroma",
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            model_name: Sentence transformer model for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.model_name = model_name

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize embedding function
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, using default: {e}")
            # Fallback to ChromaDB's default embedding function
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            self.embedding_model = None

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing collection: {collection_name}")
        except ValueError:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "AIVA's long-term memory and knowledge base"}
            )
            logger.info(f"Created new collection: {collection_name}")

    def add_conversation(self,
                        conversation_id: str,
                        messages: List[Dict[str, str]],
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a conversation to the vector store

        Args:
            conversation_id: Unique identifier for the conversation
            messages: List of message dictionaries with 'role' and 'content'
            metadata: Additional metadata for the conversation

        Returns:
            Document ID that was added
        """
        # Combine messages into searchable text
        conversation_text = self._format_conversation_text(messages)

        # Generate document ID
        doc_id = f"conv_{conversation_id}_{hashlib.md5(conversation_text.encode()).hexdigest()[:8]}"

        # Prepare metadata
        doc_metadata = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "message_count": len(messages),
            "participants": list(set(msg["role"] for msg in messages)),
            "type": "conversation"
        }

        if metadata:
            doc_metadata.update(metadata)

        # Add to collection
        self.collection.add(
            documents=[conversation_text],
            metadatas=[doc_metadata],
            ids=[doc_id]
        )

        logger.info(f"Added conversation {conversation_id} as document {doc_id}")
        return doc_id

    def add_knowledge(self,
                     title: str,
                     content: str,
                     source: str = "manual",
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add knowledge/document to the vector store

        Args:
            title: Title of the knowledge item
            content: Content text
            source: Source of the knowledge (manual, file, web, etc.)
            tags: List of tags for categorization
            metadata: Additional metadata

        Returns:
            Document ID that was added
        """
        # Generate document ID
        doc_id = f"knowledge_{hashlib.md5(f'{title}_{content}'.encode()).hexdigest()[:8]}"

        # Prepare metadata
        doc_metadata = {
            "title": title,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "type": "knowledge",
            "word_count": len(content.split())
        }

        if tags:
            doc_metadata["tags"] = json.dumps(tags)

        if metadata:
            doc_metadata.update(metadata)

        # Add to collection
        self.collection.add(
            documents=[content],
            metadatas=[doc_metadata],
            ids=[doc_id]
        )

        logger.info(f"Added knowledge '{title}' as document {doc_id}")
        return doc_id

    def search_similar(self,
                      query: str,
                      n_results: int = 5,
                      where: Optional[Dict[str, Any]] = None,
                      where_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for similar documents

        Args:
            query: Search query
            n_results: Number of results to return
            where: Metadata filters
            where_document: Document content filters

        Returns:
            Search results with documents, metadata, and distances
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            formatted_results = []
            if results['documents'] and results['metadatas']:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results.append({
                        "id": results['ids'][0][i] if results['ids'] else f"result_{i}",
                        "document": doc,
                        "metadata": metadata,
                        "distance": distance,
                        "similarity": 1 - distance  # Convert distance to similarity
                    })

            return {
                "query": query,
                "results": formatted_results,
                "total_found": len(formatted_results)
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "query": query,
                "results": [],
                "total_found": 0,
                "error": str(e)
            }

    def get_conversation_context(self,
                                current_query: str,
                                conversation_history: Optional[List[Dict[str, str]]] = None,
                                max_context_length: int = 2000) -> str:
        """
        Get relevant context for current query from past conversations

        Args:
            current_query: Current user query
            conversation_history: Recent conversation messages
            max_context_length: Maximum context length in characters

        Returns:
            Formatted context string
        """
        context_parts = []

        # Search for similar past conversations
        search_results = self.search_similar(
            current_query,
            n_results=3,
            where={"type": "conversation"}
        )

        # Add similar conversations
        if search_results["results"]:
            context_parts.append("Similar past conversations:")
            for result in search_results["results"][:2]:  # Limit to top 2
                if result["similarity"] > 0.7:  # Only include highly similar results
                    # Truncate long documents
                    doc_preview = result["document"][:500] + "..." if len(result["document"]) > 500 else result["document"]
                    context_parts.append(f"Past context: {doc_preview}")

        # Add recent conversation history
        if conversation_history:
            recent_messages = conversation_history[-5:]  # Last 5 messages
            context_parts.append("Recent conversation:")
            for msg in recent_messages:
                context_parts.append(f"{msg['role']}: {msg['content']}")

        # Combine and truncate if needed
        full_context = "\n\n".join(context_parts)

        if len(full_context) > max_context_length:
            # Truncate while keeping most recent parts
            truncated_parts = []
            current_length = 0

            for part in reversed(context_parts):
                if current_length + len(part) + 2 <= max_context_length:  # +2 for \n\n
                    truncated_parts.insert(0, part)
                    current_length += len(part) + 2
                else:
                    break

            full_context = "\n\n".join(truncated_parts)
            if truncated_parts:
                full_context = "[Earlier context truncated]\n\n" + full_context

        return full_context

    def get_knowledge_context(self,
                             query: str,
                             tags: Optional[List[str]] = None,
                             max_results: int = 3) -> str:
        """
        Get relevant knowledge context for a query

        Args:
            query: Search query
            tags: Filter by tags if provided
            max_results: Maximum number of results

        Returns:
            Formatted knowledge context
        """
        where_clause = {"type": "knowledge"}
        if tags:
            # ChromaDB doesn't support array contains directly, so we'll filter in Python
            pass

        results = self.search_similar(query, n_results=max_results * 2, where=where_clause)

        # Filter by tags if specified
        if tags:
            filtered_results = []
            for result in results["results"]:
                result_tags = result["metadata"].get("tags")
                if result_tags:
                    try:
                        doc_tags = json.loads(result_tags)
                        if any(tag in doc_tags for tag in tags):
                            filtered_results.append(result)
                    except:
                        pass
                if len(filtered_results) >= max_results:
                    break
            results["results"] = filtered_results[:max_results]

        # Format knowledge context
        if not results["results"]:
            return ""

        context_parts = ["Relevant knowledge:"]
        for result in results["results"]:
            title = result["metadata"].get("title", "Untitled")
            content_preview = result["document"][:300] + "..." if len(result["document"]) > 300 else result["document"]
            context_parts.append(f"{title}: {content_preview}")

        return "\n\n".join(context_parts)

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store

        Args:
            document_id: ID of the document to delete

        Returns:
            True if deleted successfully
        """
        try:
            self.collection.delete(ids=[document_id])
            logger.info(f"Deleted document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    def update_document(self, document_id: str, new_content: str, new_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update a document in the vector store

        Args:
            document_id: ID of the document to update
            new_content: New content
            new_metadata: New metadata (optional)

        Returns:
            True if updated successfully
        """
        try:
            update_data = {"documents": [new_content]}

            if new_metadata:
                update_data["metadatas"] = [new_metadata]

            self.collection.update(ids=[document_id], **update_data)
            logger.info(f"Updated document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()

            # Get document types
            results = self.collection.get(include=['metadatas'])
            types_count = {}
            sources_count = {}

            if results['metadatas']:
                for metadata in results['metadatas']:
                    doc_type = metadata.get('type', 'unknown')
                    source = metadata.get('source', 'unknown')

                    types_count[doc_type] = types_count.get(doc_type, 0) + 1
                    sources_count[source] = sources_count.get(source, 0) + 1

            return {
                "total_documents": count,
                "document_types": types_count,
                "sources": sources_count,
                "embedding_model": self.model_name,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection (dangerous operation)

        Returns:
            True if cleared successfully
        """
        try:
            # Get all document IDs
            results = self.collection.get(include=[])
            if results['ids']:
                self.collection.delete(ids=results['ids'])
            logger.warning("Cleared entire collection")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def _format_conversation_text(self, messages: List[Dict[str, str]]) -> str:
        """
        Format conversation messages into searchable text

        Args:
            messages: List of message dictionaries

        Returns:
            Formatted conversation text
        """
        formatted_parts = []

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Capitalize role for better formatting
            role_cap = role.capitalize()

            # Add role prefix
            formatted_parts.append(f"{role_cap}: {content}")

        return "\n".join(formatted_parts)

    def export_collection(self, export_path: str) -> bool:
        """
        Export the collection data to JSON

        Args:
            export_path: Path to export file

        Returns:
            True if exported successfully
        """
        try:
            results = self.collection.get(include=['documents', 'metadatas'])

            export_data = {
                "collection_name": self.collection_name,
                "export_timestamp": datetime.now().isoformat(),
                "documents": []
            }

            if results['documents'] and results['metadatas']:
                for doc, metadata in zip(results['documents'], results['metadatas']):
                    export_data["documents"].append({
                        "id": metadata.get("id", ""),
                        "document": doc,
                        "metadata": metadata
                    })

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported collection to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export collection: {e}")
            return False

    def import_collection(self, import_path: str) -> bool:
        """
        Import collection data from JSON

        Args:
            import_path: Path to import file

        Returns:
            True if imported successfully
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            documents = []
            metadatas = []
            ids = []

            for doc_data in import_data.get("documents", []):
                documents.append(doc_data["document"])
                metadatas.append(doc_data["metadata"])
                ids.append(doc_data["id"])

            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )

            logger.info(f"Imported {len(documents)} documents from {import_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to import collection: {e}")
            return False

# Convenience functions
def create_vector_store(collection_name: str = "aiva_memory",
                       persist_directory: str = "./data/chroma") -> AIVAVectorStore:
    """
    Create and return a new AIVA vector store instance

    Args:
        collection_name: Name of the collection
        persist_directory: Directory to persist data

    Returns:
        AIVAVectorStore instance
    """
    return AIVAVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory
    )

def test_vector_store():
    """Test function for the vector store"""
    store = create_vector_store("test_collection")

    # Add test conversation
    messages = [
        {"role": "user", "content": "How do I implement a binary search?"},
        {"role": "assistant", "content": "Binary search is a divide and conquer algorithm..."}
    ]

    doc_id = store.add_conversation("test_conv_001", messages)
    print(f"Added conversation with ID: {doc_id}")

    # Add test knowledge
    store.add_knowledge(
        title="Binary Search Algorithm",
        content="Binary search works by repeatedly dividing the search interval in half...",
        tags=["algorithms", "search"]
    )

    # Search
    results = store.search_similar("binary search implementation")
    print(f"Found {results['total_found']} results")

    # Get context
    context = store.get_conversation_context("How does binary search work?")
    print(f"Context length: {len(context)} characters")

    print("Vector store test completed successfully!")

if __name__ == "__main__":
    test_vector_store()
