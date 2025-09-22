#!/usr/bin/env python3
"""
Enhanced Embeddings System
==========================
Advanced semantic embeddings for RAG/KAG knowledge base
Replaces simple hash-based embeddings with transformer-based semantic embeddings
"""

import os
import json
import numpy as np
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

# Import sentence transformers for semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    sentence_transformers_available = False
    print("⚠️ SentenceTransformers not available, falling back to improved embeddings")

logger = logging.getLogger(__name__)

class EnhancedEmbeddingsSystem:
    """Advanced semantic embeddings for knowledge base"""

    # Class variable for sentence transformers availability
    _sentence_transformers_available = None

    @classmethod
    def _check_sentence_transformers(cls):
        if cls._sentence_transformers_available is None:
            try:
                from sentence_transformers import SentenceTransformer
                cls._sentence_transformers_available = True
            except ImportError:
                cls._sentence_transformers_available = False
                print("⚠️ SentenceTransformers not available, falling back to improved embeddings")
        return cls._sentence_transformers_available

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "embeddings_cache"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Embedding cache
        self.embedding_cache = {}
        self.cache_file = self.cache_dir / "embeddings_cache.json"

        # Initialize model
        self.model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2

        if self._check_sentence_transformers():
            try:
                self.model = SentenceTransformer(model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                print(f"✅ Initialized SentenceTransformer: {model_name} (dim: {self.embedding_dim})")
            except Exception as e:
                print(f"❌ Failed to load SentenceTransformer: {e}")
                self._sentence_transformers_available = False

        # Load existing cache
        self._load_cache()

    def generate_embeddings(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """Generate semantic embeddings for texts"""
        if not texts:
            return np.array([])

        # Check cache first
        if use_cache:
            cached_embeddings = []
            texts_to_process = []

            for text in texts:
                cache_key = self._get_cache_key(text)
                if cache_key in self.embedding_cache:
                    cached_embeddings.append(self.embedding_cache[cache_key])
                else:
                    texts_to_process.append((text, cache_key))

            # Generate embeddings for uncached texts
            if texts_to_process:
                new_texts = [text for text, _ in texts_to_process]
                if self._check_sentence_transformers() and self.model:
                    new_embeddings = self.model.encode(new_texts, convert_to_numpy=True)

                    # Cache new embeddings
                    for (text, cache_key), embedding in zip(texts_to_process, new_embeddings):
                        self.embedding_cache[cache_key] = embedding.tolist()
                else:
                    # Fallback to improved hash-based embeddings
                    new_embeddings = self._generate_improved_embeddings(new_texts)

                    # Cache new embeddings
                    for (text, cache_key), embedding in zip(texts_to_process, new_embeddings):
                        self.embedding_cache[cache_key] = embedding.tolist()

                # Save cache
                self._save_cache()

            # Combine cached and new embeddings
            all_embeddings = []
            cache_idx = 0
            new_idx = 0

            for text in texts:
                cache_key = self._get_cache_key(text)
                if cache_key in self.embedding_cache:
                    all_embeddings.append(self.embedding_cache[cache_key])
                else:
                    if new_idx < len(new_embeddings):
                        all_embeddings.append(new_embeddings[new_idx].tolist())
                        new_idx += 1

        else:
            # Generate without cache
            if self._check_sentence_transformers() and self.model:
                all_embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
            else:
                all_embeddings = self._generate_improved_embeddings(texts).tolist()

        return np.array(all_embeddings)

    def generate_single_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Generate embedding for single text"""
        embeddings = self.generate_embeddings([text], use_cache=use_cache)
        return embeddings[0] if len(embeddings) > 0 else np.array([])

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        if embedding1.size == 0 or embedding2.size == 0:
            return 0.0

        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def find_similar_texts(self, query_embedding: np.ndarray, text_embeddings: Dict[str, np.ndarray],
                          top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar texts to query embedding"""
        similarities = []

        for text_id, embedding in text_embeddings.items():
            similarity = self.compute_similarity(query_embedding, embedding)
            similarities.append((text_id, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def _generate_improved_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate improved embeddings when transformer models unavailable"""
        embeddings = []

        for text in texts:
            # Improved approach: combine multiple hash functions with word-level features
            words = text.lower().split()
            word_hashes = []

            # Multiple hash functions for better distribution
            for word in words[:20]:  # First 20 words
                hash1 = hash(word) % 1000
                hash2 = hash(word[::-1]) % 1000  # Reverse hash
                hash3 = hash(word + "salt") % 1000  # Salted hash
                word_hashes.extend([hash1, hash2, hash3])

            # Pad or truncate to fixed size
            while len(word_hashes) < self.embedding_dim:
                word_hashes.append(0)

            word_hashes = word_hashes[:self.embedding_dim]

            # Normalize to [0, 1]
            embedding = np.array(word_hashes) / 1000.0
            embeddings.append(embedding)

        return np.array(embeddings)

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _load_cache(self):
        """Load embedding cache from file"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.embedding_cache = json.load(f)
                print(f"✅ Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            print(f"⚠️ Failed to load embedding cache: {e}")
            self.embedding_cache = {}

    def _save_cache(self):
        """Save embedding cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.embedding_cache, f, indent=2)
            print(f"💾 Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            print(f"❌ Failed to save embedding cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding system statistics"""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "sentence_transformers_available": self._check_sentence_transformers(),
            "cached_embeddings": len(self.embedding_cache),
            "cache_file_size": self.cache_file.stat().st_size if self.cache_file.exists() else 0
        }

class EnhancedRAGEmbeddings:
    """Enhanced RAG system with advanced embeddings"""

    def __init__(self, embeddings_system: EnhancedEmbeddingsSystem):
        self.embeddings_system = embeddings_system
        self.document_embeddings = {}  # doc_id -> embedding
        self.document_texts = {}  # doc_id -> text

    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Add document with semantic embedding"""
        try:
            # Generate embedding
            embedding = self.embeddings_system.generate_single_embedding(text)

            if embedding.size > 0:
                self.document_embeddings[doc_id] = embedding
                self.document_texts[doc_id] = text
                return True
            else:
                return False

        except Exception as e:
            logger.warning(f"Failed to add document {doc_id}: {e}")
            return False

    def add_documents_batch(self, documents: Dict[str, str]) -> int:
        """Add multiple documents efficiently"""
        successful = 0

        for doc_id, text in documents.items():
            if self.add_document(doc_id, text):
                successful += 1

        print(f"✅ Added {successful}/{len(documents)} documents with semantic embeddings")
        return successful

    def search_similar(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Search for documents similar to query"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings_system.generate_single_embedding(query)

            if query_embedding.size == 0:
                return []

            # Find similar documents
            similar_docs = self.embeddings_system.find_similar_texts(
                query_embedding, self.document_embeddings, top_k=top_k
            )

            # Return with text content
            results = []
            for doc_id, similarity in similar_docs:
                text = self.document_texts.get(doc_id, "")
                results.append((doc_id, similarity, text))

            return results

        except Exception as e:
            logger.warning(f"Failed to search similar documents: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG embeddings statistics"""
        return {
            "total_documents": len(self.document_embeddings),
            "embedding_system": self.embeddings_system.get_stats()
        }

def upgrade_knowledge_base_embeddings(knowledge_system, embeddings_system: EnhancedEmbeddingsSystem):
    """Upgrade existing knowledge base with enhanced embeddings"""
    print("🚀 Upgrading knowledge base with enhanced embeddings...")

    try:
        # Get existing documents
        conn = knowledge_system.db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id, content FROM documents")
        documents = cursor.fetchall()

        print(f"📊 Found {len(documents)} existing documents to upgrade")

        # Initialize enhanced RAG
        enhanced_rag = EnhancedRAGEmbeddings(embeddings_system)

        # Upgrade documents in batches
        batch_size = 10
        upgraded = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_dict = {doc_id: content for doc_id, content in batch}

            successful = enhanced_rag.add_documents_batch(batch_dict)
            upgraded += successful

            print(f"   Progress: {upgraded}/{len(documents)} documents upgraded")

        # Save enhanced embeddings
        embeddings_file = Path("enhanced_rag_embeddings.json")
        with open(embeddings_file, 'w') as f:
            json.dump({
                "document_embeddings": {k: v.tolist() for k, v in enhanced_rag.document_embeddings.items()},
                "document_texts": enhanced_rag.document_texts,
                "upgrade_timestamp": datetime.now().isoformat(),
                "embedding_system": embeddings_system.get_stats()
            }, f, indent=2)

        print(f"💾 Enhanced embeddings saved to {embeddings_file}")
        print(f"✅ Successfully upgraded {upgraded}/{len(documents)} documents")

        return enhanced_rag

    except Exception as e:
        print(f"❌ Failed to upgrade knowledge base: {e}")
        return None

def main():
    """Main enhanced embeddings system demonstration"""
    print("🚀 Enhanced Embeddings System")
    print("=" * 50)

    # Initialize enhanced embeddings
    print("🔧 Initializing enhanced embeddings system...")
    embeddings_system = EnhancedEmbeddingsSystem()

    print(f"📊 Embedding System Stats:")
    stats = embeddings_system.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Test embedding generation
    print("\n🧪 Testing embedding generation...")
    test_texts = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "Machine learning is fascinating.",
        "Artificial intelligence enables automation."
    ]

    embeddings = embeddings_system.generate_embeddings(test_texts)
    print(f"✅ Generated embeddings for {len(test_texts)} texts")
    print(f"   Shape: {embeddings.shape}")

    # Test similarity computation
    print("\n🔍 Testing similarity computation...")
    query = "A cat is resting on a carpet."
    query_embedding = embeddings_system.generate_single_embedding(query)

    similarities = []
    for i, text in enumerate(test_texts):
        similarity = embeddings_system.compute_similarity(query_embedding, embeddings[i])
        similarities.append((text, similarity))
        print(".4f")

    # Initialize enhanced RAG
    print("\n🧠 Initializing Enhanced RAG with semantic embeddings...")
    enhanced_rag = EnhancedRAGEmbeddings(embeddings_system)

    # Add test documents
    test_docs = {
        "doc1": "Machine learning algorithms can predict patterns in data.",
        "doc2": "Deep learning uses neural networks to solve complex problems.",
        "doc3": "Natural language processing helps computers understand text.",
        "doc4": "Computer vision enables machines to interpret visual information."
    }

    enhanced_rag.add_documents_batch(test_docs)

    # Test semantic search
    print("\n🔎 Testing semantic search...")
    queries = [
        "How do computers learn from data?",
        "What is artificial intelligence?",
        "How do machines see images?"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = enhanced_rag.search_similar(query, top_k=2)

        for doc_id, similarity, text in results:
            print(".4f")
            print(f"      Text: {text[:60]}...")

    # Upgrade knowledge base (commented out to avoid modifying production data)
    print("\n⚠️  Knowledge base upgrade available:")
    print("   Run upgrade_knowledge_base_embeddings() to upgrade existing documents")
    print("   This will add semantic embeddings to all existing knowledge base entries")

    print("\n✅ Enhanced Embeddings System demonstration complete!")
    print("🧠 Semantic embeddings are now ready for production use!")

if __name__ == "__main__":
    main()
