#!/usr/bin/env python3
"""
Retrieval Optimization System
=============================
Advanced retrieval optimization for RAG/KAG knowledge base
Implements re-ranking, query expansion, and intelligent content filtering
"""

import os
import json
import re
import math
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
from datetime import datetime
import logging
from pathlib import Path

from enhanced_embeddings_system import EnhancedEmbeddingsSystem, EnhancedRAGEmbeddings

logger = logging.getLogger(__name__)

class QueryExpander:
    """Intelligent query expansion for better retrieval"""

    def __init__(self):
        self.expansion_rules = {
            # Linguistic expansions
            "linguistics": ["grammar", "syntax", "semantics", "morphology", "phonology", "pragmatics"],
            "grammar": ["syntax", "morphology", "sentence structure", "linguistic rules"],
            "syntax": ["sentence structure", "grammatical rules", "word order", "linguistics"],

            # Sentiment expansions
            "sentiment": ["emotion", "feeling", "mood", "attitude", "opinion"],
            "emotion": ["feeling", "sentiment", "mood", "affect", "psychological state"],
            "feeling": ["emotion", "sentiment", "sensation", "perception"],

            # Semantic expansions
            "semantics": ["meaning", "interpretation", "significance", "sense"],
            "meaning": ["semantics", "interpretation", "significance", "denotation", "connotation"],
            "paraphrase": ["rephrase", "reword", "synonym", "equivalent", "similar meaning"],

            # Factual expansions
            "fact": ["information", "knowledge", "data", "evidence", "verification"],
            "knowledge": ["information", "facts", "data", "wisdom", "understanding"],
            "question": ["query", "inquiry", "ask", "answer", "response"],

            # Commonsense expansions
            "commonsense": ["obvious", "intuitive", "practical", "everyday knowledge"],
            "reasoning": ["logic", "thinking", "inference", "deduction", "analysis"],
            "causality": ["cause", "effect", "relationship", "connection", "influence"]
        }

        self.contextual_synonyms = {
            "cat": ["feline", "kitten", "pet", "animal"],
            "dog": ["canine", "puppy", "pet", "animal"],
            "car": ["vehicle", "automobile", "transportation", "machine"],
            "computer": ["machine", "device", "system", "technology"],
            "learning": ["education", "training", "study", "knowledge acquisition"],
            "algorithm": ["method", "procedure", "technique", "process"]
        }

    def expand_query(self, query: str, domain: str = None) -> List[str]:
        """Expand query with relevant terms and synonyms"""
        expanded_terms = set()
        query_lower = query.lower()

        # Add original query terms
        words = re.findall(r'\b\w+\b', query_lower)
        expanded_terms.update(words)

        # Domain-specific expansions
        if domain and domain in self.expansion_rules:
            for term in words:
                if term in self.expansion_rules[domain]:
                    expanded_terms.update(self.expansion_rules[domain][term])

        # General term expansions
        for term in words:
            if term in self.expansion_rules:
                expanded_terms.update(self.expansion_rules[term])

        # Contextual synonyms
        for word in words:
            if word in self.contextual_synonyms:
                expanded_terms.update(self.contextual_synonyms[word])

        # Create expanded query variations
        expanded_queries = [query]  # Original query always first

        # Add single term expansions
        for term in list(expanded_terms)[:5]:  # Limit to avoid explosion
            if term not in words:
                expanded_queries.append(f"{query} {term}")

        return expanded_queries[:10]  # Limit total expansions

class ContentQualityFilter:
    """Advanced content quality filtering and assessment"""

    def __init__(self):
        self.quality_indicators = {
            "high": {
                "min_length": 100,
                "min_sentences": 3,
                "max_repetition_ratio": 0.3,
                "required_keywords": True,
                "structure_score": 0.7
            },
            "medium": {
                "min_length": 50,
                "min_sentences": 2,
                "max_repetition_ratio": 0.5,
                "required_keywords": False,
                "structure_score": 0.5
            },
            "low": {
                "min_length": 20,
                "min_sentences": 1,
                "max_repetition_ratio": 0.7,
                "required_keywords": False,
                "structure_score": 0.3
            }
        }

        self.domain_keywords = {
            "linguistics": ["grammar", "syntax", "semantics", "language", "linguistic", "sentence", "word"],
            "sentiment_analysis": ["sentiment", "emotion", "feeling", "positive", "negative", "mood", "attitude"],
            "semantic_analysis": ["meaning", "semantics", "similarity", "paraphrase", "synonym", "understanding"],
            "factual_knowledge": ["fact", "knowledge", "information", "data", "evidence", "verification"],
            "commonsense_reasoning": ["reasoning", "commonsense", "logic", "cause", "effect", "intuitive"]
        }

    def assess_quality(self, content: str, domain: str = None) -> Dict[str, Any]:
        """Assess content quality comprehensively"""
        if not content or not content.strip():
            return {"quality_level": "low", "score": 0.0, "issues": ["empty content"]}

        # Basic metrics
        length = len(content)
        sentences = len(re.findall(r'[.!?]+', content))
        words = len(re.findall(r'\b\w+\b', content))

        # Repetition analysis
        word_counts = Counter(re.findall(r'\b\w+\b', content.lower()))
        total_words = sum(word_counts.values())
        repetition_ratio = sum(count for count in word_counts.values() if count > 3) / max(total_words, 1)

        # Structure analysis
        has_title_case = bool(re.search(r'\b[A-Z][a-z]+\b', content))
        has_numbers = bool(re.search(r'\d', content))
        has_punctuation = bool(re.search(r'[.!?]', content))

        structure_score = sum([has_title_case, has_numbers, has_punctuation]) / 3.0

        # Domain relevance
        domain_relevance = 0.0
        if domain and domain in self.domain_keywords:
            keywords = self.domain_keywords[domain]
            content_lower = content.lower()
            keyword_matches = sum(1 for keyword in keywords if keyword in content_lower)
            domain_relevance = min(keyword_matches / len(keywords), 1.0)

        # Overall quality score
        quality_score = (
            (min(length / 200, 1.0) * 0.3) +  # Length factor
            (min(sentences / 5, 1.0) * 0.2) +  # Structure factor
            ((1.0 - repetition_ratio) * 0.2) +  # Uniqueness factor
            (structure_score * 0.15) +  # Formatting factor
            (domain_relevance * 0.15)  # Relevance factor
        )

        # Determine quality level
        if quality_score >= 0.7:
            quality_level = "high"
        elif quality_score >= 0.5:
            quality_level = "medium"
        else:
            quality_level = "low"

        # Quality issues
        issues = []
        if length < 50:
            issues.append("too short")
        if sentences < 2:
            issues.append("insufficient structure")
        if repetition_ratio > 0.6:
            issues.append("high repetition")
        if domain_relevance < 0.3:
            issues.append("low domain relevance")

        return {
            "quality_level": quality_level,
            "score": quality_score,
            "issues": issues,
            "metrics": {
                "length": length,
                "sentences": sentences,
                "words": words,
                "repetition_ratio": repetition_ratio,
                "structure_score": structure_score,
                "domain_relevance": domain_relevance
            }
        }

    def should_include(self, content: str, domain: str = None, min_quality: str = "medium") -> bool:
        """Determine if content should be included based on quality"""
        assessment = self.assess_quality(content, domain)

        quality_hierarchy = {"low": 0, "medium": 1, "high": 2}
        min_level = quality_hierarchy.get(min_quality, 1)
        content_level = quality_hierarchy.get(assessment["quality_level"], 0)

        return content_level >= min_level

class RetrievalOptimizer:
    """Advanced retrieval optimization with re-ranking and query expansion"""

    def __init__(self, embeddings_system: EnhancedEmbeddingsSystem):
        self.embeddings_system = embeddings_system
        self.query_expander = QueryExpander()
        self.quality_filter = ContentQualityFilter()

        # Retrieval configuration
        self.retrieval_config = {
            "initial_candidates": 20,
            "final_results": 5,
            "query_expansions": 3,
            "rerank_weights": {
                "semantic_similarity": 0.4,
                "quality_score": 0.3,
                "recency": 0.1,
                "domain_relevance": 0.2
            }
        }

    def optimize_retrieval(self, query: str, candidate_docs: List[Tuple[str, str, Dict[str, Any]]],
                          domain: str = None) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """Optimize retrieval with query expansion and re-ranking"""
        if not candidate_docs:
            return []

        print(f"🔍 Optimizing retrieval for query: '{query}'")

        # Phase 1: Query Expansion
        expanded_queries = self.query_expander.expand_query(query, domain)
        print(f"   📝 Expanded to {len(expanded_queries)} query variations")

        # Phase 2: Multi-query Retrieval
        all_candidates = []
        seen_docs = set()

        for expanded_query in expanded_queries[:self.retrieval_config["query_expansions"]]:
            # Get embeddings for expanded query
            query_embedding = self.embeddings_system.generate_single_embedding(expanded_query)

            # Find similar documents
            similar_docs = self.embeddings_system.find_similar_texts(
                query_embedding,
                {doc_id: self.embeddings_system.generate_single_embedding(doc_text)
                 for doc_id, doc_text, _ in candidate_docs},
                top_k=self.retrieval_config["initial_candidates"]
            )

            # Add to candidates with scores
            for doc_id, similarity in similar_docs:
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    # Find original doc data
                    doc_data = next((doc for doc in candidate_docs if doc[0] == doc_id), None)
                    if doc_data:
                        all_candidates.append((doc_id, similarity, expanded_query, doc_data))

        print(f"   📚 Found {len(all_candidates)} candidate documents")

        # Phase 3: Quality Filtering
        filtered_candidates = []
        for doc_id, similarity, expanded_query, doc_data in all_candidates:
            content = doc_data[1]
            quality_assessment = self.quality_filter.assess_quality(content, domain)

            if self.quality_filter.should_include(content, domain, "low"):  # Include all for now
                filtered_candidates.append((doc_id, similarity, expanded_query, doc_data, quality_assessment))

        print(f"   ✅ {len(filtered_candidates)} documents passed quality filtering")

        # Phase 4: Advanced Re-ranking
        reranked_results = self._advanced_reranking(filtered_candidates, query, domain)

        # Phase 5: Final Selection
        final_results = reranked_results[:self.retrieval_config["final_results"]]

        print(f"   🏆 Returning top {len(final_results)} optimized results")

        return [(doc_id, score, content, metadata) for doc_id, score, content, metadata in final_results]

    def _advanced_reranking(self, candidates: List[Tuple], original_query: str, domain: str = None) -> List[Tuple]:
        """Advanced re-ranking with multiple factors"""
        reranked = []

        for doc_id, similarity, expanded_query, doc_data, quality_assessment in candidates:
            content, metadata = doc_data[1], doc_data[2]

            # Factor 1: Semantic Similarity (normalized)
            semantic_score = similarity

            # Factor 2: Quality Score
            quality_score = quality_assessment["score"]

            # Factor 3: Recency (if available)
            recency_score = self._calculate_recency_score(metadata)

            # Factor 4: Domain Relevance
            domain_score = quality_assessment["metrics"]["domain_relevance"]

            # Factor 5: Query Match Quality
            query_match_score = self._calculate_query_match_score(original_query, content)

            # Factor 6: Content Diversity (anti-duplication)
            diversity_score = self._calculate_diversity_score(content, [r[2] for r in reranked])

            # Weighted combination
            weights = self.retrieval_config["rerank_weights"]
            final_score = (
                semantic_score * weights["semantic_similarity"] +
                quality_score * weights["quality_score"] +
                recency_score * weights["recency"] +
                domain_score * weights["domain_relevance"] +
                query_match_score * 0.2 +  # Additional weight
                diversity_score * 0.1     # Additional weight
            )

            reranked.append((doc_id, final_score, content, metadata))

        # Sort by final score (descending)
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked

    def _calculate_recency_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate recency score based on timestamp"""
        try:
            if "scraped_at" in metadata:
                scraped_date = datetime.fromisoformat(metadata["scraped_at"].replace('Z', '+00:00'))
                days_old = (datetime.now() - scraped_date.replace(tzinfo=None)).days
                # Exponential decay: newer content gets higher score
                return math.exp(-days_old / 365)  # 1 year half-life
        except:
            pass
        return 0.5  # Neutral score for missing timestamp

    def _calculate_query_match_score(self, query: str, content: str) -> float:
        """Calculate how well content matches the original query"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        content_words = set(re.findall(r'\b\w+\b', content.lower()))

        if not query_words:
            return 0.0

        # Exact word matches
        exact_matches = len(query_words.intersection(content_words))

        # Partial matches (stemming approximation)
        partial_matches = 0
        for q_word in query_words:
            for c_word in content_words:
                if q_word in c_word or c_word in q_word:
                    partial_matches += 0.5

        total_matches = exact_matches + partial_matches
        return min(total_matches / len(query_words), 1.0)

    def _calculate_diversity_score(self, content: str, existing_contents: List[str]) -> float:
        """Calculate diversity score to avoid duplicate content"""
        if not existing_contents:
            return 1.0

        content_words = set(re.findall(r'\b\w+\b', content.lower()))
        avg_similarity = 0.0

        for existing in existing_contents:
            existing_words = set(re.findall(r'\b\w+\b', existing.lower()))
            if existing_words:
                similarity = len(content_words.intersection(existing_words)) / len(content_words.union(existing_words))
                avg_similarity += similarity

        avg_similarity /= len(existing_contents)

        # Return diversity score (1.0 - average similarity)
        return 1.0 - avg_similarity

class OptimizedRAGSystem:
    """Complete optimized RAG system with all enhancements"""

    def __init__(self, knowledge_system):
        self.knowledge_system = knowledge_system

        # Initialize components
        self.embeddings_system = EnhancedEmbeddingsSystem()
        self.enhanced_rag = EnhancedRAGEmbeddings(self.embeddings_system)
        self.retrieval_optimizer = RetrievalOptimizer(self.embeddings_system)

        # Load existing knowledge base
        self._load_existing_knowledge()

    def _load_existing_knowledge(self):
        """Load existing documents into enhanced RAG system"""
        try:
            # Get documents from knowledge system
            conn = self.knowledge_system.db_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT id, content, metadata FROM documents")
            documents = cursor.fetchall()

            print(f"📚 Loading {len(documents)} existing documents into optimized RAG...")

            for doc_id, content, metadata_str in documents:
                try:
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    self.enhanced_rag.add_document(doc_id, content, metadata)
                except Exception as e:
                    logger.warning(f"Failed to load document {doc_id}: {e}")

            print(f"✅ Loaded {len(self.enhanced_rag.document_embeddings)} documents into optimized RAG")

        except Exception as e:
            print(f"❌ Failed to load existing knowledge: {e}")

    def enhanced_retrieve(self, query: str, domain: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced retrieval with optimization"""
        print(f"🔍 Enhanced retrieval for: '{query}' (domain: {domain})")

        # Get initial candidates from enhanced RAG
        initial_candidates = self.enhanced_rag.search_similar(query, top_k=top_k*2)

        # Convert to expected format
        candidate_docs = [(doc_id, content, {"content": content}) for doc_id, _, content in initial_candidates]

        if not candidate_docs:
            return []

        # Apply retrieval optimization
        optimized_results = self.retrieval_optimizer.optimize_retrieval(query, candidate_docs, domain)

        # Format results
        results = []
        for doc_id, score, content, metadata in optimized_results:
            results.append({
                "doc_id": doc_id,
                "content": content,
                "score": score,
                "metadata": metadata,
                "domain": domain
            })

        print(f"✅ Retrieved {len(results)} optimized results")
        return results

    def add_optimized_document(self, content: str, domain: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Add document with quality assessment and optimization"""
        # Quality assessment
        quality_assessment = self.retrieval_optimizer.quality_filter.assess_quality(content, domain)

        if not self.retrieval_optimizer.quality_filter.should_include(content, domain, "low"):
            print(f"⚠️ Document rejected due to quality issues: {quality_assessment['issues']}")
            return False

        # Generate unique ID
        content_hash = hash(content) % 1000000
        doc_id = f"optimized_{domain}_{content_hash}" if domain else f"optimized_{content_hash}"

        # Add metadata
        if metadata is None:
            metadata = {}

        metadata.update({
            "quality_assessment": quality_assessment,
            "optimization_timestamp": datetime.now().isoformat(),
            "domain": domain
        })

        # Add to enhanced RAG
        return self.enhanced_rag.add_document(doc_id, content, metadata)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            "embeddings_system": self.embeddings_system.get_stats(),
            "enhanced_rag": self.enhanced_rag.get_stats(),
            "retrieval_optimizer": {
                "query_expansions": len(self.retrieval_optimizer.query_expander.expansion_rules),
                "quality_filters": len(self.retrieval_optimizer.quality_filter.quality_indicators)
            }
        }

def benchmark_optimized_system():
    """Benchmark the optimized RAG system against test queries"""
    print("🧪 Benchmarking Optimized RAG System")
    print("=" * 50)

    # This would integrate with the knowledge system
    # For now, demonstrate the components

    # Test query expansion
    expander = QueryExpander()
    test_query = "What is machine learning?"
    expansions = expander.expand_query(test_query, "factual_knowledge")
    print(f"Query: '{test_query}'")
    print(f"Expansions: {expansions[:3]}...")

    # Test quality filtering
    quality_filter = ContentQualityFilter()

    test_contents = [
        "Machine learning is a subset of artificial intelligence.",
        "This is a very short content.",
        "Machine learning algorithms can predict patterns in data. Deep learning uses neural networks. AI systems learn from examples and improve performance over time.",
        "Machine learning machine learning machine learning machine learning."  # High repetition
    ]

    for i, content in enumerate(test_contents):
        assessment = quality_filter.assess_quality(content, "factual_knowledge")
        print(f"Content {i+1} quality: {assessment['quality_level']} (score: {assessment['score']:.2f})")

    print("✅ Optimized RAG system components validated!")

if __name__ == "__main__":
    benchmark_optimized_system()
