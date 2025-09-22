#!/usr/bin/env python3
"""
Content Preprocessing System
============================
Domain-specific content preprocessing pipelines for optimized knowledge integration
Implements specialized processing for different knowledge domains and content types
"""

import re
import nltk
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
import logging
from datetime import datetime
import json
from pathlib import Path

# Try to import NLTK data (download if needed)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

logger = logging.getLogger(__name__)

class DomainPreprocessor:
    """Base class for domain-specific preprocessing"""

    def __init__(self, domain: str):
        self.domain = domain
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.custom_stopwords = set()

    def preprocess(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main preprocessing pipeline"""
        if metadata is None:
            metadata = {}

        # Basic cleaning
        cleaned_content = self._basic_cleaning(content)

        # Domain-specific processing
        processed_content = self._domain_specific_processing(cleaned_content)

        # Extract features
        features = self._extract_features(processed_content)

        # Quality assessment
        quality_score = self._assess_quality(processed_content)

        return {
            "original_content": content,
            "processed_content": processed_content,
            "features": features,
            "quality_score": quality_score,
            "processing_metadata": {
                "domain": self.domain,
                "processed_at": datetime.now().isoformat(),
                "processing_steps": ["basic_cleaning", "domain_specific", "feature_extraction"]
            }
        }

    def _basic_cleaning(self, content: str) -> str:
        """Basic text cleaning"""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content.strip())

        # Remove URLs
        content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)

        # Remove email addresses
        content = re.sub(r'\S+@\S+', '', content)

        # Normalize quotes
        content = content.replace('"', '"').replace('"', '"')
        content = content.replace(''', "'").replace(''', "'")

        return content

    def _domain_specific_processing(self, content: str) -> str:
        """Override in subclasses for domain-specific processing"""
        return content

    def _extract_features(self, content: str) -> Dict[str, Any]:
        """Extract linguistic and semantic features"""
        sentences = nltk.sent_tokenize(content)
        words = nltk.word_tokenize(content)

        # Basic features
        features = {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "lexical_diversity": len(set(words)) / len(words) if words else 0,
        }

        # POS tagging
        try:
            pos_tags = nltk.pos_tag(words)
            pos_counts = Counter(tag for word, tag in pos_tags)
            features["pos_distribution"] = dict(pos_counts)
        except:
            features["pos_distribution"] = {}

        # Keyword extraction (simple TF-IDF approximation)
        features["keywords"] = self._extract_keywords(words)

        return features

    def _extract_keywords(self, words: List[str]) -> List[str]:
        """Extract important keywords"""
        # Remove stopwords and short words
        filtered_words = [
            word.lower() for word in words
            if len(word) > 3 and word.lower() not in self.stopwords
        ]

        # Get most common words
        word_freq = Counter(filtered_words)
        keywords = [word for word, freq in word_freq.most_common(10)]

        return keywords

    def _assess_quality(self, content: str) -> float:
        """Assess content quality"""
        score = 0.5  # Base score

        # Length factors
        word_count = len(content.split())
        if word_count > 50:
            score += 0.2
        elif word_count < 20:
            score -= 0.2

        # Structure factors
        sentences = content.count('.')
        if sentences > 2:
            score += 0.1

        # Diversity factors
        words = content.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * 0.2

        return max(0.0, min(1.0, score))

class LinguisticsPreprocessor(DomainPreprocessor):
    """Specialized preprocessor for linguistics content"""

    def __init__(self):
        super().__init__("linguistics")
        self.linguistic_terms = {
            "grammar", "syntax", "semantics", "morphology", "phonology",
            "pragmatics", "lexicon", "phoneme", "morpheme", "syntactic",
            "grammatical", "linguistic", "language", "utterance", "discourse"
        }

    def _domain_specific_processing(self, content: str) -> str:
        """Linguistics-specific processing"""
        # Preserve linguistic terminology
        # Add context for technical terms
        processed = content

        # Identify and tag linguistic concepts
        for term in self.linguistic_terms:
            if term in processed.lower():
                # Could add annotations or expansions here
                pass

        # Clean up common linguistic notation
        processed = re.sub(r'\[([^\]]+)\]', r'(linguistic annotation: \1)', processed)

        return processed

    def _extract_features(self, content: str) -> Dict[str, Any]:
        """Extract linguistics-specific features"""
        features = super()._extract_features(content)

        # Linguistic complexity metrics
        words = content.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        # Technical term density
        technical_terms = sum(1 for word in words if word.lower() in self.linguistic_terms)
        features["technical_term_density"] = technical_terms / len(words) if words else 0

        # Sentence complexity (clauses, etc.)
        sentences = nltk.sent_tokenize(content)
        complex_sentences = sum(1 for sent in sentences if len(sent.split()) > 20)
        features["complex_sentence_ratio"] = complex_sentences / len(sentences) if sentences else 0

        features["linguistic_complexity_score"] = (
            avg_word_length * 0.3 +
            features["technical_term_density"] * 0.4 +
            features["complex_sentence_ratio"] * 0.3
        )

        return features

class SentimentAnalysisPreprocessor(DomainPreprocessor):
    """Specialized preprocessor for sentiment analysis content"""

    def __init__(self):
        super().__init__("sentiment_analysis")
        self.emotion_words = {
            "happy", "sad", "angry", "fear", "surprise", "disgust",
            "joy", "grief", "rage", "anxiety", "amazement", "revulsion",
            "positive", "negative", "neutral", "emotional", "mood"
        }

        self.sentiment_indicators = {
            "very", "extremely", "slightly", "somewhat", "quite",
            "really", "absolutely", "completely", "totally"
        }

    def _domain_specific_processing(self, content: str) -> str:
        """Sentiment-specific processing"""
        processed = content

        # Preserve emotional context
        # Identify sentiment-bearing phrases
        processed = re.sub(r'\b(very|extremely|absolutely)\s+(\w+)', r'\1_\2', processed)

        return processed

    def _extract_features(self, content: str) -> Dict[str, Any]:
        """Extract sentiment-specific features"""
        features = super()._extract_features(content)

        words = content.lower().split()

        # Emotion word density
        emotion_count = sum(1 for word in words if word in self.emotion_words)
        features["emotion_density"] = emotion_count / len(words) if words else 0

        # Intensifier usage
        intensifier_count = sum(1 for word in words if word in self.sentiment_indicators)
        features["intensifier_density"] = intensifier_count / len(words) if words else 0

        # Sentiment polarity indicators
        positive_indicators = sum(1 for word in words if word in ["good", "great", "excellent", "amazing", "wonderful"])
        negative_indicators = sum(1 for word in words if word in ["bad", "terrible", "awful", "horrible", "hate"])

        features["sentiment_polarity_ratio"] = (positive_indicators - negative_indicators) / len(words) if words else 0

        # Contextual sentiment complexity
        features["sentiment_complexity"] = (
            features["emotion_density"] * 0.4 +
            features["intensifier_density"] * 0.3 +
            abs(features["sentiment_polarity_ratio"]) * 0.3
        )

        return features

class SemanticAnalysisPreprocessor(DomainPreprocessor):
    """Specialized preprocessor for semantic analysis content"""

    def __init__(self):
        super().__init__("semantic_analysis")
        self.semantic_relations = {
            "synonym", "antonym", "hyponym", "hypernym", "meronym",
            "holonym", "similar", "related", "means", "implies",
            "entails", "contradicts", "paraphrase", "equivalent"
        }

    def _domain_specific_processing(self, content: str) -> str:
        """Semantic-specific processing"""
        processed = content

        # Preserve semantic relation indicators
        # Clean up semantic notation
        processed = re.sub(r'(\w+)\s*≈\s*(\w+)', r'\1 is similar to \2', processed)
        processed = re.sub(r'(\w+)\s*≠\s*(\w+)', r'\1 is different from \2', processed)

        return processed

    def _extract_features(self, content: str) -> Dict[str, Any]:
        """Extract semantic-specific features"""
        features = super()._extract_features(content)

        words = content.lower().split()

        # Semantic relation density
        relation_count = sum(1 for word in words if word in self.semantic_relations)
        features["semantic_relation_density"] = relation_count / len(words) if words else 0

        # Word pair analysis (potential semantic relationships)
        sentences = nltk.sent_tokenize(content)
        word_pairs = []
        for sentence in sentences:
            sent_words = nltk.word_tokenize(sentence.lower())
            # Look for word pairs that might indicate semantic relationships
            for i in range(len(sent_words) - 1):
                if sent_words[i] in ["is", "means", "refers", "denotes"]:
                    word_pairs.append((sent_words[i-1] if i > 0 else "", sent_words[i+1]))

        features["semantic_pair_count"] = len(word_pairs)
        features["semantic_complexity"] = (
            features["semantic_relation_density"] * 0.5 +
            features["semantic_pair_count"] / len(sentences) * 0.5 if sentences else 0
        )

        return features

class FactualKnowledgePreprocessor(DomainPreprocessor):
    """Specialized preprocessor for factual knowledge content"""

    def __init__(self):
        super().__init__("factual_knowledge")
        self.factual_indicators = {
            "fact", "true", "false", "evidence", "proven", "verified",
            "according", "research", "study", "data", "statistics"
        }

    def _domain_specific_processing(self, content: str) -> str:
        """Factual-specific processing"""
        processed = content

        # Preserve factual citations and references
        # Clean up citation formatting
        processed = re.sub(r'\[(\d+)\]', r'(reference \1)', processed)

        # Normalize factual statements
        processed = re.sub(r'\b(it is known that|it has been shown that|research shows that)\b',
                          r'According to research,', processed, flags=re.IGNORECASE)

        return processed

    def _extract_features(self, content: str) -> Dict[str, Any]:
        """Extract factual-specific features"""
        features = super()._extract_features(content)

        words = content.lower().split()

        # Factual indicator density
        factual_count = sum(1 for word in words if word in self.factual_indicators)
        features["factual_density"] = factual_count / len(words) if words else 0

        # Citation analysis
        citations = len(re.findall(r'\(reference \d+\)', content))
        features["citation_count"] = citations

        # Claim density (sentences that make assertions)
        sentences = nltk.sent_tokenize(content)
        claim_sentences = 0
        for sentence in sentences:
            # Simple heuristic: sentences with "is" or "are" are likely claims
            if re.search(r'\bis\b|\bare\b|\bwas\b|\bwere\b', sentence.lower()):
                claim_sentences += 1

        features["claim_density"] = claim_sentences / len(sentences) if sentences else 0

        # Factual reliability score
        features["factual_reliability"] = (
            features["factual_density"] * 0.4 +
            features["citation_count"] / len(sentences) * 0.3 if sentences else 0 +
            features["claim_density"] * 0.3
        )

        return features

class CommonsenseReasoningPreprocessor(DomainPreprocessor):
    """Specialized preprocessor for commonsense reasoning content"""

    def __init__(self):
        super().__init__("commonsense_reasoning")
        self.commonsense_indicators = {
            "obviously", "clearly", "normally", "usually", "typically",
            "generally", "commonly", "naturally", "logically", "makes sense",
            "intuitive", "reasonable", "practical", "everyday"
        }

        self.causal_words = {
            "because", "so", "therefore", "thus", "consequently",
            "due to", "caused by", "results in", "leads to", "why"
        }

    def _domain_specific_processing(self, content: str) -> str:
        """Commonsense-specific processing"""
        processed = content

        # Preserve causal relationships
        # Enhance commonsense reasoning markers
        processed = re.sub(r'\b(it makes sense|obviously|clearly)\b',
                          lambda m: f"From commonsense reasoning: {m.group()}", processed, flags=re.IGNORECASE)

        return processed

    def _extract_features(self, content: str) -> Dict[str, Any]:
        """Extract commonsense-specific features"""
        features = super()._extract_features(content)

        words = content.lower().split()

        # Commonsense indicator density
        commonsense_count = sum(1 for word in words if word in self.commonsense_indicators)
        features["commonsense_density"] = commonsense_count / len(words) if words else 0

        # Causal reasoning analysis
        causal_count = sum(1 for word in words if word in self.causal_words)
        features["causal_density"] = causal_count / len(words) if words else 0

        # Reasoning chain analysis
        sentences = nltk.sent_tokenize(content)
        reasoning_chains = 0
        for sentence in sentences:
            # Look for causal chains
            if sum(1 for word in self.causal_words if word in sentence.lower()) >= 2:
                reasoning_chains += 1

        features["reasoning_chain_density"] = reasoning_chains / len(sentences) if sentences else 0

        # Commonsense reasoning complexity
        features["commonsense_complexity"] = (
            features["commonsense_density"] * 0.3 +
            features["causal_density"] * 0.4 +
            features["reasoning_chain_density"] * 0.3
        )

        return features

class ContentPreprocessingPipeline:
    """Main content preprocessing orchestration"""

    def __init__(self):
        self.preprocessors = {
            "linguistics": LinguisticsPreprocessor(),
            "sentiment_analysis": SentimentAnalysisPreprocessor(),
            "semantic_analysis": SemanticAnalysisPreprocessor(),
            "factual_knowledge": FactualKnowledgePreprocessor(),
            "commonsense_reasoning": CommonsenseReasoningPreprocessor(),
            "pragmatics": DomainPreprocessor("pragmatics"),  # Generic for pragmatics
            "general": DomainPreprocessor("general")  # Fallback
        }

    def process_content(self, content: str, domain: str = "general",
                       metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process content through appropriate preprocessing pipeline"""
        if domain not in self.preprocessors:
            domain = "general"

        preprocessor = self.preprocessors[domain]

        print(f"🔧 Processing content for domain: {domain}")
        result = preprocessor.preprocess(content, metadata)

        print(f"✅ Content processed - Quality Score: {result['quality_score']:.2f}")
        return result

    def batch_process(self, content_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple content items"""
        processed_items = []

        for item in content_items:
            content = item.get("content", "")
            domain = item.get("domain", "general")
            metadata = item.get("metadata", {})

            processed = self.process_content(content, domain, metadata)
            processed_items.append({
                **item,
                "processed_data": processed
            })

        return processed_items

    def get_domain_stats(self) -> Dict[str, Any]:
        """Get statistics about preprocessing by domain"""
        stats = {}

        for domain, preprocessor in self.preprocessors.items():
            # This would track actual processing stats in production
            stats[domain] = {
                "available": True,
                "custom_features": len(preprocessor.__class__.__dict__) - len(DomainPreprocessor.__dict__)
            }

        return stats

def demonstrate_preprocessing():
    """Demonstrate content preprocessing capabilities"""
    print("🔧 Content Preprocessing System Demonstration")
    print("=" * 60)

    # Initialize preprocessing pipeline
    pipeline = ContentPreprocessingPipeline()

    # Test content for different domains
    test_content = {
        "linguistics": """
        Syntax refers to the arrangement of words in sentences. Grammatical rules govern
        how words can be combined to form acceptable utterances in a language. Morphology
        deals with the structure of words and how they are formed from morphemes.
        """,

        "sentiment_analysis": """
        The user expressed great joy and satisfaction with the amazing product.
        However, they were extremely disappointed with the terrible customer service.
        The overall experience was quite positive despite some minor issues.
        """,

        "semantic_analysis": """
        The word "bank" can refer to a financial institution or the side of a river.
        These meanings are related but distinct. A synonym for happy is joyful,
        while an antonym would be sad. Understanding context is crucial for
        determining the correct interpretation.
        """,

        "factual_knowledge": """
        According to recent research published in Nature (2023), machine learning
        algorithms have achieved 95% accuracy in medical diagnosis tasks.
        The study, conducted by researchers at Stanford University, analyzed
        over 100,000 patient records and demonstrated significant improvements
        in early disease detection.
        """,

        "commonsense_reasoning": """
        Obviously, you can't fit an elephant in a refrigerator. It makes sense that
        people wear coats in cold weather because they get chilly otherwise.
        Due to gravity, objects fall down when you drop them. Normally,
        restaurants serve food because that's their purpose.
        """
    }

    print("📊 Domain-Specific Preprocessing Results:")
    print("-" * 50)

    for domain, content in test_content.items():
        print(f"\n🎯 Domain: {domain.upper()}")
        print(f"Content: {content.strip()[:100]}...")

        result = pipeline.process_content(content, domain)

        print(".2f")
        print(f"Keywords: {', '.join(result['features'].get('keywords', [])[:5])}")

        # Domain-specific metrics
        if domain == "linguistics" and "linguistic_complexity_score" in result["features"]:
            print(".2f")
        elif domain == "sentiment_analysis" and "sentiment_complexity" in result["features"]:
            print(".2f")
        elif domain == "semantic_analysis" and "semantic_complexity" in result["features"]:
            print(".2f")
        elif domain == "factual_knowledge" and "factual_reliability" in result["features"]:
            print(".2f")
        elif domain == "commonsense_reasoning" and "commonsense_complexity" in result["features"]:
            print(".2f")

    # Batch processing demonstration
    print("\n🔄 Batch Processing Demonstration:")
    batch_items = [
        {"content": "This is a test sentence.", "domain": "linguistics"},
        {"content": "I feel very happy today.", "domain": "sentiment_analysis"},
        {"content": "Words have meanings.", "domain": "semantic_analysis"}
    ]

    batch_results = pipeline.batch_process(batch_items)
    print(f"✅ Processed {len(batch_results)} items in batch")

    # System statistics
    stats = pipeline.get_domain_stats()
    print("\n📈 System Statistics:")
    for domain, domain_stats in stats.items():
        print(f"   • {domain}: {'✅' if domain_stats['available'] else '❌'} available, {domain_stats['custom_features']} custom features")

    print("\n✅ Content preprocessing system demonstration complete!")
    print("🧠 Domain-specific content processing is now operational!")

if __name__ == "__main__":
    demonstrate_preprocessing()
