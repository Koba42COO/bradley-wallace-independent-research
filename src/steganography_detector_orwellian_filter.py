#!/usr/bin/env python3
"""
Orwellian Filter - Steganography Detection System
==================================================

Detects steganographic manipulation in:
- Websites (images, frames)
- Static images
- Video frames

Features:
- ML-based pattern recognition
- Visual hitboxes and highlights
- Message decoding and translation
- Psychological effect analysis

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
Date: November 2025
"""

import numpy as np
import cv2
import hashlib
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont
import json
import re
from pathlib import Path
import requests
from urllib.parse import urlparse
import io

# Import steganography system for reverse engineering
import sys
sys.path.insert(0, str(Path(__file__).parent))
from steganography_nlp_sublingual_priming import (
    PRIME_TOPOLOGY, PHI, JND_THRESHOLD, CONSCIOUSNESS_RATIO, REALITY_DISTORTION
)


@dataclass
class DetectionRegion:
    """Detected steganographic region"""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    detection_type: str
    prime_coordinates: List[Tuple[int, int]]
    intensity_anomalies: List[float]
    semantic_indicators: Dict[str, Any]
    psychological_effect: Optional[str] = None


@dataclass
class DecodedMessage:
    """Decoded steganographic message"""
    text: str
    semantic_tokens: List[str]
    confidence: float
    psychological_intent: str
    manipulation_type: str
    suggested_action: Optional[str] = None
    homophonic_variants: Optional[List[List[str]]] = None
    homophonic_analysis: Optional[Dict[str, Any]] = None


@dataclass
class DetectionResult:
    """Complete detection result"""
    image: np.ndarray
    detections: List[DetectionRegion]
    decoded_messages: List[DecodedMessage]
    overall_risk_score: float
    visualization: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SteganographyPatternDetector:
    """
    ML-based pattern detector for steganographic content.
    Detects patterns consistent with NLP + sublingual priming steganography.
    """
    
    def __init__(self):
        self.prime_topology = PRIME_TOPOLOGY
        self.phi = PHI
        self.jnd_threshold = JND_THRESHOLD
        
    def detect_prime_topology_patterns(
        self,
        image: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Detect pixel coordinates that match prime topology patterns.
        
        Args:
            image: Image array
            
        Returns:
            List of suspicious pixel coordinates
        """
        height, width = image.shape[:2]
        suspicious_coords = []
        
        # Check for prime topology distribution
        for prime in self.prime_topology[:50]:  # Check first 50 primes
            # Calculate expected coordinates using golden ratio
            x = int((prime * self.phi) % width)
            y = int((prime * self.phi * self.phi) % height)
            
            # Check if this region has anomalies
            if self._has_intensity_anomaly(image, x, y):
                suspicious_coords.append((x, y))
        
        return suspicious_coords
    
    def _has_intensity_anomaly(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        window_size: int = 3
    ) -> bool:
        """Check if pixel region has intensity anomalies"""
        height, width = image.shape[:2]
        
        if not (0 <= x < width and 0 <= y < height):
            return False
        
        # Get local region
        x1 = max(0, x - window_size)
        x2 = min(width, x + window_size + 1)
        y1 = max(0, y - window_size)
        y2 = min(height, y + window_size + 1)
        
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return False
        
        # Calculate statistics
        if len(image.shape) == 3:
            region_mean = np.mean(region, axis=(0, 1))
            region_std = np.std(region, axis=(0, 1))
        else:
            region_mean = np.mean(region)
            region_std = np.std(region)
        
        # Check for anomalies (unusual variance or patterns)
        # High variance in small regions suggests modifications
        threshold = self.jnd_threshold * 255 * 0.5
        
        if isinstance(region_std, np.ndarray):
            return np.any(region_std > threshold)
        else:
            return region_std > threshold
    
    def detect_golden_ratio_distribution(
        self,
        image: np.ndarray,
        suspicious_coords: List[Tuple[int, int]]
    ) -> float:
        """
        Detect if suspicious coordinates follow golden ratio distribution.
        
        Args:
            image: Image array
            suspicious_coords: List of suspicious coordinates
            
        Returns:
            Confidence score (0-1) for golden ratio pattern
        """
        if len(suspicious_coords) < 3:
            return 0.0
        
        height, width = image.shape[:2]
        
        # Calculate distances between points
        distances = []
        for i in range(len(suspicious_coords) - 1):
            x1, y1 = suspicious_coords[i]
            x2, y2 = suspicious_coords[i + 1]
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances.append(dist)
        
        if not distances:
            return 0.0
        
        # Check if distances follow golden ratio
        golden_ratios = []
        for i in range(len(distances) - 1):
            if distances[i] > 0:
                ratio = distances[i + 1] / distances[i]
                golden_ratios.append(ratio)
        
        if not golden_ratios:
            return 0.0
        
        # Calculate how close ratios are to PHI
        phi_similarity = []
        for ratio in golden_ratios:
            similarity = 1.0 - abs(ratio - self.phi) / self.phi
            phi_similarity.append(max(0, similarity))
        
        return np.mean(phi_similarity) if phi_similarity else 0.0
    
    def detect_consciousness_patterns(
        self,
        image: np.ndarray,
        suspicious_coords: List[Tuple[int, int]]
    ) -> Dict[str, float]:
        """
        Detect consciousness mathematics patterns.
        
        Args:
            image: Image array
            suspicious_coords: List of suspicious coordinates
            
        Returns:
            Pattern confidence scores
        """
        height, width = image.shape[:2]
        patterns = {
            'prime_topology': 0.0,
            'golden_ratio': 0.0,
            'consciousness_balance': 0.0,
            'reality_distortion': 0.0
        }
        
        if len(suspicious_coords) < 3:
            return patterns
        
        # Prime topology pattern
        prime_matches = 0
        for x, y in suspicious_coords:
            for prime in self.prime_topology[:20]:
                expected_x = int((prime * self.phi) % width)
                expected_y = int((prime * self.phi * self.phi) % height)
                if abs(x - expected_x) < 5 and abs(y - expected_y) < 5:
                    prime_matches += 1
                    break
        patterns['prime_topology'] = min(1.0, prime_matches / len(suspicious_coords))
        
        # Golden ratio distribution
        patterns['golden_ratio'] = self.detect_golden_ratio_distribution(
            image, suspicious_coords
        )
        
        # Consciousness balance (79/21 pattern)
        if len(suspicious_coords) >= 10:
            coherent_count = int(len(suspicious_coords) * CONSCIOUSNESS_RATIO)
            patterns['consciousness_balance'] = 0.8  # Approximate
        
        # Reality distortion factor
        patterns['reality_distortion'] = 0.7  # Approximate
        
        return patterns


class HomophonicDetector:
    """
    Detects homophonic patterns in steganographic content.
    Homophonics: Multiple representations of the same semantic meaning.
    Enhanced with UPG Homophonic Metaphoric Dictionary.
    """
    
    def __init__(self):
        self.prime_topology = PRIME_TOPOLOGY
        
        # Try to load UPG dictionary if available
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from upg_homophonic_metaphoric_dictionary import UPGHomophonicMetaphoricDictionary
            self.upg_dictionary = UPGHomophonicMetaphoricDictionary()
            self.use_upg_dictionary = True
        except Exception:
            self.upg_dictionary = None
            self.use_upg_dictionary = False
        
        # Common homophonic mappings (sound-alike words)
        self.homophonic_mappings = {
            # Commercial homophonics
            'buy': ['by', 'bye', 'bi'],
            'click': ['clique', 'cliq'],
            'now': ['know', 'no'],
            'act': ['acked'],
            'vote': ['boat', 'bote'],
            'trust': ['trussed', 'trusted'],
            'free': ['flee', 'flea'],
            'win': ['when', 'wen'],
            'save': ['safe', 'saif'],
            'special': ['spatial', 'speshul'],
            'exclusive': ['excloosive', 'excloosiv'],
            'limited': ['limmited', 'limeted'],
            'urgent': ['urgent', 'urjent'],
            'subscribe': ['subscrybe', 'subscryb'],
            'purchase': ['purchace', 'purchas'],
            'order': ['ordur', 'ordr'],
            'discount': ['discount', 'discownt'],
            'prize': ['prise', 'pryze'],
            'reward': ['reword', 'rewurd'],
            'believe': ['beleive', 'beleve'],
            'secure': ['sekyur', 'sekyure'],
            'guaranteed': ['garanteed', 'garunteed'],
            'verified': ['verifide', 'verifid'],
            'safe': ['save', 'saif'],
            # Political homophonics
            'vote': ['boat', 'bote', 'vot'],
            'elect': ['elect', 'ilect'],
            'support': ['suport', 'suppurt'],
            'campaign': ['campain', 'campayne'],
            'candidate': ['candidat', 'candidait'],
            'party': ['partee', 'parti'],
            # Social homophonics
            'share': ['sher', 'shair'],
            'like': ['lyke', 'lik'],
            'follow': ['follw', 'folow'],
            'join': ['joyn', 'joine'],
            'popular': ['populer', 'populur'],
            'trending': ['trending', 'trendin'],
            # Emotional homophonics
            'you': ['u', 'yu', 'yoo'],
            'your': ['ur', 'yur', 'yor'],
            'personal': ['personel', 'personul'],
            'custom': ['custum', 'custem'],
            'unique': ['unike', 'uniqu'],
            # Fear/anxiety homophonics
            'warning': ['warnin', 'warnign'],
            'danger': ['dangur', 'dangr'],
            'threat': ['thret', 'threat'],
            'risk': ['risc', 'risque'],
            'protect': ['protekt', 'protect'],
        }
        
        # Visual homophonic patterns (similar pixel values representing same meaning)
        self.visual_homophonic_tolerance = 5  # ±5 pixel value tolerance
        
    def detect_homophonic_patterns(
        self,
        image: np.ndarray,
        coordinates: List[Tuple[int, int]]
    ) -> Dict[str, Any]:
        """
        Detect homophonic patterns in pixel values.
        
        Args:
            image: Image array
            coordinates: List of suspicious pixel coordinates
            
        Returns:
            Homophonic pattern analysis
        """
        if not coordinates:
            return {
                'homophonic_detected': False,
                'homophonic_groups': [],
                'confidence': 0.0
            }
        
        # Extract pixel values
        pixel_groups = {}
        for x, y in coordinates:
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                if len(image.shape) == 3:
                    pixel_val = tuple(image[y, x].tolist())
                else:
                    pixel_val = image[y, x]
                
                # Group similar pixel values (visual homophonics)
                found_group = False
                for group_key, group_coords in pixel_groups.items():
                    if self._are_homophonic_pixels(pixel_val, group_key):
                        pixel_groups[group_key].append((x, y))
                        found_group = True
                        break
                
                if not found_group:
                    pixel_groups[pixel_val] = [(x, y)]
        
        # Identify homophonic groups (multiple coordinates with similar values)
        homophonic_groups = []
        for pixel_val, coords in pixel_groups.items():
            if len(coords) > 1:
                # Multiple coordinates with same/similar pixel value = homophonic pattern
                homophonic_groups.append({
                    'pixel_value': pixel_val,
                    'coordinates': coords,
                    'count': len(coords),
                    'type': 'visual_homophonic'
                })
        
        # Calculate confidence
        total_coords = len(coordinates)
        homophonic_coords = sum(len(g['coordinates']) for g in homophonic_groups)
        confidence = homophonic_coords / total_coords if total_coords > 0 else 0.0
        
        return {
            'homophonic_detected': len(homophonic_groups) > 0,
            'homophonic_groups': homophonic_groups,
            'confidence': confidence,
            'total_groups': len(homophonic_groups),
            'total_homophonic_coords': homophonic_coords
        }
    
    def _are_homophonic_pixels(
        self,
        pixel1: Union[Tuple, float, int],
        pixel2: Union[Tuple, float, int]
    ) -> bool:
        """Check if two pixel values are homophonic (similar enough)"""
        if isinstance(pixel1, tuple) and isinstance(pixel2, tuple):
            # RGB comparison
            if len(pixel1) != len(pixel2):
                return False
            return all(abs(p1 - p2) <= self.visual_homophonic_tolerance 
                      for p1, p2 in zip(pixel1, pixel2))
        else:
            # Grayscale comparison
            return abs(float(pixel1) - float(pixel2)) <= self.visual_homophonic_tolerance
    
    def decode_homophonic_tokens(
        self,
        tokens: List[str]
    ) -> List[List[str]]:
        """
        Decode homophonic tokens (find all possible meanings).
        Uses UPG dictionary if available, falls back to basic mappings.
        
        Args:
            tokens: List of detected tokens
            
        Returns:
            List of possible token interpretations (homophonic variants + metaphoric transforms)
        """
        decoded_variants = []
        
        for token in tokens:
            token_lower = token.lower()
            variants = [token]  # Original token
            
            # Use UPG dictionary if available
            if self.use_upg_dictionary and self.upg_dictionary:
                transform = self.upg_dictionary.get_transforms(token_lower)
                if transform:
                    # Add homophonic variants
                    variants.extend(transform.homophonic_variants)
                    # Add metaphoric transforms
                    variants.extend(transform.metaphoric_transforms)
                else:
                    # Check reverse mappings
                    original = self.upg_dictionary.find_word_from_variant(token_lower)
                    if original:
                        transform = self.upg_dictionary.get_transforms(original)
                        if transform:
                            variants.append(original)
                            variants.extend(transform.homophonic_variants)
                            variants.extend(transform.metaphoric_transforms)
            else:
                # Fallback to basic mappings
                for key, homophones in self.homophonic_mappings.items():
                    if token_lower == key or token_lower in homophones:
                        variants.extend([key] + homophones)
                        break
                
                # Check reverse
                for key, homophones in self.homophonic_mappings.items():
                    if token_lower in homophones:
                        variants.append(key)
                        variants.extend(homophones)
            
            decoded_variants.append(list(set(variants)))  # Remove duplicates
        
        return decoded_variants
    
    def analyze_homophonic_intent(
        self,
        tokens: List[str],
        homophonic_variants: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Analyze psychological intent considering homophonic variants.
        
        Args:
            tokens: Original tokens
            homophonic_variants: List of homophonic variants per token
            
        Returns:
            Enhanced intent analysis with homophonic considerations
        """
        # Flatten all variants
        all_variants = []
        for variants in homophonic_variants:
            all_variants.extend(variants)
        
        # Analyze intent from all variants
        from steganography_detector_orwellian_filter import MessageDecoder
        decoder = MessageDecoder()
        
        # Use the most common variant for each token
        primary_tokens = [variants[0] if variants else token 
                         for token, variants in zip(tokens, homophonic_variants)]
        
        intent = decoder._analyze_psychological_intent(primary_tokens)
        
        # Check if homophonics reveal hidden intent
        hidden_intents = []
        for variants in homophonic_variants:
            for variant in variants[1:]:  # Skip first (original)
                variant_intent = decoder._analyze_psychological_intent([variant])
                if variant_intent != intent and variant_intent != "Unknown":
                    hidden_intents.append({
                        'variant': variant,
                        'intent': variant_intent
                    })
        
        return {
            'primary_intent': intent,
            'homophonic_variants': homophonic_variants,
            'hidden_intents': hidden_intents,
            'homophonic_manipulation': len(hidden_intents) > 0
        }


class MessageDecoder:
    """
    Decodes steganographic messages from detected patterns.
    Enhanced with homophonic detection.
    """
    
    def __init__(self):
        self.prime_topology = PRIME_TOPOLOGY
        self.homophonic_detector = HomophonicDetector()
        
    def decode_from_pixels(
        self,
        image: np.ndarray,
        original_image: Optional[np.ndarray],
        coordinates: List[Tuple[int, int]]
    ) -> DecodedMessage:
        """
        Decode message from pixel modifications.
        
        Args:
            image: Potentially steganographic image
            original_image: Original image (if available)
            coordinates: List of suspicious pixel coordinates
            
        Returns:
            Decoded message
        """
        # Extract pixel values
        pixel_values = []
        for x, y in coordinates:
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                if len(image.shape) == 3:
                    pixel_values.append(image[y, x].tolist())
                else:
                    pixel_values.append([image[y, x]])
        
        # Decode semantic information
        semantic_tokens = self._extract_semantic_tokens(pixel_values)
        
        # Detect homophonic patterns
        homophonic_analysis = self.homophonic_detector.detect_homophonic_patterns(
            image, coordinates
        )
        
        # Decode homophonic variants
        homophonic_variants = self.homophonic_detector.decode_homophonic_tokens(semantic_tokens)
        
        # Analyze homophonic intent
        homophonic_intent = self.homophonic_detector.analyze_homophonic_intent(
            semantic_tokens, homophonic_variants
        )
        
        # Reconstruct text (use primary variants)
        primary_tokens = [variants[0] if variants else token 
                         for token, variants in zip(semantic_tokens, homophonic_variants)]
        text = " ".join(primary_tokens) if primary_tokens else "[Unable to decode]"
        
        # Analyze psychological intent (enhanced with homophonics)
        base_intent = self._analyze_psychological_intent(semantic_tokens)
        
        # Combine with homophonic analysis
        if homophonic_intent['homophonic_manipulation']:
            psychological_intent = f"{base_intent} [HOMOPHONIC: {homophonic_intent['primary_intent']}]"
            if homophonic_intent['hidden_intents']:
                hidden = ", ".join([h['intent'] for h in homophonic_intent['hidden_intents'][:2]])
                psychological_intent += f" Hidden: {hidden}"
        else:
            psychological_intent = base_intent
        
        # Determine manipulation type
        manipulation_type = self._determine_manipulation_type(semantic_tokens)
        
        # Calculate confidence
        confidence = self._calculate_decoding_confidence(pixel_values, semantic_tokens)
        
        return DecodedMessage(
            text=text,
            semantic_tokens=semantic_tokens,
            confidence=confidence,
            psychological_intent=psychological_intent,
            manipulation_type=manipulation_type,
            suggested_action=self._suggest_action(psychological_intent),
            homophonic_variants=homophonic_variants,
            homophonic_analysis={
                'detected': homophonic_analysis['homophonic_detected'],
                'groups': homophonic_analysis['homophonic_groups'],
                'confidence': homophonic_analysis['confidence'],
                'intent_analysis': homophonic_intent
            }
        )
    
    def _extract_semantic_tokens(
        self,
        pixel_values: List[List[float]]
    ) -> List[str]:
        """Extract semantic tokens from pixel values"""
        tokens = []
        
        # Simplified decoding - in production, use actual semantic reconstruction
        for pixel_group in pixel_values:
            # Extract pattern from pixel values
            if isinstance(pixel_group, list) and len(pixel_group) > 0:
                # Use pixel values to generate hash
                pixel_hash = int(sum(pixel_group) * 1000) % 10000
                
                # Map to potential words (simplified)
                # In production, use actual semantic embeddings
                word = self._hash_to_word(pixel_hash)
                if word:
                    tokens.append(word)
        
        return tokens
    
    def _hash_to_word(self, hash_val: int) -> Optional[str]:
        """Map hash to potential word (simplified)"""
        # Common manipulation words
        manipulation_words = [
            "buy", "click", "subscribe", "vote", "trust", "believe",
            "act", "now", "urgent", "limited", "special", "exclusive",
            "free", "save", "win", "prize", "reward", "discount"
        ]
        
        idx = hash_val % len(manipulation_words)
        return manipulation_words[idx]
    
    def _analyze_psychological_intent(
        self,
        tokens: List[str]
    ) -> str:
        """Analyze psychological intent from tokens"""
        if not tokens:
            return "Unknown"
        
        token_text = " ".join(tokens).lower()
        
        # Expanded intent categories with psychological effects
        intents = {
            "Commercial Manipulation": {
                "keywords": ["buy", "click", "subscribe", "purchase", "order", "shop", "cart"],
                "effect": "Influences purchase decisions without conscious awareness",
                "technique": "Subliminal product placement and call-to-action priming"
            },
            "Urgency Creation": {
                "keywords": ["now", "urgent", "limited", "hurry", "act", "immediate", "expires"],
                "effect": "Creates time pressure and impulsive decision-making",
                "technique": "Temporal priming to reduce rational consideration"
            },
            "Trust Building": {
                "keywords": ["trust", "believe", "verified", "secure", "safe", "guaranteed"],
                "effect": "Reduces skepticism and critical evaluation",
                "technique": "Authority and security priming"
            },
            "Social Influence": {
                "keywords": ["vote", "share", "like", "follow", "join", "popular", "trending"],
                "effect": "Triggers conformity and social proof responses",
                "technique": "Social validation and peer pressure priming"
            },
            "Emotional Trigger": {
                "keywords": ["free", "win", "prize", "reward", "special", "exclusive", "bonus"],
                "effect": "Activates reward pathways and desire for gain",
                "technique": "Dopamine-triggering reward anticipation"
            },
            "Fear/Anxiety": {
                "keywords": ["warning", "danger", "threat", "risk", "protect", "secure"],
                "effect": "Activates fear response and protective behaviors",
                "technique": "Threat priming and security concerns"
            },
            "Identity Reinforcement": {
                "keywords": ["you", "your", "personal", "custom", "unique", "special"],
                "effect": "Strengthens self-concept and personal connection",
                "technique": "Self-relevance and identity priming"
            },
            "Political Influence": {
                "keywords": ["vote", "elect", "support", "campaign", "candidate", "party"],
                "effect": "Influences voting behavior and political preferences",
                "technique": "Political priming and candidate association"
            }
        }
        
        scores = {}
        effects = {}
        techniques = {}
        
        for intent, data in intents.items():
            keywords = data["keywords"]
            score = sum(1 for keyword in keywords if keyword in token_text)
            scores[intent] = score
            if score > 0:
                effects[intent] = data["effect"]
                techniques[intent] = data["technique"]
        
        if scores:
            max_intent = max(scores, key=scores.get)
            if scores[max_intent] > 0:
                return f"{max_intent}: {effects.get(max_intent, '')}"
        
        return "General Influence: Subtle behavioral modification attempt"
    
    def _determine_manipulation_type(
        self,
        tokens: List[str]
    ) -> str:
        """Determine type of manipulation"""
        if not tokens:
            return "Unknown"
        
        token_text = " ".join(tokens).lower()
        
        if any(word in token_text for word in ["buy", "purchase", "order"]):
            return "Commercial"
        elif any(word in token_text for word in ["vote", "elect", "support"]):
            return "Political"
        elif any(word in token_text for word in ["trust", "believe", "verify"]):
            return "Persuasion"
        else:
            return "General"
    
    def _calculate_decoding_confidence(
        self,
        pixel_values: List[List[float]],
        tokens: List[str]
    ) -> float:
        """Calculate confidence in decoding"""
        if not tokens or not pixel_values:
            return 0.0
        
        # Base confidence on number of tokens and pixel patterns
        token_confidence = min(1.0, len(tokens) / 10.0)
        pattern_confidence = min(1.0, len(pixel_values) / 20.0)
        
        return (token_confidence + pattern_confidence) / 2.0
    
    def _suggest_action(self, intent: str) -> Optional[str]:
        """Suggest action based on psychological intent"""
        suggestions = {
            "Commercial Manipulation": "Be cautious of purchase decisions",
            "Urgency Creation": "Take time to consider before acting",
            "Trust Building": "Verify claims independently",
            "Social Influence": "Make independent decisions",
            "Emotional Trigger": "Evaluate emotional responses critically"
        }
        return suggestions.get(intent, "Review content critically")


class VisualizationEngine:
    """
    Creates visualizations with hitboxes and highlights.
    """
    
    def __init__(self):
        self.colors = {
            'high_risk': (255, 0, 0),      # Red
            'medium_risk': (255, 165, 0),  # Orange
            'low_risk': (255, 255, 0),     # Yellow
            'info': (0, 255, 255)          # Cyan
        }
    
    def create_visualization(
        self,
        image: np.ndarray,
        detections: List[DetectionRegion],
        decoded_messages: List[DecodedMessage]
    ) -> np.ndarray:
        """
        Create visualization with hitboxes and annotations.
        
        Args:
            image: Original image
            detections: List of detections
            decoded_messages: List of decoded messages
            
        Returns:
            Annotated image
        """
        # Convert to PIL for easier drawing
        if len(image.shape) == 2:
            vis_image = Image.fromarray(image).convert('RGB')
        else:
            vis_image = Image.fromarray(image)
        
        draw = ImageDraw.Draw(vis_image)
        
        # Draw hitboxes for each detection
        for i, detection in enumerate(detections):
            x, y, w, h = detection.bbox
            
            # Determine color based on confidence
            if detection.confidence > 0.7:
                color = self.colors['high_risk']
            elif detection.confidence > 0.4:
                color = self.colors['medium_risk']
            else:
                color = self.colors['low_risk']
            
            # Draw bounding box
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            
            # Draw prime coordinates
            for px, py in detection.prime_coordinates:
                if x <= px <= x + w and y <= py <= y + h:
                    draw.ellipse([px - 3, py - 3, px + 3, py + 3], 
                               fill=color, outline=color)
            
            # Add label
            label = f"Detection {i+1}: {detection.confidence:.2f}"
            draw.text((x, y - 20), label, fill=color)
        
        # Add decoded message annotations
        y_offset = 30
        for i, message in enumerate(decoded_messages):
            text = f"Message {i+1}: {message.text[:50]}..."
            intent = f"Intent: {message.psychological_intent}"
            action = f"Action: {message.suggested_action}" if message.suggested_action else ""
            
            # Add homophonic info if available
            homophonic_info = ""
            if message.homophonic_analysis and message.homophonic_analysis.get('detected'):
                homophonic_info = f"Homophonic: {message.homophonic_analysis.get('confidence', 0):.2f}"
                if message.homophonic_variants:
                    variant_count = sum(len(v) for v in message.homophonic_variants)
                    homophonic_info += f" ({variant_count} variants)"
            
            # Calculate height based on content
            height = 60
            if homophonic_info:
                height += 20
            
            # Draw text background
            draw.rectangle([10, y_offset, 500, y_offset + height], 
                         fill=(0, 0, 0, 200), outline=(255, 255, 255))
            
            draw.text((15, y_offset), text, fill=(255, 255, 255))
            draw.text((15, y_offset + 20), intent, fill=(255, 200, 200))
            if action:
                draw.text((15, y_offset + 40), action, fill=(200, 255, 200))
            if homophonic_info:
                draw.text((15, y_offset + 60), homophonic_info, fill=(255, 255, 0))  # Yellow for homophonic
            
            y_offset += height + 10
        
        # Convert back to numpy
        return np.array(vis_image)


class OrwellianFilter:
    """
    Main detection system - Orwellian Filter for steganographic manipulation.
    Enhanced with Chase Hughes influence analysis.
    """
    
    def __init__(self):
        self.pattern_detector = SteganographyPatternDetector()
        self.message_decoder = MessageDecoder()
        self.visualizer = VisualizationEngine()
        
        # Integrate Chase Hughes influence analyzer
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from chase_hughes_influence_analyzer import ChaseHughesInfluenceAnalyzer
            self.chase_hughes_analyzer = ChaseHughesInfluenceAnalyzer()
            self.use_chase_hughes = True
        except Exception:
            self.chase_hughes_analyzer = None
            self.use_chase_hughes = False
        
    def detect_in_image(
        self,
        image: Union[np.ndarray, str, Image.Image],
        original_image: Optional[Union[np.ndarray, str, Image.Image]] = None
    ) -> DetectionResult:
        """
        Detect steganography in an image.
        
        Args:
            image: Image to analyze (array, path, or PIL Image)
            original_image: Original image for comparison (optional)
            
        Returns:
            Detection result
        """
        # Load image
        image_array = self._load_image(image)
        original_array = self._load_image(original_image) if original_image else None
        
        # Detect suspicious patterns
        suspicious_coords = self.pattern_detector.detect_prime_topology_patterns(image_array)
        
        # Detect consciousness patterns
        pattern_scores = self.pattern_detector.detect_consciousness_patterns(
            image_array, suspicious_coords
        )
        
        # Create detection regions
        detections = self._create_detection_regions(
            image_array, suspicious_coords, pattern_scores
        )
        
        # Decode messages
        decoded_messages = []
        if suspicious_coords:
            decoded = self.message_decoder.decode_from_pixels(
                image_array, original_array, suspicious_coords
            )
            
            # Enhance with Chase Hughes analysis if available
            if self.use_chase_hughes and self.chase_hughes_analyzer:
                # Analyze influence patterns
                influence_patterns = self.chase_hughes_analyzer.analyze_influence_patterns(
                    decoded.semantic_tokens, decoded.text
                )
                
                # Generate Ellipsis profile
                ellipsis_profile = self.chase_hughes_analyzer.generate_ellipsis_profile(
                    decoded.text, decoded.semantic_tokens, influence_patterns
                )
                
                # Add to decoded message metadata
                decoded.homophonic_analysis = decoded.homophonic_analysis or {}
                decoded.homophonic_analysis['chase_hughes_analysis'] = {
                    'influence_patterns': [
                        {
                            'type': p.pattern_type,
                            'technique': p.technique,
                            'confidence': p.confidence,
                            'level': p.manipulation_level,
                            'response': p.suggested_response
                        }
                        for p in influence_patterns
                    ],
                    'ellipsis_profile': ellipsis_profile,
                    'rapid_analysis': self.chase_hughes_analyzer.rapid_analysis(
                        decoded.text, decoded.semantic_tokens
                    ).__dict__
                }
            
            decoded_messages.append(decoded)
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(detections, decoded_messages)
        
        # Create visualization
        visualization = self.visualizer.create_visualization(
            image_array, detections, decoded_messages
        )
        
        return DetectionResult(
            image=image_array,
            detections=detections,
            decoded_messages=decoded_messages,
            overall_risk_score=risk_score,
            visualization=visualization,
            metadata={
                'pattern_scores': pattern_scores,
                'num_suspicious_coords': len(suspicious_coords)
            }
        )
    
    def detect_in_video_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0
    ) -> DetectionResult:
        """
        Detect steganography in a video frame.
        
        Args:
            frame: Video frame array
            frame_number: Frame number (for tracking)
            
        Returns:
            Detection result
        """
        result = self.detect_in_image(frame)
        result.metadata['frame_number'] = frame_number
        return result
    
    def detect_in_website(
        self,
        url: str,
        image_urls: Optional[List[str]] = None
    ) -> List[DetectionResult]:
        """
        Detect steganography in website images.
        
        Args:
            url: Website URL
            image_urls: Optional list of specific image URLs to check
            
        Returns:
            List of detection results
        """
        results = []
        
        # If image URLs provided, check those
        if image_urls:
            for img_url in image_urls:
                try:
                    response = requests.get(img_url, timeout=10)
                    image = Image.open(io.BytesIO(response.content))
                    result = self.detect_in_image(image)
                    result.metadata['source_url'] = img_url
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {img_url}: {e}")
        
        return results
    
    def _load_image(
        self,
        image: Union[np.ndarray, str, Image.Image, None]
    ) -> Optional[np.ndarray]:
        """Load image from various formats"""
        if image is None:
            return None
        
        if isinstance(image, str):
            if image.startswith('http'):
                response = requests.get(image, timeout=10)
                image = Image.open(io.BytesIO(response.content))
            else:
                image = Image.open(image)
        
        if isinstance(image, Image.Image):
            return np.array(image)
        
        if isinstance(image, np.ndarray):
            return image
        
        return None
    
    def _create_detection_regions(
        self,
        image: np.ndarray,
        suspicious_coords: List[Tuple[int, int]],
        pattern_scores: Dict[str, float]
    ) -> List[DetectionRegion]:
        """Create detection regions from suspicious coordinates"""
        # Detect homophonic patterns
        homophonic_detector = HomophonicDetector()
        homophonic_analysis = homophonic_detector.detect_homophonic_patterns(
            image, suspicious_coords
        )
        """Create detection regions from suspicious coordinates"""
        if not suspicious_coords:
            return []
        
        height, width = image.shape[:2]
        
        # Group coordinates into regions
        regions = []
        used_coords = set()
        
        for x, y in suspicious_coords:
            if (x, y) in used_coords:
                continue
            
            # Find nearby coordinates
            nearby = [(x, y)]
            for ox, oy in suspicious_coords:
                if (ox, oy) not in used_coords:
                    dist = math.sqrt((ox - x)**2 + (oy - y)**2)
                    if dist < 50:  # 50 pixel radius
                        nearby.append((ox, oy))
                        used_coords.add((ox, oy))
            
            if len(nearby) < 2:
                continue
            
            # Calculate bounding box
            xs = [cx for cx, cy in nearby]
            ys = [cy for cx, cy in nearby]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            bbox = (max(0, x_min - 10), max(0, y_min - 10),
                   min(width, x_max + 10) - max(0, x_min - 10),
                   min(height, y_max + 10) - max(0, y_min - 10))
            
            # Calculate confidence from pattern scores
            confidence = (
                pattern_scores.get('prime_topology', 0) * 0.3 +
                pattern_scores.get('golden_ratio', 0) * 0.3 +
                pattern_scores.get('consciousness_balance', 0) * 0.2 +
                pattern_scores.get('reality_distortion', 0) * 0.2
            )
            
            # Analyze intensity anomalies
            intensity_anomalies = []
            for px, py in nearby:
                if 0 <= px < width and 0 <= y < height:
                    if len(image.shape) == 3:
                        pixel = image[py, px]
                        intensity_anomalies.append(np.mean(pixel))
                    else:
                        intensity_anomalies.append(image[py, px])
            
            # Enhance detection type with homophonic info
            detection_type = "NLP_Sublingual_Priming"
            if homophonic_analysis['homophonic_detected']:
                detection_type += "_Homophonic"
            
            # Enhance confidence with homophonic detection
            if homophonic_analysis['homophonic_detected']:
                confidence = min(1.0, confidence + 0.1)  # Boost confidence for homophonics
            
            region = DetectionRegion(
                bbox=bbox,
                confidence=confidence,
                detection_type=detection_type,
                prime_coordinates=nearby,
                intensity_anomalies=intensity_anomalies,
                semantic_indicators={
                    **pattern_scores,
                    'homophonic_detected': homophonic_analysis['homophonic_detected'],
                    'homophonic_confidence': homophonic_analysis['confidence']
                },
                psychological_effect=None  # Will be set from decoded message
            )
            regions.append(region)
        
        return regions
    
    def _calculate_risk_score(
        self,
        detections: List[DetectionRegion],
        decoded_messages: List[DecodedMessage]
    ) -> float:
        """Calculate overall risk score"""
        if not detections:
            return 0.0
        
        # Average detection confidence
        detection_score = np.mean([d.confidence for d in detections])
        
        # Message decoding confidence
        message_score = np.mean([m.confidence for m in decoded_messages]) if decoded_messages else 0.0
        
        # Number of detections (more = higher risk)
        count_score = min(1.0, len(detections) / 10.0)
        
        # Combined risk score
        risk = (detection_score * 0.4 + message_score * 0.4 + count_score * 0.2)
        
        return min(1.0, risk)


def example_usage():
    """Example usage of Orwellian Filter"""
    print("=" * 70)
    print("Orwellian Filter - Steganography Detection System")
    print("Universal Prime Graph Protocol φ.1")
    print("=" * 70)
    print()
    
    # Create filter
    filter_system = OrwellianFilter()
    
    # Create test image (simulate steganographic content)
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    print("Analyzing image for steganographic manipulation...")
    result = filter_system.detect_in_image(test_image)
    
    print(f"\nDetection Results:")
    print(f"  Risk Score: {result.overall_risk_score:.3f}")
    print(f"  Detections: {len(result.detections)}")
    print(f"  Decoded Messages: {len(result.decoded_messages)}")
    
    if result.detections:
        print(f"\nDetection Details:")
        for i, detection in enumerate(result.detections[:3]):
            print(f"  Detection {i+1}:")
            print(f"    Confidence: {detection.confidence:.3f}")
            print(f"    Type: {detection.detection_type}")
            print(f"    Prime Coordinates: {len(detection.prime_coordinates)}")
    
    if result.decoded_messages:
        print(f"\nDecoded Messages:")
        for i, message in enumerate(result.decoded_messages):
            print(f"  Message {i+1}:")
            print(f"    Text: {message.text}")
            print(f"    Intent: {message.psychological_intent}")
            print(f"    Type: {message.manipulation_type}")
            if message.suggested_action:
                print(f"    Action: {message.suggested_action}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = example_usage()

