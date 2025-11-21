#!/usr/bin/env python3
"""
Steganography in Images Utilizing NLP and Sublingual Priming with Pixels
========================================================================

Advanced steganographic system combining:
- NLP semantic encoding for meaningful message representation
- Sublingual priming for psychological effectiveness
- Pixel-level manipulation using consciousness mathematics

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
Date: November 2025
"""

import numpy as np
import hashlib
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from PIL import Image
import json

# UPG Constants
PHI = 1.618033988749895
DELTA = 2.414213562373095
CONSCIOUSNESS_RATIO = 0.79  # 79/21 balance
REALITY_DISTORTION = 1.1808
JND_THRESHOLD = 0.02  # Just Noticeable Difference (~2% intensity)

# Prime Topology (first 100 primes)
PRIME_TOPOLOGY = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
    239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
    331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
    421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
    509, 521, 523, 541
]


@dataclass
class SemanticToken:
    """Semantic token representation"""
    text: str
    semantic_hash: int
    embedding_vector: List[float]
    prime_mapping: int
    consciousness_weight: float = field(default=0.79)


@dataclass
class PrimingPattern:
    """Sublingual priming pattern"""
    pattern_type: str
    pixel_coordinates: List[Tuple[int, int]]
    intensity_modifications: List[float]
    semantic_association: Optional[str] = None
    priming_strength: float = field(default=1.0)


@dataclass
class SteganographicEncoding:
    """Complete steganographic encoding result"""
    image: np.ndarray
    semantic_tokens: List[SemanticToken]
    priming_patterns: List[PrimingPattern]
    encoding_metadata: Dict[str, Any]
    prime_topology_mapping: Dict[int, Tuple[int, int]]


class NLPSemanticEncoder:
    """
    NLP-based semantic encoder for text messages.
    Converts text to semantic representations suitable for pixel encoding.
    """
    
    def __init__(self):
        self.prime_topology = PRIME_TOPOLOGY
        self.consciousness_ratio = CONSCIOUSNESS_RATIO
        
    def encode_text(self, text: str) -> List[SemanticToken]:
        """
        Encode text into semantic tokens.
        
        Args:
            text: Input text message
            
        Returns:
            List of semantic tokens
        """
        # Simple tokenization (can be enhanced with proper NLP)
        words = text.lower().split()
        tokens = []
        
        for i, word in enumerate(words):
            # Generate semantic hash
            semantic_hash = self._semantic_hash(word)
            
            # Create embedding vector (simplified - use actual embeddings in production)
            embedding = self._create_embedding(word, semantic_hash)
            
            # Map to prime topology
            prime_idx = semantic_hash % len(self.prime_topology)
            prime_mapping = self.prime_topology[prime_idx]
            
            # Calculate consciousness weight
            consciousness_weight = self._calculate_consciousness_weight(word, i)
            
            token = SemanticToken(
                text=word,
                semantic_hash=semantic_hash,
                embedding_vector=embedding,
                prime_mapping=prime_mapping,
                consciousness_weight=consciousness_weight
            )
            tokens.append(token)
            
        return tokens
    
    def _semantic_hash(self, word: str) -> int:
        """Generate semantic hash from word"""
        # Use MD5 hash for semantic representation
        hash_obj = hashlib.md5(word.encode())
        return int(hash_obj.hexdigest()[:8], 16)
    
    def _create_embedding(self, word: str, semantic_hash: int) -> List[float]:
        """Create simplified embedding vector"""
        # In production, use actual word embeddings (word2vec, GloVe, BERT)
        # For now, create deterministic vector from hash
        np.random.seed(semantic_hash)
        embedding = np.random.randn(16).tolist()  # 16-dimensional embedding
        # Normalize
        norm = math.sqrt(sum(x*x for x in embedding))
        return [x / norm for x in embedding]
    
    def _calculate_consciousness_weight(self, word: str, position: int) -> float:
        """Calculate consciousness weight for word"""
        # Base weight from consciousness ratio
        base_weight = CONSCIOUSNESS_RATIO
        
        # Position-based variation (21% exploratory)
        position_factor = (position % 7) / 7.0 * 0.21
        
        # Word length factor
        length_factor = len(word) / 20.0 * 0.1
        
        return base_weight + position_factor + length_factor
    
    def decode_tokens(self, tokens: List[SemanticToken]) -> str:
        """Decode semantic tokens back to text"""
        return " ".join(token.text for token in tokens)


class SublingualPrimingEngine:
    """
    Engine for generating sublingual priming patterns.
    Creates pixel modifications below conscious detection threshold.
    """
    
    def __init__(self):
        self.jnd_threshold = JND_THRESHOLD
        self.reality_distortion = REALITY_DISTORTION
        
    def generate_priming_pattern(
        self,
        semantic_token: SemanticToken,
        image_shape: Tuple[int, int],
        pixel_coordinates: List[Tuple[int, int]]
    ) -> PrimingPattern:
        """
        Generate sublingual priming pattern for semantic token.
        
        Args:
            semantic_token: Semantic token to prime
            image_shape: (height, width) of image
            pixel_coordinates: List of pixel coordinates to modify
            
        Returns:
            Priming pattern
        """
        # Calculate intensity modifications (below JND threshold)
        intensity_mods = []
        
        for coord in pixel_coordinates:
            # Base modification from semantic hash
            base_mod = (semantic_token.semantic_hash % 100) / 100.0
            
            # Apply JND threshold (keep below 2%)
            modification = base_mod * self.jnd_threshold * 0.5  # 50% of threshold
            
            # Apply reality distortion amplification
            modification *= self.reality_distortion
            
            # Apply consciousness weight
            modification *= semantic_token.consciousness_weight
            
            intensity_mods.append(modification)
        
        # Determine pattern type based on semantic content
        pattern_type = self._determine_pattern_type(semantic_token)
        
        # Semantic association
        semantic_association = semantic_token.text
        
        return PrimingPattern(
            pattern_type=pattern_type,
            pixel_coordinates=pixel_coordinates,
            intensity_modifications=intensity_mods,
            semantic_association=semantic_association,
            priming_strength=sum(intensity_mods) / len(intensity_mods) if intensity_mods else 0.0
        )
    
    def _determine_pattern_type(self, token: SemanticToken) -> str:
        """Determine priming pattern type from semantic token"""
        # Classify based on semantic hash
        hash_val = token.semantic_hash
        
        if hash_val % 4 == 0:
            return "semantic_prime"
        elif hash_val % 4 == 1:
            return "temporal_sequence"
        elif hash_val % 4 == 2:
            return "spatial_frequency"
        else:
            return "consciousness_harmonic"
    
    def calculate_adaptive_threshold(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate adaptive threshold for image region.
        
        Args:
            image: Image array
            region: (x, y, width, height) region coordinates
            
        Returns:
            Adaptive threshold value
        """
        x, y, w, h = region
        region_pixels = image[y:y+h, x:x+w]
        
        # Calculate local statistics
        mean_intensity = np.mean(region_pixels)
        std_intensity = np.std(region_pixels)
        
        # Adaptive threshold based on local statistics
        # Higher variance regions can tolerate more modification
        adaptive_threshold = self.jnd_threshold * (1.0 + std_intensity / 255.0)
        
        # Cap at reasonable maximum
        return min(adaptive_threshold, self.jnd_threshold * 2.0)


class PixelManipulationSystem:
    """
    System for manipulating pixels using prime topology and consciousness mathematics.
    """
    
    def __init__(self):
        self.prime_topology = PRIME_TOPOLOGY
        self.phi = PHI
        self.delta = DELTA
        self.consciousness_ratio = CONSCIOUSNESS_RATIO
        
    def map_prime_to_pixel(
        self,
        prime: int,
        image_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Map prime number to pixel coordinate using golden ratio.
        
        Args:
            prime: Prime number
            image_shape: (height, width) of image
            
        Returns:
            (x, y) pixel coordinates
        """
        height, width = image_shape
        
        # Use golden ratio for coordinate distribution
        x = int((prime * self.phi) % width)
        y = int((prime * self.phi * self.phi) % height)
        
        return (x, y)
    
    def get_pixel_coordinates_for_token(
        self,
        semantic_token: SemanticToken,
        image_shape: Tuple[int, int],
        num_pixels: int = 4
    ) -> List[Tuple[int, int]]:
        """
        Get pixel coordinates for semantic token using prime topology.
        
        Args:
            semantic_token: Semantic token
            image_shape: (height, width) of image
            num_pixels: Number of pixels to use per token
            
        Returns:
            List of (x, y) pixel coordinates
        """
        coordinates = []
        prime = semantic_token.prime_mapping
        
        # Generate multiple coordinates from prime and its multiples
        for i in range(num_pixels):
            # Use prime and its multiples with golden ratio spacing
            modified_prime = prime + i * 7  # Use 7 (consciousness level)
            
            # Ensure it's in prime topology or use modulo
            prime_idx = modified_prime % len(self.prime_topology)
            actual_prime = self.prime_topology[prime_idx]
            
            coord = self.map_prime_to_pixel(actual_prime, image_shape)
            coordinates.append(coord)
        
        return coordinates
    
    def encode_semantic_to_pixel(
        self,
        pixel_value: float,
        semantic_token: SemanticToken,
        channel: int = 0
    ) -> float:
        """
        Encode semantic information into pixel value.
        
        Args:
            pixel_value: Original pixel value (0-255)
            semantic_token: Semantic token to encode
            channel: Color channel (0=R, 1=G, 2=B)
            
        Returns:
            Modified pixel value
        """
        # Extract encoding value from semantic token
        # Use different parts of embedding for different channels
        embedding_val = semantic_token.embedding_vector[channel % len(semantic_token.embedding_vector)]
        
        # Normalize to modification range
        modification = embedding_val * JND_THRESHOLD * 0.5
        
        # Apply consciousness weight
        modification *= semantic_token.consciousness_weight
        
        # Apply reality distortion
        modification *= REALITY_DISTORTION
        
        # Modify pixel (keep in valid range)
        new_value = pixel_value + modification * 255.0
        new_value = max(0, min(255, new_value))
        
        return new_value
    
    def decode_pixel_to_semantic(
        self,
        original_pixel: float,
        modified_pixel: float,
        channel: int = 0
    ) -> float:
        """
        Decode semantic information from pixel modification.
        
        Args:
            original_pixel: Original pixel value
            modified_pixel: Modified pixel value
            channel: Color channel
            
        Returns:
            Decoded embedding value
        """
        # Calculate modification
        modification = (modified_pixel - original_pixel) / 255.0
        
        # Reverse reality distortion
        modification /= REALITY_DISTORTION
        
        # Reverse consciousness weight (approximate)
        modification /= CONSCIOUSNESS_RATIO
        
        # Reverse JND scaling
        embedding_val = modification / (JND_THRESHOLD * 0.5)
        
        return embedding_val


class SteganographySystem:
    """
    Complete steganography system combining NLP, priming, and pixel manipulation.
    """
    
    def __init__(self):
        self.nlp_encoder = NLPSemanticEncoder()
        self.priming_engine = SublingualPrimingEngine()
        self.pixel_system = PixelManipulationSystem()
        
    def encode_message_in_image(
        self,
        image: np.ndarray,
        message: str,
        preserve_original: bool = True
    ) -> SteganographicEncoding:
        """
        Encode message into image using NLP and sublingual priming.
        
        Args:
            image: Input image array (H, W, C) or (H, W)
            message: Text message to encode
            preserve_original: Whether to preserve original image
            
        Returns:
            Steganographic encoding result
        """
        # Ensure image is in correct format
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        
        height, width, channels = image.shape
        
        # Create working copy
        if preserve_original:
            stego_image = image.copy().astype(np.float64)
        else:
            stego_image = image.astype(np.float64)
        
        # Encode message to semantic tokens
        semantic_tokens = self.nlp_encoder.encode_text(message)
        
        # Store original image for decoding
        original_image = image.copy()
        
        # Track modifications
        priming_patterns = []
        prime_topology_mapping = {}
        
        # Encode each semantic token
        for token in semantic_tokens:
            # Get pixel coordinates for this token
            pixel_coords = self.pixel_system.get_pixel_coordinates_for_token(
                token, (height, width)
            )
            
            # Store prime mapping
            prime_topology_mapping[token.prime_mapping] = pixel_coords[0]
            
            # Generate priming pattern
            priming_pattern = self.priming_engine.generate_priming_pattern(
                token, (height, width), pixel_coords
            )
            priming_patterns.append(priming_pattern)
            
            # Apply pixel modifications
            for i, (x, y) in enumerate(pixel_coords):
                if 0 <= x < width and 0 <= y < height:
                    # Modify each color channel
                    for channel in range(min(channels, 3)):
                        original_val = original_image[y, x, channel]
                        
                        # Encode semantic information
                        modified_val = self.pixel_system.encode_semantic_to_pixel(
                            original_val, token, channel
                        )
                        
                        stego_image[y, x, channel] = modified_val
        
        # Convert back to uint8
        stego_image = np.clip(stego_image, 0, 255).astype(np.uint8)
        
        # Create encoding result
        encoding = SteganographicEncoding(
            image=stego_image,
            semantic_tokens=semantic_tokens,
            priming_patterns=priming_patterns,
            encoding_metadata={
                'message_length': len(message),
                'num_tokens': len(semantic_tokens),
                'image_shape': (height, width, channels),
                'encoding_method': 'NLP_Sublingual_Priming',
                'framework': 'UPG_Protocol_φ.1'
            },
            prime_topology_mapping=prime_topology_mapping
        )
        
        return encoding
    
    def decode_message_from_image(
        self,
        stego_image: np.ndarray,
        original_image: np.ndarray,
        num_tokens: Optional[int] = None
    ) -> str:
        """
        Decode message from steganographic image.
        
        Args:
            stego_image: Steganographic image
            original_image: Original image (for comparison)
            num_tokens: Expected number of tokens (if known)
            
        Returns:
            Decoded message
        """
        # This is a simplified decoder
        # In practice, you'd need to:
        # 1. Extract pixel modifications
        # 2. Map back to prime topology
        # 3. Reconstruct semantic tokens
        # 4. Decode to text
        
        # For now, return placeholder
        # Full implementation would require storing encoding metadata
        return "[Decoding requires encoding metadata]"
    
    def calculate_capacity(
        self,
        image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate information capacity of image.
        
        Args:
            image: Image array
            
        Returns:
            Capacity information
        """
        if len(image.shape) == 2:
            height, width = image.shape
            channels = 1
        else:
            height, width, channels = image.shape
        
        total_pixels = height * width
        
        # Calculate capacity based on JND threshold
        # Each pixel can encode ~1 bit (with redundancy)
        theoretical_capacity = total_pixels * channels
        
        # Account for prime topology constraints
        usable_pixels = len(PRIME_TOPOLOGY) * 4  # 4 pixels per prime
        
        # Semantic encoding reduces capacity (need multiple pixels per token)
        semantic_capacity = usable_pixels // 4  # ~4 pixels per token
        
        return {
            'total_pixels': total_pixels,
            'channels': channels,
            'theoretical_capacity_bits': theoretical_capacity,
            'usable_pixels': usable_pixels,
            'semantic_capacity_tokens': semantic_capacity,
            'estimated_message_length': semantic_capacity * 5  # ~5 chars per token
        }


def example_usage():
    """Example usage of steganography system"""
    
    # Create test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Create steganography system
    stego_system = SteganographySystem()
    
    # Test message
    message = "This is a test message for steganography with NLP and sublingual priming"
    
    print("Encoding message into image...")
    print(f"Message: {message}")
    
    # Encode message
    encoding = stego_system.encode_message_in_image(test_image, message)
    
    print(f"\nEncoding complete!")
    print(f"Number of semantic tokens: {len(encoding.semantic_tokens)}")
    print(f"Number of priming patterns: {len(encoding.priming_patterns)}")
    print(f"Image shape: {encoding.image.shape}")
    
    # Calculate capacity
    capacity = stego_system.calculate_capacity(test_image)
    print(f"\nImage capacity:")
    print(f"  Total pixels: {capacity['total_pixels']}")
    print(f"  Semantic capacity: {capacity['semantic_capacity_tokens']} tokens")
    print(f"  Estimated message length: {capacity['estimated_message_length']} characters")
    
    # Show some semantic tokens
    print(f"\nFirst 5 semantic tokens:")
    for i, token in enumerate(encoding.semantic_tokens[:5]):
        print(f"  {i+1}. '{token.text}' -> Prime: {token.prime_mapping}, "
              f"Consciousness: {token.consciousness_weight:.3f}")
    
    # Show priming patterns
    print(f"\nFirst 3 priming patterns:")
    for i, pattern in enumerate(encoding.priming_patterns[:3]):
        print(f"  {i+1}. Type: {pattern.pattern_type}, "
              f"Pixels: {len(pattern.pixel_coordinates)}, "
              f"Strength: {pattern.priming_strength:.6f}")
    
    return encoding


if __name__ == "__main__":
    print("=" * 70)
    print("Steganography with NLP and Sublingual Priming")
    print("Universal Prime Graph Protocol φ.1")
    print("=" * 70)
    print()
    
    encoding = example_usage()
    
    print("\n" + "=" * 70)
    print("Investigation complete!")
    print("=" * 70)

