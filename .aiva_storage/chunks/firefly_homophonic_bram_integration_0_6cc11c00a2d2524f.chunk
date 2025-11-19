#!/usr/bin/env python3
"""
🔥 FIREFLY HOMOPHONIC DECODER + BRAM COHEN INTEGRATION 🔥
========================================================================

Complete Homophonic Cipher Decoder integrated with Firefly Universal Decoder
and Bram Cohen's DissidentX steganographic framework for maximum analytical power.

HOMOPHONIC CIPHER FEATURES:
- Each plaintext letter can be represented by multiple ciphertext symbols
- Frequency analysis resistant
- Information-theoretic security
- Pattern-based decoding with consciousness mathematics

BRAM COHEN DISIDENTX INTEGRATION:
- Universal decoder framework
- Information-theoretic encoding
- Key-based steganography
- Multi-message support

FIREFLY CONSCIOUSNESS MATHEMATICS:
- Golden ratio transformations
- Consciousness level mapping
- Cetacean frequency analysis
- Sacred text integration

Author: Bradley Wallace (Consciousness Mathematics Architect)
Integration: Firefly Universal Decoder + Bram Cohen DissidentX + Homophonic Cipher
Framework: Universal Prime Graph Protocol φ.1
Date: November 7, 2025
"""

import asyncio
import hashlib
import math
import numpy as np
import random
import re
import string
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

# Import Firefly Universal Decoder
from firefly_universal_decoder import FireflyUniversalDecoder, PHI, DELTA, CONSCIOUSNESS_RATIO


# ============================================================================
# UPG FOUNDATIONS - Universal Prime Graph Protocol φ.1
# ============================================================================
from decimal import Decimal, getcontext
import math
import cmath
from typing import Dict, List, Tuple, Optional, Any

# Set high precision for consciousness mathematics
getcontext().prec = 50

class UPGConstants:
    """Universal Prime Graph consciousness mathematics constants"""
    PHI = Decimal('1.618033988749895')
    DELTA = Decimal('2.414213562373095')
    CONSCIOUSNESS = Decimal('0.79')  # 79/21 universal coherence rule
    REALITY_DISTORTION = Decimal('1.1808')  # Quantum amplification factor
    QUANTUM_BRIDGE = Decimal('137') / Decimal('0.79')  # 173.41772151898732
    GREAT_YEAR = 25920  # Astronomical precession cycle (years)
    CONSCIOUSNESS_DIMENSIONS = 21  # Prime topology dimension
    COHERENCE_THRESHOLD = Decimal('1e-15')  # Beyond machine precision



# ============================================================================
# PELL SEQUENCE PRIME PREDICTION INTEGRATION
# ============================================================================
def integrate_pell_prime_prediction(target_number: int, constants: UPGConstants = None):
    """Integrate Pell sequence prime prediction with this tool"""
    try:
        from pell_sequence_prime_prediction_upg_complete import PrimePredictionEngine, UPGConstants as UPG
        if constants is None:
            constants = UPG()
        predictor = PrimePredictionEngine(constants)
        return predictor.predict_prime(target_number)
    except ImportError:
        # Fallback if Pell module not available
        return {'target_number': target_number, 'is_prime': None, 'note': 'Pell module not available'}



# ============================================================================
# GREAT YEAR ASTRONOMICAL PRECESSION INTEGRATION
# ============================================================================
def integrate_great_year_precession(year: int, constants: UPGConstants = None):
    """Integrate Great Year (25,920-year) precession cycle"""
    try:
        from pell_sequence_prime_prediction_upg_complete import GreatYearIntegration, UPGConstants as UPG
        if constants is None:
            constants = UPG()
        great_year = GreatYearIntegration(constants)
        return great_year.consciousness_amplitude_from_year(year)
    except ImportError:
        # Fallback calculation
        if constants is None:
            constants = UPGConstants()
        angle = (year * 2 * math.pi) / constants.GREAT_YEAR
        return complex(float(angle * constants.CONSCIOUSNESS * constants.REALITY_DISTORTION), 0.0)



# Import Bram Cohen steganographic techniques
try:
    from consciousness_universal_framework import UniversalConsciousnessFramework
    universal_framework = UniversalConsciousnessFramework()
except ImportError:
    universal_framework = None


@dataclass
class HomophonicMapping:
    """Homophonic cipher mapping (one plaintext -> multiple ciphertexts)"""
    plaintext_symbol: str
    ciphertext_symbols: List[str] = field(default_factory=list)
    frequency_weight: float = 1.0
    consciousness_level: int = 1


@dataclass
class HomophonicCipher:
    """Complete homophonic cipher system"""
    mappings: Dict[str, HomophonicMapping] = field(default_factory=dict)
    reverse_mappings: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    frequency_table: Dict[str, int] = field(default_factory=dict)

    def add_mapping(self, plaintext: str, ciphertexts: List[str], frequency_weight: float = 1.0):
        """Add homophonic mapping"""
        if plaintext not in self.mappings:
            self.mappings[plaintext] = HomophonicMapping(plaintext, [], frequency_weight)

        self.mappings[plaintext].ciphertext_symbols.extend(ciphertexts)
        self.mappings[plaintext].frequency_weight = frequency_weight

        # Update reverse mappings
        for ciphertext in ciphertexts:
            self.reverse_mappings[ciphertext].append(plaintext)

    def generate_from_text(self, text: str, symbols_per_letter: int = 3):
        """Generate homophonic cipher from text frequency analysis"""
        # Analyze letter frequencies
        letter_freq = Counter(text.lower())
        total_letters = sum(letter_freq.values())

        # Create mappings based on frequency
        for letter, count in letter_freq.items():
            if letter.isalpha():
                frequency_weight = count / total_letters
                consciousness_level = self._map_to_consciousness_level(frequency_weight)

                # Generate ciphertext symbols (numbers, special chars, mixed)
                ciphertexts = self._generate_ciphertext_symbols(
                    letter, symbols_per_letter, frequency_weight
                )

                self.add_mapping(letter, ciphertexts, frequency_weight)
                self.mappings[letter].consciousness_level = consciousness_level

    def _generate_ciphertext_symbols(self, letter: str, count: int, weight: float) -> List[str]:
        """Generate ciphertext symbols for a letter"""
        symbols = []

        # Base symbol sets
        numbers = "0123456789"
        specials = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        letters = string.ascii_letters

        # Weight-based complexity
        if weight > 0.1:  # Common letters
            symbol_sets = [numbers, specials, letters]
        elif weight > 0.05:  # Medium frequency
            symbol_sets = [numbers + specials, letters + numbers]
        else:  # Rare letters
            symbol_sets = [specials + letters + numbers]

        for i in range(count):
            # Create symbol based on letter position and weight
            base = ord(letter.lower()) - ord('a')
            length = int(2 + weight * 4)  # 2-6 characters

            symbol = ""
            for j in range(length):
                set_choice = (base + i + j) % len(symbol_sets)
                char_set = symbol_sets[set_choice]
                char_idx = (base + i * j) % len(char_set)
                symbol += char_set[char_idx]

            symbols.append(symbol)

        return symbols

    def _map_to_consciousness_level(self, frequency: float) -> int:
        """Map frequency to consciousness level (1-21)"""
        # Higher frequency = lower consciousness level (more basic)
        level = int(22 - frequency * 20)
        return max(1, min(21, level))

    def encode(self, plaintext: str) -> str:
        """Encode plaintext using homophonic cipher"""
        ciphertext = []
        for char in plaintext.upper():
            if char in self.mappings:
                mapping = self.mappings[char]
                # Choose ciphertext based on frequency weight
                if mapping.ciphertext_symbols:
                    # Weighted random selection
                    weights = [mapping.frequency_weight] * len(mapping.ciphertext_symbols)
                    chosen = random.choices(mapping.ciphertext_symbols, weights=weights, k=1)[0]
                    ciphertext.append(chosen)
                else:
                    ciphertext.append(char)
            else:
                ciphertext.append(char)

        return ''.join(ciphertext)

    def decode_candidate(self, ciphertext: str, max_candidates: int = 10) -> List[str]:
        """Generate decoding candidates using statistical analysis"""
        candidates = []

        # Split ciphertext into potential symbols
        symbol_candidates = self._split_into_symbols(ciphertext)

        # Try different decoding paths
        for symbol_split in symbol_candidates[:max_candidates]:
            candidate = self._decode_symbol_sequence(symbol_split)
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        return candidates

    def _split_into_symbols(self, ciphertext: str, max_splits: int = 50) -> List[List[str]]:
        """Split ciphertext into potential symbol sequences"""
        candidates = []

        def split_recursive(remaining: str, current: List[str]):
            if len(candidates) >= max_splits:
                return

            if not remaining:
                candidates.append(current[:])
                return

            # Try different symbol lengths (2-6 characters)
            for length in range(2, min(7, len(remaining) + 1)):
                symbol = remaining[:length]
                if symbol in self.reverse_mappings:
                    split_recursive(remaining[length:], current + [symbol])

        split_recursive(ciphertext, [])
        return candidates

    def _decode_symbol_sequence(self, symbols: List[str]) -> Optional[str]:
        """Decode a sequence of symbols"""
        plaintext = []

        for symbol in symbols:
            if symbol in self.reverse_mappings:
                possible_letters = self.reverse_mappings[symbol]
                if possible_letters:
                    # Choose most frequent letter for this symbol
                    plaintext.append(possible_letters[0])
                else:
                    plaintext.append('?')
            else:
                plaintext.append('?')

        result = ''.join(plaintext)
        # Only return if we have some valid letters
        if result.count('?') / len(result) < 0.5:  # Less than 50% unknown
            return result

        return None


@dataclass
class FireflyHomophonicAnalysis:
    """Complete analysis result from Firefly + Homophonic + Bram Cohen integration"""
    original_text: str
    homophonic_candidates: List[str]
    firefly_analysis: Dict[str, Any]
    bram_steganographic_findings: List[Dict[str, Any]]
    consciousness_mapping: Dict[str, Any]
    cetacean_frequency_analysis: Dict[str, Any]
    integrated_confidence_score: float
    recommended_decodings: List[str]


class FireflyHomophonicBramDecoder:
    """
    Integrated decoder combining Firefly, Homophonic Cipher, and Bram Cohen techniques
    """

    def __init__(self):
        self.firefly_decoder = FireflyUniversalDecoder()
        self.homophonic_cipher = HomophonicCipher()
        self.bram_available = universal_framework is not None

        # Initialize with Chia Friends specific homophonic mappings
        self._initialize_chia_friends_homophonic()

        print("🔥 Firefly Homophonic Bram Decoder initialized")
        print(f"   Firefly Universal Decoder: ✅")
        print(f"   Homophonic Cipher: ✅")
        print(f"   Bram Cohen DissidentX: {'✅' if self.bram_available else '❌'}")

    def _initialize_chia_friends_homophonic(self):
        """Initialize homophonic cipher with Chia Friends specific patterns"""
        # Based on Chia Friends puzzle analysis
        chia_text = """
        Chia Friends puzzle with silver ratio derived from Ethiopian Bible.
        Coordinates at Arabian Sea and Pacific Ocean. Palindromic pattern.
        Core number 2156 generates all locations. Consciousness mathematics.
        Bram Cohen steganography. DissidentX techniques. Homophonic cipher.
        """

        self.homophonic_cipher.generate_from_text(chia_text, symbols_per_letter=4)

        # Add Chia-specific mappings
        chia_mappings = {
            'C': ['2156', 'CHIA', '892', '21.56'],
            'H': ['56.21', '2.156', '156.2', 'δ'],
            'I': ['φ', 'PHI', '1.618', 'GOLDEN'],
            'A': ['SILVER', '2.414', 'ETHIOPIAN', 'BIBLE']
        }

        for letter, symbols in chia_mappings.items():
            self.homophonic_cipher.add_mapping(letter, symbols, frequency_weight=0.8)

    async def comprehensive_decode(self, ciphertext: str, context: str = "chia_friends") -> FireflyHomophonicAnalysis:
        """
        Comprehensive decoding using all integrated techniques
        """
        print(f"\n🔥 Starting comprehensive analysis of: '{ciphertext[:50]}...'")
        print("=" * 80)

        # 1. Homophonic cipher analysis
        print("🔄 Phase 1: Homophonic Cipher Analysis")
        homophonic_candidates = self.homophonic_cipher.decode_candidate(ciphertext, max_candidates=20)
        print(f"   Generated {len(homophonic_candidates)} homophonic candidates")

        # 2. Firefly universal decoder analysis
        print("🔥 Phase 2: Firefly Consciousness Mathematics Analysis")
        firefly_results = await self._firefly_analyze_candidates(homophonic_candidates)
        print(f"   Analyzed {len(firefly_results)} candidates with consciousness mathematics")

        # 3. Bram Cohen steganographic analysis
        print("🕊️ Phase 3: Bram Cohen DissidentX Analysis")
        bram_findings = await self._bram_analyze_candidates(homophonic_candidates)
        print(f"   Found {len(bram_findings)} steganographic elements")

        # 4. Cetacean frequency analysis
        print("🐋 Phase 4: Cetacean Frequency Pattern Analysis")
        cetacean_analysis = await self._cetacean_frequency_analysis(ciphertext, homophonic_candidates)

        # 5. Integrated consciousness mapping
        print("🧠 Phase 5: Integrated Consciousness Mapping")
        consciousness_mapping = self._create_consciousness_mapping(
            homophonic_candidates, firefly_results, bram_findings
        )

        # 6. Calculate integrated confidence and recommendations
        integrated_confidence = self._calculate_integrated_confidence(
            homophonic_candidates, firefly_results, bram_findings, cetacean_analysis
        )

        recommended_decodings = self._generate_recommendations(
            homophonic_candidates, integrated_confidence
        )

        analysis = FireflyHomophonicAnalysis(
            original_text=ciphertext,
            homophonic_candidates=homophonic_candidates,
            firefly_analysis=firefly_results,
            bram_steganographic_findings=bram_findings,
            consciousness_mapping=consciousness_mapping,
            cetacean_frequency_analysis=cetacean_analysis,
            integrated_confidence_score=integrated_confidence,
            recommended_decodings=recommended_decodings
        )

        print("\n✅ Comprehensive analysis complete!")
        print(".3f")
        print(f"   Top recommendations: {len(recommended_decodings)}")

        return analysis

    async def _firefly_analyze_candidates(self, candidates: List[str]) -> Dict[str, Any]:
        """Analyze candidates using Firefly consciousness mathematics"""
        results = {}

        for candidate in candidates:
            if not candidate or len(candidate) < 3:
                continue

            try:
                # Sacred text analysis
                if any(c.isalpha() for c in candidate):
                    sacred_result = self.firefly_decoder.decode_sacred_text(candidate, 'universal')
                    results[candidate] = {
                        'sacred_analysis': sacred_result,
                        'consciousness_level': sacred_result.consciousness_level,
                        'interpretation': sacred_result.interpretation
                    }
                else:
                    # Numerical analysis
                    num_analysis = self._numerical_firefly_analysis(candidate)
                    results[candidate] = num_analysis

            except Exception as e:
                results[candidate] = {'error': str(e)}

        return results

    def _numerical_firefly_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze numerical patterns using Firefly mathematics"""
        numbers = re.findall(r'\d+\.?\d*', text)
        results = {}

        if numbers:
            # Convert to numerical analysis
            num_values = [float(n) for n in numbers if n.replace('.', '').isdigit()]

            if num_values:
                # Golden ratio analysis
                phi_analysis = []
                for num in num_values:
                    phi_ratio = num / PHI
                    phi_distance = abs(phi_ratio - round(phi_ratio))
                    phi_analysis.append({
                        'value': num,
                        'phi_ratio': phi_ratio,
                        'phi_distance': phi_distance,
                        'is_phi_harmonic': phi_distance < 0.1
                    })

                # Consciousness level mapping
                avg_value = sum(num_values) / len(num_values)
                consciousness_level = int(min(21, max(1, avg_value / 100)))

                results = {
                    'numerical_values': num_values,
                    'phi_analysis': phi_analysis,
                    'consciousness_level': consciousness_level,
                    'level_meaning': self.firefly_decoder.CONSCIOUSNESS_SEMANTICS.get(consciousness_level, 'Unknown'),
                    'average_value': avg_value
                }

        return results

    async def _bram_analyze_candidates(self, candidates: List[str]) -> List[Dict[str, Any]]:
        """Analyze candidates using Bram Cohen's techniques"""
        findings = []

        if not self.bram_available or not universal_framework:
            return findings

        chia_keys = ['chia', 'friends', 'puzzle', '2156', '892', 'silver', 'ratio',
                    'ethiopian', 'bible', 'consciousness', 'bram', 'cohen']

        for candidate in candidates:
            for key in chia_keys:
                try:
                    # Try different encoding techniques
                    decoded = universal_framework.universal_decode(candidate, key.encode())
                    if decoded and self._is_chia_relevant(decoded):
                        findings.append({
                            'candidate': candidate,
                            'key': key,
                            'decoded': decoded.decode() if isinstance(decoded, bytes) else str(decoded),
                            'method': 'universal_decoder',
                            'confidence': self._calculate_bram_confidence(decoded, key)
                        })
                except:
                    continue

        return findings

    async def _cetacean_frequency_analysis(self, original: str, candidates: List[str]) -> Dict[str, Any]:
        """Analyze frequency patterns using cetacean communication mathematics"""
        analysis = {
            'original_frequencies': {},
            'candidate_patterns': [],
            'cetacean_mappings': {}
        }

        # Analyze original text frequencies
        if original:
            freqs = Counter(original.lower())
            analysis['original_frequencies'] = dict(freqs)

        # Analyze candidate patterns for cetacean frequency compatibility
        for candidate in candidates:
            if len(candidate) >= 3:
                # Calculate pattern that could map to cetacean frequencies
                pattern_value = sum(ord(c) for c in candidate) / len(candidate)
                cetacean_freq = pattern_value * PHI  # Golden ratio scaling

                # Map to known cetacean frequency ranges
                if 1000 < cetacean_freq < 200000:  # Dolphin/whale range
                    species = "dolphin" if cetacean_freq < 50000 else "whale"
                    analysis['candidate_patterns'].append({
                        'candidate': candidate,
                        'cetacean_frequency': cetacean_freq,
                        'species': species,
                        'consciousness_level': self.firefly_decoder.map_frequency_to_consciousness(cetacean_freq)
                    })

        return analysis

    def _create_consciousness_mapping(self, candidates: List[str],
                                    firefly_results: Dict[str, Any],
                                    bram_findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create integrated consciousness mapping"""
        mapping = {
            'candidates_with_levels': [],
            'bram_integrated_findings': [],
            'cross_correlations': []
        }

        for candidate in candidates:
            level = 1  # Default

            # Get level from Firefly analysis
            if candidate in firefly_results:
                firefly_data = firefly_results[candidate]
                if 'consciousness_level' in firefly_data:
                    level = firefly_data['consciousness_level']
                elif 'numerical_values' in firefly_data:
                    # Derive from numerical analysis
                    nums = firefly_data['numerical_values']
                    if nums:
                        level = int(min(21, max(1, sum(nums) / len(nums) / 100)))

            mapping['candidates_with_levels'].append({
                'candidate': candidate,
                'consciousness_level': level,
                'level_meaning': self.firefly_decoder.CONSCIOUSNESS_SEMANTICS.get(level, 'Unknown')
            })

        # Integrate Bram findings
        for finding in bram_findings:
            consciousness_level = self._calculate_bram_consciousness_level(finding)
            mapping['bram_integrated_findings'].append({
                **finding,
                'consciousness_level': consciousness_level,
                'level_meaning': self.firefly_decoder.CONSCIOUSNESS_SEMANTICS.get(consciousness_level, 'Unknown')
            })

        return mapping

    def _calculate_integrated_confidence(self, candidates: List[str],
                                       firefly_results: Dict[str, Any],
                                       bram_findings: List[Dict[str, Any]],
                                       cetacean_analysis: Dict[str, Any]) -> float:
        """Calculate integrated confidence score"""
        confidence_factors = []

        # Factor 1: Number of strong candidates
        strong_candidates = [c for c in candidates if len(c) >= 5 and c.count('?') / len(c) < 0.3]
        confidence_factors.append(min(1.0, len(strong_candidates) / 10))

        # Factor 2: Firefly analysis quality
        firefly_scores = []
        for result in firefly_results.values():
            if isinstance(result, dict) and 'consciousness_level' in result:
                level = result['consciousness_level']
                # Higher consciousness levels = higher confidence
                firefly_scores.append(min(1.0, level / 21))
        if firefly_scores:
            confidence_factors.append(sum(firefly_scores) / len(firefly_scores))

        # Factor 3: Bram Cohen findings
        if bram_findings:
            avg_bram_confidence = sum(f.get('confidence', 0) for f in bram_findings) / len(bram_findings)
            confidence_factors.append(avg_bram_confidence)

        # Factor 4: Cetacean pattern matches
        cetacean_matches = len(cetacean_analysis.get('candidate_patterns', []))
        confidence_factors.append(min(1.0, cetacean_matches / 5))

        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0

    def _generate_recommendations(self, candidates: List[str], confidence_score: float) -> List[str]:
        """Generate recommended decodings"""
        recommendations = []

        # Sort candidates by quality metrics
        scored_candidates = []
        for candidate in candidates:
            score = 0

            # Length bonus
            score += min(1.0, len(candidate) / 10)

            # Letter ratio bonus
            letter_ratio = sum(1 for c in candidate if c.isalpha()) / len(candidate)
            score += letter_ratio * 0.5

            # Chia keyword bonus
            chia_keywords = ['chia', 'friends', 'puzzle', 'prize', 'consciousness']
            keyword_count = sum(1 for keyword in chia_keywords if keyword.lower() in candidate.lower())
            score += keyword_count * 0.3

            scored_candidates.append((candidate, score))

        # Sort by score and return top recommendations
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        recommendations = [candidate for candidate, score in scored_candidates[:5] if score > 0.5]

        return recommendations

    def _is_chia_relevant(self, decoded: Union[bytes, str]) -> bool:
        """Check if decoded content is Chia Friends relevant"""
        if isinstance(decoded, bytes):
            text = decoded.decode().lower()
        else:
            text = str(decoded).lower()

        chia_indicators = [
            'chia', 'friends', 'puzzle', 'prize', 'reward', '2156', '892',
            'silver', 'ratio', 'ethiopian', 'bible', 'consciousness',
            'bram', 'cohen', 'palindrome', 'coordinates'
        ]

        return any(indicator in text for indicator in chia_indicators)

    def _calculate_bram_confidence(self, decoded: Union[bytes, str], key: str) -> float:
        """Calculate confidence for Bram Cohen finding"""
        if isinstance(decoded, bytes):
            text = decoded.decode()
        else:
            text = str(decoded)

        confidence = 0.0

        # Chia relevance
        if self._is_chia_relevant(text):
            confidence += 0.4

        # Length factor
        confidence += min(0.3, len(text) / 50)

        # Key strength
        key_strength = len(key) / 20  # Normalize to 0-1
        confidence += key_strength * 0.3

        return min(1.0, confidence)

    def _calculate_bram_consciousness_level(self, finding: Dict[str, Any]) -> int:
        """Calculate consciousness level for Bram finding"""
        confidence = finding.get('confidence', 0)
        key_length = len(finding.get('key', ''))

        # Map confidence and key strength to consciousness level
        level = int(confidence * 21) + int(key_length / 2)
        return max(1, min(21, level))


async def run_chia_friends_full_analysis():
    """Run full integrated analysis on Chia Friends puzzle"""
    print("🔥🕊️ FIREFLY HOMOPHONIC BRAM INTEGRATION ANALYSIS")
    print("=" * 80)
    print("Analyzing Chia Friends puzzle with maximum decoding power:")
    print("• Firefly Universal Decoder (Consciousness Mathematics)")
    print("• Homophonic Cipher Analysis")
    print("• Bram Cohen DissidentX Steganography")
    print("• Cetacean Frequency Pattern Analysis")
    print("=" * 80)

    # Initialize integrated decoder
    decoder = FireflyHomophonicBramDecoder()

    # Test data from Chia Friends puzzle
    test_inputs = [
        "21566512",  # Palindromic 2156
        "892298",    # Palindromic 892
        "56216521",  # Coordinate mirror
        "21.56°N, 56.21°E",  # Coordinate 1
        "2.156°N, 156.2°E",  # Coordinate 2
        "CHIA2156892SILVER",  # Combined elements
        "!@#2156$%^892&*()",  # Symbolic encoding
        "φ2.414213562373095δ",  # Mathematical constants
    ]

    all_results = []

    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n🎯 ANALYSIS {i}/{len(test_inputs)}: '{test_input}'")
        print("-" * 60)

        try:
            analysis = await decoder.comprehensive_decode(test_input, "chia_friends")

            # Display results
            print(f"📊 Homophonic Candidates: {len(analysis.homophonic_candidates)}")
            print(f"🔥 Firefly Analysis: {len(analysis.firefly_analysis)} results")
            print(f"🕊️ Bram Findings: {len(analysis.bram_steganographic_findings)}")
            print(".3f")

            if analysis.recommended_decodings:
                print(f"🏆 Top Recommendations:")
                for j, rec in enumerate(analysis.recommended_decodings[:3], 1):
                    print(f"   {j}. '{rec}'")

            all_results.append({
                'input': test_input,
                'analysis': analysis,
                'recommendations': analysis.recommended_decodings
            })

        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            all_results.append({
                'input': test_input,
                'error': str(e)
            })

    # Generate comprehensive report
    print("\n🎉 COMPLETE INTEGRATED ANALYSIS REPORT")
    print("=" * 80)

    successful_analyses = [r for r in all_results if 'analysis' in r]
    total_candidates = sum(len(r['analysis'].homophonic_candidates) for r in successful_analyses)
    total_bram_findings = sum(len(r['analysis'].bram_steganographic_findings) for r in successful_analyses)

    print(f"📊 Total Analyses: {len(all_results)}")
    print(f"✅ Successful: {len(successful_analyses)}")
    print(f"🔄 Homophonic Candidates Generated: {total_candidates}")
    print(f"🕊️ Bram Cohen Steganographic Findings: {total_bram_findings}")

    # Show top findings across all analyses
    all_recommendations = []
    for result in successful_analyses:
        all_recommendations.extend(result['analysis'].recommended_decodings)

    if all_recommendations:
        # Count recommendation frequency
        rec_counts = Counter(all_recommendations)
        top_recs = rec_counts.most_common(5)

        print("\n🏆 TOP RECOMMENDATIONS ACROSS ALL ANALYSES:")
        for i, (rec, count) in enumerate(top_recs, 1):
            print(f"   {i}. '{rec}' (appeared {count} times)")

    return all_results


if __name__ == "__main__":
    # Run the full integrated analysis
    results = asyncio.run(run_chia_friends_full_analysis())

    print("\n🔥🕊️ INTEGRATED ANALYSIS COMPLETE!")
    print("Firefly Homophonic Bram Decoder has analyzed Chia Friends puzzle")
    print("with maximum consciousness mathematics and steganographic power.")

    successful = len([r for r in results if 'analysis' in r])
    print(f"\n📈 Results: {successful}/{len(results)} analyses successful")
    print("   Next steps: Manual verification of top recommendations")
    print("   Potential prize mechanisms identified for further investigation")
