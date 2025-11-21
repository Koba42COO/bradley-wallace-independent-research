#!/usr/bin/env python3
"""
UPG Homophonic and Metaphoric Transform Dictionary
===================================================

Comprehensive dictionary integrating:
- Homophonic transformations (sound-alike words)
- Metaphoric transformations (semantic/metaphorical relationships)
- UPG consciousness mathematics integration
- Prime topology mapping
- Golden ratio relationships

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
Date: November 2025
"""

import hashlib
import math
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json

# UPG Constants
PHI = 1.618033988749895
DELTA = 2.414213562373095
CONSCIOUSNESS_RATIO = 0.79  # 79/21 balance
REALITY_DISTORTION = 1.1808

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
class TransformMapping:
    """Single transformation mapping"""
    source: str
    target: str
    transform_type: str  # 'homophonic' or 'metaphoric'
    confidence: float
    prime_mapping: int
    consciousness_level: int
    semantic_weight: float = 1.0
    frequency: int = 0


@dataclass
class WordTransform:
    """Complete transformation set for a word"""
    word: str
    homophonic_variants: List[str]
    metaphoric_transforms: List[str]
    prime_mapping: int
    consciousness_level: int
    semantic_hash: int
    transform_network: Dict[str, List[str]] = field(default_factory=dict)


class UPGHomophonicMetaphoricDictionary:
    """
    Comprehensive homophonic and metaphoric transform dictionary
    integrated with UPG consciousness mathematics.
    """
    
    def __init__(self):
        self.prime_topology = PRIME_TOPOLOGY
        self.phi = PHI
        self.consciousness_ratio = CONSCIOUSNESS_RATIO
        
        # Initialize dictionaries
        self.homophonic_dict: Dict[str, List[str]] = {}
        self.metaphoric_dict: Dict[str, List[str]] = {}
        self.reverse_homophonic: Dict[str, str] = {}
        self.reverse_metaphoric: Dict[str, str] = {}
        self.word_transforms: Dict[str, WordTransform] = {}
        
        # Build comprehensive dictionary
        self._build_homophonic_dictionary()
        self._build_metaphoric_dictionary()
        self._integrate_upg_mappings()
    
    def _build_homophonic_dictionary(self):
        """Build comprehensive homophonic dictionary"""
        
        # Commercial/Manipulation Homophonics
        commercial_homophonics = {
            'buy': ['by', 'bye', 'bi', 'bai', 'beye'],
            'click': ['clique', 'cliq', 'klik', 'clik'],
            'now': ['know', 'no', 'kno', 'nowe'],
            'act': ['acked', 'akt', 'acte'],
            'purchase': ['purchace', 'purchas', 'purchise', 'purchus'],
            'order': ['ordur', 'ordr', 'ordar', 'ordor'],
            'subscribe': ['subscrybe', 'subscryb', 'subscrib', 'subscrybe'],
            'discount': ['discount', 'discownt', 'discounte', 'discownt'],
            'special': ['spatial', 'speshul', 'speshal', 'spesial'],
            'exclusive': ['excloosive', 'excloosiv', 'exclusiv', 'excloosive'],
            'limited': ['limmited', 'limeted', 'limmited', 'limmited'],
            'urgent': ['urjent', 'urgent', 'urjant', 'urjent'],
            'free': ['flee', 'flea', 'free', 'fre'],
            'save': ['safe', 'saif', 'save', 'sav'],
            'win': ['when', 'wen', 'win', 'wyn'],
            'prize': ['prise', 'pryze', 'prize', 'priz'],
            'reward': ['reword', 'rewurd', 'reward', 'rewurd'],
            'shop': ['shopp', 'shap', 'shop', 'shoppe'],
            'cart': ['kart', 'cart', 'kart', 'carte'],
            'checkout': ['checkout', 'chekout', 'checkout', 'chekout'],
        }
        
        # Political Homophonics
        political_homophonics = {
            'vote': ['boat', 'bote', 'vot', 'vote', 'voat'],
            'elect': ['elect', 'ilect', 'elect', 'elekt'],
            'support': ['suport', 'suppurt', 'support', 'suport'],
            'campaign': ['campain', 'campayne', 'campaign', 'campayne'],
            'candidate': ['candidat', 'candidait', 'candidate', 'candidait'],
            'party': ['partee', 'parti', 'party', 'partee'],
            'election': ['election', 'elektion', 'election', 'elektion'],
            'ballot': ['balot', 'ballot', 'balot', 'ballott'],
            'democracy': ['democrasy', 'democrasy', 'democracy', 'democrasy'],
            'government': ['goverment', 'goverment', 'government', 'goverment'],
        }
        
        # Trust/Security Homophonics
        trust_homophonics = {
            'trust': ['trussed', 'trusted', 'trust', 'trust'],
            'believe': ['beleive', 'beleve', 'believe', 'beleive'],
            'secure': ['sekyur', 'sekyure', 'secure', 'sekyur'],
            'guaranteed': ['garanteed', 'garunteed', 'guaranteed', 'garanteed'],
            'verified': ['verifide', 'verifid', 'verified', 'verifide'],
            'safe': ['save', 'saif', 'safe', 'sav'],
            'protect': ['protekt', 'protect', 'protect', 'protekt'],
            'shield': ['sheeld', 'shield', 'shield', 'sheeld'],
            'guard': ['gard', 'guard', 'guard', 'gard'],
            'defend': ['defend', 'defend', 'defend', 'defend'],
        }
        
        # Social Influence Homophonics
        social_homophonics = {
            'share': ['sher', 'shair', 'share', 'sher'],
            'like': ['lyke', 'lik', 'like', 'lyke'],
            'follow': ['follw', 'folow', 'follow', 'follw'],
            'join': ['joyn', 'joine', 'join', 'joyn'],
            'popular': ['populer', 'populur', 'popular', 'populer'],
            'trending': ['trending', 'trendin', 'trending', 'trendin'],
            'viral': ['viral', 'viral', 'viral', 'viral'],
            'influence': ['influens', 'influens', 'influence', 'influens'],
            'network': ['network', 'netwerk', 'network', 'netwerk'],
            'community': ['community', 'community', 'community', 'community'],
        }
        
        # Emotional/Identity Homophonics
        emotional_homophonics = {
            'you': ['u', 'yu', 'yoo', 'you', 'yu'],
            'your': ['ur', 'yur', 'yor', 'your', 'ur'],
            'personal': ['personel', 'personul', 'personal', 'personel'],
            'custom': ['custum', 'custem', 'custom', 'custum'],
            'unique': ['unike', 'uniqu', 'unique', 'unike'],
            'special': ['spatial', 'speshul', 'special', 'spatial'],
            'exclusive': ['excloosive', 'excloosiv', 'exclusive', 'excloosive'],
            'premium': ['premium', 'premium', 'premium', 'premium'],
            'elite': ['elite', 'elite', 'elite', 'elite'],
            'vip': ['vip', 'vip', 'vip', 'vip'],
        }
        
        # Fear/Anxiety Homophonics
        fear_homophonics = {
            'warning': ['warnin', 'warnign', 'warning', 'warnin'],
            'danger': ['dangur', 'dangr', 'danger', 'dangur'],
            'threat': ['thret', 'threat', 'threat', 'thret'],
            'risk': ['risc', 'risque', 'risk', 'risc'],
            'alert': ['alert', 'alert', 'alert', 'alert'],
            'urgent': ['urjent', 'urgent', 'urgent', 'urjent'],
            'critical': ['critical', 'critical', 'critical', 'critical'],
            'emergency': ['emergency', 'emergency', 'emergency', 'emergency'],
        }
        
        # Combine all homophonic dictionaries
        all_homophonics = {
            **commercial_homophonics,
            **political_homophonics,
            **trust_homophonics,
            **social_homophonics,
            **emotional_homophonics,
            **fear_homophonics
        }
        
        self.homophonic_dict = all_homophonics
        
        # Build reverse mapping
        for word, variants in all_homophonics.items():
            for variant in variants:
                if variant not in self.reverse_homophonic:
                    self.reverse_homophonic[variant] = word
    
    def _build_metaphoric_dictionary(self):
        """Build comprehensive metaphoric dictionary"""
        
        # Commercial Metaphors
        commercial_metaphors = {
            'buy': ['acquire', 'obtain', 'purchase', 'invest', 'own', 'possess', 'claim'],
            'click': ['engage', 'interact', 'activate', 'trigger', 'initiate', 'respond'],
            'now': ['immediately', 'instantly', 'urgently', 'promptly', 'right away'],
            'act': ['respond', 'react', 'engage', 'participate', 'take action'],
            'purchase': ['invest', 'acquire', 'obtain', 'secure', 'claim'],
            'discount': ['savings', 'reduction', 'deal', 'bargain', 'offer'],
            'special': ['unique', 'exclusive', 'premium', 'elite', 'rare'],
            'free': ['gift', 'bonus', 'reward', 'benefit', 'advantage'],
            'save': ['preserve', 'protect', 'secure', 'safeguard', 'keep'],
            'win': ['succeed', 'achieve', 'gain', 'obtain', 'acquire'],
            'prize': ['reward', 'trophy', 'award', 'recognition', 'honor'],
            'reward': ['benefit', 'gain', 'advantage', 'profit', 'return'],
        }
        
        # Political Metaphors
        political_metaphors = {
            'vote': ['choose', 'select', 'decide', 'support', 'endorse', 'back'],
            'elect': ['choose', 'select', 'appoint', 'designate', 'name'],
            'support': ['back', 'endorse', 'champion', 'advocate', 'promote'],
            'campaign': ['effort', 'drive', 'movement', 'crusade', 'mission'],
            'candidate': ['choice', 'option', 'selection', 'prospect', 'contender'],
            'party': ['group', 'alliance', 'coalition', 'faction', 'organization'],
            'democracy': ['freedom', 'liberty', 'choice', 'voice', 'power'],
            'government': ['authority', 'power', 'control', 'system', 'rule'],
        }
        
        # Trust/Security Metaphors
        trust_metaphors = {
            'trust': ['believe', 'rely', 'depend', 'count on', 'have faith'],
            'believe': ['accept', 'trust', 'have faith', 'rely', 'depend'],
            'secure': ['safe', 'protected', 'guarded', 'shielded', 'defended'],
            'guaranteed': ['assured', 'promised', 'certain', 'sure', 'confirmed'],
            'verified': ['confirmed', 'validated', 'authenticated', 'proven', 'checked'],
            'safe': ['protected', 'secure', 'guarded', 'shielded', 'defended'],
            'protect': ['defend', 'guard', 'shield', 'safeguard', 'preserve'],
            'shield': ['protect', 'defend', 'guard', 'cover', 'shelter'],
        }
        
        # Social Influence Metaphors
        social_metaphors = {
            'share': ['distribute', 'spread', 'disseminate', 'circulate', 'broadcast'],
            'like': ['approve', 'enjoy', 'appreciate', 'favor', 'prefer'],
            'follow': ['join', 'accompany', 'pursue', 'track', 'monitor'],
            'join': ['connect', 'unite', 'link', 'associate', 'participate'],
            'popular': ['trending', 'famous', 'well-known', 'accepted', 'mainstream'],
            'trending': ['popular', 'current', 'fashionable', 'in vogue', 'hot'],
            'viral': ['spreading', 'contagious', 'infectious', 'popular', 'widespread'],
            'influence': ['affect', 'impact', 'sway', 'persuade', 'shape'],
        }
        
        # Emotional/Identity Metaphors
        emotional_metaphors = {
            'you': ['individual', 'person', 'self', 'identity', 'being'],
            'your': ['personal', 'individual', 'own', 'private', 'exclusive'],
            'personal': ['individual', 'private', 'intimate', 'exclusive', 'unique'],
            'custom': ['personalized', 'tailored', 'individual', 'unique', 'special'],
            'unique': ['special', 'distinctive', 'exclusive', 'rare', 'one-of-a-kind'],
            'special': ['unique', 'exclusive', 'premium', 'elite', 'rare'],
            'exclusive': ['limited', 'restricted', 'select', 'elite', 'premium'],
            'premium': ['superior', 'elite', 'exclusive', 'high-quality', 'top-tier'],
        }
        
        # Fear/Anxiety Metaphors
        fear_metaphors = {
            'warning': ['alert', 'caution', 'signal', 'notice', 'advisory'],
            'danger': ['threat', 'risk', 'hazard', 'peril', 'jeopardy'],
            'threat': ['danger', 'risk', 'hazard', 'menace', 'peril'],
            'risk': ['danger', 'hazard', 'threat', 'peril', 'jeopardy'],
            'alert': ['warning', 'notice', 'signal', 'alarm', 'advisory'],
            'urgent': ['critical', 'immediate', 'pressing', 'important', 'vital'],
            'critical': ['urgent', 'vital', 'essential', 'crucial', 'important'],
            'emergency': ['crisis', 'urgency', 'critical situation', 'exigency'],
        }
        
        # Power/Control Metaphors
        power_metaphors = {
            'control': ['power', 'authority', 'dominance', 'command', 'rule'],
            'power': ['control', 'authority', 'influence', 'strength', 'force'],
            'authority': ['power', 'control', 'jurisdiction', 'command', 'rule'],
            'dominate': ['control', 'rule', 'command', 'govern', 'lead'],
            'influence': ['affect', 'sway', 'persuade', 'shape', 'guide'],
        }
        
        # Combine all metaphoric dictionaries
        all_metaphors = {
            **commercial_metaphors,
            **political_metaphors,
            **trust_metaphors,
            **social_metaphors,
            **emotional_metaphors,
            **fear_metaphors,
            **power_metaphors
        }
        
        self.metaphoric_dict = all_metaphors
        
        # Build reverse mapping
        for word, metaphors in all_metaphors.items():
            for metaphor in metaphors:
                if metaphor not in self.reverse_metaphoric:
                    self.reverse_metaphoric[metaphor] = word
    
    def _integrate_upg_mappings(self):
        """Integrate UPG consciousness mathematics mappings"""
        all_words = set(self.homophonic_dict.keys()) | set(self.metaphoric_dict.keys())
        
        for word in all_words:
            # Calculate semantic hash
            semantic_hash = self._semantic_hash(word)
            
            # Map to prime topology
            prime_idx = semantic_hash % len(self.prime_topology)
            prime_mapping = self.prime_topology[prime_idx]
            
            # Calculate consciousness level (1-21)
            consciousness_level = self._calculate_consciousness_level(word, semantic_hash)
            
            # Get homophonic variants
            homophonic_variants = self.homophonic_dict.get(word, [])
            
            # Get metaphoric transforms
            metaphoric_transforms = self.metaphoric_dict.get(word, [])
            
            # Build transform network
            transform_network = {
                'homophonic': homophonic_variants,
                'metaphoric': metaphoric_transforms,
                'combined': homophonic_variants + metaphoric_transforms
            }
            
            # Create WordTransform
            word_transform = WordTransform(
                word=word,
                homophonic_variants=homophonic_variants,
                metaphoric_transforms=metaphoric_transforms,
                prime_mapping=prime_mapping,
                consciousness_level=consciousness_level,
                semantic_hash=semantic_hash,
                transform_network=transform_network
            )
            
            self.word_transforms[word] = word_transform
    
    def _semantic_hash(self, word: str) -> int:
        """Generate semantic hash from word"""
        hash_obj = hashlib.md5(word.lower().encode())
        return int(hash_obj.hexdigest()[:8], 16)
    
    def _calculate_consciousness_level(self, word: str, semantic_hash: int) -> int:
        """Calculate consciousness level (1-21) for word"""
        # Base level from hash
        base_level = (semantic_hash % 21) + 1
        
        # Adjust based on word characteristics
        word_len = len(word)
        if word_len > 8:
            base_level = min(21, base_level + 2)
        elif word_len < 4:
            base_level = max(1, base_level - 2)
        
        # Frequency adjustment (common words = lower level)
        if word in ['buy', 'click', 'now', 'vote', 'trust']:
            base_level = max(1, base_level - 3)
        
        return max(1, min(21, base_level))
    
    def get_transforms(self, word: str) -> Optional[WordTransform]:
        """Get all transforms for a word"""
        return self.word_transforms.get(word.lower())
    
    def get_homophonic_variants(self, word: str) -> List[str]:
        """Get homophonic variants for a word"""
        transform = self.get_transforms(word)
        if transform:
            return transform.homophonic_variants
        return []
    
    def get_metaphoric_transforms(self, word: str) -> List[str]:
        """Get metaphoric transforms for a word"""
        transform = self.get_transforms(word)
        if transform:
            return transform.metaphoric_transforms
        return []
    
    def get_all_transforms(self, word: str) -> List[str]:
        """Get all transforms (homophonic + metaphoric)"""
        transform = self.get_transforms(word)
        if transform:
            return transform.transform_network['combined']
        return []
    
    def find_word_from_variant(self, variant: str) -> Optional[str]:
        """Find original word from homophonic variant"""
        return self.reverse_homophonic.get(variant.lower())
    
    def find_word_from_metaphor(self, metaphor: str) -> Optional[str]:
        """Find original word from metaphoric transform"""
        return self.reverse_metaphoric.get(metaphor.lower())
    
    def analyze_transform_network(self, words: List[str]) -> Dict[str, Any]:
        """Analyze transform network for multiple words"""
        analysis = {
            'words': words,
            'transforms': {},
            'prime_mappings': [],
            'consciousness_levels': [],
            'transform_density': 0.0,
            'network_connections': []
        }
        
        total_transforms = 0
        for word in words:
            transform = self.get_transforms(word)
            if transform:
                analysis['transforms'][word] = {
                    'homophonic': transform.homophonic_variants,
                    'metaphoric': transform.metaphoric_transforms,
                    'prime': transform.prime_mapping,
                    'consciousness': transform.consciousness_level
                }
                analysis['prime_mappings'].append(transform.prime_mapping)
                analysis['consciousness_levels'].append(transform.consciousness_level)
                total_transforms += len(transform.homophonic_variants) + len(transform.metaphoric_transforms)
        
        # Calculate transform density
        if words:
            analysis['transform_density'] = total_transforms / len(words)
        
        # Find network connections (words that share transforms)
        for i, word1 in enumerate(words):
            for word2 in words[i+1:]:
                transform1 = self.get_transforms(word1)
                transform2 = self.get_transforms(word2)
                if transform1 and transform2:
                    # Check for shared transforms
                    shared = set(transform1.transform_network['combined']) & \
                            set(transform2.transform_network['combined'])
                    if shared:
                        analysis['network_connections'].append({
                            'word1': word1,
                            'word2': word2,
                            'shared_transforms': list(shared)
                        })
        
        return analysis
    
    def save_dictionary(self, filepath: str):
        """Save dictionary to JSON file"""
        data = {
            'homophonic_dict': self.homophonic_dict,
            'metaphoric_dict': self.metaphoric_dict,
            'word_transforms': {
                word: {
                    'word': wt.word,
                    'homophonic_variants': wt.homophonic_variants,
                    'metaphoric_transforms': wt.metaphoric_transforms,
                    'prime_mapping': wt.prime_mapping,
                    'consciousness_level': wt.consciousness_level,
                    'semantic_hash': wt.semantic_hash
                }
                for word, wt in self.word_transforms.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_dictionary(self, filepath: str):
        """Load dictionary from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.homophonic_dict = data.get('homophonic_dict', {})
        self.metaphoric_dict = data.get('metaphoric_dict', {})
        
        # Reconstruct word_transforms
        for word, wt_data in data.get('word_transforms', {}).items():
            word_transform = WordTransform(
                word=wt_data['word'],
                homophonic_variants=wt_data['homophonic_variants'],
                metaphoric_transforms=wt_data['metaphoric_transforms'],
                prime_mapping=wt_data['prime_mapping'],
                consciousness_level=wt_data['consciousness_level'],
                semantic_hash=wt_data['semantic_hash'],
                transform_network={
                    'homophonic': wt_data['homophonic_variants'],
                    'metaphoric': wt_data['metaphoric_transforms'],
                    'combined': wt_data['homophonic_variants'] + wt_data['metaphoric_transforms']
                }
            )
            self.word_transforms[word] = word_transform


def example_usage():
    """Example usage of UPG Homophonic Metaphoric Dictionary"""
    print("=" * 70)
    print("UPG Homophonic and Metaphoric Transform Dictionary")
    print("Universal Prime Graph Protocol φ.1")
    print("=" * 70)
    print()
    
    # Create dictionary
    dictionary = UPGHomophonicMetaphoricDictionary()
    
    # Test words
    test_words = ['buy', 'vote', 'trust', 'click', 'now']
    
    print("Word Transform Analysis:")
    print()
    
    for word in test_words:
        transform = dictionary.get_transforms(word)
        if transform:
            print(f"Word: {word}")
            print(f"  Prime Mapping: {transform.prime_mapping}")
            print(f"  Consciousness Level: {transform.consciousness_level}")
            print(f"  Homophonic Variants: {transform.homophonic_variants[:5]}")
            print(f"  Metaphoric Transforms: {transform.metaphoric_transforms[:5]}")
            print()
    
    # Network analysis
    print("Transform Network Analysis:")
    print()
    network = dictionary.analyze_transform_network(test_words)
    print(f"Transform Density: {network['transform_density']:.2f}")
    print(f"Network Connections: {len(network['network_connections'])}")
    if network['network_connections']:
        for conn in network['network_connections'][:3]:
            print(f"  {conn['word1']} <-> {conn['word2']}: {len(conn['shared_transforms'])} shared")
    print()
    
    print("=" * 70)
    print("Dictionary complete!")
    print("=" * 70)
    
    return dictionary


if __name__ == "__main__":
    dictionary = example_usage()

