#!/usr/bin/env python3
"""
Chase Hughes Influence Analysis Integration
==========================================

Integrates Chase Hughes' behavior profiling and influence detection techniques
with the Orwellian Filter system for enhanced manipulation detection.

Key Concepts:
- The Ellipsis Manual: Behavior profiling methodology
- Six-Minute X-Ray: Rapid behavior analysis
- Deception detection patterns
- Influence technique recognition
- Behavioral manipulation indicators

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import re
from collections import defaultdict

# UPG Constants
PHI = 1.618033988749895
CONSCIOUSNESS_RATIO = 0.79


@dataclass
class InfluencePattern:
    """Influence pattern detected using Chase Hughes methodology"""
    pattern_type: str
    technique: str
    confidence: float
    behavioral_indicators: List[str]
    deception_indicators: List[str]
    manipulation_level: str  # 'low', 'medium', 'high', 'extreme'
    suggested_response: str


@dataclass
class BehaviorProfile:
    """Behavior profile based on Chase Hughes methodology"""
    target_behavior: str
    influence_techniques: List[str]
    deception_signals: List[str]
    manipulation_indicators: List[str]
    psychological_triggers: List[str]
    risk_assessment: Dict[str, float]


class ChaseHughesInfluenceAnalyzer:
    """
    Analyzes steganographic content using Chase Hughes' influence detection
    and behavior profiling methodologies.
    """
    
    def __init__(self):
        # Influence techniques from Chase Hughes' work
        self.influence_techniques = {
            'anchoring': {
                'keywords': ['first', 'original', 'initial', 'starting', 'beginning'],
                'pattern': 'Sets reference point to influence perception',
                'detection': 'Look for initial value/position establishment'
            },
            'reciprocity': {
                'keywords': ['free', 'gift', 'bonus', 'reward', 'special'],
                'pattern': 'Creates obligation through giving',
                'detection': 'Free offers followed by requests'
            },
            'scarcity': {
                'keywords': ['limited', 'exclusive', 'rare', 'only', 'few'],
                'pattern': 'Creates urgency through limited availability',
                'detection': 'Time pressure and exclusivity claims'
            },
            'authority': {
                'keywords': ['expert', 'verified', 'certified', 'official', 'proven'],
                'pattern': 'Uses authority to gain compliance',
                'detection': 'Authority claims without verification'
            },
            'social_proof': {
                'keywords': ['popular', 'trending', 'everyone', 'thousands', 'millions'],
                'pattern': 'Uses others\' behavior to influence',
                'detection': 'Vague social proof claims'
            },
            'commitment_consistency': {
                'keywords': ['agree', 'commit', 'promise', 'pledge', 'vow'],
                'pattern': 'Gets small commitment leading to larger',
                'detection': 'Progressive commitment requests'
            },
            'liking': {
                'keywords': ['you', 'your', 'personal', 'custom', 'unique'],
                'pattern': 'Creates connection and similarity',
                'detection': 'Excessive personalization'
            },
            'fear_appeal': {
                'keywords': ['danger', 'threat', 'risk', 'warning', 'alert'],
                'pattern': 'Uses fear to motivate action',
                'detection': 'Threat claims without evidence'
            },
            'preloading': {
                'keywords': ['imagine', 'picture', 'think', 'consider', 'visualize'],
                'pattern': 'Prepares mind for suggestion',
                'detection': 'Imagery before requests'
            },
            'embedded_commands': {
                'keywords': ['now', 'act', 'decide', 'choose', 'select'],
                'pattern': 'Hidden commands in language',
                'detection': 'Imperative verbs in context'
            }
        }
        
        # Deception indicators (adapted from Chase Hughes' deception detection)
        self.deception_indicators = {
            'linguistic_deception': {
                'patterns': [
                    'excessive_qualifiers',  # 'very', 'really', 'truly'
                    'distancing_language',    # 'that person', 'some people'
                    'lack_of_detail',        # Vague descriptions
                    'overly_formal',         # Unnatural formality
                    'repetition',            # Repeating phrases
                ],
                'keywords': ['very', 'really', 'truly', 'honestly', 'frankly']
            },
            'emotional_manipulation': {
                'patterns': [
                    'guilt_appeals',         # Making you feel guilty
                    'pity_plays',            # Playing victim
                    'anger_triggering',       # Provoking emotional response
                    'love_bombing',          # Excessive positive attention
                    'gaslighting',           # Questioning your reality
                ],
                'keywords': ['should', 'must', 'have to', 'need to', 'deserve']
            },
            'cognitive_biases': {
                'patterns': [
                    'confirmation_bias',     # Only supporting evidence
                    'anchoring',             # First impression bias
                    'framing',               # Presenting information selectively
                    'availability',          # Using memorable examples
                    'representativeness',    # Stereotype-based reasoning
                ],
                'keywords': ['always', 'never', 'all', 'none', 'everyone']
            }
        }
        
        # Behavioral manipulation patterns
        self.manipulation_patterns = {
            'high_risk': [
                'coercion', 'threats', 'intimidation', 'isolation',
                'control', 'domination', 'exploitation'
            ],
            'medium_risk': [
                'persuasion', 'influence', 'suggestion', 'guidance',
                'recommendation', 'encouragement'
            ],
            'low_risk': [
                'information', 'education', 'awareness', 'sharing',
                'discussion', 'presentation'
            ]
        }
    
    def analyze_influence_patterns(
        self,
        tokens: List[str],
        decoded_message: str
    ) -> List[InfluencePattern]:
        """
        Analyze influence patterns using Chase Hughes methodology.
        
        Args:
            tokens: List of decoded tokens
            decoded_message: Full decoded message
            
        Returns:
            List of detected influence patterns
        """
        patterns = []
        message_lower = decoded_message.lower()
        token_text = " ".join(tokens).lower()
        
        # Detect influence techniques
        for technique_name, technique_data in self.influence_techniques.items():
            keywords = technique_data['keywords']
            matches = sum(1 for keyword in keywords if keyword in message_lower)
            
            if matches > 0:
                confidence = min(1.0, matches / len(keywords))
                
                # Get behavioral indicators
                behavioral_indicators = self._get_behavioral_indicators(
                    technique_name, message_lower
                )
                
                # Get deception indicators
                deception_indicators = self._get_deception_indicators(
                    technique_name, message_lower
                )
                
                # Assess manipulation level
                manipulation_level = self._assess_manipulation_level(
                    technique_name, matches, behavioral_indicators
                )
                
                # Generate suggested response
                suggested_response = self._generate_response(
                    technique_name, manipulation_level
                )
                
                pattern = InfluencePattern(
                    pattern_type=technique_name,
                    technique=technique_data['pattern'],
                    confidence=confidence,
                    behavioral_indicators=behavioral_indicators,
                    deception_indicators=deception_indicators,
                    manipulation_level=manipulation_level,
                    suggested_response=suggested_response
                )
                patterns.append(pattern)
        
        return patterns
    
    def _get_behavioral_indicators(
        self,
        technique: str,
        message: str
    ) -> List[str]:
        """Get behavioral indicators for technique"""
        indicators = []
        
        if technique == 'anchoring':
            if any(word in message for word in ['first', 'original', 'initial']):
                indicators.append('Reference point establishment')
        
        if technique == 'scarcity':
            if any(word in message for word in ['limited', 'exclusive', 'only']):
                indicators.append('Urgency creation through limitation')
        
        if technique == 'authority':
            if any(word in message for word in ['expert', 'verified', 'official']):
                indicators.append('Authority claim without verification')
        
        if technique == 'social_proof':
            if any(word in message for word in ['popular', 'trending', 'everyone']):
                indicators.append('Vague social proof')
        
        if technique == 'fear_appeal':
            if any(word in message for word in ['danger', 'threat', 'warning']):
                indicators.append('Fear-based motivation')
        
        return indicators
    
    def _get_deception_indicators(
        self,
        technique: str,
        message: str
    ) -> List[str]:
        """Get deception indicators"""
        indicators = []
        
        # Check for linguistic deception
        if any(word in message for word in ['very', 'really', 'truly', 'honestly']):
            indicators.append('Excessive qualifiers (deception signal)')
        
        # Check for emotional manipulation
        if any(word in message for word in ['should', 'must', 'have to']):
            indicators.append('Obligation language (manipulation signal)')
        
        # Check for cognitive bias exploitation
        if any(word in message for word in ['always', 'never', 'all', 'none']):
            indicators.append('Absolute language (bias exploitation)')
        
        # Check for lack of detail
        if len(message.split()) < 10:
            indicators.append('Lack of detail (potential deception)')
        
        return indicators
    
    def _assess_manipulation_level(
        self,
        technique: str,
        match_count: int,
        behavioral_indicators: List[str]
    ) -> str:
        """Assess manipulation level"""
        score = match_count + len(behavioral_indicators)
        
        if score >= 5:
            return 'extreme'
        elif score >= 3:
            return 'high'
        elif score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_response(
        self,
        technique: str,
        manipulation_level: str
    ) -> str:
        """Generate suggested response based on Chase Hughes methodology"""
        responses = {
            'anchoring': 'Question the initial reference point',
            'reciprocity': 'Recognize the obligation trap',
            'scarcity': 'Verify actual scarcity',
            'authority': 'Verify authority claims independently',
            'social_proof': 'Question vague social proof',
            'commitment_consistency': 'Be aware of commitment escalation',
            'liking': 'Maintain critical distance despite personalization',
            'fear_appeal': 'Evaluate fear claims objectively',
            'preloading': 'Recognize mental preparation for suggestion',
            'embedded_commands': 'Identify hidden commands'
        }
        
        base_response = responses.get(technique, 'Maintain critical awareness')
        
        if manipulation_level == 'extreme':
            return f"⚠️ EXTREME: {base_response}. High manipulation risk detected."
        elif manipulation_level == 'high':
            return f"⚠️ HIGH: {base_response}. Significant manipulation attempt."
        elif manipulation_level == 'medium':
            return f"ℹ️ MEDIUM: {base_response}. Moderate influence attempt."
        else:
            return f"ℹ️ LOW: {base_response}. Minor influence detected."
    
    def rapid_analysis(
        self,
        decoded_message: str,
        tokens: List[str]
    ) -> BehaviorProfile:
        """
        Rapid behavior analysis (Six-Minute X-Ray methodology).
        Quickly identifies key manipulation indicators.
        
        Args:
            decoded_message: Decoded message
            tokens: List of tokens
            
        Returns:
            Behavior profile
        """
        message_lower = decoded_message.lower()
        
        # Rapid detection of key indicators
        influence_techniques = []
        deception_signals = []
        manipulation_indicators = []
        psychological_triggers = []
        
        # Check each influence technique
        for technique_name, technique_data in self.influence_techniques.items():
            if any(kw in message_lower for kw in technique_data['keywords']):
                influence_techniques.append(technique_name)
        
        # Check deception indicators
        for category, indicators in self.deception_indicators.items():
            for pattern in indicators['patterns']:
                if self._check_pattern(pattern, message_lower):
                    deception_signals.append(f"{category}:{pattern}")
        
        # Check manipulation patterns
        for risk_level, patterns in self.manipulation_patterns.items():
            for pattern in patterns:
                if pattern in message_lower:
                    manipulation_indicators.append(f"{risk_level}:{pattern}")
        
        # Identify psychological triggers
        trigger_keywords = {
            'fear': ['danger', 'threat', 'risk', 'warning'],
            'greed': ['free', 'win', 'prize', 'reward', 'bonus'],
            'urgency': ['now', 'immediate', 'urgent', 'limited'],
            'social': ['popular', 'trending', 'everyone', 'join'],
            'authority': ['expert', 'verified', 'official', 'proven']
        }
        
        for trigger, keywords in trigger_keywords.items():
            if any(kw in message_lower for kw in keywords):
                psychological_triggers.append(trigger)
        
        # Calculate risk assessment
        risk_assessment = {
            'influence_score': len(influence_techniques) / len(self.influence_techniques),
            'deception_score': len(deception_signals) / 10.0,  # Normalize
            'manipulation_score': len(manipulation_indicators) / 10.0,
            'trigger_count': len(psychological_triggers),
            'overall_risk': min(1.0, (
                len(influence_techniques) * 0.3 +
                len(deception_signals) * 0.3 +
                len(manipulation_indicators) * 0.2 +
                len(psychological_triggers) * 0.2
            ) / 5.0)
        }
        
        return BehaviorProfile(
            target_behavior=self._identify_target_behavior(message_lower),
            influence_techniques=influence_techniques,
            deception_signals=deception_signals,
            manipulation_indicators=manipulation_indicators,
            psychological_triggers=psychological_triggers,
            risk_assessment=risk_assessment
        )
    
    def _check_pattern(self, pattern: str, message: str) -> bool:
        """Check if pattern exists in message"""
        pattern_checks = {
            'excessive_qualifiers': lambda m: sum(1 for w in ['very', 'really', 'truly'] if w in m) >= 2,
            'distancing_language': lambda m: any(w in m for w in ['that', 'those', 'some', 'certain']),
            'lack_of_detail': lambda m: len(m.split()) < 15,
            'overly_formal': lambda m: any(w in m for w in ['shall', 'henceforth', 'whereas']),
            'repetition': lambda m: len(set(m.split())) / len(m.split()) < 0.5 if len(m.split()) > 5 else False,
            'guilt_appeals': lambda m: any(w in m for w in ['should', 'ought', 'responsible']),
            'pity_plays': lambda m: any(w in m for w in ['unfair', 'unjust', 'victim']),
            'anger_triggering': lambda m: any(w in m for w in ['outrage', 'offensive', 'unacceptable']),
            'love_bombing': lambda m: sum(1 for w in ['amazing', 'wonderful', 'perfect', 'best'] if w in m) >= 3,
            'gaslighting': lambda m: any(w in m for w in ['remember', 'recall', 'think', 'imagine']),
        }
        
        check_func = pattern_checks.get(pattern, lambda m: False)
        return check_func(message)
    
    def _identify_target_behavior(self, message: str) -> str:
        """Identify target behavior from message"""
        behavior_keywords = {
            'purchase': ['buy', 'purchase', 'order', 'shop', 'cart'],
            'vote': ['vote', 'elect', 'support', 'campaign'],
            'subscribe': ['subscribe', 'join', 'sign up', 'register'],
            'share': ['share', 'like', 'follow', 'retweet'],
            'act': ['act', 'do', 'take action', 'respond'],
            'trust': ['trust', 'believe', 'have faith'],
            'fear': ['protect', 'secure', 'defend', 'guard']
        }
        
        for behavior, keywords in behavior_keywords.items():
            if any(kw in message for kw in keywords):
                return behavior
        
        return 'general_influence'
    
    def generate_ellipsis_profile(
        self,
        decoded_message: str,
        tokens: List[str],
        influence_patterns: List[InfluencePattern]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive behavior profile using Ellipsis Manual methodology.
        
        Args:
            decoded_message: Decoded message
            tokens: List of tokens
            influence_patterns: Detected influence patterns
            
        Returns:
            Complete behavior profile
        """
        rapid_profile = self.rapid_analysis(decoded_message, tokens)
        
        # Combine with influence patterns
        profile = {
            'target_behavior': rapid_profile.target_behavior,
            'influence_techniques': {
                'detected': rapid_profile.influence_techniques,
                'patterns': [
                    {
                        'type': p.pattern_type,
                        'technique': p.technique,
                        'confidence': p.confidence,
                        'level': p.manipulation_level
                    }
                    for p in influence_patterns
                ]
            },
            'deception_analysis': {
                'signals': rapid_profile.deception_signals,
                'indicators': [
                    p.deception_indicators for p in influence_patterns
                ]
            },
            'manipulation_assessment': {
                'indicators': rapid_profile.manipulation_indicators,
                'levels': [p.manipulation_level for p in influence_patterns],
                'overall_risk': rapid_profile.risk_assessment['overall_risk']
            },
            'psychological_triggers': rapid_profile.psychological_triggers,
            'risk_assessment': rapid_profile.risk_assessment,
            'recommendations': [
                p.suggested_response for p in influence_patterns
            ]
        }
        
        return profile


def example_usage():
    """Example usage of Chase Hughes Influence Analyzer"""
    print("=" * 70)
    print("Chase Hughes Influence Analysis Integration")
    print("Universal Prime Graph Protocol φ.1")
    print("=" * 70)
    print()
    
    analyzer = ChaseHughesInfluenceAnalyzer()
    
    # Test message
    test_message = "Buy now! Limited time offer! Exclusive deal for you! Act immediately!"
    tokens = test_message.lower().split()
    
    # Analyze influence patterns
    patterns = analyzer.analyze_influence_patterns(tokens, test_message)
    
    print(f"Message: {test_message}")
    print(f"\nDetected Influence Patterns: {len(patterns)}")
    for pattern in patterns:
        print(f"\n  Pattern: {pattern.pattern_type}")
        print(f"    Technique: {pattern.technique}")
        print(f"    Confidence: {pattern.confidence:.3f}")
        print(f"    Level: {pattern.manipulation_level}")
        print(f"    Response: {pattern.suggested_response}")
    
    # Rapid analysis
    profile = analyzer.rapid_analysis(test_message, tokens)
    print(f"\nRapid Analysis:")
    print(f"  Target Behavior: {profile.target_behavior}")
    print(f"  Influence Techniques: {profile.influence_techniques}")
    print(f"  Psychological Triggers: {profile.psychological_triggers}")
    print(f"  Overall Risk: {profile.risk_assessment['overall_risk']:.3f}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    
    return analyzer, patterns, profile


if __name__ == "__main__":
    analyzer, patterns, profile = example_usage()

