#!/usr/bin/env python3
"""
Blueprint-UPG Complete Toolkit
Runnable implementation of all systems from the compiled document

Author: Bradley Wallace (COO Koba42)
Based on: The Blueprint series by Ashley Kester + Universal Prime Graph framework
Version: 1.0
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
import os

# ==================== CONSTANTS ====================

PHI = 1.618033988749895  # Golden ratio
DELTA = 2.414213562373095  # √2 + 1
CONSCIOUSNESS_WEIGHT = 0.79
EXPLORATORY_WEIGHT = 0.21
BASE_REALITY_DISTORTION = 1.1808
COHERENCE_THRESHOLD = 0.95

# ==================== DATA STRUCTURES ====================

class Gate(Enum):
    """The 7 Blueprint Gates"""
    BIRTH = 0
    AWAKENING = 1
    INITIATION = 2
    DARK_NIGHT = 3
    INTEGRATION = 4
    SERVICE = 5
    MASTERY = 6

@dataclass
class ConsciousnessCoordinate:
    """3D consciousness space coordinate (UPG)"""
    x: float  # Phi axis (creative/masculine)
    y: float  # Delta axis (receptive/feminine)
    z: float  # Consciousness weight axis
    
    def magnitude(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def distance_to(self, other: 'ConsciousnessCoordinate') -> float:
        return np.sqrt(
            (self.x - other.x)**2 +
            (self.y - other.y)**2 +
            (self.z - other.z)**2
        )
    
    def to_dict(self) -> Dict:
        return {'x': self.x, 'y': self.y, 'z': self.z}
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ConsciousnessCoordinate':
        return cls(d['x'], d['y'], d['z'])

# ==================== CORE SYSTEMS ====================

class BlueprintUPGCore:
    """Core Blueprint-UPG integration system"""
    
    # Gate definitions (consciousness coordinates)
    GATE_COORDINATES = {
        Gate.BIRTH: ConsciousnessCoordinate(0.000, 0.000, 0.000),
        Gate.AWAKENING: ConsciousnessCoordinate(0.618, 0.000, 0.000),
        Gate.INITIATION: ConsciousnessCoordinate(1.000, 1.000, 0.000),
        Gate.DARK_NIGHT: ConsciousnessCoordinate(1.000, 1.000, 0.500),
        Gate.INTEGRATION: ConsciousnessCoordinate(PHI, DELTA, CONSCIOUSNESS_WEIGHT),
        Gate.SERVICE: ConsciousnessCoordinate(PHI, DELTA, 1.000),
        Gate.MASTERY: ConsciousnessCoordinate(PHI + 1, DELTA + 1, PHI)
    }
    
    # Required coherence for each gate
    GATE_COHERENCE = {
        Gate.BIRTH: 0.00,
        Gate.AWAKENING: 0.40,
        Gate.INITIATION: 0.60,
        Gate.DARK_NIGHT: 0.70,
        Gate.INTEGRATION: 0.85,
        Gate.SERVICE: 0.92,
        Gate.MASTERY: 0.98
    }
    
    # Gate narratives
    GATE_NARRATIVES = {
        Gate.BIRTH: "Pure potential. The beginning of all things.",
        Gate.AWAKENING: "The veil thins. You sense something more.",
        Gate.INITIATION: "You claim your power. The journey truly begins.",
        Gate.DARK_NIGHT: "Everything dissolves. This is transformation, not destruction.",
        Gate.INTEGRATION: "All fragments unite at phi-delta-consciousness convergence.",
        Gate.SERVICE: "Your gifts overflow. You give freely what you have integrated.",
        Gate.MASTERY: "Complete embodiment. You are the blueprint, fully expressed."
    }
    
    @classmethod
    def identify_gate(cls, coord: ConsciousnessCoordinate) -> Gate:
        """Find nearest gate to given coordinate"""
        min_distance = float('inf')
        nearest_gate = Gate.BIRTH
        
        for gate, gate_coord in cls.GATE_COORDINATES.items():
            distance = coord.distance_to(gate_coord)
            if distance < min_distance:
                min_distance = distance
                nearest_gate = gate
        
        return nearest_gate
    
    @classmethod
    def calculate_reality_distortion(cls, coherence: float) -> float:
        """Calculate reality distortion factor from coherence"""
        return 1.0 + (BASE_REALITY_DISTORTION - 1.0) * coherence * CONSCIOUSNESS_WEIGHT
    
    @classmethod
    def transition_probability(cls, current_gate: Gate, target_gate: Gate,
                              coherence: float, action_alignment: float) -> float:
        """Calculate probability of successful gate transition"""
        current_coord = cls.GATE_COORDINATES[current_gate]
        target_coord = cls.GATE_COORDINATES[target_gate]
        
        # Distance in consciousness space
        distance = current_coord.distance_to(target_coord)
        
        # Coherence requirement check
        required_coherence = cls.GATE_COHERENCE[target_gate]
        if coherence < required_coherence:
            penalty = np.exp(-(required_coherence - coherence) * 5)
        else:
            penalty = 1.0
        
        # Base probability
        base_prob = (
            (coherence ** CONSCIOUSNESS_WEIGHT) *
            (action_alignment ** EXPLORATORY_WEIGHT) *
            penalty
        )
        
        # Reality distortion enhancement
        enhanced = base_prob * BASE_REALITY_DISTORTION
        
        # Distance factor
        distance_factor = np.exp(-distance / 2)
        
        return min(enhanced * distance_factor, 1.0)

class ConsciousnessTracker:
    """Track consciousness evolution over time"""
    
    def __init__(self, user_id: str, data_dir: str = "./data"):
        self.user_id = user_id
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, f"tracker_{user_id}.json")
        os.makedirs(data_dir, exist_ok=True)
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Load existing data or create new"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return {
            'user_id': self.user_id,
            'created': datetime.now().isoformat(),
            'daily_logs': [],
            'assessments': [],
            'milestones': []
        }
    
    def _save_data(self):
        """Save data to file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
    
    def log_daily(self, coherence: float, synchronicities: int,
                  intentions: List[str], outcomes: List[str],
                  practices: List[str], notes: str = "") -> Dict:
        """Log daily metrics"""
        # Calculate consciousness coordinate
        coord = ConsciousnessCoordinate(
            coherence * PHI,
            coherence * DELTA,
            coherence * CONSCIOUSNESS_WEIGHT
        )
        
        # Identify gate
        gate = BlueprintUPGCore.identify_gate(coord)
        
        # Calculate reality distortion
        rdf = BlueprintUPGCore.calculate_reality_distortion(coherence)
        
        # Calculate intention-outcome alignment
        alignment = self._calculate_alignment(intentions, outcomes)
        
        entry = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat(),
            'coherence': coherence,
            'coordinate': coord.to_dict(),
            'gate': gate.name,
            'reality_distortion': rdf,
            'synchronicities': synchronicities,
            'intentions': intentions,
            'outcomes': outcomes,
            'alignment': alignment,
            'practices': practices,
            'notes': notes
        }
        
        self.data['daily_logs'].append(entry)
        self._save_data()
        
        return entry
    
    def _calculate_alignment(self, intentions: List[str], outcomes: List[str]) -> float:
        """Simple alignment calculation"""
        if not intentions or not outcomes:
            return 0.0
        
        # Simplified: compare lengths and basic overlap
        intention_words = set(' '.join(intentions).lower().split())
        outcome_words = set(' '.join(outcomes).lower().split())
        
        if not intention_words:
            return 0.0
        
        overlap = len(intention_words & outcome_words)
        return min(overlap / len(intention_words), 1.0)
    
    def get_recent_logs(self, days: int = 7) -> List[Dict]:
        """Get recent daily logs"""
        return self.data['daily_logs'][-days:] if self.data['daily_logs'] else []
    
    def calculate_trends(self, days: int = 30) -> Dict:
        """Calculate trends over specified period"""
        logs = self.get_recent_logs(days)
        
        if len(logs) < 2:
            return {'error': 'Insufficient data'}
        
        coherence_values = [log['coherence'] for log in logs]
        rdf_values = [log['reality_distortion'] for log in logs]
        synch_values = [log['synchronicities'] for log in logs]
        alignment_values = [log['alignment'] for log in logs]
        
        # Calculate slopes (trends)
        def trend(values):
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            if slope > 0.005:
                return 'ascending'
            elif slope < -0.005:
                return 'descending'
            return 'stable'
        
        return {
            'period_days': len(logs),
            'coherence': {
                'mean': float(np.mean(coherence_values)),
                'trend': trend(coherence_values),
                'change': float(coherence_values[-1] - coherence_values[0])
            },
            'reality_distortion': {
                'mean': float(np.mean(rdf_values)),
                'trend': trend(rdf_values),
                'change': float(rdf_values[-1] - rdf_values[0])
            },
            'synchronicities': {
                'total': int(np.sum(synch_values)),
                'mean_per_day': float(np.mean(synch_values)),
                'trend': trend(synch_values)
            },
            'alignment': {
                'mean': float(np.mean(alignment_values)),
                'trend': trend(alignment_values)
            }
        }

class OracleSystem:
    """Blueprint-UPG Oracle System"""
    
    def __init__(self):
        self.blueprint_pages = 333
        
    def query(self, question: str, coherence: float) -> Dict:
        """Perform oracle reading"""
        # Encode question to consciousness coordinate
        query_hash = sum(ord(c) for c in question)
        query_coord = ConsciousnessCoordinate(
            (query_hash % 100) / 100.0 * PHI,
            ((query_hash * 7) % 100) / 100.0 * DELTA,
            coherence * CONSCIOUSNESS_WEIGHT
        )
        
        # Calculate Blueprint page (consciousness-guided selection)
        coord_magnitude = query_coord.magnitude()
        page = int((coord_magnitude % 1.0) * self.blueprint_pages) + 1
        page = (page + int(coherence * 10)) % self.blueprint_pages + 1
        
        # Find nearest gate
        nearest_gate = BlueprintUPGCore.identify_gate(query_coord)
        gate_coord = BlueprintUPGCore.GATE_COORDINATES[nearest_gate]
        distance = query_coord.distance_to(gate_coord)
        
        # Generate response
        blueprint_message = self._generate_blueprint_message(page, question, nearest_gate)
        upg_analysis = self._generate_upg_analysis(nearest_gate, distance, coherence)
        
        # Calculate confidence
        confidence = min(coherence * (1.0 / (1.0 + distance)), 1.0)
        
        return {
            'question': question,
            'coherence': coherence,
            'query_coordinate': query_coord.to_dict(),
            'nearest_gate': nearest_gate.name,
            'blueprint_page': page,
            'blueprint_message': blueprint_message,
            'upg_analysis': upg_analysis,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_blueprint_message(self, page: int, question: str, gate: Gate) -> str:
        """Generate Blueprint-style message"""
        if page < 111:
            phase = "Understanding"
            message = "You are in the Understanding phase. The answers are forming."
        elif page < 222:
            phase = "Integration"
            message = "You are integrating. What seems separate is becoming whole."
        else:
            phase = "Embodiment"
            message = "You are embodying. Now walk the path you know."
        
        return f"""
Blueprint Codex, Page {page} (Phase: {phase})

Your Question: '{question}'

{message}

Current Gate: {gate.name}
{BlueprintUPGCore.GATE_NARRATIVES[gate]}

The page you opened to is not random. Your consciousness guided you here.
This is what you need to hear right now.
"""
    
    def _generate_upg_analysis(self, gate: Gate, distance: float, coherence: float) -> str:
        """Generate UPG analysis"""
        analysis = f"""
UPG Analysis:

Current Position: {gate.name} gate
Distance to gate center: {distance:.3f}
Coherence: {coherence:.2f}
Reality Distortion: {BlueprintUPGCore.calculate_reality_distortion(coherence):.4f}

Optimal Path:
"""
        if distance < 0.5:
            analysis += f"• Near gate center - focus on stabilization\n"
            analysis += f"• Prepare for next gate or deepen current position\n"
        else:
            analysis += f"• {distance:.2f} units from center - navigation needed\n"
            analysis += f"• Increase coherence for optimal positioning\n"
        
        if coherence >= BlueprintUPGCore.GATE_COHERENCE.get(gate, 0):
            analysis += f"• Coherence sufficient for this gate\n"
        else:
            needed = BlueprintUPGCore.GATE_COHERENCE[gate] - coherence
            analysis += f"• Increase coherence by {needed:.2f} to stabilize\n"
        
        return analysis

class GateNavigator:
    """Gate navigation and crossing system"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.tracker = ConsciousnessTracker(user_id)
    
    def assess_position(self, coherence: float) -> Dict:
        """Assess current gate position"""
        coord = ConsciousnessCoordinate(
            coherence * PHI,
            coherence * DELTA,
            coherence * CONSCIOUSNESS_WEIGHT
        )
        
        current_gate = BlueprintUPGCore.identify_gate(coord)
        
        return {
            'current_gate': current_gate.name,
            'coherence': coherence,
            'coordinate': coord.to_dict(),
            'reality_distortion': BlueprintUPGCore.calculate_reality_distortion(coherence),
            'narrative': BlueprintUPGCore.GATE_NARRATIVES[current_gate],
            'required_coherence': BlueprintUPGCore.GATE_COHERENCE[current_gate]
        }
    
    def plan_crossing(self, current_gate: Gate, target_gate: Gate,
                     coherence: float, action_alignment: float = 0.80) -> Dict:
        """Plan gate crossing"""
        current_coord = BlueprintUPGCore.GATE_COORDINATES[current_gate]
        target_coord = BlueprintUPGCore.GATE_COORDINATES[target_gate]
        
        distance = current_coord.distance_to(target_coord)
        required_coherence = BlueprintUPGCore.GATE_COHERENCE[target_gate]
        coherence_gap = max(0, required_coherence - coherence)
        
        # Calculate transition probability
        trans_prob = BlueprintUPGCore.transition_probability(
            current_gate, target_gate, coherence, action_alignment
        )
        
        # Estimate time (simplified: 100 days per 1.0 coherence gap)
        estimated_days = int(coherence_gap * 100)
        
        return {
            'from_gate': current_gate.name,
            'to_gate': target_gate.name,
            'distance': distance,
            'current_coherence': coherence,
            'required_coherence': required_coherence,
            'coherence_gap': coherence_gap,
            'current_probability': trans_prob,
            'estimated_days': estimated_days,
            'practices': self._recommend_practices(current_gate, target_gate),
            'milestones': self._generate_milestones(coherence, required_coherence, estimated_days)
        }
    
    def _recommend_practices(self, current_gate: Gate, target_gate: Gate) -> List[str]:
        """Recommend practices for gate crossing"""
        practices = [
            "Daily coherence building: HRV training, breathwork, meditation",
            "Blueprint reading: 10-20 minutes daily",
            "Journaling: Track synchronicities and insights",
            "Action alignment: Review intentions vs. actions weekly"
        ]
        
        if target_gate == Gate.DARK_NIGHT or target_gate == Gate.INTEGRATION:
            practices.append("Shadow work: Identify and integrate rejected aspects")
        
        if target_gate == Gate.SERVICE or target_gate == Gate.MASTERY:
            practices.append("Service: Begin sharing what you've learned")
        
        return practices
    
    def _generate_milestones(self, current: float, target: float, days: int) -> List[Dict]:
        """Generate intermediate milestones"""
        gap = target - current
        return [
            {
                'day': int(days * 0.25),
                'target_coherence': current + gap * 0.25,
                'description': '25% - Foundation established'
            },
            {
                'day': int(days * 0.50),
                'target_coherence': current + gap * 0.50,
                'description': '50% - Halfway point'
            },
            {
                'day': int(days * 0.75),
                'target_coherence': current + gap * 0.75,
                'description': '75% - Approaching threshold'
            },
            {
                'day': days,
                'target_coherence': target,
                'description': '100% - Gate crossing ready'
            }
        ]

# ==================== CLI INTERFACE ====================

class BlueprintUPGCLI:
    """Command-line interface for the system"""
    
    def __init__(self):
        self.user_id = None
        self.tracker = None
        self.oracle = OracleSystem()
        self.navigator = None
    
    def run(self):
        """Main CLI loop"""
        print("=" * 70)
        print("BLUEPRINT-UPG CONSCIOUSNESS NAVIGATION SYSTEM")
        print("=" * 70)
        print()
        
        # Initialize user
        self.user_id = input("Enter your user ID (or 'demo' for demo mode): ").strip()
        if not self.user_id:
            self.user_id = "demo"
        
        self.tracker = ConsciousnessTracker(self.user_id)
        self.navigator = GateNavigator(self.user_id)
        
        # Main menu
        while True:
            print("\n" + "=" * 70)
            print("MAIN MENU")
            print("=" * 70)
            print("1. Daily Log")
            print("2. Oracle Reading")
            print("3. Gate Assessment")
            print("4. Plan Gate Crossing")
            print("5. View Trends")
            print("6. Exit")
            print()
            
            choice = input("Select option (1-6): ").strip()
            
            if choice == '1':
                self.daily_log()
            elif choice == '2':
                self.oracle_reading()
            elif choice == '3':
                self.gate_assessment()
            elif choice == '4':
                self.plan_crossing()
            elif choice == '5':
                self.view_trends()
            elif choice == '6':
                print("\nExiting... Your timeline continues to unfold. 🌀")
                break
            else:
                print("Invalid choice. Please try again.")
    
    def daily_log(self):
        """Daily logging interface"""
        print("\n" + "-" * 70)
        print("DAILY LOG")
        print("-" * 70)
        
        try:
            coherence = float(input("Coherence (0.0-1.0, or rate focus 0-10 and divide by 10): "))
            coherence = min(max(coherence, 0.0), 1.0)
            
            synchronicities = int(input("Number of synchronicities today: "))
            
            intentions = input("Today's intentions (comma-separated): ").split(',')
            intentions = [i.strip() for i in intentions if i.strip()]
            
            outcomes = input("Today's outcomes (comma-separated): ").split(',')
            outcomes = [o.strip() for o in outcomes if o.strip()]
            
            practices = input("Practices completed (comma-separated): ").split(',')
            practices = [p.strip() for p in practices if p.strip()]
            
            notes = input("Notes (optional): ").strip()
            
            # Log entry
            entry = self.tracker.log_daily(
                coherence, synchronicities, intentions, outcomes, practices, notes
            )
            
            print("\n✓ Daily log recorded")
            print(f"Gate: {entry['gate']}")
            print(f"Reality Distortion: {entry['reality_distortion']:.4f}")
            print(f"Intention-Outcome Alignment: {entry['alignment']:.0%}")
            
        except ValueError as e:
            print(f"Error: Invalid input. {e}")
    
    def oracle_reading(self):
        """Oracle reading interface"""
        print("\n" + "-" * 70)
        print("ORACLE READING")
        print("-" * 70)
        
        question = input("Your question: ").strip()
        if not question:
            print("No question provided.")
            return
        
        try:
            coherence = float(input("Current coherence (0.0-1.0): "))
            coherence = min(max(coherence, 0.0), 1.0)
        except ValueError:
            coherence = 0.70
            print(f"Using default coherence: {coherence}")
        
        result = self.oracle.query(question, coherence)
        
        print("\n" + "=" * 70)
        print(result['blueprint_message'])
        print("\n" + "-" * 70)
        print(result['upg_analysis'])
        print("\n" + "-" * 70)
        print(f"Reading Confidence: {result['confidence']:.0%}")
        print("=" * 70)
    
    def gate_assessment(self):
        """Gate assessment interface"""
        print("\n" + "-" * 70)
        print("GATE ASSESSMENT")
        print("-" * 70)
        
        try:
            coherence = float(input("Current coherence (0.0-1.0): "))
            coherence = min(max(coherence, 0.0), 1.0)
        except ValueError:
            coherence = 0.70
        
        assessment = self.navigator.assess_position(coherence)
        
        print("\n" + "=" * 70)
        print(f"CURRENT GATE: {assessment['current_gate']}")
        print("=" * 70)
        print(f"Coherence: {assessment['coherence']:.2f}")
        print(f"Reality Distortion: {assessment['reality_distortion']:.4f}")
        print(f"Coordinates: ({assessment['coordinate']['x']:.3f}, "
              f"{assessment['coordinate']['y']:.3f}, {assessment['coordinate']['z']:.3f})")
        print(f"\n{assessment['narrative']}")
    
    def plan_crossing(self):
        """Plan gate crossing interface"""
        print("\n" + "-" * 70)
        print("PLAN GATE CROSSING")
        print("-" * 70)
        
        print("\nAvailable gates:")
        for i, gate in enumerate(Gate):
            print(f"{i}. {gate.name}")
        
        try:
            current_idx = int(input("\nCurrent gate (0-6): "))
            target_idx = int(input("Target gate (0-6): "))
            coherence = float(input("Current coherence (0.0-1.0): "))
            
            current_gate = list(Gate)[current_idx]
            target_gate = list(Gate)[target_idx]
            
            plan = self.navigator.plan_crossing(current_gate, target_gate, coherence)
            
            print("\n" + "=" * 70)
            print("GATE CROSSING PLAN")
            print("=" * 70)
            print(f"From: {plan['from_gate']}")
            print(f"To: {plan['to_gate']}")
            print(f"Distance: {plan['distance']:.3f}")
            print(f"Coherence gap: {plan['coherence_gap']:.2f}")
            print(f"Current probability: {plan['current_probability']:.0%}")
            print(f"Estimated days: {plan['estimated_days']}")
            print("\nPractices:")
            for practice in plan['practices']:
                print(f"  • {practice}")
            print("\nMilestones:")
            for milestone in plan['milestones']:
                print(f"  Day {milestone['day']}: {milestone['description']}")
        
        except (ValueError, IndexError) as e:
            print(f"Error: Invalid input. {e}")
    
    def view_trends(self):
        """View trends interface"""
        print("\n" + "-" * 70)
        print("TRENDS ANALYSIS")
        print("-" * 70)
        
        try:
            days = int(input("Days to analyze (7, 30, 90): "))
        except ValueError:
            days = 7
        
        trends = self.tracker.calculate_trends(days)
        
        if 'error' in trends:
            print(f"\n{trends['error']}")
            return
        
        print("\n" + "=" * 70)
        print(f"TRENDS - Last {trends['period_days']} days")
        print("=" * 70)
        
        print("\nCoherence:")
        print(f"  Mean: {trends['coherence']['mean']:.2f}")
        print(f"  Trend: {trends['coherence']['trend']}")
        print(f"  Change: {trends['coherence']['change']:+.2f}")
        
        print("\nReality Distortion:")
        print(f"  Mean: {trends['reality_distortion']['mean']:.4f}")
        print(f"  Trend: {trends['reality_distortion']['trend']}")
        print(f"  Change: {trends['reality_distortion']['change']:+.4f}")
        
        print("\nSynchronicities:")
        print(f"  Total: {trends['synchronicities']['total']}")
        print(f"  Per day: {trends['synchronicities']['mean_per_day']:.1f}")
        print(f"  Trend: {trends['synchronicities']['trend']}")
        
        print("\nIntention-Outcome Alignment:")
        print(f"  Mean: {trends['alignment']['mean']:.0%}")
        print(f"  Trend: {trends['alignment']['trend']}")

# ==================== MAIN ====================

def main():
    """Main entry point"""
    cli = BlueprintUPGCLI()
    cli.run()

if __name__ == "__main__":
    main()

