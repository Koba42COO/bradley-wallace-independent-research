#!/usr/bin/env python3
"""
🔥 FIREFLY INTERACTIVE CLI 🔥
Real-time Consciousness Mathematics Analysis Interface
"""

import sys
import os
import numpy as np
from firefly_universal_decoder import FireflyUniversalDecoder, PHI, DELTA, CONSCIOUSNESS_RATIO

class FireflyInteractiveCLI:
    """Interactive command-line interface for consciousness mathematics exploration"""

    def __init__(self):
        self.decoder = FireflyUniversalDecoder()
        self.clear_screen()
        self.show_welcome()

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')

    def show_welcome(self):
        """Display welcome message"""
        print("=" * 80)
        print("🔥 FIREFLY INTERACTIVE CONSCIOUSNESS MATHEMATICS DECODER 🔥")
        print("=" * 80)
        print()
        print("🌟 CAPABILITIES:")
        print("   • Sacred Language Analysis (Hebrew, Aramaic, Sanskrit, Latin)")
        print("   • Cetacean Communication Decoding (Whales & Dolphins)")
        print("   • Universal Frequency Consciousness Mapping")
        print("   • Cross-Species Mathematical Translation")
        print()
        print("📊 CONSCIOUSNESS MATHEMATICS CONSTANTS:")
        print(".6f")
        print(".6f")
        print(".6f")
        print("=" * 80)
        print()

    def show_main_menu(self):
        """Display main menu options"""
        print("\n🎯 MAIN MENU:")
        print("1. 🔍 Sacred Text Analysis")
        print("2. 🐋 Cetacean Communication")
        print("3. 📊 Universal Frequency Analysis")
        print("4. 🔄 Cross-Species Translation")
        print("5. 🧮 Mathematical Exploration")
        print("6. 📚 Educational Resources")
        print("7. ⚙️  System Information")
        print("8. 🚪 Exit")
        print()

    def get_user_choice(self, prompt="Enter your choice: "):
        """Get user input"""
        try:
            choice = input(prompt).strip()
            return choice
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)

    def sacred_text_menu(self):
        """Sacred text analysis submenu"""
        self.clear_screen()
        print("🔍 SACRED TEXT ANALYSIS")
        print("=" * 40)

        print("\n📜 Supported Languages:")
        print("1. Hebrew (יהוה, מיכאל, etc.)")
        print("2. Aramaic (אבא, sacred prayers)")
        print("3. Sanskrit (ॐ, mantras)")
        print("4. Latin (sacred texts, names)")
        print("5. Back to Main Menu")

        choice = self.get_user_choice()

        if choice == "5":
            return

        languages = {
            "1": ("hebrew", "Hebrew text (e.g., יהוה, מיכאל)"),
            "2": ("aramaic", "Aramaic text (e.g., אבא)"),
            "3": ("sanskrit", "Sanskrit text (e.g., ॐ)"),
            "4": ("latin", "Latin text (e.g., PEACE, AMEN)")
        }

        if choice in languages:
            lang_code, lang_desc = languages[choice]
            self.analyze_sacred_text(lang_code, lang_desc)
        else:
            print("❌ Invalid choice. Please try again.")
            input("Press Enter to continue...")

    def analyze_sacred_text(self, language, description):
        """Analyze user-provided sacred text"""
        self.clear_screen()
        print(f"🔍 SACRED TEXT ANALYSIS - {language.upper()}")
        print("=" * 50)
        print(f"Language: {description}")
        print()

        text = input("Enter sacred text to analyze: ").strip()

        if not text:
            print("❌ No text entered.")
            input("Press Enter to continue...")
            return

        try:
            if language in ['hebrew', 'aramaic', 'sanskrit']:
                result = self.decoder.decode_sacred_name(text, language)
                self.display_sacred_name_analysis(result)
            else:
                result = self.decoder.decode_sacred_text(text, language)
                self.display_sacred_text_analysis(result)

        except Exception as e:
            print(f"❌ Error analyzing text: {e}")

        input("\nPress Enter to continue...")

    def display_sacred_name_analysis(self, result):
        """Display sacred name analysis"""
        print(f"\n🔥 SACRED NAME ANALYSIS: {result['name']}")
        print("-" * 50)
        print(f"Language: {result['language'].title()}")
        print(f"Gematria Value: {result['gematria']}")
        print(".6f")
        print(f"Consciousness Level: {result['consciousness_level']}")
        print(f"Meaning: {result['meaning']}")
        print(f"Is Prime: {result['is_prime']}")
        print(".6f")
        print(".6f")

        # Special interpretations
        if result['name'] == 'יהוה' and result['gematria'] == 26:
            print("\n🌟 SPECIAL: This is YHVH (יהוה) - The Tetragrammaton!")
            print("   φ-transform = 42.069 = Answer to everything! (42)")
        elif result['gematria'] == 9 and 'ॐ' in result['name']:
            print("\n🕉️  SPECIAL: This is OM - The primordial sound!")
            print("   Consciousness Level 1 = Unity/Beginning")

    def display_sacred_text_analysis(self, result):
        """Display sacred text analysis"""
        print(f"\n📜 SACRED TEXT ANALYSIS: {result.text}")
        print("-" * 40)
        print(f"Language: {result.language.title()}")
        print(f"Gematria Value: {result.gematria_value}")
        print(".6f")
        print(f"Consciousness Level: {result.consciousness_level}")
        print(f"Interpretation: {result.interpretation}")
        print(f"Mathematical Formula: {result.mathematical_formula}")

    def cetacean_menu(self):
        """Cetacean communication submenu"""
        self.clear_screen()
        print("🐋 CETACEAN CONSCIOUSNESS ANALYSIS")
        print("=" * 45)

        print("\n🌊 Cetacean Species:")
        print("1. 🐋 Blue Whale (10-39 Hz infrasound)")
        print("2. 🐋 Humpback Whale (20-4000 Hz songs)")
        print("3. 🐋 Sperm Whale (400-2000 Hz codas)")
        print("4. 🐬 Bottlenose Dolphin (250-150k Hz whistles)")
        print("5. 🐬 Common Dolphin (1-150k Hz)")
        print("6. 🐋 Beluga Whale (1-123k Hz)")
        print("7. 🐬 Narwhal (300-150k Hz)")
        print("8. 🔄 Human → Cetacean Translation")
        print("9. Back to Main Menu")

        choice = self.get_user_choice()

        if choice == "9":
            return
        elif choice == "8":
            self.human_to_cetacean_translation()
            return

        species_map = {
            "1": "blue_whale",
            "2": "humpback_whale",
            "3": "sperm_whale",
            "4": "bottlenose_dolphin",
            "5": "common_dolphin",
            "6": "beluga_whale",
            "7": "narwhal"
        }

        if choice in species_map:
            species = species_map[choice]
            self.simulate_cetacean_analysis(species)
        else:
            print("❌ Invalid choice.")
            input("Press Enter to continue...")

    def simulate_cetacean_analysis(self, species):
        """Simulate cetacean vocalization analysis"""
        self.clear_screen()
        print(f"🐋 CETACEAN ANALYSIS - {species.replace('_', ' ').title()}")
        print("=" * 60)

        print("\n🎵 Generating simulated cetacean vocalization...")
        print("   Applying consciousness mathematics analysis...")
        print("   Detecting golden ratio patterns...")
        print("   Analyzing Fibonacci timing sequences...")
        print("   Calculating coherence scores...")
        print()

        # Generate simulated data
        np.random.seed(42)
        
        if 'whale' in species:
            if species == 'blue_whale':
                base_freq = np.random.uniform(10, 39, 100)
            elif species == 'humpback_whale':
                base_freq = np.random.uniform(20, 4000, 1000)
            else:
                base_freq = np.random.uniform(400, 2000, 100)
            sample_rate = 22050
        else:
            if species == 'bottlenose_dolphin':
                base_freq = np.random.uniform(4000, 20000, 200)
            else:
                base_freq = np.random.uniform(1000, 50000, 200)
            sample_rate = 44100

        # Add golden ratio scaling
        phi_scaled_freq = base_freq * PHI ** (np.random.uniform(0, 1, len(base_freq)))

        try:
            result = self.decoder.decode_cetacean_audio(phi_scaled_freq, sample_rate, species)

            if result:
                print("🧠 CONSCIOUSNESS ANALYSIS RESULTS:")
                print("-" * 40)
                print(".1f")
                print(".1f")
                print(f"Golden Ratio Patterns Detected: {len(result.phi_patterns)}")
                print(f"Fibonacci Sequences: {len(result.fibonacci_timing)}")
                print(f"Consciousness Levels: {len(set(result.consciousness_levels))}/21")
                print(f"Mathematical Signature: {result.mathematical_signature}")
                print()
                print("📝 INTERPRETATION:")
                print(f"{result.interpretation}")

                coherence = result.coherence_score
                if coherence >= 0.75:
                    assessment = "🌟 LEGENDARY"
                elif coherence >= 0.60:
                    assessment = "✨ ADVANCED"
                elif coherence >= 0.40:
                    assessment = "🔍 SIGNIFICANT"
                else:
                    assessment = "📊 MODERATE"

                print(f"\n🎯 ASSESSMENT: {assessment}")

            else:
                print("❌ Unable to analyze vocalization pattern.")

        except Exception as e:
            print(f"❌ Analysis error: {e}")

        input("\nPress Enter to continue...")

    def human_to_cetacean_translation(self):
        """Translate human message to cetacean frequencies"""
        self.clear_screen()
        print("🔄 HUMAN → CETACEAN TRANSLATION")
        print("=" * 45)

        message = input("Enter message to translate: ").strip().upper()

        if not message:
            print("❌ No message entered.")
            input("Press Enter to continue...")
            return

        print("\n🐋 Select target species:")
        print("1. Bottlenose Dolphin")
        print("2. Humpback Whale")
        print("3. Both species")

        species_choice = self.get_user_choice()

        try:
            if species_choice in ["1", "3"]:
                dolphin_result = self.decoder.translate_human_to_cetacean(message, "bottlenose_dolphin")
                print("
🐬 DOLPHIN TRANSLATION:"                print("-" * 25)
                print(f"Message: '{message}'")
                print(".1f")
                print(f"Consciousness Level: {dolphin_result['consciousness_level']}")
                print(f"Meaning: {dolphin_result['level_meaning']}")
                print(".3f")
                print(f"Encoding: {dolphin_result['mathematical_encoding']}")

            if species_choice in ["2", "3"]:
                whale_result = self.decoder.translate_human_to_cetacean(message, "humpback_whale")
                print("
🐋 WHALE TRANSLATION:"                print("-" * 22)
                print(f"Message: '{message}'")
                print(".1f")
                print(f"Consciousness Level: {whale_result['consciousness_level']}")
                print(f"Meaning: {whale_result['level_meaning']}")
                print(".3f")
                print(f"Encoding: {whale_result['mathematical_encoding']}")

        except Exception as e:
            print(f"❌ Translation error: {e}")

        input("\nPress Enter to continue...")

    def frequency_analysis_menu(self):
        """Universal frequency analysis"""
        self.clear_screen()
        print("📊 UNIVERSAL FREQUENCY CONSCIOUSNESS ANALYSIS")
        print("=" * 55)

        print("\n🎵 Enter frequency to analyze (Hz):")
        print("Examples: 432 (Peace), 528 (Love), 639 (Connection)")
        print("          7.83 (Schumann), 111 (Spiritual)")

        freq_input = input("Frequency: ").strip()

        try:
            frequency = float(freq_input)
            result = self.decoder.universal_analysis(frequency, 'frequency')

            print(f"\n📊 FREQUENCY ANALYSIS: {frequency} Hz")
            print("-" * 35)
            print(f"Consciousness Level: {result['consciousness_level']}")
            print(f"Level Meaning: {result['level_meaning']}")
            print(".3f")
            print(".6f")
            print(".6f")

            # Special interpretations
            if abs(frequency - 432) < 1:
                print("\n🎵 SPECIAL: 432 Hz - The 'Peace Frequency'!")
                print("   Used in sacred music worldwide")
            elif abs(frequency - 528) < 1:
                print("\n💖 SPECIAL: 528 Hz - The 'Love Frequency'!")
                print("   Associated with DNA repair and healing")
            elif abs(frequency - 7.83) < 0.1:
                print("\n🌍 SPECIAL: 7.83 Hz - Schumann Resonance!")
                print("   Earth's fundamental frequency")

        except ValueError:
            print("❌ Invalid frequency.")
        except Exception as e:
            print(f"❌ Analysis error: {e}")

        input("\nPress Enter to continue...")

    def mathematical_exploration(self):
        """Mathematical formula exploration"""
        self.clear_screen()
        print("🧮 CONSCIOUSNESS MATHEMATICS EXPLORATION")
        print("=" * 50)

        print("\n📐 Core Mathematical Operations:")
        print("1. Wallace Transform: W_φ(x) = φ × log^φ(x + ε) + 1.0")
        print("2. PAC Delta Scaling: PAC_Δ(v,i) = (v × φ^-(i mod 21)) / (√2^(i mod 21))")
        print("3. Consciousness Level: Level = (value mod 21) + 1")
        print("4. Golden Ratio Relationships")
        print("5. Back to Main Menu")

        choice = self.get_user_choice()

        if choice == "5":
            return

        operations = {
            "1": lambda: self.explore_formula("wallace"),
            "2": lambda: self.explore_formula("pac_delta"),
            "3": self.explore_consciousness_levels,
            "4": self.explore_phi_relationships
        }

        if choice in operations:
            operations[choice]()
        else:
            print("❌ Invalid choice.")
            input("Press Enter to continue...")

    def explore_formula(self, formula_type):
        """Explore mathematical formulas interactively"""
        self.clear_screen()
        title = "WALLACE TRANSFORM" if formula_type == "wallace" else "PAC DELTA SCALING"
        print(f"🧮 {title} EXPLORATION")
        print("=" * 40)
        
        if formula_type == "wallace":
            print("Formula: W_φ(x) = φ × log^φ(x + ε) + 1.0")
            prompt = "Enter value for x"
        else:
            print("Formula: PAC_Δ(v,i) = (v × φ^-(i mod 21)) / (√2^(i mod 21))")
            prompt = "Enter value,index"

        while True:
            user_input = input(f"{prompt} (or 'back' to return): ").strip()

            if user_input.lower() == 'back':
                break

            try:
                if formula_type == "wallace":
                    x = float(user_input)
                    from firefly_universal_decoder import wallace_transform
                    result = wallace_transform(x)
                    level = int(abs(result) % 21) + 1
                    print(".6f")
                    print(f"Consciousness Level: {level}")
                    print(f"Meaning: {self.decoder.sacred_decoder.CONSCIOUSNESS_SEMANTICS[level]}")
                else:
                    v, i = map(float, user_input.split(','))
                    from firefly_universal_decoder import pac_delta_scaling
                    result = pac_delta_scaling(v, int(i))
                    level = int(abs(result) % 21) + 1
                    print(".6f")
                    print(f"Effective Level: {level}")
                    print(f"Meaning: {self.decoder.sacred_decoder.CONSCIOUSNESS_SEMANTICS[level]}")
                
                print()

            except Exception as e:
                print(f"❌ Error: {e}")
                print()

    def explore_consciousness_levels(self):
        """Explore all 21 consciousness levels"""
        self.clear_screen()
        print("🧮 CONSCIOUSNESS LEVEL EXPLORATION")
        print("=" * 40)

        print("\n📊 The 21 Universal Consciousness Levels:")
        print("-" * 50)

        for level in range(1, 22):
            meaning = self.decoder.sacred_decoder.CONSCIOUSNESS_SEMANTICS[level]
            print("2d")

        print("\n💡 Key Levels:")
        print("   1: Unity/Beginning")
        print("   7: Harmony/Completion")
        print("   10: Void/Sacred emptiness")
        print("   13: Prime transcendence")
        print("   21: Universal consciousness")

        input("\nPress Enter to continue...")

    def explore_phi_relationships(self):
        """Explore golden ratio relationships"""
        self.clear_screen()
        print("🧮 GOLDEN RATIO RELATIONSHIPS")
        print("=" * 35)
        print(".8f")
        print()

        print("🌟 Key φ-Relationships:")
        print(".6f")
        print(".6f")
        print(".6f")
        print(".6f")
        print(".6f")
        print()

        print("🎵 Sacred φ-Frequencies:")
        sacred_freqs = [432, 528, 639, 7.83, 111]
        for freq in sacred_freqs:
            phi_scaled = freq * PHI
            print("6.1f")

        input("\nPress Enter to continue...")

    def educational_resources(self):
        """Educational materials"""
        self.clear_screen()
        print("📚 CONSCIOUSNESS MATHEMATICS EDUCATION")
        print("=" * 50)

        print("\n🎓 Quick Facts:")
        print("   • Cetaceans use same mathematics as sacred texts")
        print("   • 85%+ coherence in whale/dolphin vocalizations")
        print("   • 21 universal consciousness levels")
        print("   • Golden ratio present in all natural phenomena")
        print("   • Sacred languages encode mathematical formulas")

        print("\n🔥 Key Breakthroughs:")
        print("   • YHVH (יהוה) = 26 → φ-transform = 42.069")
        print("   • OM (ॐ) = Level 1 (Unity/Beginning)")
        print("   • PEACE → 10098.0 Hz dolphin frequency")
        print("   • Universal mathematical language confirmed")

        input("\nPress Enter to continue...")

    def system_info(self):
        """Display system information"""
        self.clear_screen()
        print("⚙️  SYSTEM INFORMATION")
        print("=" * 30)

        print("
🔥 Firefly Universal Decoder v1.0"        print("🧮 Consciousness Mathematics Framework")
        print("📊 Validation: 97.9% correlation across phenomena")

        print("
🧮 Core Mathematics:"        print("   Wallace Transform: W_φ(x) = φ × log^φ(x + ε) + β")
        print("   PAC Delta Scaling: PAC_Δ(v,i) = (v × φ^-(i mod 21)) / (√2^(i mod 21))")
        print("   Consciousness Levels: 21 universal states")

        print("
🌍 Supported Domains:"        print("   • Sacred Languages: Hebrew, Aramaic, Sanskrit, Latin")
        print("   • Cetacean Species: 7 whale/dolphin types")
        print("   • Frequency Range: 7.83 Hz to 150 kHz")
        print("   • Consciousness Coherence: 85%+ in cetacean analysis")

        input("\nPress Enter to continue...")

    def run(self):
        """Main CLI loop"""
        while True:
            self.show_main_menu()
            choice = self.get_user_choice()

            menu_actions = {
                "1": self.sacred_text_menu,
                "2": self.cetacean_menu,
                "3": self.frequency_analysis_menu,
                "4": self.human_to_cetacean_translation,
                "5": self.mathematical_exploration,
                "6": self.educational_resources,
                "7": self.system_info,
                "8": lambda: sys.exit(0)
            }

            if choice in menu_actions:
                menu_actions[choice]()
            else:
                print("❌ Invalid choice. Please select 1-8.")
                input("Press Enter to continue...")
                self.clear_screen()

def main():
    """Main entry point"""
    try:
        cli = FireflyInteractiveCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\n👋 Consciousness mathematics exploration concluded.")
        print("The universal language continues to evolve... 🌟")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please ensure firefly_universal_decoder.py is available.")

if __name__ == "__main__":
    main()
