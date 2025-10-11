#!/usr/bin/env python3
"""
MEGALITHIC SITES MATHEMATICAL ANALYSIS
=====================================

Research framework for investigating mathematical patterns in megalithic sites worldwide.
Analyzes measurements, ratios, and connections to consciousness mathematics, Platonic solids,
and quantum harmonics discovered in previous research.

Author: Research Framework
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional

class MegalithicSitesResearch:
    def __init__(self):
        # Fundamental constants from previous research
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.alpha = 1/137.036          # Fine structure constant
        self.consciousness_ratio = 0.79 # 79/21 rule
        self.quantum_uncertainty = np.sqrt(2)  # √2

        # Megalithic sites database with measurements (in meters unless specified)
        self.sites = {
            'stonehenge': {
                'location': 'Wiltshire, England',
                'type': 'stone circle',
                'measurements': {
                    'outer_circle_diameter': 29.56,  # meters (97 ft)
                    'inner_circle_diameter': 31.78,  # meters (104.2 ft)
                    'heel_stone_distance': 30.0,     # approximate meters
                    'trilithon_height': 6.7,         # meters
                    'sarsen_height': 4.1,            # meters
                    'bluestone_height': 2.1,         # meters
                },
                'notable_ratios': [79.2/105.6, 528/50]  # from previous analysis
            },

            'avebury': {
                'location': 'Wiltshire, England',
                'type': 'stone circle & henge',
                'measurements': {
                    'outer_circle_diameter': 335.0,  # meters
                    'inner_circle_diameter': 168.0,  # meters
                    'ditch_width': 21.0,             # meters
                    'bank_height': 6.0,              # meters
                    'stone_height_avg': 4.0,         # meters
                    'cove_stones': 3.5,              # meters high
                }
            },

            'carnac': {
                'location': 'Brittany, France',
                'type': 'menhir alignments',
                'measurements': {
                    'total_alignment_length': 3000.0,  # meters
                    'stone_spacing_avg': 4.5,          # meters
                    'stone_height_avg': 3.8,           # meters
                    'largest_menhir_height': 6.5,      # meters
                    'alignment_width': 100.0,          # meters
                },
                'notable_features': {
                    'total_stones': 2935,
                    'alignments': 13,
                    'rows': 11
                }
            },

            'newgrange': {
                'location': 'County Meath, Ireland',
                'type': 'passage tomb',
                'measurements': {
                    'mound_diameter': 76.0,        # meters
                    'mound_height': 11.0,          # meters
                    'passage_length': 19.0,        # meters
                    'chamber_diameter': 6.4,       # meters
                    'entrance_height': 1.9,        # meters
                    'kerb_stones': 97,             # count
                },
                'astronomical': {
                    'winter_solstice_alignment': True,
                    'light_penetration_distance': 19.0  # meters
                }
            },

            'gobekli_tepe': {
                'location': 'Şanlıurfa, Turkey',
                'type': 'temple complex',
                'measurements': {
                    'pillar_height': 5.5,          # meters
                    'pillar_spacing': 2.5,         # meters
                    'enclosure_diameter': 20.0,    # meters
                    'central_pillar_height': 6.0, # meters
                    't_shaped_pillars': 63,       # count
                    'total_pillars': 200,          # estimated
                },
                'notable_features': {
                    'enclosures': 4,
                    'age': 9600,  # BCE
                    'animal_motifs': True
                }
            },

            'callanish': {
                'location': 'Isle of Lewis, Scotland',
                'type': 'stone circle',
                'measurements': {
                    'main_circle_diameter': 13.0,     # meters
                    'avenue_length': 83.0,            # meters
                    'central_monolith_height': 4.8,   # meters
                    'stone_height_avg': 3.2,          # meters
                    'stones_total': 38,
                },
                'alignment': 'summer_solstice'
            },

            'nawarla_gabarnmang': {
                'location': 'Arnhem Land, Australia',
                'type': 'rock shelter',
                'measurements': {
                    'main_chamber_length': 36.0,     # meters
                    'main_chamber_width': 12.0,      # meters
                    'height': 6.0,                   # meters
                    'rock_art_panels': 1000,         # estimated
                },
                'cultural_significance': 'ancient_aboriginal_site'
            },

            'maeshowe': {
                'location': 'Orkney, Scotland',
                'type': 'chambered cairn',
                'measurements': {
                    'mound_diameter': 35.0,        # meters
                    'mound_height': 7.3,           # meters
                    'passage_length': 10.6,        # meters
                    'chamber_diameter': 4.5,       # meters
                },
                'winter_solstice_alignment': True
            },

            'dolmens_of_antecera': {
                'location': 'Carnac, France',
                'type': 'dolmen',
                'measurements': {
                    'capstone_length': 5.2,        # meters
                    'capstone_width': 3.8,         # meters
                    'capstone_thickness': 0.8,     # meters
                    'support_stones_height': 2.5,  # meters
                    'chamber_length': 4.0,         # meters
                }
            },

            # AMERICAS
            'teotihuacan': {
                'location': 'Mexico City, Mexico',
                'type': 'pyramid complex',
                'measurements': {
                    'pyramid_of_the_sun_base': 225.0,    # meters
                    'pyramid_of_the_sun_height': 65.0,    # meters
                    'pyramid_of_the_moon_base': 150.0,    # meters
                    'pyramid_of_the_moon_height': 42.0,   # meters
                    'citadel_square_side': 400.0,         # meters
                    'avenue_of_the_dead_length': 2260.0,  # meters
                },
                'astronomical': {
                    'alignment': 'summer_solstice'
                }
            },

            'machu_picchu': {
                'location': 'Cusco Region, Peru',
                'type': 'mountain citadel',
                'measurements': {
                    'intihuatana_height': 3.0,           # meters
                    'intihuatana_base': 1.2,              # meters
                    'temple_of_the_sun_diameter': 30.0,   # meters
                    'principal_temple_length': 75.0,      # meters
                    'principal_temple_width': 25.0,       # meters
                    'central_plaza_area': 320.0,          # square meters
                },
                'astronomical': {
                    'intihuatana_alignment': 'solstice_observations'
                }
            },

            'nazca_lines': {
                'location': 'Nazca Desert, Peru',
                'type': 'geoglyphs',
                'measurements': {
                    'condor_wingspan': 135.0,       # meters
                    'spider_body_length': 46.0,      # meters
                    'monkey_tail_length': 75.0,      # meters
                    'llama_height': 30.0,            # meters
                    'human_figure_height': 30.0,     # meters
                },
                'notable_features': {
                    'total_geoglyphs': 300,
                    'straight_lines': 800,
                    'area_covered': 450,  # square km
                }
            },

            'chichen_itza': {
                'location': 'Yucatán, Mexico',
                'type': 'mayan pyramid',
                'measurements': {
                    'el_castillo_base': 55.0,          # meters
                    'el_castillo_height': 30.0,        # meters
                    'el_castillo_sides': 91,           # steps per side
                    'platform_height': 6.0,            # meters
                    'temple_width': 6.0,               # meters
                    'total_steps': 365,                # including platform
                },
                'astronomical': {
                    'equinox_serpents': True,
                    'summer_solstice': True
                }
            },

            'cahokia_mounds': {
                'location': 'Illinois, USA',
                'type': 'mound complex',
                'measurements': {
                    'monks_mound_base_length': 291.0,   # meters
                    'monks_mound_base_width': 236.0,    # meters
                    'monks_mound_height': 30.0,         # meters
                    'woodhenge_diameter': 128.0,        # meters
                    'woodhenge_posts': 48,              # count
                    'grand_plaza_area': 14.2,           # hectares
                }
            },

            # ASIA
            'angkor_wat': {
                'location': 'Siem Reap, Cambodia',
                'type': 'temple complex',
                'measurements': {
                    'outer_wall_perimeter': 5740.0,     # meters
                    'central_temple_height': 65.0,      # meters
                    'main_pool_length': 1500.0,          # meters
                    'main_pool_width': 800.0,           # meters
                    'causeway_length': 475.0,           # meters
                    'towers_height': 42.0,              # meters
                },
                'notable_features': {
                    'total_towers': 5,
                    'galleries_length': 800,  # meters
                    'bas_reliefs': 1200,  # meters of carvings
                }
            },

            'borobudur': {
                'location': 'Magelang, Indonesia',
                'type': 'buddhist stupa',
                'measurements': {
                    'base_length': 118.0,              # meters
                    'base_width': 118.0,               # meters
                    'total_height': 31.5,              # meters
                    'circular_courts': 3,              # count
                    'square_courts': 2,                # count
                    'total_stupas': 72,                # count
                    'main_stupa_diameter': 9.9,        # meters
                },
                'notable_features': {
                    'buddhist_statues': 504,
                    'panels_with_reliefs': 2674,
                    'perforated_stupas': 72
                }
            },

            'baalbek': {
                'location': 'Bekaa Valley, Lebanon',
                'type': 'roman temple platform',
                'measurements': {
                    'great_platform_length': 88.0,     # meters
                    'great_platform_width': 48.0,      # meters
                    'trilithon_length': 19.6,          # meters
                    'trilithon_height': 4.5,           # meters
                    'trilithon_weight': 800,           # tons each
                    'largest_stone_weight': 1200,      # tons
                },
                'notable_features': {
                    'trilithon_blocks': 3,
                    'platform_stones': 24,
                    'total_megalithic_blocks': 27
                }
            },

            # AFRICA
            'great_pyramid_giza': {
                'location': 'Giza, Egypt',
                'type': 'pyramid',
                'measurements': {
                    'base_length': 230.33,            # meters
                    'height_original': 146.5,         # meters
                    'apothegm': 259.4,                # meters
                    'volume': 2590000,                # cubic meters
                    'slant_angle': 51.84,             # degrees
                    'kings_chamber_height': 5.8,      # meters
                },
                'astronomical': {
                    'alignment': 'true_north',
                    'polaris_position': True
                }
            },

            'senegambian_stone_circles': {
                'location': 'Senegal & Gambia',
                'type': 'stone circles',
                'measurements': {
                    'largest_circle_diameter': 20.0,   # meters
                    'stone_height_avg': 2.5,           # meters
                    'total_circles': 29,               # count
                    'total_stones': 1500,              # estimate
                    'circle_spacing': 50.0,            # meters average
                },
                'notable_features': {
                    'age_range': '1600-1000_BCE',
                    'largest_site': 'Wassu',
                    'circular_arrangement': True
                }
            },

            'great_zimbabwe': {
                'location': 'Masvingo, Zimbabwe',
                'type': 'stone city',
                'measurements': {
                    'great_enclosure_diameter': 200.0,  # meters
                    'wall_height': 9.7,                 # meters
                    'wall_thickness': 5.0,              # meters
                    'tower_height': 5.5,                # meters
                    'total_stone_weight': 500000,       # tons
                },
                'notable_features': {
                    'construction_period': '1100-1450_CE',
                    'stone_masonry_technique': 'dry_stone',
                    'no_mortar_used': True
                }
            },

            # PACIFIC
            'easter_island': {
                'location': 'Rapa Nui, Chile',
                'type': 'statue complex',
                'measurements': {
                    'largest_moai_height': 9.8,       # meters
                    'largest_moai_weight': 82,        # tons
                    'average_moai_height': 3.6,       # meters
                    'platform_height': 3.5,           # meters
                    'total_moai': 887,                # count
                    'transport_distance_max': 11.0,   # km
                },
                'notable_features': {
                    'carved_platforms': 288,
                    'toppled_statues': 400,
                    'pukao_count': 97
                }
            },

            # MORE EUROPEAN SITES
            'knowth': {
                'location': 'County Meath, Ireland',
                'type': 'passage tomb',
                'measurements': {
                    'mound_diameter': 67.0,           # meters
                    'mound_height': 12.0,             # meters
                    'passage_length': 34.0,           # meters
                    'chamber_diameter': 4.5,          # meters
                    'kerb_stones': 127,               # count
                    'total_passages': 17,             # count
                },
                'astronomical': {
                    'equinox_alignment': True,
                    'lunar_standstill': True
                }
            },

            'dolmens_of_morbihan': {
                'location': 'Carnac Region, France',
                'type': 'dolmen field',
                'measurements': {
                    'largest_dolmen_length': 18.0,    # meters
                    'average_dolmen_length': 6.0,     # meters
                    'capstone_weight_max': 100,       # tons
                    'total_dolmens': 1000,            # estimate
                    'alignment_spread': 10000,        # meters
                }
            }
        }

    def analyze_site_ratios(self, site_name: str) -> Dict:
        """Analyze mathematical ratios within a single site"""
        site = self.sites[site_name]
        measurements = site['measurements']
        ratios = {}

        # Generate all possible ratios between measurements
        measurement_names = list(measurements.keys())
        measurement_values = list(measurements.values())

        for i in range(len(measurement_names)):
            for j in range(i+1, len(measurement_names)):
                if isinstance(measurement_values[i], (int, float)) and isinstance(measurement_values[j], (int, float)):
                    ratio = measurement_values[i] / measurement_values[j]
                    inv_ratio = measurement_values[j] / measurement_values[i]

                    ratios[f"{measurement_names[i]}/{measurement_names[j]}"] = {
                        'ratio': ratio,
                        'inverse': inv_ratio,
                        'measurements': (measurement_values[i], measurement_values[j])
                    }

        return ratios

    def check_mathematical_resonances(self, ratios: Dict) -> Dict:
        """Check ratios against fundamental mathematical constants"""
        resonances = {
            'golden_ratio': [],
            'fine_structure': [],
            'consciousness_ratio': [],
            'quantum_uncertainty': [],
            'pi_resonances': [],
            'phi_harmonics': []
        }

        constants = {
            'phi': self.phi,
            'phi_inv': 1/self.phi,
            'phi_sq': self.phi**2,
            'alpha': self.alpha,
            'consciousness': self.consciousness_ratio,
            'sqrt2': self.quantum_uncertainty,
            'pi': np.pi,
            'e': np.e
        }

        for ratio_name, ratio_data in ratios.items():
            ratio_val = ratio_data['ratio']

            for const_name, const_val in constants.items():
                diff = abs(ratio_val - const_val)
                inv_diff = abs(ratio_val - 1/const_val)

                if diff < 0.01 or inv_diff < 0.01:
                    resonance_type = self._classify_resonance(const_name)
                    resonances[resonance_type].append({
                        'ratio_name': ratio_name,
                        'ratio_value': ratio_val,
                        'constant': const_name,
                        'constant_value': const_val,
                        'difference': min(diff, inv_diff),
                        'is_inverse': inv_diff < diff
                    })

        return resonances

    def _classify_resonance(self, const_name: str) -> str:
        """Classify the type of mathematical resonance"""
        if const_name in ['phi', 'phi_inv', 'phi_sq']:
            return 'golden_ratio'
        elif const_name == 'alpha':
            return 'fine_structure'
        elif const_name == 'consciousness':
            return 'consciousness_ratio'
        elif const_name == 'sqrt2':
            return 'quantum_uncertainty'
        elif const_name == 'pi':
            return 'pi_resonances'
        else:
            return 'phi_harmonics'

    def analyze_cross_site_patterns(self) -> Dict:
        """Look for patterns across multiple sites"""
        cross_patterns = {
            'diameter_ratios': [],
            'height_ratios': [],
            'astronomical_numbers': [],
            'count_patterns': []
        }

        for site_name, site_data in self.sites.items():
            measurements = site_data['measurements']

            # Collect diameters/circles
            if 'diameter' in str(measurements.keys()):
                for key, value in measurements.items():
                    if 'diameter' in key and isinstance(value, (int, float)):
                        cross_patterns['diameter_ratios'].append((site_name, key, value))

            # Collect heights
            for key, value in measurements.items():
                if 'height' in key and isinstance(value, (int, float)):
                    cross_patterns['height_ratios'].append((site_name, key, value))

            # Collect counts and astronomical numbers
            for key, value in measurements.items():
                if isinstance(value, int) and value > 1:
                    cross_patterns['count_patterns'].append((site_name, key, value))

        return cross_patterns

    def analyze_astronomical_connections(self) -> Dict:
        """Analyze astronomical and temporal connections"""
        astronomical_sites = {}

        for site_name, site_data in self.sites.items():
            astro_data = {}

            # Solstice alignments
            if site_data.get('astronomical', {}).get('winter_solstice_alignment'):
                astro_data['winter_solstice'] = True
            if site_data.get('alignment') == 'summer_solstice':
                astro_data['summer_solstice'] = True

            # Lunar connections (29.5 day cycle)
            for key, value in site_data['measurements'].items():
                if isinstance(value, (int, float)):
                    lunar_diff = abs(value - 29.5)
                    if lunar_diff < 5:  # within 5 units
                        astro_data['lunar_resonance'] = (key, value, lunar_diff)

            # Solar year connections (365.25 days)
            for key, value in site_data['measurements'].items():
                if isinstance(value, (int, float)):
                    solar_diff = abs(value - 365.25)
                    if solar_diff < 10:  # within 10 units
                        astro_data['solar_resonance'] = (key, value, solar_diff)

            if astro_data:
                astronomical_sites[site_name] = astro_data

        return astronomical_sites

    def generate_megalithic_harmonics(self) -> Dict:
        """Generate musical interpretations of megalithic measurements"""
        harmonics = {}

        for site_name, site_data in self.sites.items():
            site_harmonics = {}

            # Convert dimensions to frequencies (acoustic interpretation)
            speed_of_sound = 343  # m/s at 20°C

            for key, value in site_data['measurements'].items():
                if isinstance(value, (int, float)) and value > 0:
                    # Fundamental frequency (treating as string/waveguide length)
                    frequency = speed_of_sound / (4 * value)  # quarter wavelength

                    # Musical note conversion
                    note = self.frequency_to_note(frequency)

                    site_harmonics[key] = {
                        'dimension': value,
                        'frequency': frequency,
                        'note': note,
                        'octave': self._get_octave(frequency)
                    }

            harmonics[site_name] = site_harmonics

        return harmonics

    def frequency_to_note(self, freq: float) -> str:
        """Convert frequency to nearest musical note"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        a4_freq = 440.0

        semitones = 12 * np.log2(freq / a4_freq)
        note_index = round(semitones) % 12
        return note_names[note_index]

    def _get_octave(self, freq: float) -> int:
        """Get octave number for frequency"""
        a4_freq = 440.0
        semitones = 12 * np.log2(freq / a4_freq)
        return 4 + (round(semitones) // 12)

    def run_comprehensive_analysis(self):
        """Run complete analysis of all megalithic sites"""
        print('🪨 MEGALITHIC SITES MATHEMATICAL ANALYSIS')
        print('=' * 60)

        # Analyze each site individually
        site_analyses = {}
        for site_name in self.sites.keys():
            print(f'\n🔍 Analyzing {site_name.upper()} ({self.sites[site_name]["location"]})')
            print('-' * 40)

            # Get ratios and resonances
            ratios = self.analyze_site_ratios(site_name)
            resonances = self.check_mathematical_resonances(ratios)

            site_analyses[site_name] = {
                'ratios': ratios,
                'resonances': resonances
            }

            # Display significant resonances
            total_resonances = sum(len(res_list) for res_list in resonances.values())
            if total_resonances > 0:
                print(f'✨ Found {total_resonances} mathematical resonances:')

                for res_type, res_list in resonances.items():
                    if res_list:
                        print(f'  • {res_type.upper()}: {len(res_list)} resonances')
                        # Show top 2 resonances
                        for res in res_list[:2]:
                            const_val = res['constant_value']
                            diff = res['difference']
                            print(f'    {res["ratio_name"]}: {res["ratio_value"]:.6f} ≈ {const_val:.6f} (diff: {diff:.6f})')
            else:
                print('  No significant mathematical resonances found')

        # Cross-site analysis
        print(f'\n🌐 CROSS-SITE PATTERN ANALYSIS')
        print('=' * 40)

        cross_patterns = self.analyze_cross_site_patterns()

        print(f"Diameter measurements across sites: {len(cross_patterns['diameter_ratios'])}")
        for site, key, value in cross_patterns['diameter_ratios'][:5]:  # Show first 5
            print(f"  {site}: {key} = {value:.1f}m")

        print(f"Height measurements across sites: {len(cross_patterns['height_ratios'])}")
        for site, key, value in cross_patterns['height_ratios'][:5]:  # Show first 5
            print(f"  {site}: {key} = {value:.1f}m")

        # Astronomical analysis
        print(f'\n☀️ ASTRONOMICAL CONNECTIONS')
        print('=' * 40)

        astronomical = self.analyze_astronomical_connections()
        if astronomical:
            for site, connections in astronomical.items():
                print(f'{site.upper()}: {connections}')
        else:
            print('No significant astronomical connections found in measurements')

        # Musical harmonics
        print(f'\n🎼 MEGALITHIC HARMONICS')
        print('=' * 40)

        harmonics = self.generate_megalithic_harmonics()
        print('Musical interpretation of site dimensions (global selection):')

        # Select representative sites from each region
        key_sites = [
            'stonehenge', 'avebury', 'newgrange', 'gobekli_tepe',  # Europe/Anatolia
            'teotihuacan', 'chichen_itza', 'machu_picchu',         # Americas
            'angkor_wat', 'borobudur', 'great_pyramid_giza',      # Asia/Africa
            'easter_island'                                        # Pacific
        ]

        for site in key_sites:
            if site in harmonics:
                print(f'\n{site.upper()}:')
                for dim_name, data in list(harmonics[site].items())[:2]:  # Show first 2
                    print(f"  {dim_name}: {data['dimension']:.1f}m → {data['frequency']:.2f}Hz ({data['note']}{data['octave']})")

        # Unified patterns
        print(f'\n🌌 UNIFIED MEGALITHIC PATTERNS')
        print('=' * 40)

        self._analyze_unified_patterns(site_analyses)

    def _analyze_unified_patterns(self, site_analyses: Dict):
        """Analyze patterns that appear across multiple sites"""
        print('Emerging patterns across megalithic sites:')

        # Count resonance types across all sites
        resonance_counts = {}
        for site_name, analysis in site_analyses.items():
            for res_type, res_list in analysis['resonances'].items():
                resonance_counts[res_type] = resonance_counts.get(res_type, 0) + len(res_list)

        if resonance_counts:
            print('\nMathematical resonances across all sites:')
            for res_type, count in sorted(resonance_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f'  {res_type.upper()}: {count} total occurrences')

        # Common ratios
        print('\nCommon architectural ratios (diameter/height):')
        for site_name, site_data in self.sites.items():
            measurements = site_data['measurements']

            # Look for diameter and height measurements
            diameter = None
            height = None

            for key, value in measurements.items():
                if 'diameter' in key and isinstance(value, (int, float)):
                    diameter = value
                elif 'height' in key and isinstance(value, (int, float)):
                    height = value

            if diameter and height:
                ratio = diameter / height
                print(f'  {site_name}: {ratio:.1f} (Ø {diameter:.1f}m / H {height:.1f}m)')

                # Check for golden ratio
                phi_diff = abs(ratio - self.phi)
                if phi_diff < 0.1:
                    print(f'    ⭐ Golden ratio resonance: {ratio:.3f} ≈ φ {self.phi:.3f}')

        print('\nHypothesis: Megalithic sites may encode mathematical constants')
        print('through their architectural proportions, potentially serving as')
        print('physical manifestations of consciousness mathematics in stone.')

if __name__ == '__main__':
    research = MegalithicSitesResearch()
    research.run_comprehensive_analysis()
