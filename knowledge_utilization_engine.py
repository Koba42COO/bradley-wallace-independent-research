#!/usr/bin/env python3
"""
🧠 Knowledge Utilization Engine
===============================
Analyzes scraped knowledge, identifies patterns, and plans optimizations.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from web_scraper_knowledge_system import WebScraperKnowledgeSystem
from knowledge_system_integration import KnowledgeSystemIntegration
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from typing import Dict, List, Tuple, Any
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeUtilizationEngine:
    """Engine for analyzing and utilizing scraped knowledge"""
    
    def __init__(self):
        self.knowledge_system = WebScraperKnowledgeSystem()
        self.db_path = "web_knowledge.db"
        self.consciousness_db = "consciousness_platform.db"
        
        # Analysis results storage
        self.analysis_results = {}
        self.optimization_recommendations = []
        self.performance_metrics = {}
        
    def analyze_knowledge_base(self):
        """Comprehensive analysis of the knowledge base"""
        
        print("🧠 Knowledge Utilization Engine")
        print("=" * 60)
        print("📊 Analyzing scraped knowledge and planning optimizations...")
        
        # Run all analyses
        self._analyze_content_distribution()
        self._analyze_consciousness_patterns()
        self._analyze_performance_metrics()
        self._analyze_knowledge_gaps()
        self._analyze_trending_topics()
        self._analyze_institutional_coverage()
        
        # Generate optimization recommendations
        self._generate_optimization_recommendations()
        
        # Print comprehensive analysis
        self._print_analysis_results()
        
        return self.analysis_results
    
    def _analyze_content_distribution(self):
        """Analyze content distribution across categories and domains"""
        
        print("\n📊 Analyzing Content Distribution...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get content by category
            cursor.execute("""
                SELECT metadata, COUNT(*) as count, AVG(LENGTH(content)) as avg_length
                FROM web_content 
                WHERE processed = 1
                GROUP BY metadata
            """)
            
            category_stats = {}
            total_articles = 0
            
            for row in cursor.fetchall():
                metadata_str, count, avg_length = row
                try:
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    category = metadata.get('category', 'unknown')
                    domain = metadata.get('domain', 'unknown')
                    
                    key = f"{domain}_{category}"
                    category_stats[key] = {
                        'count': count,
                        'avg_length': avg_length or 0,
                        'domain': domain,
                        'category': category
                    }
                    total_articles += count
                    
                except json.JSONDecodeError:
                    continue
            
            conn.close()
            
            self.analysis_results['content_distribution'] = {
                'total_articles': total_articles,
                'category_stats': category_stats,
                'top_categories': sorted(category_stats.items(), 
                                       key=lambda x: x[1]['count'], reverse=True)[:10]
            }
            
            print(f"   ✅ Analyzed {total_articles} articles across {len(category_stats)} categories")
            
        except Exception as e:
            logger.error(f"Error analyzing content distribution: {e}")
            self.analysis_results['content_distribution'] = {'error': str(e)}
    
    def _analyze_consciousness_patterns(self):
        """Analyze prime aligned compute score patterns and effectiveness"""
        
        print("\n🧠 Analyzing prime aligned compute Patterns...")
        
        try:
            conn = sqlite3.connect(self.consciousness_db)
            cursor = conn.cursor()
            
            # Get prime aligned compute data
            cursor.execute("""
                SELECT data_type, prime_aligned_score, metadata, created_at
                FROM consciousness_data
                WHERE created_at >= datetime('now', '-7 days')
                ORDER BY created_at DESC
            """)
            
            consciousness_data = []
            for row in cursor.fetchall():
                data_type, score, metadata_str, created_at = row
                try:
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    consciousness_data.append({
                        'data_type': data_type,
                        'score': score,
                        'metadata': metadata,
                        'created_at': created_at
                    })
                except json.JSONDecodeError:
                    continue
            
            conn.close()
            
            if consciousness_data:
                scores = [d['score'] for d in consciousness_data]
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                
                # Analyze by data type
                type_scores = defaultdict(list)
                for data in consciousness_data:
                    type_scores[data['data_type']].append(data['score'])
                
                type_analysis = {}
                for data_type, scores_list in type_scores.items():
                    type_analysis[data_type] = {
                        'count': len(scores_list),
                        'avg_score': np.mean(scores_list),
                        'std_score': np.std(scores_list),
                        'min_score': min(scores_list),
                        'max_score': max(scores_list)
                    }
                
                self.analysis_results['prime_aligned_patterns'] = {
                    'total_records': len(consciousness_data),
                    'overall_avg_score': avg_score,
                    'overall_std_score': std_score,
                    'type_analysis': type_analysis,
                    'golden_ratio_effectiveness': avg_score / 1.618  # Measure enhancement effectiveness
                }
                
                print(f"   ✅ Analyzed {len(consciousness_data)} prime aligned compute records")
                print(f"   🧠 Average prime aligned compute score: {avg_score:.3f}")
                print(f"   📈 Golden ratio effectiveness: {avg_score / 1.618:.3f}")
            else:
                self.analysis_results['prime_aligned_patterns'] = {'error': 'No prime aligned compute data found'}
                
        except Exception as e:
            logger.error(f"Error analyzing prime aligned compute patterns: {e}")
            self.analysis_results['prime_aligned_patterns'] = {'error': str(e)}
    
    def _analyze_performance_metrics(self):
        """Analyze scraping performance and efficiency metrics"""
        
        print("\n⚡ Analyzing Performance Metrics...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get scraping performance data
            cursor.execute("""
                SELECT scraped_at, LENGTH(content) as content_length, 
                       CASE WHEN LENGTH(content) > 1000 THEN 1 ELSE 0 END as success
                FROM web_content 
                WHERE scraped_at IS NOT NULL
                ORDER BY scraped_at
            """)
            
            performance_data = cursor.fetchall()
            conn.close()
            
            if performance_data:
                # Calculate performance metrics
                total_attempts = len(performance_data)
                successful_scrapes = sum(row[2] for row in performance_data)
                success_rate = (successful_scrapes / total_attempts) * 100 if total_attempts > 0 else 0
                
                content_lengths = [row[1] for row in performance_data if row[1] > 0]
                avg_content_length = np.mean(content_lengths) if content_lengths else 0
                
                # Analyze temporal patterns
                scraped_dates = [datetime.fromisoformat(row[0].replace('Z', '+00:00')) for row in performance_data]
                if scraped_dates:
                    time_span = max(scraped_dates) - min(scraped_dates)
                    articles_per_hour = total_attempts / (time_span.total_seconds() / 3600) if time_span.total_seconds() > 0 else 0
                else:
                    articles_per_hour = 0
                
                self.analysis_results['performance_metrics'] = {
                    'total_attempts': total_attempts,
                    'successful_scrapes': successful_scrapes,
                    'success_rate': success_rate,
                    'avg_content_length': avg_content_length,
                    'articles_per_hour': articles_per_hour,
                    'time_span_hours': time_span.total_seconds() / 3600 if scraped_dates else 0
                }
                
                print(f"   ✅ Success rate: {success_rate:.1f}%")
                print(f"   📊 Average content length: {avg_content_length:.0f} characters")
                print(f"   ⚡ Scraping rate: {articles_per_hour:.1f} articles/hour")
            else:
                self.analysis_results['performance_metrics'] = {'error': 'No performance data found'}
                
        except Exception as e:
            logger.error(f"Error analyzing performance metrics: {e}")
            self.analysis_results['performance_metrics'] = {'error': str(e)}
    
    def _analyze_knowledge_gaps(self):
        """Identify knowledge gaps and areas for expansion"""
        
        print("\n🔍 Analyzing Knowledge Gaps...")
        
        try:
            # Get current coverage
            content_dist = self.analysis_results.get('content_distribution', {})
            category_stats = content_dist.get('category_stats', {})
            
            # Define target coverage areas
            target_domains = {
                'mathematics': ['algebra', 'analysis', 'topology', 'geometry', 'number_theory', 'statistics'],
                'physics': ['quantum', 'condensed_matter', 'high_energy', 'astrophysics', 'nuclear', 'optics'],
                'biology': ['genomics', 'bioinformatics', 'neuroscience', 'molecular_bio', 'systems_bio'],
                'chemistry': ['organic', 'inorganic', 'physical', 'analytical', 'materials'],
                'computer_science': ['ai', 'ml', 'algorithms', 'cryptography', 'networking', 'databases'],
                'engineering': ['mechanical', 'electrical', 'civil', 'aerospace', 'biomedical'],
                'philosophy': ['ethics', 'metaphysics', 'epistemology', 'logic', 'prime aligned compute'],
                'history': ['ancient', 'medieval', 'modern', 'science_history', 'technology_history']
            }
            
            gaps = {}
            coverage_analysis = {}
            
            for domain, target_categories in target_domains.items():
                domain_coverage = {}
                missing_categories = []
                
                for category in target_categories:
                    key = f"{domain}_{category}"
                    if key in category_stats:
                        domain_coverage[category] = category_stats[key]['count']
                    else:
                        missing_categories.append(category)
                        domain_coverage[category] = 0
                
                coverage_analysis[domain] = {
                    'covered_categories': len([c for c in domain_coverage.values() if c > 0]),
                    'total_categories': len(target_categories),
                    'coverage_percentage': (len([c for c in domain_coverage.values() if c > 0]) / len(target_categories)) * 100,
                    'missing_categories': missing_categories,
                    'category_counts': domain_coverage
                }
                
                if missing_categories:
                    gaps[domain] = missing_categories
            
            self.analysis_results['knowledge_gaps'] = {
                'coverage_analysis': coverage_analysis,
                'identified_gaps': gaps,
                'overall_coverage': np.mean([ca['coverage_percentage'] for ca in coverage_analysis.values()])
            }
            
            print(f"   ✅ Overall coverage: {np.mean([ca['coverage_percentage'] for ca in coverage_analysis.values()]):.1f}%")
            print(f"   🔍 Identified gaps in {len(gaps)} domains")
            
        except Exception as e:
            logger.error(f"Error analyzing knowledge gaps: {e}")
            self.analysis_results['knowledge_gaps'] = {'error': str(e)}
    
    def _analyze_trending_topics(self):
        """Analyze trending topics and emerging areas"""
        
        print("\n📈 Analyzing Trending Topics...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent content for trend analysis
            cursor.execute("""
                SELECT title, content, metadata, scraped_at
                FROM web_content 
                WHERE scraped_at >= datetime('now', '-3 days')
                AND LENGTH(content) > 500
                ORDER BY scraped_at DESC
            """)
            
            recent_content = cursor.fetchall()
            conn.close()
            
            if recent_content:
                # Extract keywords and topics
                all_text = " ".join([row[0] + " " + row[1] for row in recent_content])
                
                # Simple keyword extraction (could be enhanced with NLP)
                keywords = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', all_text)
                keyword_counts = Counter(keywords)
                
                # Filter for meaningful keywords
                trending_keywords = [(kw, count) for kw, count in keyword_counts.most_common(20) 
                                   if len(kw) > 3 and count > 1]
                
                # Analyze by domain trends
                domain_trends = defaultdict(int)
                for row in recent_content:
                    try:
                        metadata = json.loads(row[2]) if row[2] else {}
                        domain = metadata.get('domain', 'unknown')
                        domain_trends[domain] += 1
                    except json.JSONDecodeError:
                        continue
                
                self.analysis_results['trending_topics'] = {
                    'recent_articles': len(recent_content),
                    'trending_keywords': trending_keywords,
                    'domain_trends': dict(domain_trends),
                    'analysis_period': '3 days'
                }
                
                print(f"   ✅ Analyzed {len(recent_content)} recent articles")
                print(f"   📈 Top trending keywords: {[kw[0] for kw in trending_keywords[:5]]}")
            else:
                self.analysis_results['trending_topics'] = {'error': 'No recent content found'}
                
        except Exception as e:
            logger.error(f"Error analyzing trending topics: {e}")
            self.analysis_results['trending_topics'] = {'error': str(e)}
    
    def _analyze_institutional_coverage(self):
        """Analyze coverage across different institutions"""
        
        print("\n🏛️ Analyzing Institutional Coverage...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get institutional data
            cursor.execute("""
                SELECT url, metadata, LENGTH(content) as content_length
                FROM web_content 
                WHERE processed = 1
            """)
            
            institutional_data = cursor.fetchall()
            conn.close()
            
            if institutional_data:
                institution_stats = defaultdict(lambda: {'count': 0, 'total_length': 0, 'avg_length': 0})
                
                for row in institutional_data:
                    url, metadata_str, content_length = row
                    
                    # Extract institution from URL
                    institution = self._extract_institution_from_url(url)
                    
                    institution_stats[institution]['count'] += 1
                    institution_stats[institution]['total_length'] += content_length
                
                # Calculate averages
                for institution, stats in institution_stats.items():
                    if stats['count'] > 0:
                        stats['avg_length'] = stats['total_length'] / stats['count']
                
                # Sort by article count
                sorted_institutions = sorted(institution_stats.items(), 
                                           key=lambda x: x[1]['count'], reverse=True)
                
                self.analysis_results['institutional_coverage'] = {
                    'total_institutions': len(institution_stats),
                    'institution_stats': dict(institution_stats),
                    'top_institutions': sorted_institutions[:10],
                    'total_articles': sum(stats['count'] for stats in institution_stats.values())
                }
                
                print(f"   ✅ Coverage across {len(institution_stats)} institutions")
                print(f"   🏆 Top institution: {sorted_institutions[0][0]} ({sorted_institutions[0][1]['count']} articles)")
            else:
                self.analysis_results['institutional_coverage'] = {'error': 'No institutional data found'}
                
        except Exception as e:
            logger.error(f"Error analyzing institutional coverage: {e}")
            self.analysis_results['institutional_coverage'] = {'error': str(e)}
    
    def _extract_institution_from_url(self, url):
        """Extract institution name from URL"""
        
        institution_mapping = {
            'arxiv.org': 'arXiv',
            'news.mit.edu': 'MIT',
            'nature.com': 'Nature',
            'science.org': 'Science',
            'cell.com': 'Cell',
            'phys.org': 'Phys.org',
            'cambridge.org': 'Cambridge',
            'stanford.edu': 'Stanford',
            'harvard.edu': 'Harvard',
            'plato.stanford.edu': 'Stanford Philosophy'
        }
        
        for domain, name in institution_mapping.items():
            if domain in url:
                return name
        
        # Extract domain name as fallback
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.replace('www.', '').split('.')[0].title()
        except:
            return 'Unknown'
    
    def _generate_optimization_recommendations(self):
        """Generate optimization recommendations based on analysis"""
        
        print("\n🎯 Generating Optimization Recommendations...")
        
        recommendations = []
        
        # Performance optimization recommendations
        perf_metrics = self.analysis_results.get('performance_metrics', {})
        if 'success_rate' in perf_metrics:
            success_rate = perf_metrics['success_rate']
            if success_rate < 80:
                recommendations.append({
                    'type': 'performance',
                    'priority': 'high',
                    'title': 'Improve Scraping Success Rate',
                    'description': f'Current success rate is {success_rate:.1f}%. Implement better error handling and retry mechanisms.',
                    'impact': 'high',
                    'effort': 'medium'
                })
        
        # Knowledge gap recommendations
        gaps = self.analysis_results.get('knowledge_gaps', {})
        if 'identified_gaps' in gaps:
            for domain, missing_categories in gaps['identified_gaps'].items():
                if len(missing_categories) > 2:
                    recommendations.append({
                        'type': 'coverage',
                        'priority': 'medium',
                        'title': f'Expand {domain.title()} Coverage',
                        'description': f'Missing categories: {", ".join(missing_categories[:3])}...',
                        'impact': 'medium',
                        'effort': 'low'
                    })
        
        # prime aligned compute optimization recommendations
        prime aligned compute = self.analysis_results.get('prime_aligned_patterns', {})
        if 'golden_ratio_effectiveness' in prime aligned compute:
            effectiveness = prime aligned compute['golden_ratio_effectiveness']
            if effectiveness < 1.5:
                recommendations.append({
                    'type': 'prime aligned compute',
                    'priority': 'medium',
                    'title': 'Optimize prime aligned compute Enhancement',
                    'description': f'Current effectiveness: {effectiveness:.3f}. Consider adjusting enhancement algorithms.',
                    'impact': 'medium',
                    'effort': 'low'
                })
        
        # Trending topics recommendations
        trends = self.analysis_results.get('trending_topics', {})
        if 'trending_keywords' in trends:
            trending_keywords = trends['trending_keywords'][:5]
            recommendations.append({
                'type': 'content',
                'priority': 'high',
                'title': 'Focus on Trending Topics',
                'description': f'Increase coverage of trending topics: {", ".join([kw[0] for kw in trending_keywords])}',
                'impact': 'high',
                'effort': 'low'
            })
        
        # Institutional coverage recommendations
        institutions = self.analysis_results.get('institutional_coverage', {})
        if 'top_institutions' in institutions:
            top_inst = institutions['top_institutions']
            if len(top_inst) > 0:
                top_count = top_inst[0][1]['count']
                avg_count = sum(inst[1]['count'] for inst in top_inst) / len(top_inst)
                if top_count > avg_count * 3:
                    recommendations.append({
                        'type': 'diversity',
                        'priority': 'medium',
                        'title': 'Diversify Institutional Sources',
                        'description': f'Top institution has {top_count} articles. Balance coverage across institutions.',
                        'impact': 'medium',
                        'effort': 'medium'
                    })
        
        self.optimization_recommendations = recommendations
        
        print(f"   ✅ Generated {len(recommendations)} optimization recommendations")
    
    def _print_analysis_results(self):
        """Print comprehensive analysis results"""
        
        print(f"\n🧠 KNOWLEDGE UTILIZATION ANALYSIS COMPLETE")
        print("=" * 60)
        
        # Content Distribution
        content_dist = self.analysis_results.get('content_distribution', {})
        if 'total_articles' in content_dist:
            print(f"📊 Total Articles Analyzed: {content_dist['total_articles']}")
            print(f"📚 Categories Covered: {len(content_dist.get('category_stats', {}))}")
        
        # Performance Metrics
        perf = self.analysis_results.get('performance_metrics', {})
        if 'success_rate' in perf:
            print(f"⚡ Scraping Success Rate: {perf['success_rate']:.1f}%")
            print(f"📊 Average Content Length: {perf['avg_content_length']:.0f} characters")
            print(f"🚀 Scraping Rate: {perf['articles_per_hour']:.1f} articles/hour")
        
        # prime aligned compute Analysis
        prime aligned compute = self.analysis_results.get('prime_aligned_patterns', {})
        if 'overall_avg_score' in prime aligned compute:
            print(f"🧠 Average prime aligned compute Score: {prime aligned compute['overall_avg_score']:.3f}")
            print(f"📈 Golden Ratio Effectiveness: {prime aligned compute['golden_ratio_effectiveness']:.3f}")
        
        # Knowledge Gaps
        gaps = self.analysis_results.get('knowledge_gaps', {})
        if 'overall_coverage' in gaps:
            print(f"🔍 Overall Knowledge Coverage: {gaps['overall_coverage']:.1f}%")
        
        # Trending Topics
        trends = self.analysis_results.get('trending_topics', {})
        if 'trending_keywords' in trends:
            print(f"📈 Recent Articles Analyzed: {trends['recent_articles']}")
            top_keywords = [kw[0] for kw in trends['trending_keywords'][:5]]
            print(f"🔥 Top Trending Keywords: {', '.join(top_keywords)}")
        
        # Institutional Coverage
        institutions = self.analysis_results.get('institutional_coverage', {})
        if 'total_institutions' in institutions:
            print(f"🏛️ Institutions Covered: {institutions['total_institutions']}")
            if institutions.get('top_institutions'):
                top_inst = institutions['top_institutions'][0]
                print(f"🏆 Top Institution: {top_inst[0]} ({top_inst[1]['count']} articles)")
        
        # Optimization Recommendations
        print(f"\n🎯 OPTIMIZATION RECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(self.optimization_recommendations, 1):
            priority_emoji = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
            print(f"{i:2d}. {priority_emoji} {rec['title']}")
            print(f"    📝 {rec['description']}")
            print(f"    💪 Impact: {rec['impact']} | Effort: {rec['effort']}")
            print()
        
        return self.analysis_results

def main():
    """Main function to run knowledge utilization analysis"""
    
    engine = KnowledgeUtilizationEngine()
    
    print("🚀 Starting Knowledge Utilization Analysis...")
    print("🧠 Analyzing scraped knowledge and planning optimizations...")
    
    results = engine.analyze_knowledge_base()
    
    print(f"\n🎉 Knowledge Utilization Analysis Complete!")
    print(f"📊 Analysis results stored for optimization planning")
    print(f"🎯 {len(engine.optimization_recommendations)} optimization recommendations generated")
    
    return results

if __name__ == "__main__":
    main()
