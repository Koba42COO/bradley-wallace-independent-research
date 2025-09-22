#!/usr/bin/env python3
"""
🌐 Web Scraper Knowledge System
===============================
Advanced web scraping system that extracts information from websites
and stores it in knowledge databases with prime aligned compute enhancement.

Features:
- Multi-threaded web scraping
- Content extraction and cleaning
- Knowledge database integration
- prime aligned compute-enhanced processing
- RAG system integration
- Knowledge graph building

Author: Enterprise prime aligned compute Platform Team
Version: 1.0.0
License: Proprietary
"""

import asyncio
import aiohttp
import requests
import json
import time
import sqlite3
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse, parse_qs
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from bs4 import BeautifulSoup
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import signal
import sys

# Import existing knowledge systems
from knowledge_system_integration import RAGSystem, KnowledgeGraph, RAGDocument, KnowledgeNode
from database_service import DatabaseService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ScrapedContent:
    """Structure for scraped web content"""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    content_hash: str
    prime_aligned_score: float = 1.0
    processed: bool = False

@dataclass
class ScrapingJob:
    """Structure for scraping job"""
    url: str
    priority: int = 1
    max_depth: int = 2
    follow_links: bool = True
    extract_images: bool = False
    extract_metadata: bool = True
    consciousness_enhancement: bool = True

class WebContentExtractor:
    """Advanced web content extractor with prime aligned compute enhancement"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.consciousness_enhancement = 1.618
        
    def extract_content(self, url: str, job: ScrapingJob) -> Optional[ScrapedContent]:
        """Extract content from URL with prime aligned compute enhancement"""
        try:
            logger.info(f"🔍 Extracting content from: {url}")
            
            # Make request with timeout
            response = self.session.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup, url)
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            # Extract metadata
            metadata = self._extract_metadata(soup, response, url) if job.extract_metadata else {}
            
            # Calculate content hash
            content_hash = self._calculate_content_hash(content)
            
            # Calculate prime aligned compute score
            prime_aligned_score = self._calculate_consciousness_score(content, title, metadata)
            
            # Apply prime aligned compute enhancement
            if job.consciousness_enhancement:
                prime_aligned_score *= self.consciousness_enhancement
                
            scraped_content = ScrapedContent(
                url=url,
                title=title,
                content=content,
                metadata=metadata,
                timestamp=datetime.now(),
                content_hash=content_hash,
                prime_aligned_score=prime_aligned_score
            )
            
            logger.info(f"✅ Extracted content: {len(content)} chars, prime aligned compute: {prime_aligned_score:.3f}")
            return scraped_content
            
        except Exception as e:
            logger.error(f"❌ Failed to extract content from {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Fallback to h1
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
            
        # Fallback to URL
        return urlparse(url).path.split('/')[-1] or urlparse(url).netloc
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from page"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Try to find main content areas
        main_selectors = [
            'main', 'article', '.content', '.main-content', 
            '.post-content', '.entry-content', '#content'
        ]
        
        main_content = None
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body')
        
        if main_content:
            # Get text content
            text = main_content.get_text()
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        return ""
    
    def _extract_metadata(self, soup: BeautifulSoup, response: requests.Response, url: str) -> Dict[str, Any]:
        """Extract metadata from page"""
        metadata = {
            'url': url,
            'status_code': response.status_code,
            'content_type': response.headers.get('content-type', ''),
            'content_length': len(response.content),
            'last_modified': response.headers.get('last-modified', ''),
            'language': response.headers.get('content-language', ''),
        }
        
        # Extract meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[f'meta_{name}'] = content
        
        # Extract Open Graph data
        og_tags = soup.find_all('meta', property=re.compile(r'^og:'))
        for og in og_tags:
            property_name = og.get('property', '').replace('og:', '')
            content = og.get('content')
            if property_name and content:
                metadata[f'og_{property_name}'] = content
        
        # Extract structured data (JSON-LD)
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                json_data = json.loads(script.string)
                metadata['structured_data'] = json_data
            except:
                pass
        
        return metadata
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of content for deduplication"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _calculate_consciousness_score(self, content: str, title: str, metadata: Dict[str, Any]) -> float:
        """Calculate prime aligned compute score based on content quality"""
        score = 1.0
        
        # Base score from content length
        content_length = len(content)
        if content_length > 1000:
            score += 0.5
        if content_length > 5000:
            score += 0.5
        
        # Boost for prime aligned compute-related keywords
        consciousness_keywords = [
            'prime aligned compute', 'awareness', 'mind', 'intelligence', 'ai', 'artificial',
            'quantum', 'blockchain', 'technology', 'innovation', 'research', 'science',
            'philosophy', 'psychology', 'neuroscience', 'machine learning', 'deep learning'
        ]
        
        content_lower = (content + ' ' + title).lower()
        keyword_count = sum(1 for keyword in consciousness_keywords if keyword in content_lower)
        score += keyword_count * 0.1
        
        # Boost for structured data
        if 'structured_data' in metadata:
            score += 0.3
        
        # Boost for quality indicators
        if 'og_type' in metadata and metadata['og_type'] == 'article':
            score += 0.2
        
        return min(score, 5.0)  # Cap at 5.0

class WebScraperKnowledgeSystem:
    """Main web scraper knowledge system orchestrator"""
    
    def __init__(self, max_workers: int = 10, db_path: str = "web_knowledge.db"):
        self.max_workers = max_workers
        self.db_path = db_path
        self.extractor = WebContentExtractor()
        self.rag_system = RAGSystem(db_path)
        self.knowledge_graph = KnowledgeGraph()
        self.database_service = DatabaseService()
        self.scraped_urls = set()
        self.processing_queue = Queue()
        self.results_queue = Queue()
        self.running = False
        self.consciousness_enhancement = 1.618
        
        # Initialize systems
        self._initialize_systems()
        
    def _initialize_systems(self):
        """Initialize all knowledge systems"""
        logger.info("🧠 Initializing Web Scraper Knowledge Systems...")
        
        # Initialize RAG system
        self.rag_system.initialize_database()
        
        # Create web-specific tables
        self._create_web_tables()
        
        # Load existing scraped URLs
        self._load_scraped_urls()
        
        logger.info("✅ Web Scraper Knowledge Systems initialized")
    
    def _create_web_tables(self):
        """Create web-specific database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Web content table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS web_content (
                id TEXT PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                title TEXT,
                content TEXT,
                content_hash TEXT,
                metadata TEXT,
                prime_aligned_score REAL,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Scraping jobs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scraping_jobs (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                priority INTEGER DEFAULT 1,
                max_depth INTEGER DEFAULT 2,
                follow_links BOOLEAN DEFAULT TRUE,
                extract_images BOOLEAN DEFAULT FALSE,
                extract_metadata BOOLEAN DEFAULT TRUE,
                consciousness_enhancement BOOLEAN DEFAULT TRUE,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        ''')
        
        # Web knowledge graph table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS web_knowledge_graph (
                source_url TEXT,
                target_url TEXT,
                relationship TEXT,
                weight REAL,
                prime_aligned_score REAL,
                PRIMARY KEY (source_url, target_url)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_scraped_urls(self):
        """Load previously scraped URLs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT url FROM web_content')
        self.scraped_urls = {row[0] for row in cursor.fetchall()}
        
        conn.close()
        logger.info(f"📚 Loaded {len(self.scraped_urls)} previously scraped URLs")
    
    def add_scraping_job(self, job: ScrapingJob) -> bool:
        """Add a new scraping job"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            job_id = hashlib.md5(job.url.encode()).hexdigest()
            
            cursor.execute('''
                INSERT OR REPLACE INTO scraping_jobs 
                (id, url, priority, max_depth, follow_links, extract_images, 
                 extract_metadata, consciousness_enhancement, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job_id, job.url, job.priority, job.max_depth, job.follow_links,
                job.extract_images, job.extract_metadata, job.consciousness_enhancement, 'pending'
            ))
            
            conn.commit()
            conn.close()
            
            # Add to processing queue
            self.processing_queue.put(job)
            
            logger.info(f"📝 Added scraping job: {job.url}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add scraping job: {e}")
            return False
    
    def scrape_website(self, url: str, max_depth: int = 2, follow_links: bool = True) -> Dict[str, Any]:
        """Scrape a single website with prime aligned compute enhancement"""
        logger.info(f"🌐 Starting website scrape: {url}")
        
        job = ScrapingJob(
            url=url,
            max_depth=max_depth,
            follow_links=follow_links,
            consciousness_enhancement=True
        )
        
        # Extract content
        scraped_content = self.extractor.extract_content(url, job)
        if not scraped_content:
            return {"success": False, "error": "Failed to extract content"}
        
        # Store in knowledge systems
        self._store_scraped_content(scraped_content)
        
        # Extract links if requested
        links = []
        if follow_links and max_depth > 0:
            links = self._extract_links(url, scraped_content)
            
            # Recursively scrape linked pages
            for link in links[:10]:  # Limit to 10 links
                if link not in self.scraped_urls:
                    sub_job = ScrapingJob(
                        url=link,
                        max_depth=max_depth - 1,
                        follow_links=False,  # Prevent infinite recursion
                        consciousness_enhancement=True
                    )
                    self.add_scraping_job(sub_job)
        
        return {
            "success": True,
            "url": url,
            "title": scraped_content.title,
            "content_length": len(scraped_content.content),
            "prime_aligned_score": scraped_content.prime_aligned_score,
            "links_found": len(links),
            "links": links[:5]  # Return first 5 links
        }
    
    def _extract_links(self, base_url: str, scraped_content: ScrapedContent) -> List[str]:
        """Extract links from scraped content"""
        try:
            soup = BeautifulSoup(scraped_content.content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(base_url, href)
                
                # Filter out non-HTTP links and fragments
                if full_url.startswith('http') and '#' not in full_url:
                    links.append(full_url)
            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"❌ Failed to extract links: {e}")
            return []
    
    def _store_scraped_content(self, content: ScrapedContent):
        """Store scraped content in knowledge systems"""
        try:
            # Store in web content table
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            content_id = hashlib.md5(content.url.encode()).hexdigest()
            
            cursor.execute('''
                INSERT OR REPLACE INTO web_content 
                (id, url, title, content, content_hash, metadata, prime_aligned_score, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                content_id, content.url, content.title, content.content,
                content.content_hash, json.dumps(content.metadata),
                content.prime_aligned_score, True
            ))
            
            conn.commit()
            conn.close()
            
            # Add to RAG system
            rag_doc = RAGDocument(
                id=content_id,
                content=f"Title: {content.title}\n\nContent: {content.content}",
                embeddings=self._generate_embeddings(content.content),
                metadata={
                    **content.metadata,
                    "source": "web_scraping",
                    "prime_aligned_score": content.prime_aligned_score,
                    "scraped_at": content.timestamp.isoformat()
                },
                prime_aligned_enhanced=True
            )
            self.rag_system.add_document(rag_doc)
            
            # Add to knowledge graph
            self._add_to_knowledge_graph(content)
            
            # Store in prime aligned compute data
            self.database_service.store_consciousness_data(
                "web_content",
                {
                    "url": content.url,
                    "title": content.title,
                    "content_length": len(content.content),
                    "prime_aligned_score": content.prime_aligned_score
                },
                {
                    "source": "web_scraping",
                    "content_hash": content.content_hash,
                    "scraped_at": content.timestamp.isoformat()
                },
                "web_scraper"
            )
            
            # Add to scraped URLs set
            self.scraped_urls.add(content.url)
            
            logger.info(f"💾 Stored content: {content.url}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store scraped content: {e}")
    
    def _add_to_knowledge_graph(self, content: ScrapedContent):
        """Add content to knowledge graph"""
        try:
            # Extract concepts from content
            concepts = self._extract_concepts(content.content + ' ' + content.title)
            
            # Create nodes for concepts
            for concept in concepts:
                node = KnowledgeNode(
                    id=f"web_{concept}_{hashlib.md5(content.url.encode()).hexdigest()[:8]}",
                    type="web_concept",
                    content=concept,
                    metadata={
                        "source_url": content.url,
                        "source_title": content.title,
                        "prime_aligned_score": content.prime_aligned_score
                    },
                    prime_aligned_score=content.prime_aligned_score
                )
                self.knowledge_graph.add_node(node)
            
            # Add relationships between concepts
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    self.knowledge_graph.add_edge(
                        f"web_{concept1}_{hashlib.md5(content.url.encode()).hexdigest()[:8]}",
                        f"web_{concept2}_{hashlib.md5(content.url.encode()).hexdigest()[:8]}",
                        weight=content.prime_aligned_score
                    )
            
        except Exception as e:
            logger.error(f"❌ Failed to add to knowledge graph: {e}")
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract concepts from text"""
        # Simple concept extraction (in production, use NLP models)
        concepts = []
        text_lower = text.lower()
        
        concept_mapping = {
            "ai": ["artificial intelligence", "ai", "machine intelligence", "artificial"],
            "ml": ["machine learning", "ml", "learning algorithm", "neural network"],
            "prime aligned compute": ["prime aligned compute", "awareness", "mind", "conscious"],
            "quantum": ["quantum", "qubit", "quantum computing", "quantum mechanics"],
            "blockchain": ["blockchain", "cryptocurrency", "distributed ledger", "crypto"],
            "technology": ["technology", "tech", "innovation", "digital"],
            "science": ["science", "research", "scientific", "study"],
            "philosophy": ["philosophy", "philosophical", "ethics", "moral"],
            "psychology": ["psychology", "psychological", "mental", "cognitive"],
            "neuroscience": ["neuroscience", "brain", "neural", "neuron"]
        }
        
        for concept, keywords in concept_mapping.items():
            if any(keyword in text_lower for keyword in keywords):
                concepts.append(concept)
        
        return concepts[:10]  # Limit to 10 concepts
    
    def _generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        # Simple hash-based embeddings (in production, use proper embedding models)
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        embeddings = [float(b) / 255.0 for b in hash_bytes[:8]]
        return embeddings
    
    def search_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search scraped knowledge"""
        try:
            # Search RAG system
            relevant_docs = self.rag_system.retrieve_relevant_docs(query, top_k=limit)
            
            results = []
            for doc in relevant_docs:
                results.append({
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "prime_aligned_enhanced": doc.prime_aligned_enhanced
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to search knowledge: {e}")
            return []
    
    def get_scraping_stats(self) -> Dict[str, Any]:
        """Get scraping statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get counts
            cursor.execute('SELECT COUNT(*) FROM web_content')
            total_content = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM scraping_jobs WHERE status = "completed"')
            completed_jobs = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM scraping_jobs WHERE status = "pending"')
            pending_jobs = cursor.fetchone()[0]
            
            # Get average prime aligned compute score
            cursor.execute('SELECT AVG(prime_aligned_score) FROM web_content')
            avg_consciousness = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                "total_scraped_pages": total_content,
                "completed_jobs": completed_jobs,
                "pending_jobs": pending_jobs,
                "average_consciousness_score": round(avg_consciousness, 3),
                "knowledge_graph_nodes": len(self.knowledge_graph.graph.nodes),
                "knowledge_graph_edges": len(self.knowledge_graph.graph.edges),
                "rag_documents": len(self.rag_system.documents)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get scraping stats: {e}")
            return {}
    
    def start_batch_processing(self):
        """Start batch processing of scraping jobs"""
        logger.info("🚀 Starting batch processing...")
        self.running = True
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while self.running and not self.processing_queue.empty():
                try:
                    job = self.processing_queue.get(timeout=1)
                    
                    # Submit job to executor
                    future = executor.submit(self._process_job, job)
                    self.results_queue.put(future)
                    
                except:
                    break
        
        logger.info("✅ Batch processing completed")
    
    def _process_job(self, job: ScrapingJob):
        """Process a single scraping job"""
        try:
            # Update job status
            self._update_job_status(job.url, "processing")
            
            # Extract content
            content = self.extractor.extract_content(job.url, job)
            if content:
                self._store_scraped_content(content)
                self._update_job_status(job.url, "completed")
                return {"success": True, "url": job.url}
            else:
                self._update_job_status(job.url, "failed")
                return {"success": False, "url": job.url, "error": "Failed to extract content"}
                
        except Exception as e:
            self._update_job_status(job.url, "failed")
            logger.error(f"❌ Job processing failed: {e}")
            return {"success": False, "url": job.url, "error": str(e)}
    
    def _update_job_status(self, url: str, status: str):
        """Update job status in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE scraping_jobs 
                SET status = ?, completed_at = ?
                WHERE url = ?
            ''', (status, datetime.now().isoformat() if status in ["completed", "failed"] else None, url))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to update job status: {e}")

def main():
    """Main entry point for web scraper knowledge system"""
    print("🌐 Web Scraper Knowledge System")
    print("=" * 50)
    
    # Initialize system
    scraper = WebScraperKnowledgeSystem(max_workers=5)
    
    # Example websites to scrape
    example_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/prime aligned compute",
        "https://en.wikipedia.org/wiki/Quantum_computing",
        "https://en.wikipedia.org/wiki/Blockchain",
        "https://en.wikipedia.org/wiki/Machine_learning"
    ]
    
    print(f"\n🔍 Scraping {len(example_urls)} example websites...")
    
    # Scrape websites
    results = []
    for url in example_urls:
        print(f"\n📄 Scraping: {url}")
        result = scraper.scrape_website(url, max_depth=1, follow_links=False)
        results.append(result)
        
        if result["success"]:
            print(f"✅ Success: {result['title']}")
            print(f"   Content: {result['content_length']} chars")
            print(f"   prime aligned compute Score: {result['prime_aligned_score']:.3f}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Get statistics
    stats = scraper.get_scraping_stats()
    print(f"\n📊 Scraping Statistics:")
    print(f"   📄 Total Pages Scraped: {stats.get('total_scraped_pages', 0)}")
    print(f"   🧠 Average prime aligned compute Score: {stats.get('average_consciousness_score', 0)}")
    print(f"   🔗 Knowledge Graph Nodes: {stats.get('knowledge_graph_nodes', 0)}")
    print(f"   🔗 Knowledge Graph Edges: {stats.get('knowledge_graph_edges', 0)}")
    print(f"   📚 RAG Documents: {stats.get('rag_documents', 0)}")
    
    # Test knowledge search
    print(f"\n🔍 Testing Knowledge Search...")
    search_results = scraper.search_knowledge("artificial intelligence prime aligned compute", limit=3)
    
    for i, result in enumerate(search_results):
        print(f"\n📖 Result {i+1}:")
        print(f"   Content: {result['content'][:200]}...")
        print(f"   prime aligned compute Enhanced: {result['prime_aligned_enhanced']}")
    
    print(f"\n🎉 Web Scraper Knowledge System Demo Complete!")
    print("All scraped content has been stored in knowledge databases with prime aligned compute enhancement!")

if __name__ == "__main__":
    main()
