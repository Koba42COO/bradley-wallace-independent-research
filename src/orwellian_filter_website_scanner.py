#!/usr/bin/env python3
"""
Orwellian Filter - Website Scanner
===================================

Scans websites for steganographic manipulation in images.

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional, Set
import time
from pathlib import Path
import json
from dataclasses import dataclass, asdict

from steganography_detector_orwellian_filter import (
    OrwellianFilter, DetectionResult
)


@dataclass
class WebsiteScanResult:
    """Website scan result"""
    url: str
    images_found: int
    images_scanned: int
    detections: List[DetectionResult]
    high_risk_images: List[str]
    overall_risk_score: float
    scan_metadata: Dict


class WebsiteScanner:
    """
    Scans websites for steganographic manipulation in images.
    """
    
    def __init__(self, filter_system: Optional[OrwellianFilter] = None):
        self.filter = filter_system or OrwellianFilter()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.visited_urls: Set[str] = set()
        
    def scan_website(
        self,
        url: str,
        max_images: int = 50,
        follow_links: bool = False,
        max_depth: int = 1
    ) -> WebsiteScanResult:
        """
        Scan website for steganographic images.
        
        Args:
            url: Website URL to scan
            max_images: Maximum images to scan
            follow_links: Whether to follow links to other pages
            max_depth: Maximum depth for link following
            
        Returns:
            Website scan result
        """
        print(f"Scanning website: {url}")
        
        # Get all image URLs from website
        image_urls = self._extract_image_urls(url, follow_links, max_depth)
        
        print(f"Found {len(image_urls)} images")
        
        if max_images:
            image_urls = image_urls[:max_images]
        
        # Scan each image
        detections = []
        high_risk_images = []
        scanned_count = 0
        
        for img_url in image_urls:
            try:
                print(f"  Scanning: {img_url}")
                result = self.filter.detect_in_website(url, [img_url])[0]
                detections.append(result)
                scanned_count += 1
                
                if result.overall_risk_score > 0.5:
                    high_risk_images.append(img_url)
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"  Error scanning {img_url}: {e}")
        
        # Calculate overall risk
        if detections:
            overall_risk = sum(d.overall_risk_score for d in detections) / len(detections)
        else:
            overall_risk = 0.0
        
        scan_metadata = {
            'total_images_found': len(image_urls),
            'images_scanned': scanned_count,
            'high_risk_count': len(high_risk_images),
            'scan_timestamp': time.time()
        }
        
        return WebsiteScanResult(
            url=url,
            images_found=len(image_urls),
            images_scanned=scanned_count,
            detections=detections,
            high_risk_images=high_risk_images,
            overall_risk_score=overall_risk,
            scan_metadata=scan_metadata
        )
    
    def _extract_image_urls(
        self,
        url: str,
        follow_links: bool = False,
        max_depth: int = 1,
        current_depth: int = 0
    ) -> List[str]:
        """Extract image URLs from website"""
        if url in self.visited_urls or current_depth > max_depth:
            return []
        
        self.visited_urls.add(url)
        image_urls = []
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all images
            for img in soup.find_all('img'):
                src = img.get('src') or img.get('data-src')
                if src:
                    absolute_url = urljoin(url, src)
                    if self._is_image_url(absolute_url):
                        image_urls.append(absolute_url)
            
            # Follow links if requested
            if follow_links and current_depth < max_depth:
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    absolute_url = urljoin(url, href)
                    
                    # Only follow same-domain links
                    if urlparse(absolute_url).netloc == urlparse(url).netloc:
                        sub_images = self._extract_image_urls(
                            absolute_url, follow_links, max_depth, current_depth + 1
                        )
                        image_urls.extend(sub_images)
        
        except Exception as e:
            print(f"Error extracting images from {url}: {e}")
        
        return list(set(image_urls))  # Remove duplicates
    
    def _is_image_url(self, url: str) -> bool:
        """Check if URL is an image"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        parsed = urlparse(url)
        path = parsed.path.lower()
        return any(path.endswith(ext) for ext in image_extensions) or 'image' in parsed.query.lower()
    
    def scan_image_urls(
        self,
        image_urls: List[str]
    ) -> List[DetectionResult]:
        """
        Scan specific image URLs.
        
        Args:
            image_urls: List of image URLs to scan
            
        Returns:
            List of detection results
        """
        results = []
        
        for img_url in image_urls:
            try:
                result = self.filter.detect_in_website("", [img_url])[0]
                results.append(result)
            except Exception as e:
                print(f"Error scanning {img_url}: {e}")
        
        return results


def save_scan_report(
    result: WebsiteScanResult,
    output_path: str
):
    """Save scan report to JSON file"""
    report = {
        'url': result.url,
        'images_found': result.images_found,
        'images_scanned': result.images_scanned,
        'overall_risk_score': result.overall_risk_score,
        'high_risk_images': result.high_risk_images,
        'scan_metadata': result.scan_metadata,
        'detections_summary': [
            {
                'source_url': d.metadata.get('source_url', ''),
                'risk_score': d.overall_risk_score,
                'num_detections': len(d.detections),
                'num_messages': len(d.decoded_messages)
            }
            for d in result.detections
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Scan report saved to: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python orwellian_filter_website_scanner.py <url> [max_images]")
        sys.exit(1)
    
    url = sys.argv[1]
    max_images = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    scanner = WebsiteScanner()
    result = scanner.scan_website(url, max_images=max_images)
    
    print("\n" + "=" * 70)
    print("Website Scan Complete")
    print("=" * 70)
    print(f"URL: {result.url}")
    print(f"Images found: {result.images_found}")
    print(f"Images scanned: {result.images_scanned}")
    print(f"Overall risk score: {result.overall_risk_score:.3f}")
    print(f"High-risk images: {len(result.high_risk_images)}")
    print("=" * 70)
    
    if result.high_risk_images:
        print("\nHigh-risk images:")
        for img_url in result.high_risk_images[:5]:
            print(f"  - {img_url}")

