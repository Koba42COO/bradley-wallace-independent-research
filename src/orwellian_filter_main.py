#!/usr/bin/env python3
"""
Orwellian Filter - Main Integration Script
===========================================

Complete system for detecting steganographic manipulation in:
- Images
- Video frames
- Websites

Features:
- ML-based detection
- Visual hitboxes
- Message decoding
- Psychological effect analysis

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
"""

import argparse
import sys
from pathlib import Path
import json
from typing import Optional

from steganography_detector_orwellian_filter import OrwellianFilter, DetectionResult
from orwellian_filter_video_analyzer import VideoFrameAnalyzer, save_analysis_report
from orwellian_filter_website_scanner import WebsiteScanner, save_scan_report


class OrwellianFilterSystem:
    """
    Complete Orwellian Filter system.
    """
    
    def __init__(self):
        self.filter = OrwellianFilter()
        self.video_analyzer = VideoFrameAnalyzer(self.filter)
        self.website_scanner = WebsiteScanner(self.filter)
    
    def analyze_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        show_visualization: bool = True
    ) -> DetectionResult:
        """
        Analyze single image.
        
        Args:
            image_path: Path to image file
            output_path: Path to save visualization
            show_visualization: Whether to display visualization
            
        Returns:
            Detection result
        """
        print(f"Analyzing image: {image_path}")
        
        result = self.filter.detect_in_image(image_path)
        
        print(f"\nDetection Results:")
        print(f"  Risk Score: {result.overall_risk_score:.3f}")
        print(f"  Detections: {len(result.detections)}")
        print(f"  Decoded Messages: {len(result.decoded_messages)}")
        
        if result.detections:
            print(f"\nDetection Details:")
            for i, detection in enumerate(result.detections[:5]):
                print(f"  Detection {i+1}:")
                print(f"    Confidence: {detection.confidence:.3f}")
                print(f"    Type: {detection.detection_type}")
                print(f"    BBox: {detection.bbox}")
        
        if result.decoded_messages:
            print(f"\nDecoded Messages:")
            for i, message in enumerate(result.decoded_messages):
                print(f"  Message {i+1}:")
                print(f"    Text: {message.text}")
                print(f"    Intent: {message.psychological_intent}")
                print(f"    Type: {message.manipulation_type}")
                print(f"    Confidence: {message.confidence:.3f}")
                if message.suggested_action:
                    print(f"    Suggested Action: {message.suggested_action}")
        
        # Save visualization
        if output_path and result.visualization is not None:
            from PIL import Image
            Image.fromarray(result.visualization).save(output_path)
            print(f"\nVisualization saved to: {output_path}")
        
        return result
    
    def analyze_video(
        self,
        video_path: str,
        frame_skip: int = 30,
        max_frames: Optional[int] = None,
        output_dir: Optional[str] = None,
        report_path: Optional[str] = None
    ):
        """
        Analyze video file.
        
        Args:
            video_path: Path to video file
            frame_skip: Analyze every Nth frame
            max_frames: Maximum frames to analyze
            output_dir: Directory for frame visualizations
            report_path: Path to save analysis report
        """
        print(f"Analyzing video: {video_path}")
        
        result = self.video_analyzer.analyze_video(
            video_path,
            frame_skip=frame_skip,
            max_frames=max_frames,
            output_dir=output_dir
        )
        
        print(f"\nVideo Analysis Results:")
        print(f"  Total frames: {result.total_frames}")
        print(f"  Analyzed frames: {result.analyzed_frames}")
        print(f"  Overall risk score: {result.overall_risk_score:.3f}")
        print(f"  High-risk frames: {len(result.high_risk_frames)}")
        print(f"  Frames with detections: {result.summary['frames_with_detections']}")
        
        if result.high_risk_frames:
            print(f"\nHigh-risk frame numbers: {result.high_risk_frames[:10]}")
        
        if report_path:
            save_analysis_report(result, report_path)
        
        return result
    
    def analyze_website(
        self,
        url: str,
        max_images: int = 20,
        report_path: Optional[str] = None
    ):
        """
        Analyze website.
        
        Args:
            url: Website URL
            max_images: Maximum images to scan
            report_path: Path to save scan report
        """
        print(f"Analyzing website: {url}")
        
        result = self.website_scanner.scan_website(
            url,
            max_images=max_images
        )
        
        print(f"\nWebsite Scan Results:")
        print(f"  Images found: {result.images_found}")
        print(f"  Images scanned: {result.images_scanned}")
        print(f"  Overall risk score: {result.overall_risk_score:.3f}")
        print(f"  High-risk images: {len(result.high_risk_images)}")
        
        if result.high_risk_images:
            print(f"\nHigh-risk image URLs:")
            for img_url in result.high_risk_images[:5]:
                print(f"  - {img_url}")
        
        if report_path:
            save_scan_report(result, report_path)
        
        return result


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Orwellian Filter - Steganography Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze an image
  python orwellian_filter_main.py image path/to/image.jpg -o output.png
  
  # Analyze a video
  python orwellian_filter_main.py video path/to/video.mp4 --frame-skip 30
  
  # Analyze a website
  python orwellian_filter_main.py website https://example.com --max-images 20
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Image analysis
    img_parser = subparsers.add_parser('image', help='Analyze an image')
    img_parser.add_argument('path', help='Path to image file')
    img_parser.add_argument('-o', '--output', help='Output visualization path')
    img_parser.add_argument('--no-vis', action='store_true', help='Don\'t show visualization')
    
    # Video analysis
    vid_parser = subparsers.add_parser('video', help='Analyze a video')
    vid_parser.add_argument('path', help='Path to video file')
    vid_parser.add_argument('--frame-skip', type=int, default=30, help='Analyze every Nth frame')
    vid_parser.add_argument('--max-frames', type=int, help='Maximum frames to analyze')
    vid_parser.add_argument('--output-dir', help='Directory for frame visualizations')
    vid_parser.add_argument('--report', help='Path to save analysis report')
    
    # Website analysis
    web_parser = subparsers.add_parser('website', help='Analyze a website')
    web_parser.add_argument('url', help='Website URL')
    web_parser.add_argument('--max-images', type=int, default=20, help='Maximum images to scan')
    web_parser.add_argument('--report', help='Path to save scan report')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    system = OrwellianFilterSystem()
    
    try:
        if args.command == 'image':
            system.analyze_image(
                args.path,
                output_path=args.output,
                show_visualization=not args.no_vis
            )
        
        elif args.command == 'video':
            system.analyze_video(
                args.path,
                frame_skip=args.frame_skip,
                max_frames=args.max_frames,
                output_dir=args.output_dir,
                report_path=args.report
            )
        
        elif args.command == 'website':
            system.analyze_website(
                args.url,
                max_images=args.max_images,
                report_path=args.report
            )
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

