#!/usr/bin/env python3
"""
Orwellian Filter - Video Frame Analyzer
========================================

Analyzes video files frame-by-frame for steganographic manipulation.

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass, asdict

from steganography_detector_orwellian_filter import (
    OrwellianFilter, DetectionResult
)


@dataclass
class VideoAnalysisResult:
    """Complete video analysis result"""
    video_path: str
    total_frames: int
    analyzed_frames: int
    detections_per_frame: List[int]
    risk_scores_per_frame: List[float]
    high_risk_frames: List[int]
    overall_risk_score: float
    frame_results: List[DetectionResult]
    summary: Dict


class VideoFrameAnalyzer:
    """
    Analyzes video files for steganographic content frame-by-frame.
    """
    
    def __init__(self, filter_system: Optional[OrwellianFilter] = None):
        self.filter = filter_system or OrwellianFilter()
        self.frame_skip = 1  # Analyze every Nth frame (1 = all frames)
        
    def analyze_video(
        self,
        video_path: str,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> VideoAnalysisResult:
        """
        Analyze video for steganographic manipulation.
        
        Args:
            video_path: Path to video file
            frame_skip: Analyze every Nth frame
            max_frames: Maximum frames to analyze (None = all)
            output_dir: Directory to save frame visualizations
            
        Returns:
            Video analysis result
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_results = []
        detections_per_frame = []
        risk_scores_per_frame = []
        high_risk_frames = []
        
        frame_number = 0
        analyzed_count = 0
        
        print(f"Analyzing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps:.2f}")
        print(f"Frame skip: {frame_skip}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and analyzed_count >= max_frames:
                break
            
            # Skip frames if needed
            if frame_number % frame_skip != 0:
                frame_number += 1
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect steganography
            result = self.filter.detect_in_video_frame(frame_rgb, frame_number)
            
            frame_results.append(result)
            detections_per_frame.append(len(result.detections))
            risk_scores_per_frame.append(result.overall_risk_score)
            
            # Track high-risk frames
            if result.overall_risk_score > 0.5:
                high_risk_frames.append(frame_number)
            
            # Save visualization if output directory provided
            if output_dir and result.visualization is not None:
                output_path = Path(output_dir) / f"frame_{frame_number:06d}.png"
                cv2.imwrite(str(output_path), 
                           cv2.cvtColor(result.visualization, cv2.COLOR_RGB2BGR))
            
            analyzed_count += 1
            frame_number += 1
            
            if analyzed_count % 100 == 0:
                print(f"  Analyzed {analyzed_count} frames...")
        
        cap.release()
        
        # Calculate overall statistics
        overall_risk = np.mean(risk_scores_per_frame) if risk_scores_per_frame else 0.0
        
        summary = {
            'total_frames': total_frames,
            'analyzed_frames': analyzed_count,
            'high_risk_frames': len(high_risk_frames),
            'avg_risk_score': overall_risk,
            'max_risk_score': max(risk_scores_per_frame) if risk_scores_per_frame else 0.0,
            'frames_with_detections': sum(1 for d in detections_per_frame if d > 0),
            'total_detections': sum(detections_per_frame)
        }
        
        return VideoAnalysisResult(
            video_path=video_path,
            total_frames=total_frames,
            analyzed_frames=analyzed_count,
            detections_per_frame=detections_per_frame,
            risk_scores_per_frame=risk_scores_per_frame,
            high_risk_frames=high_risk_frames,
            overall_risk_score=overall_risk,
            frame_results=frame_results,
            summary=summary
        )
    
    def analyze_video_segment(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int
    ) -> VideoAnalysisResult:
        """
        Analyze specific segment of video.
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame number
            end_frame: Ending frame number
            
        Returns:
            Video analysis result for segment
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_results = []
        detections_per_frame = []
        risk_scores_per_frame = []
        high_risk_frames = []
        
        frame_number = start_frame
        
        while frame_number <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.filter.detect_in_video_frame(frame_rgb, frame_number)
            
            frame_results.append(result)
            detections_per_frame.append(len(result.detections))
            risk_scores_per_frame.append(result.overall_risk_score)
            
            if result.overall_risk_score > 0.5:
                high_risk_frames.append(frame_number)
            
            frame_number += 1
        
        cap.release()
        
        overall_risk = np.mean(risk_scores_per_frame) if risk_scores_per_frame else 0.0
        
        summary = {
            'segment_start': start_frame,
            'segment_end': end_frame,
            'analyzed_frames': len(frame_results),
            'high_risk_frames': len(high_risk_frames),
            'avg_risk_score': overall_risk
        }
        
        return VideoAnalysisResult(
            video_path=video_path,
            total_frames=end_frame - start_frame + 1,
            analyzed_frames=len(frame_results),
            detections_per_frame=detections_per_frame,
            risk_scores_per_frame=risk_scores_per_frame,
            high_risk_frames=high_risk_frames,
            overall_risk_score=overall_risk,
            frame_results=frame_results,
            summary=summary
        )


def save_analysis_report(
    result: VideoAnalysisResult,
    output_path: str
):
    """Save analysis report to JSON file"""
    report = {
        'video_path': result.video_path,
        'total_frames': result.total_frames,
        'analyzed_frames': result.analyzed_frames,
        'overall_risk_score': result.overall_risk_score,
        'summary': result.summary,
        'high_risk_frames': result.high_risk_frames,
        'detections_per_frame': result.detections_per_frame[:100],  # Limit size
        'risk_scores_per_frame': [float(x) for x in result.risk_scores_per_frame[:100]]
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Analysis report saved to: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python orwellian_filter_video_analyzer.py <video_path> [frame_skip] [max_frames]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    frame_skip = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    analyzer = VideoFrameAnalyzer()
    result = analyzer.analyze_video(video_path, frame_skip=frame_skip, max_frames=max_frames)
    
    print("\n" + "=" * 70)
    print("Video Analysis Complete")
    print("=" * 70)
    print(f"Total frames: {result.total_frames}")
    print(f"Analyzed frames: {result.analyzed_frames}")
    print(f"Overall risk score: {result.overall_risk_score:.3f}")
    print(f"High-risk frames: {len(result.high_risk_frames)}")
    print(f"Frames with detections: {result.summary['frames_with_detections']}")
    print("=" * 70)

