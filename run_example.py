#!/usr/bin/env python3
"""
Example script for processing a video file using Eulerian Video Magnification for heart rate detection.
This script demonstrates how to use the EulerianMagnification class directly without the GUI.
"""

import sys
import os
import matplotlib.pyplot as plt
from eulerian_magnification import EulerianMagnification

def process_video(video_path, roi=None, low_bpm=50, high_bpm=150, amplification=50, save_path=None):
    """
    Process a video to detect heart rate.
    
    Parameters:
    - video_path: Path to the video file
    - roi: Region of interest (x, y, width, height) or None for full frame
    - low_bpm: Minimum heart rate in BPM (default: 50)
    - high_bpm: Maximum heart rate in BPM (default: 150)
    - amplification: Amplification factor (default: 50)
    - save_path: Path to save the processed video or None to skip saving
    
    Returns:
    - Estimated heart rate in BPM
    """
    # Convert BPM to Hz
    low_cutoff = low_bpm / 60.0
    high_cutoff = high_bpm / 60.0
    
    # Create and configure the processor
    processor = EulerianMagnification(
        video_path=video_path,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        amplification=amplification,
        pyramid_levels=4
    )
    
    # Set ROI if provided
    if roi:
        processor.set_roi(*roi)
    
    # Process the video
    print(f"Processing video: {video_path}")
    print(f"Parameters: {low_bpm}-{high_bpm} BPM, Amplification: {amplification}")
    
    frames, _ = processor.process_video(display=False)
    
    # Calculate heart rate
    heart_rate, freqs, power = processor.calculate_heart_rate()
    print(f"Estimated Heart Rate: {heart_rate:.1f} BPM")
    
    # Plot the results
    processor.plot_signal()
    
    # Save results if requested
    if save_path:
        processor.save_results(save_path, frames)
        print(f"Results saved to {save_path}")
    
    return heart_rate

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_example.py <video_path> [x y width height] [low_bpm high_bpm] [amplification]")
        print("  video_path: Path to the video file")
        print("  x y width height: Optional ROI coordinates")
        print("  low_bpm high_bpm: Optional heart rate range in BPM (default: 50 150)")
        print("  amplification: Optional amplification factor (default: 50)")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Parse optional ROI
    roi = None
    if len(sys.argv) >= 6:
        try:
            x = int(sys.argv[2])
            y = int(sys.argv[3])
            width = int(sys.argv[4])
            height = int(sys.argv[5])
            roi = (x, y, width, height)
            print(f"Using ROI: {roi}")
        except ValueError:
            print("Invalid ROI coordinates. Using full frame.")
    
    # Parse optional BPM range
    low_bpm = 50
    high_bpm = 150
    if len(sys.argv) >= 8:
        try:
            low_bpm = int(sys.argv[6])
            high_bpm = int(sys.argv[7])
            print(f"Using BPM range: {low_bpm}-{high_bpm}")
        except ValueError:
            print(f"Invalid BPM range. Using default: {low_bpm}-{high_bpm}")
    
    # Parse optional amplification
    amplification = 50
    if len(sys.argv) >= 9:
        try:
            amplification = int(sys.argv[8])
            print(f"Using amplification: {amplification}")
        except ValueError:
            print(f"Invalid amplification. Using default: {amplification}")
    
    # Generate save path based on input file
    save_path = os.path.splitext(video_path)[0] + "_processed.avi"
    
    try:
        # Process the video
        heart_rate = process_video(
            video_path=video_path,
            roi=roi,
            low_bpm=low_bpm,
            high_bpm=high_bpm,
            amplification=amplification,
            save_path=save_path
        )
        
        print(f"Processing complete. Heart rate: {heart_rate:.1f} BPM")
        print(f"Results saved to {save_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 