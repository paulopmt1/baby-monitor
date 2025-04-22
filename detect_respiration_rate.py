#!/usr/bin/env python3
"""
Script for detecting respiration (breathing) rate from a video file.
This uses Eulerian Video Magnification to amplify subtle movements
associated with breathing.
"""

import sys
import cv2
import numpy as np
import gc
import matplotlib.pyplot as plt
from eulerian_magnification import EulerianMagnification

def detect_respiration_rate(video_path):
    """Detect respiration rate (breaths per minute) from a video file."""
    # Open the video to get the first frame for ROI selection
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties to check size
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Print video info
    print(f"Video information:")
    print(f"- Resolution: {width}x{height}")
    print(f"- Frame count: {frame_count}")
    print(f"- FPS: {fps}")
    print(f"- Duration: {frame_count/fps:.1f} seconds")
    
    # Warn if video is large
    if frame_count > 1000 or (width * height > 1000000):
        print("\nWARNING: This video is quite large which may require significant memory.")
        print("It's highly recommended to select a region of interest (ROI) to reduce processing time.\n")
    
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame")
        cap.release()
        return
    
    # Let the user select a ROI interactively
    print("Select a region of interest (ROI) showing the torso/chest area and press ENTER")
    print("Press ESC to cancel and use the full frame")
    roi = cv2.selectROI("Select ROI", frame, False)
    cv2.destroyWindow("Select ROI")
    
    # Check if a valid ROI was selected
    if roi[2] <= 0 or roi[3] <= 0:
        print("No ROI selected or invalid ROI. Using the full frame.")
        roi = None
    else:
        print(f"Using ROI: {roi}")
        # Calculate the ROI size to determine appropriate pyramid levels
        roi_width = roi[2]
        roi_height = roi[3]
        print(f"ROI size: {roi_width}x{roi_height}")
    
    # Release the video capture
    cap.release()
    
    # Calculate appropriate pyramid levels
    min_dimension = min(roi[2], roi[3]) if roi else min(width, height)
    max_levels = 0
    size = min_dimension
    while size >= 16:  # Minimum size threshold
        size = size // 2
        max_levels += 1
    
    recommended_levels = min(3, max_levels)  # Default to 3 or less if the image is small
    print(f"Recommended pyramid levels: {recommended_levels} (based on image size)")
    
    # For respiration, we use a lower frequency range (typically 0.2-0.5 Hz which corresponds to 12-30 BPM)
    # Normal breathing rate for adults is 12-20 breaths per minute
    # For babies, it can be 30-60 breaths per minute
    print("Initializing Eulerian Video Magnification...")
    processor = EulerianMagnification(
        video_path=video_path,
        low_cutoff=0.1,     # 6 BPM (very slow breathing)
        high_cutoff=1.0,    # 60 BPM (rapid breathing)
        amplification=20,   # Lower amplification to avoid artifacts
        pyramid_levels=recommended_levels
    )
    
    # Set ROI if selected
    if roi and roi[2] > 0 and roi[3] > 0:
        processor.set_roi(*roi)
    
    # Process the video
    print("Processing video... This may take a while.")
    try:
        # Process video with progress updates
        frames, _ = processor.process_video(display=False)
        
        # Calculate respiration rate (using the same method as heart rate)
        resp_rate, freqs, power = processor.calculate_heart_rate()
        print(f"\nEstimated Respiration Rate: {resp_rate:.1f} breaths per minute")
        
        # Plot the signal with customized labels for respiration
        print("Showing results plot...")
        plt.figure(figsize=(12, 6))
        
        # Plot the raw signal
        plt.subplot(2, 1, 1)
        time = np.arange(len(processor.raw_signal)) / processor.fps
        plt.plot(time, processor.raw_signal)
        plt.title('Breathing Signal Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Plot the power spectrum
        plt.subplot(2, 1, 2)
        plt.plot(freqs, power)
        plt.axvline(x=resp_rate/60, color='r', linestyle='--')
        plt.title(f'Power Spectrum (Respiration Rate: {resp_rate:.1f} BPM)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        
        plt.tight_layout()
        plt.show()
        
        # Save the results
        output_path = "output_respiration_video.avi"
        print(f"Saving processed video to {output_path}...")
        processor.save_results(output_path, frames)
        print(f"Processed video saved to {output_path}")
        
        # Output textual results
        print("\nAnalysis Results:")
        print(f"- Respiration Rate: {resp_rate:.1f} breaths per minute")
        print(f"- Video FPS: {processor.fps}")
        print(f"- Frequency range: {processor.low_cutoff*60:.1f}-{processor.high_cutoff*60:.1f} BPM")
        print(f"- Amplification factor: {processor.amplification}")
        if processor.roi:
            print(f"- ROI: {processor.roi}")
        
        return resp_rate
    
    except MemoryError:
        print("\nERROR: Ran out of memory while processing the video.")
        print("Try using a smaller ROI or a shorter video clip.")
        return None
    
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Try using a smaller ROI or adjusting the pyramid levels.")
        return None
    
    finally:
        # Ensure memory is freed
        gc.collect()

def main():
    """Main function to run the respiration detection script."""
    if len(sys.argv) < 2:
        print("Usage: python detect_respiration_rate.py <video_path>")
        print("\nThis script detects breathing rate from a video. For best results:")
        print("1. Record video of the subject's chest/torso area")
        print("2. Keep the camera stable")
        print("3. Ensure good lighting")
        print("4. Record for at least 30 seconds")
        print("\nExample: python detect_respiration_rate.py baby_sleeping.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    try:
        resp_rate = detect_respiration_rate(video_path)
        if resp_rate is not None:
            print("\nProcessing completed successfully.")
        else:
            print("\nProcessing failed.")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 