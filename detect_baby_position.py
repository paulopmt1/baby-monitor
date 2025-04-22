#!/usr/bin/env python3
"""
Baby Position Monitor

This script detects if a baby is in an unsafe sleeping position.
For safety, babies should always sleep on their backs (supine position).
The script uses computer vision to analyze video feeds and alert if the baby
is not in a safe position (e.g., on their side or stomach).
"""

import cv2
import numpy as np
import time
import argparse
import os
import sys
from datetime import datetime

# Import optional deep learning-based pose estimation if available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not found. Using basic image processing for position detection.")

# Constants
ALERT_THRESHOLD = 5  # Number of consecutive frames to trigger an alert
POSITION_HISTORY_SIZE = 10  # Number of frames to keep for position history
POSITION_CHECK_INTERVAL = 1  # Seconds between position checks

# Safe position is supine (on back, face up)
POSITION_LABELS = {
    0: "SAFE - On Back",
    1: "UNSAFE - On Side",
    2: "UNSAFE - On Stomach"
}

class BabyPositionMonitor:
    """Class to monitor baby position from video feed."""
    
    def __init__(self, source=0, output_dir="alerts", use_mediapipe=True):
        """
        Initialize the baby position monitor.
        
        Args:
            source: Video source (0 for webcam, path to video file)
            output_dir: Directory to save alert screenshots
            use_mediapipe: Whether to use MediaPipe for pose estimation
        """
        self.source = source
        self.output_dir = output_dir
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        
        # Initialize video capture
        self.cap = None
        
        # Position tracking
        self.position_history = []
        self.last_alert_time = 0
        self.current_position = 0  # 0: back (safe), 1: side, 2: stomach (unsafe)
        self.alert_counter = 0
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize MediaPipe if available
        if self.use_mediapipe:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
    
    def start(self):
        """Start the baby position monitoring."""
        # Initialize video capture
        if isinstance(self.source, int) or self.source.isdigit():
            self.cap = cv2.VideoCapture(int(self.source))
        else:
            self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video source {self.source}")
            return False
        
        print(f"Starting baby position monitor...")
        print(f"Press 'q' to quit, 's' to save a screenshot")
        
        # Main processing loop
        last_check_time = time.time()
        
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                # If video file ends, restart from beginning
                if not isinstance(self.source, int) and not self.source.isdigit():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            # Make a copy for display
            display_frame = frame.copy()
            
            # Check position periodically to reduce CPU usage
            current_time = time.time()
            if current_time - last_check_time >= POSITION_CHECK_INTERVAL:
                # Detect baby position
                position, confidence = self.detect_position(frame)
                
                # Update position history
                self.position_history.append(position)
                if len(self.position_history) > POSITION_HISTORY_SIZE:
                    self.position_history.pop(0)
                
                # Update current position by majority voting
                if len(self.position_history) >= 3:
                    self.current_position = max(set(self.position_history), key=self.position_history.count)
                
                # Check for unsafe position
                if self.current_position > 0:  # Not on back
                    self.alert_counter += 1
                    if self.alert_counter >= ALERT_THRESHOLD:
                        # Alert if it's been at least 10 seconds since last alert
                        if current_time - self.last_alert_time >= 10:
                            self.generate_alert(display_frame)
                            self.last_alert_time = current_time
                else:
                    self.alert_counter = 0
                
                last_check_time = current_time
            
            # Display status information on frame
            self.draw_status(display_frame)
            
            # Show the frame
            cv2.imshow("Baby Position Monitor", display_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_screenshot(display_frame)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        if self.use_mediapipe:
            self.pose.close()
        
        return True
    
    def detect_position(self, frame):
        """
        Detect the baby's sleeping position.
        
        Args:
            frame: Video frame to analyze
            
        Returns:
            position: Position code (0: back, 1: side, 2: stomach)
            confidence: Confidence level (0-1)
        """
        if self.use_mediapipe:
            return self._detect_position_mediapipe(frame)
        else:
            return self._detect_position_basic(frame)
    
    def _detect_position_mediapipe(self, frame):
        """
        Detect position using MediaPipe pose estimation.
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = self.pose.process(frame_rgb)
        
        # Default position (back) if no pose detected
        if not results.pose_landmarks:
            return 0, 0.5
        
        # Extract relevant landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Check if sufficient landmarks are visible
        if not (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.5 and 
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.5):
            return 0, 0.5  # Default to safe position if can't determine clearly
        
        # Get key landmarks
        left_shoulder = np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                 landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y])
        right_shoulder = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                  landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
        left_hip = np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y])
        right_hip = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y])
        
        # Calculate shoulder width in image space
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        
        # Check for side position - shoulders will appear closer together
        if shoulder_width < 0.1:  # Threshold determined empirically
            return 1, 0.7  # Side position
        
        # Check for stomach-down position
        # When face down, typically the shoulders are below the hips in image space
        if (left_shoulder[1] > left_hip[1] and right_shoulder[1] > right_hip[1]):
            return 2, 0.8  # Stomach position (prone)
        
        # Default to back position (supine)
        return 0, 0.9
    
    def _detect_position_basic(self, frame):
        """
        Basic detection using image processing when MediaPipe is not available.
        This is less accurate but can provide a basic estimation.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (likely the baby)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box and orientation
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            # Calculate aspect ratio of the bounding box
            width = rect[1][0]
            height = rect[1][1]
            aspect_ratio = max(width, height) / (min(width, height) + 1e-5)  # Avoid division by zero
            
            # Get angle
            angle = rect[2]
            if width < height:
                angle += 90
            
            # A baby on their back usually has a wider profile from above
            if aspect_ratio > 1.5:
                return 0, 0.6  # Likely on back
            
            # Check the angle of the bounding box
            if 45 <= abs(angle) <= 135:
                return 1, 0.5  # Possibly on side
            else:
                return 2, 0.4  # Possibly on stomach
        
        # Default to safe position if no contour found
        return 0, 0.3
    
    def draw_status(self, frame):
        """Draw status information on the frame."""
        h, w = frame.shape[:2]
        
        # Create a semi-transparent overlay for the status area
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw position status
        position_text = POSITION_LABELS[self.current_position]
        if self.current_position == 0:  # Safe
            cv2.putText(frame, position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 255, 0), 2)
        else:  # Unsafe
            cv2.putText(frame, position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 0, 255), 2)
            
            # Add warning for unsafe position
            if self.alert_counter >= ALERT_THRESHOLD:
                cv2.putText(frame, "WARNING: Baby in unsafe position!", (10, 55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def generate_alert(self, frame):
        """Generate an alert for unsafe position."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"position_alert_{timestamp}.jpg")
        
        # Save the frame
        cv2.imwrite(filename, frame)
        
        # Print alert
        position_text = POSITION_LABELS[self.current_position]
        print(f"ALERT! Baby detected in unsafe position: {position_text}")
        print(f"Alert image saved to {filename}")
        
        # Add alert overlay to the frame
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, "ALERT! UNSAFE POSITION", (w//2 - 200, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    def save_screenshot(self, frame):
        """Save a screenshot manually."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"position_screenshot_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved to {filename}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Baby Position Monitor")
    parser.add_argument("--source", "-s", default=0, 
                        help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--output", "-o", default="position_alerts",
                        help="Output directory for alerts")
    parser.add_argument("--basic", "-b", action="store_true",
                        help="Use basic image processing instead of MediaPipe")
    
    args = parser.parse_args()
    
    # Print info about available detection methods
    if not MEDIAPIPE_AVAILABLE and not args.basic:
        print("MediaPipe not available. Using basic image processing instead.")
        args.basic = True
    
    # Create and start the monitor
    monitor = BabyPositionMonitor(
        source=args.source,
        output_dir=args.output,
        use_mediapipe=not args.basic
    )
    
    monitor.start()

if __name__ == "__main__":
    main() 