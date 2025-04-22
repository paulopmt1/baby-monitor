#!/usr/bin/env python3
"""
Baby Position Monitor GUI

A graphical interface for monitoring baby sleeping position to ensure safety.
This application analyzes video to detect if a baby is in an unsafe position
(on their stomach or side) rather than on their back (the safe position).
"""

import sys
import os
import cv2
import numpy as np
import threading
import time
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QComboBox, QFileDialog, QMessageBox,
                            QGroupBox, QCheckBox, QSlider, QSpinBox, QRadioButton,
                            QButtonGroup, QProgressBar, QSplitter, QFrame)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QSettings

# Import the position detection module
from detect_baby_position import BabyPositionMonitor, POSITION_LABELS, MEDIAPIPE_AVAILABLE

class VideoThread(QThread):
    """Thread for processing video frames without blocking the GUI."""
    update_frame = pyqtSignal(np.ndarray, int, float)  # frame, position, confidence
    alert_signal = pyqtSignal(np.ndarray, int)  # frame, position
    error_signal = pyqtSignal(str)  # error message
    
    def __init__(self, source=0, use_mediapipe=True):
        super().__init__()
        self.source = source
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        self.running = False
        self.monitor = None
        self.paused = False
    
    def run(self):
        """Main thread function to process video frames."""
        try:
            self.running = True
            self.monitor = BabyPositionMonitor(
                source=self.source,
                output_dir="position_alerts",
                use_mediapipe=self.use_mediapipe
            )
            
            # Initialize video capture
            if isinstance(self.source, int) or self.source.isdigit():
                cap = cv2.VideoCapture(int(self.source))
            else:
                cap = cv2.VideoCapture(self.source)
            
            if not cap.isOpened():
                self.error_signal.emit(f"Could not open video source: {self.source}")
                return
            
            # Position tracking
            position_history = []
            last_alert_time = 0
            current_position = 0
            alert_counter = 0
            check_interval = 0.5  # Check position every 0.5 seconds
            last_check_time = time.time()
            
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue
                    
                ret, frame = cap.read()
                
                if not ret:
                    # If video file ends, restart from beginning
                    if not isinstance(self.source, int) and not self.source.isdigit():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                
                # Check position periodically to reduce CPU usage
                current_time = time.time()
                if current_time - last_check_time >= check_interval:
                    # Detect baby position
                    position, confidence = self.monitor.detect_position(frame)
                    
                    # Update position history
                    position_history.append(position)
                    if len(position_history) > 10:  # Keep last 10 positions
                        position_history.pop(0)
                    
                    # Update current position by majority voting
                    if len(position_history) >= 3:
                        current_position = max(set(position_history), key=position_history.count)
                    else:
                        current_position = position
                    
                    # Check for unsafe position
                    if current_position > 0:  # Not on back
                        alert_counter += 1
                        if alert_counter >= 5:  # Alert after 5 consecutive checks
                            # Alert if it's been at least 10 seconds since last alert
                            if current_time - last_alert_time >= 10:
                                self.alert_signal.emit(frame.copy(), current_position)
                                last_alert_time = current_time
                    else:
                        alert_counter = 0
                    
                    last_check_time = current_time
                    
                    # Emit the frame with position information
                    self.update_frame.emit(frame.copy(), current_position, confidence)
                else:
                    # Just emit the frame without position check
                    self.update_frame.emit(frame.copy(), current_position, 0.0)
                
                # Limit frame rate to reduce CPU usage
                time.sleep(0.033)  # ~30 FPS
            
            # Cleanup
            cap.release()
            
        except Exception as e:
            self.error_signal.emit(f"Error in video processing: {str(e)}")
    
    def stop(self):
        """Stop the thread."""
        self.running = False
        self.wait()
    
    def toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
        return self.paused

class PositionMonitorGUI(QMainWindow):
    """Main GUI window for the baby position monitor."""
    
    def __init__(self):
        super().__init__()
        self.video_thread = None
        self.camera_sources = []
        self.alert_count = 0
        self.position_history = []
        
        self.init_ui()
        self.load_settings()
        
        # Initial detection of available cameras
        self.detect_cameras()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Baby Position Monitor")
        self.setMinimumSize(800, 600)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create splitter for resizable areas
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # Video display area
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        
        # Video feed display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        video_layout.addWidget(self.video_label)
        
        # Status bar below video
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.StyledPanel)
        status_frame.setStyleSheet("background-color: rgba(0, 0, 0, 0.1);")
        status_layout = QHBoxLayout(status_frame)
        
        # Position status
        self.position_label = QLabel("Status: Not monitoring")
        self.position_label.setFont(QFont("Arial", 12, QFont.Bold))
        status_layout.addWidget(self.position_label)
        
        # Alert counter
        self.alert_label = QLabel("Alerts: 0")
        self.alert_label.setFont(QFont("Arial", 12))
        status_layout.addWidget(self.alert_label, alignment=Qt.AlignRight)
        
        video_layout.addWidget(status_frame)
        
        # Add video widget to splitter
        splitter.addWidget(video_widget)
        
        # Controls area
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # Source selection group
        source_group = QGroupBox("Video Source")
        source_layout = QHBoxLayout(source_group)
        
        self.source_combo = QComboBox()
        self.source_combo.addItem("Select Video Source", None)
        self.source_combo.setMinimumWidth(250)
        source_layout.addWidget(self.source_combo)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.detect_cameras)
        source_layout.addWidget(self.refresh_btn)
        
        self.file_btn = QPushButton("Open Video File")
        self.file_btn.clicked.connect(self.open_video_file)
        source_layout.addWidget(self.file_btn)
        
        controls_layout.addWidget(source_group)
        
        # Detection settings group
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QHBoxLayout(settings_group)
        
        # Position detection method
        self.method_checkbox = QCheckBox("Use MediaPipe (more accurate)")
        self.method_checkbox.setChecked(MEDIAPIPE_AVAILABLE)
        self.method_checkbox.setEnabled(MEDIAPIPE_AVAILABLE)
        if not MEDIAPIPE_AVAILABLE:
            self.method_checkbox.setText("MediaPipe not available (using basic detection)")
        settings_layout.addWidget(self.method_checkbox)
        
        # Alert settings
        alert_label = QLabel("Save alerts to:")
        settings_layout.addWidget(alert_label)
        
        self.alert_folder = QPushButton("position_alerts")
        self.alert_folder.clicked.connect(self.select_alert_folder)
        settings_layout.addWidget(self.alert_folder)
        
        controls_layout.addWidget(settings_group)
        
        # Control buttons group
        buttons_group = QGroupBox("Controls")
        buttons_layout = QHBoxLayout(buttons_group)
        
        self.start_btn = QPushButton("Start Monitoring")
        self.start_btn.clicked.connect(self.toggle_monitoring)
        self.start_btn.setEnabled(False)
        buttons_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        buttons_layout.addWidget(self.pause_btn)
        
        self.screenshot_btn = QPushButton("Take Screenshot")
        self.screenshot_btn.clicked.connect(self.take_screenshot)
        self.screenshot_btn.setEnabled(False)
        buttons_layout.addWidget(self.screenshot_btn)
        
        controls_layout.addWidget(buttons_group)
        
        # Add controls widget to splitter
        splitter.addWidget(controls_widget)
        
        # Set the default splitter proportions
        splitter.setSizes([500, 100])
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Set up timers
        self.status_check_timer = QTimer()
        self.status_check_timer.timeout.connect(self.update_status)
        self.status_check_timer.start(1000)  # Check status every second
    
    def detect_cameras(self):
        """Detect available camera sources."""
        self.source_combo.clear()
        self.source_combo.addItem("Select Video Source", None)
        
        self.camera_sources = []
        # Check for webcams (typically 0-9 is sufficient for most users)
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                self.camera_sources.append(i)
                self.source_combo.addItem(f"Camera {i}", i)
        
        self.statusBar().showMessage(f"Found {len(self.camera_sources)} camera(s)")
    
    def open_video_file(self):
        """Open a video file for processing."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_path:
            # Add to combo box if not already there
            found = False
            for i in range(self.source_combo.count()):
                if self.source_combo.itemData(i) == file_path:
                    found = True
                    self.source_combo.setCurrentIndex(i)
                    break
            
            if not found:
                self.source_combo.addItem(os.path.basename(file_path), file_path)
                self.source_combo.setCurrentIndex(self.source_combo.count() - 1)
            
            self.start_btn.setEnabled(True)
            self.statusBar().showMessage(f"Loaded video file: {os.path.basename(file_path)}")
    
    def select_alert_folder(self):
        """Select folder to save alert screenshots."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Alert Output Folder", "position_alerts"
        )
        
        if folder:
            self.alert_folder.setText(os.path.basename(folder))
            self.statusBar().showMessage(f"Alert screenshots will be saved to: {folder}")
    
    def toggle_monitoring(self):
        """Start or stop the monitoring process."""
        if self.video_thread is None or not self.video_thread.isRunning():
            # Get selected source
            index = self.source_combo.currentIndex()
            if index <= 0:
                QMessageBox.warning(self, "No Source", "Please select a video source first.")
                return
                
            source = self.source_combo.itemData(index)
            use_mediapipe = self.method_checkbox.isChecked() and MEDIAPIPE_AVAILABLE
            
            # Start the video thread
            self.video_thread = VideoThread(source=source, use_mediapipe=use_mediapipe)
            self.video_thread.update_frame.connect(self.update_frame)
            self.video_thread.alert_signal.connect(self.handle_alert)
            self.video_thread.error_signal.connect(self.handle_error)
            self.video_thread.start()
            
            self.start_btn.setText("Stop Monitoring")
            self.pause_btn.setEnabled(True)
            self.screenshot_btn.setEnabled(True)
            self.position_label.setText("Status: Initializing...")
            self.statusBar().showMessage("Monitoring started")
            
            # Reset alert count
            self.alert_count = 0
            self.alert_label.setText(f"Alerts: {self.alert_count}")
            
        else:
            # Stop the video thread
            if self.video_thread:
                self.video_thread.stop()
                self.video_thread = None
            
            self.start_btn.setText("Start Monitoring")
            self.pause_btn.setText("Pause")
            self.pause_btn.setEnabled(False)
            self.screenshot_btn.setEnabled(False)
            self.position_label.setText("Status: Not monitoring")
            self.statusBar().showMessage("Monitoring stopped")
    
    def toggle_pause(self):
        """Pause or resume the video processing."""
        if self.video_thread and self.video_thread.isRunning():
            is_paused = self.video_thread.toggle_pause()
            if is_paused:
                self.pause_btn.setText("Resume")
                self.statusBar().showMessage("Monitoring paused")
            else:
                self.pause_btn.setText("Pause")
                self.statusBar().showMessage("Monitoring resumed")
    
    def take_screenshot(self):
        """Save the current frame as a screenshot."""
        if hasattr(self, 'current_frame'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = self.alert_folder.text()
            if not os.path.exists(folder):
                os.makedirs(folder)
                
            filename = os.path.join(folder, f"screenshot_{timestamp}.jpg")
            cv2.imwrite(filename, self.current_frame)
            self.statusBar().showMessage(f"Screenshot saved: {filename}")
    
    def update_frame(self, frame, position, confidence):
        """Update the video display with the latest frame."""
        self.current_frame = frame
        
        # Add position label to the frame
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        
        position_text = POSITION_LABELS[position]
        if position == 0:  # Safe
            color = (0, 255, 0)  # Green
        else:  # Unsafe
            color = (0, 0, 255)  # Red
        
        cv2.putText(frame, position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, color, 2)
        
        # Add confidence level if available
        if confidence > 0:
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (w-200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Convert to RGB for Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        
        # Convert to QImage
        qt_image = QImage(rgb_frame.data, w, h, w * ch, QImage.Format_RGB888)
        
        # Resize to fit the label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(), 
            self.video_label.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
        
        # Update position status
        self.position_label.setText(f"Status: {position_text}")
        if position == 0:  # Safe
            self.position_label.setStyleSheet("color: green; font-weight: bold;")
        else:  # Unsafe
            self.position_label.setStyleSheet("color: red; font-weight: bold;")
        
        # Update position history for trend analysis
        self.position_history.append(position)
        if len(self.position_history) > 100:
            self.position_history.pop(0)
    
    def handle_alert(self, frame, position):
        """Handle an alert for unsafe position."""
        self.alert_count += 1
        self.alert_label.setText(f"Alerts: {self.alert_count}")
        
        # Save alert screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.alert_folder.text()
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        filename = os.path.join(folder, f"alert_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        
        # Display alert message
        alert_msg = QMessageBox(self)
        alert_msg.setIcon(QMessageBox.Warning)
        alert_msg.setWindowTitle("Position Alert")
        alert_msg.setText(f"ALERT: Baby in unsafe position: {POSITION_LABELS[position]}")
        alert_msg.setInformativeText("Baby should be on their back for safe sleep.")
        alert_msg.setStandardButtons(QMessageBox.Ok)
        
        # Show the message without blocking (will autoclose after 5 seconds)
        def auto_close():
            alert_msg.done(0)
        
        QTimer.singleShot(5000, auto_close)
        alert_msg.show()
    
    def handle_error(self, error_message):
        """Handle errors from the video thread."""
        QMessageBox.critical(self, "Error", error_message)
        self.statusBar().showMessage(f"Error: {error_message}")
        
        # Stop monitoring
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        
        self.start_btn.setText("Start Monitoring")
        self.pause_btn.setEnabled(False)
        self.screenshot_btn.setEnabled(False)
    
    def update_status(self):
        """Update status and UI elements periodically."""
        # Check if source is selected
        self.start_btn.setEnabled(self.source_combo.currentIndex() > 0)
        
        # Update source selection if a camera was selected
        index = self.source_combo.currentIndex()
        if index > 0:
            source = self.source_combo.itemData(index)
            if isinstance(source, str) and os.path.exists(source):
                # File exists, keep it enabled
                pass
            elif source in self.camera_sources:
                # Camera exists, keep it enabled
                pass
            else:
                # Invalid source, reset selection
                self.source_combo.setCurrentIndex(0)
                if self.video_thread and self.video_thread.isRunning():
                    self.toggle_monitoring()
    
    def load_settings(self):
        """Load saved settings."""
        settings = QSettings("BabyMonitor", "PositionMonitor")
        alert_folder = settings.value("alert_folder", "position_alerts")
        self.alert_folder.setText(alert_folder)
        
        use_mediapipe = settings.value("use_mediapipe", MEDIAPIPE_AVAILABLE, type=bool)
        if MEDIAPIPE_AVAILABLE:
            self.method_checkbox.setChecked(use_mediapipe)
    
    def save_settings(self):
        """Save current settings."""
        settings = QSettings("BabyMonitor", "PositionMonitor")
        settings.setValue("alert_folder", self.alert_folder.text())
        settings.setValue("use_mediapipe", self.method_checkbox.isChecked())
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop the video thread
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        
        # Save settings
        self.save_settings()
        
        event.accept()

def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for a modern look
    
    # Set application palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 245, 245))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    app.setPalette(palette)
    
    # Create and show the main window
    window = PositionMonitorGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 