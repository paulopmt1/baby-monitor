#!/usr/bin/env python3
"""
GUI application for detecting respiration (breathing) rate from video.
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QLabel, QFileDialog, QWidget, QProgressBar,
                            QMessageBox, QSlider, QSpinBox, QGroupBox, QGridLayout,
                            QSplitter, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor

from eulerian_magnification import EulerianMagnification
from detect_respiration_rate import detect_respiration_rate

class VideoProcessingThread(QThread):
    """Thread for processing video without blocking the GUI."""
    # Define signals to communicate with the main thread
    progress_update = pyqtSignal(int, str)
    processing_complete = pyqtSignal(dict)
    processing_error = pyqtSignal(str)
    
    def __init__(self, video_path, roi=None, low_cutoff=0.1, high_cutoff=1.0, amplification=20, pyramid_levels=3):
        super().__init__()
        self.video_path = video_path
        self.roi = roi
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.amplification = amplification
        self.pyramid_levels = pyramid_levels
    
    def run(self):
        """Run video processing in a separate thread."""
        try:
            # Get video info
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.processing_error.emit(f"Could not open video file: {self.video_path}")
                return
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Update progress
            self.progress_update.emit(5, "Initializing processor...")
            
            # Initialize the processor
            processor = EulerianMagnification(
                video_path=self.video_path,
                low_cutoff=self.low_cutoff,
                high_cutoff=self.high_cutoff,
                amplification=self.amplification,
                pyramid_levels=self.pyramid_levels
            )
            
            # Set ROI if provided
            if self.roi and self.roi[2] > 0 and self.roi[3] > 0:
                processor.set_roi(*self.roi)
                
            # Hook into processor's progress updates
            def process_chunk_callback(chunk_idx, num_chunks):
                progress = 10 + int(85 * (chunk_idx / num_chunks))  # 10% to 95% progress
                msg = f"Processing video chunk {chunk_idx}/{num_chunks}..."
                self.progress_update.emit(progress, msg)
            
            # Set a callback function
            processor.set_progress_callback = process_chunk_callback
            
            # Process the video
            self.progress_update.emit(10, "Starting video processing...")
            frames, _ = processor.process_video()
            
            # Calculate respiration rate
            self.progress_update.emit(95, "Calculating respiration rate...")
            resp_rate, freqs, power = processor.calculate_heart_rate()
            
            # Save output video
            self.progress_update.emit(98, "Saving results...")
            output_path = os.path.join(os.path.dirname(self.video_path), "output_respiration_video.avi")
            processor.save_results(output_path, frames)
            
            # Complete
            self.progress_update.emit(100, "Processing complete!")
            
            # Return results
            results = {
                'resp_rate': resp_rate,
                'freqs': freqs,
                'power': power,
                'raw_signal': processor.raw_signal,
                'fps': processor.fps,
                'output_path': output_path,
                'roi': processor.roi,
                'low_cutoff': self.low_cutoff,
                'high_cutoff': self.high_cutoff,
                'amplification': self.amplification
            }
            
            self.processing_complete.emit(results)
            
        except MemoryError:
            self.processing_error.emit("Ran out of memory while processing the video. Try using a smaller ROI or a shorter video clip.")
        except Exception as e:
            self.processing_error.emit(f"Error during processing: {str(e)}")


class RoiSelectionWindow(QMainWindow):
    """Window for selecting a region of interest from a video frame."""
    roi_selected = pyqtSignal(tuple)
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.roi = None
        self.selecting = False
        self.start_point = None
        self.current_point = None
        
        self.setWindowTitle("Select Region of Interest")
        
        # Read the first frame
        cap = cv2.VideoCapture(video_path)
        ret, self.frame = cap.read()
        cap.release()
        
        if not ret:
            raise Exception("Could not read the first frame")
        
        # Resize if too large
        h, w = self.frame.shape[:2]
        max_dim = 800
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            self.frame = cv2.resize(self.frame, (int(w * scale), int(h * scale)))
        
        self.h, self.w = self.frame.shape[:2]
        
        # Set fixed window size based on frame
        self.setFixedSize(self.w, self.h)
        
        # Create label to display the image
        self.frame_label = QLabel(self)
        self.frame_label.setGeometry(0, 0, self.w, self.h)
        self.frame_label.setMouseTracking(True)
        self.setMouseTracking(True)
        
        # Show the image
        self.update_display()
        
        # Show instructions
        self.show_instructions()
    
    def show_instructions(self):
        """Show instructions dialog for ROI selection."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("ROI Selection Instructions")
        msg.setText("Select region of interest (ROI)")
        msg.setInformativeText("Click and drag to select the region showing the chest/torso area.\n\n"
                              "Press Enter to confirm selection.\n"
                              "Press ESC to cancel and use full frame.")
        msg.exec_()
    
    def update_display(self):
        """Update the display with the current frame and ROI rectangle."""
        display_img = self.frame.copy()
        
        # Draw ROI rectangle if we're selecting
        if self.selecting and self.start_point and self.current_point:
            cv2.rectangle(display_img, self.start_point, self.current_point, (0, 255, 0), 2)
        
        # Draw final ROI if selected
        elif self.roi:
            x, y, w, h = self.roi
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Convert OpenCV image to Qt format
        rgb_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_image.data, self.w, self.h, self.w * 3, QImage.Format_RGB888)
        self.frame_label.setPixmap(QPixmap.fromImage(qt_image))
    
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.LeftButton:
            self.selecting = True
            self.start_point = (event.x(), event.y())
            self.current_point = self.start_point
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        if self.selecting:
            self.current_point = (event.x(), event.y())
            self.update_display()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.LeftButton and self.selecting:
            self.selecting = False
            # Calculate ROI
            x = min(self.start_point[0], self.current_point[0])
            y = min(self.start_point[1], self.current_point[1])
            w = abs(self.current_point[0] - self.start_point[0])
            h = abs(self.current_point[1] - self.start_point[1])
            
            # Check if ROI is valid
            if w > 10 and h > 10:
                self.roi = (x, y, w, h)
                self.update_display()
                
                # Confirm ROI selection
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Question)
                msg.setWindowTitle("Confirm ROI")
                msg.setText("Use this region of interest?")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                response = msg.exec_()
                
                if response == QMessageBox.Yes:
                    self.roi_selected.emit(self.roi)
                    self.close()
                else:
                    self.roi = None
                    self.update_display()
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Accept current ROI
            if self.roi:
                self.roi_selected.emit(self.roi)
                self.close()
        elif event.key() == Qt.Key_Escape:
            # Cancel selection
            self.roi_selected.emit(None)
            self.close()


class RespirationMonitorApp(QMainWindow):
    """Main application window for respiration monitoring."""
    def __init__(self):
        super().__init__()
        
        self.video_path = None
        self.roi = None
        self.results = None
        self.processing_thread = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Baby Respiration Monitor")
        self.setMinimumSize(800, 600)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create splitter for resizable areas
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # Top area - Controls
        controls_frame = QFrame()
        controls_layout = QVBoxLayout(controls_frame)
        splitter.addWidget(controls_frame)
        
        # File selection
        file_box = QGroupBox("Video Selection")
        file_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setWordWrap(True)
        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(self.load_button)
        file_box.setLayout(file_layout)
        controls_layout.addWidget(file_box)
        
        # Parameter controls
        params_box = QGroupBox("Analysis Parameters")
        params_layout = QGridLayout()
        
        # Low cutoff
        params_layout.addWidget(QLabel("Low cutoff (BPM):"), 0, 0)
        self.low_cutoff_spin = QSpinBox()
        self.low_cutoff_spin.setRange(2, 30)
        self.low_cutoff_spin.setValue(6)  # 0.1 Hz = 6 BPM
        params_layout.addWidget(self.low_cutoff_spin, 0, 1)
        
        # High cutoff
        params_layout.addWidget(QLabel("High cutoff (BPM):"), 0, 2)
        self.high_cutoff_spin = QSpinBox()
        self.high_cutoff_spin.setRange(15, 100)
        self.high_cutoff_spin.setValue(60)  # 1.0 Hz = 60 BPM
        params_layout.addWidget(self.high_cutoff_spin, 0, 3)
        
        # Amplification
        params_layout.addWidget(QLabel("Amplification:"), 1, 0)
        self.amplification_spin = QSpinBox()
        self.amplification_spin.setRange(1, 100)
        self.amplification_spin.setValue(20)
        params_layout.addWidget(self.amplification_spin, 1, 1)
        
        # Pyramid levels
        params_layout.addWidget(QLabel("Pyramid levels:"), 1, 2)
        self.pyramid_spin = QSpinBox()
        self.pyramid_spin.setRange(1, 6)
        self.pyramid_spin.setValue(3)
        params_layout.addWidget(self.pyramid_spin, 1, 3)
        
        # ROI selection
        self.roi_button = QPushButton("Select ROI")
        self.roi_button.clicked.connect(self.select_roi)
        self.roi_button.setEnabled(False)
        params_layout.addWidget(self.roi_button, 2, 0, 1, 2)
        
        # ROI status
        self.roi_status_label = QLabel("ROI: None")
        params_layout.addWidget(self.roi_status_label, 2, 2, 1, 2)
        
        params_box.setLayout(params_layout)
        controls_layout.addWidget(params_box)
        
        # Analysis controls
        analysis_box = QGroupBox("Analysis")
        analysis_layout = QHBoxLayout()
        
        # Start analysis button
        self.analyze_button = QPushButton("Analyze Video")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setEnabled(False)
        self.analyze_button.setStyleSheet("QPushButton { font-weight: bold; }")
        analysis_layout.addWidget(self.analyze_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        analysis_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        analysis_layout.addWidget(self.status_label)
        
        analysis_box.setLayout(analysis_layout)
        controls_layout.addWidget(analysis_box)
        
        # Bottom area - Results
        results_frame = QFrame()
        results_layout = QVBoxLayout(results_frame)
        splitter.addWidget(results_frame)
        
        # Results box
        results_box = QGroupBox("Results")
        results_box_layout = QVBoxLayout()
        
        # Matplotlib figure for plots
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        results_box_layout.addWidget(self.canvas)
        
        # Results info
        self.results_label = QLabel("No results available")
        self.results_label.setAlignment(Qt.AlignCenter)
        self.results_label.setStyleSheet("QLabel { font-size: 14pt; }")
        results_box_layout.addWidget(self.results_label)
        
        results_box.setLayout(results_box_layout)
        results_layout.addWidget(results_box)
        
        # Set splitter default sizes (1:2 ratio)
        splitter.setSizes([200, 400])
        
        # Show the main window
        self.show()
    
    def load_video(self):
        """Open a file dialog to select a video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.file_path_label.setText(os.path.basename(file_path))
            self.roi = None
            self.roi_status_label.setText("ROI: None")
            self.roi_button.setEnabled(True)
            self.analyze_button.setEnabled(True)
            self.status_label.setText("Video loaded. Select ROI or start analysis.")
            self.results = None
            self.update_results_display()
    
    def select_roi(self):
        """Open the ROI selection window."""
        try:
            self.roi_window = RoiSelectionWindow(self.video_path)
            self.roi_window.roi_selected.connect(self.on_roi_selected)
            self.roi_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open video for ROI selection: {str(e)}")
    
    def on_roi_selected(self, roi):
        """Handle ROI selection."""
        self.roi = roi
        if roi:
            self.roi_status_label.setText(f"ROI: {roi[0]},{roi[1]} {roi[2]}x{roi[3]}")
        else:
            self.roi_status_label.setText("ROI: Full frame")
    
    def start_analysis(self):
        """Start video processing in a separate thread."""
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "Please load a video first.")
            return
        
        # Disable controls during processing
        self.analyze_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.roi_button.setEnabled(False)
        
        # Reset progress
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing...")
        
        # Get parameters
        low_cutoff = self.low_cutoff_spin.value() / 60.0  # Convert BPM to Hz
        high_cutoff = self.high_cutoff_spin.value() / 60.0
        amplification = self.amplification_spin.value()
        pyramid_levels = self.pyramid_spin.value()
        
        # Create and start the processing thread
        self.processing_thread = VideoProcessingThread(
            video_path=self.video_path,
            roi=self.roi,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            amplification=amplification,
            pyramid_levels=pyramid_levels
        )
        
        # Connect signals
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.processing_complete.connect(self.on_processing_complete)
        self.processing_thread.processing_error.connect(self.on_processing_error)
        
        # Start thread
        self.processing_thread.start()
    
    def update_progress(self, progress, message):
        """Update progress bar and status message."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def on_processing_complete(self, results):
        """Handle processing completion."""
        self.results = results
        self.update_results_display()
        
        # Re-enable controls
        self.analyze_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.roi_button.setEnabled(True)
        
        self.status_label.setText("Analysis complete!")
        
        # Show success message
        QMessageBox.information(
            self,
            "Analysis Complete",
            f"Respiration rate: {results['resp_rate']:.1f} breaths per minute\n\n"
            f"Processed video saved to:\n{results['output_path']}"
        )
    
    def on_processing_error(self, error_message):
        """Handle processing errors."""
        QMessageBox.critical(self, "Processing Error", error_message)
        
        # Re-enable controls
        self.analyze_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.roi_button.setEnabled(True)
        
        self.progress_bar.setValue(0)
        self.status_label.setText("Error occurred during processing")
    
    def update_results_display(self):
        """Update the results display area."""
        # Clear the figure
        self.figure.clear()
        
        if not self.results:
            self.results_label.setText("No results available")
            self.canvas.draw()
            return
        
        # Get data from results
        resp_rate = self.results['resp_rate']
        freqs = self.results['freqs']
        power = self.results['power']
        raw_signal = self.results['raw_signal']
        fps = self.results['fps']
        
        # Update results label
        self.results_label.setText(f"Respiration Rate: {resp_rate:.1f} breaths per minute")
        
        # Create plots
        # Plot the raw signal
        ax1 = self.figure.add_subplot(2, 1, 1)
        time = np.arange(len(raw_signal)) / fps
        ax1.plot(time, raw_signal)
        ax1.set_title('Breathing Signal Over Time')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        
        # Plot the power spectrum
        ax2 = self.figure.add_subplot(2, 1, 2)
        ax2.plot(freqs, power)
        ax2.axvline(x=resp_rate/60, color='r', linestyle='--')
        ax2.set_title(f'Power Spectrum (Respiration Rate: {resp_rate:.1f} BPM)')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power')
        
        self.figure.tight_layout()
        self.canvas.draw()


def main():
    """Main function to run the respiration monitor GUI."""
    # Handle high DPI displays
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Set application palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    # Start the application
    window = RespirationMonitorApp()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 