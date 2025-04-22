import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QSlider, QSpinBox, QDoubleSpinBox, QGroupBox, 
                           QFrame, QSplitter, QSizePolicy, QComboBox,
                           QStatusBar, QMessageBox, QGridLayout, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import gc

from eulerian_magnification import EulerianMagnification

class HeartRateCanvas(FigureCanvas):
    """Canvas for plotting heart rate data."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes1 = self.fig.add_subplot(211)  # Raw signal plot
        self.axes2 = self.fig.add_subplot(212)  # Frequency plot
        
        self.fig.tight_layout()
        
        super(HeartRateCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self,
                                  QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
    
    def plot_data(self, time, raw_signal, freqs, power, heart_rate):
        """Plot heart rate data on the canvas."""
        # Clear previous plots
        self.axes1.clear()
        self.axes2.clear()
        
        # Plot raw signal
        self.axes1.plot(time, raw_signal)
        self.axes1.set_title('Raw Signal Over Time')
        self.axes1.set_xlabel('Time (s)')
        self.axes1.set_ylabel('Amplitude')
        
        # Plot power spectrum
        self.axes2.plot(freqs, power)
        if heart_rate > 0:
            self.axes2.axvline(x=heart_rate/60, color='r', linestyle='--')
        self.axes2.set_title(f'Power Spectrum (Heart Rate: {heart_rate:.1f} BPM)')
        self.axes2.set_xlabel('Frequency (Hz)')
        self.axes2.set_ylabel('Power')
        
        self.fig.tight_layout()
        self.draw()

class VideoDisplay(QLabel):
    """Widget for displaying video frames with ROI selection."""
    def __init__(self, parent=None):
        super(VideoDisplay, self).__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # ROI selection variables
        self.roi_start = None
        self.roi_end = None
        self.selecting_roi = False
        self.roi = None
        
        # Default background
        self.setStyleSheet("QLabel { background-color: black; color: white; }")
        self.setText("No video loaded")
    
    def mousePressEvent(self, event):
        """Handle mouse press events for ROI selection."""
        if self.pixmap() is not None:
            self.roi_start = (event.x(), event.y())
            self.selecting_roi = True
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for ROI selection."""
        if self.selecting_roi and self.pixmap() is not None:
            self.roi_end = (event.x(), event.y())
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events for ROI selection."""
        if self.selecting_roi and self.pixmap() is not None:
            self.roi_end = (event.x(), event.y())
            
            # Calculate ROI coordinates
            x = min(self.roi_start[0], self.roi_end[0])
            y = min(self.roi_start[1], self.roi_end[1])
            w = abs(self.roi_start[0] - self.roi_end[0])
            h = abs(self.roi_start[1] - self.roi_end[1])
            
            # Get the pixmap size
            pixmap_size = self.pixmap().size()
            
            # Clip ROI to the pixmap bounds
            x = max(0, min(x, pixmap_size.width()))
            y = max(0, min(y, pixmap_size.height()))
            w = max(10, min(w, pixmap_size.width() - x))
            h = max(10, min(h, pixmap_size.height() - y))
            
            self.roi = (x, y, w, h)
            self.selecting_roi = False
            self.update()
    
    def paintEvent(self, event):
        """Custom paint event to draw ROI rectangle."""
        super(VideoDisplay, self).paintEvent(event)
        
        if self.pixmap() is not None:
            from PyQt5.QtGui import QPainter, QPen
            from PyQt5.QtCore import QRect
            
            painter = QPainter(self)
            
            # Draw ROI rectangle if selecting or already selected
            if self.selecting_roi and self.roi_start and self.roi_end:
                # Draw rectangle during selection
                pen = QPen(Qt.green, 2, Qt.SolidLine)
                painter.setPen(pen)
                
                x = min(self.roi_start[0], self.roi_end[0])
                y = min(self.roi_start[1], self.roi_end[1])
                w = abs(self.roi_start[0] - self.roi_end[0])
                h = abs(self.roi_start[1] - self.roi_end[1])
                
                painter.drawRect(QRect(x, y, w, h))
            elif self.roi:
                # Draw rectangle for existing ROI
                pen = QPen(Qt.green, 2, Qt.SolidLine)
                painter.setPen(pen)
                
                x, y, w, h = self.roi
                painter.drawRect(QRect(x, y, w, h))

class ProcessingThread(QThread):
    """Thread for processing video without freezing the UI."""
    progress_update = pyqtSignal(str)
    processing_complete = pyqtSignal(tuple)
    processing_error = pyqtSignal(str)
    
    def __init__(self, video_path, roi, low_cutoff, high_cutoff, amplification, pyramid_levels):
        super(ProcessingThread, self).__init__()
        self.video_path = video_path
        self.roi = roi
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.amplification = amplification
        self.pyramid_levels = pyramid_levels
        self.processor = None
    
    def run(self):
        try:
            # Create processor
            self.processor = EulerianMagnification(
                video_path=self.video_path,
                low_cutoff=self.low_cutoff,
                high_cutoff=self.high_cutoff,
                amplification=self.amplification,
                pyramid_levels=self.pyramid_levels
            )
            
            # Set the progress update callback
            self.progress_update.emit("Initializing processor...")
            
            # Set ROI if selected
            if self.roi is not None:
                self.processor.set_roi(*self.roi)
                self.progress_update.emit(f"ROI set to {self.roi}")
            
            # Process video
            self.progress_update.emit("Processing video... This may take a while.")
            frames, _ = self.processor.process_video(display=False)
            
            # Calculate heart rate
            self.progress_update.emit("Calculating heart rate...")
            heart_rate, freqs, power = self.processor.calculate_heart_rate()
            
            # Emit signal with results
            self.processing_complete.emit((self.processor, frames, heart_rate, freqs, power))
            
        except Exception as e:
            self.processing_error.emit(str(e))
            # Cleanup in case of error
            if self.processor is not None:
                del self.processor
            gc.collect()

class BabyHeartRateMonitor(QMainWindow):
    """Main application window for Baby Heart Rate Monitor."""
    def __init__(self):
        super(BabyHeartRateMonitor, self).__init__()
        
        self.setWindowTitle("Baby Heart Rate Monitor")
        self.setMinimumSize(1000, 700)
        
        # Initialize variables
        self.video_path = None
        self.cap = None
        self.current_frame = None
        self.processor = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.processing_thread = None
        
        # Create main widget and layout
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        
        # Create top layout for video controls
        self.top_layout = QHBoxLayout()
        
        # Create video display
        self.video_display = VideoDisplay()
        
        # Create control panel
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        
        # Create file selection group
        self.file_group = QGroupBox("Video Source")
        self.file_layout = QVBoxLayout(self.file_group)
        
        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        
        self.file_layout.addWidget(self.load_button)
        self.file_layout.addWidget(self.file_label)
        
        # Create parameters group
        self.params_group = QGroupBox("Analysis Parameters")
        self.params_layout = QGridLayout(self.params_group)
        
        self.lower_freq_label = QLabel("Min Heart Rate (BPM):")
        self.lower_freq_spin = QSpinBox()
        self.lower_freq_spin.setRange(30, 100)
        self.lower_freq_spin.setValue(50)
        
        self.upper_freq_label = QLabel("Max Heart Rate (BPM):")
        self.upper_freq_spin = QSpinBox()
        self.upper_freq_spin.setRange(100, 250)
        self.upper_freq_spin.setValue(150)
        
        self.amplification_label = QLabel("Amplification:")
        self.amplification_spin = QSpinBox()
        self.amplification_spin.setRange(10, 200)
        self.amplification_spin.setValue(50)
        
        self.pyramid_label = QLabel("Pyramid Levels:")
        self.pyramid_spin = QSpinBox()
        self.pyramid_spin.setRange(1, 6)
        self.pyramid_spin.setValue(3)  # Use a lower default value
        
        self.params_layout.addWidget(self.lower_freq_label, 0, 0)
        self.params_layout.addWidget(self.lower_freq_spin, 0, 1)
        self.params_layout.addWidget(self.upper_freq_label, 1, 0)
        self.params_layout.addWidget(self.upper_freq_spin, 1, 1)
        self.params_layout.addWidget(self.amplification_label, 2, 0)
        self.params_layout.addWidget(self.amplification_spin, 2, 1)
        self.params_layout.addWidget(self.pyramid_label, 3, 0)
        self.params_layout.addWidget(self.pyramid_spin, 3, 1)
        
        # Create ROI group
        self.roi_group = QGroupBox("Region of Interest")
        self.roi_layout = QVBoxLayout(self.roi_group)
        
        self.roi_info_label = QLabel("Drag on the video to select a region of interest")
        self.roi_clear_button = QPushButton("Clear ROI")
        self.roi_clear_button.clicked.connect(self.clear_roi)
        
        self.roi_layout.addWidget(self.roi_info_label)
        self.roi_layout.addWidget(self.roi_clear_button)
        
        # Create processing group
        self.processing_group = QGroupBox("Processing")
        self.processing_layout = QVBoxLayout(self.processing_group)
        
        self.process_button = QPushButton("Analyze Video")
        self.process_button.clicked.connect(self.process_video)
        self.process_button.setEnabled(False)
        
        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.hide()
        
        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.hide()
        
        self.processing_layout.addWidget(self.process_button)
        self.processing_layout.addWidget(self.save_button)
        self.processing_layout.addWidget(self.progress_bar)
        self.processing_layout.addWidget(self.progress_label)
        
        # Add groups to control layout
        self.control_layout.addWidget(self.file_group)
        self.control_layout.addWidget(self.params_group)
        self.control_layout.addWidget(self.roi_group)
        self.control_layout.addWidget(self.processing_group)
        self.control_layout.addStretch()
        
        # Add widgets to top layout
        self.top_layout.addWidget(self.video_display, 3)
        self.top_layout.addWidget(self.control_panel, 1)
        
        # Create bottom layout for results
        self.bottom_layout = QVBoxLayout()
        
        # Create plots widget
        self.results_group = QGroupBox("Analysis Results")
        self.results_layout = QVBoxLayout(self.results_group)
        
        self.heart_rate_label = QLabel("Heart Rate: -- BPM")
        self.heart_rate_label.setAlignment(Qt.AlignCenter)
        self.heart_rate_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        
        self.canvas = HeartRateCanvas(self.results_group, width=10, height=4)
        
        self.results_layout.addWidget(self.heart_rate_label)
        self.results_layout.addWidget(self.canvas)
        
        # Add layouts to main layout
        self.main_layout.addLayout(self.top_layout, 3)
        self.main_layout.addWidget(self.results_group, 2)
        
        # Set the central widget
        self.setCentralWidget(self.main_widget)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Disable results panel initially
        self.results_group.setEnabled(False)
    
    def load_video(self):
        """Open a file dialog to load a video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_path:
            # Close any existing video
            if self.cap is not None:
                self.cap.release()
            
            # Open the new video
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", f"Could not open video file: {file_path}")
                return
            
            # Update UI
            self.video_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.status_bar.showMessage(f"Loaded video: {os.path.basename(file_path)}")
            
            # Enable process button
            self.process_button.setEnabled(True)
            
            # Display the first frame
            ret, self.current_frame = self.cap.read()
            if ret:
                self.display_frame(self.current_frame)
            
            # Start timer to show video preview
            self.timer.start(30)  # Update every 30ms
            
            # Clear ROI
            self.clear_roi()
            
            # Reset results
            self.heart_rate_label.setText("Heart Rate: -- BPM")
            self.results_group.setEnabled(False)
            self.save_button.setEnabled(False)
            
            # Check video properties and warn if too large
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if frame_count > 1000 or (width * height > 1000000):
                QMessageBox.warning(
                    self, 
                    "Large Video", 
                    "This video is quite large which may require significant memory and processing time. "
                    "Consider selecting a smaller region of interest (ROI) to reduce processing time."
                )
    
    def update_frame(self):
        """Update the video display with the next frame."""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.display_frame(frame)
            else:
                # Restart video when it ends
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def display_frame(self, frame):
        """Display a frame on the video display widget."""
        # Convert the OpenCV BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create QImage from the frame
        h, w, c = rgb_frame.shape
        bytes_per_line = c * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Create and set pixmap from QImage
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale pixmap to fit the label while maintaining aspect ratio
        pixmap = pixmap.scaled(self.video_display.size(), 
                              Qt.KeepAspectRatio, 
                              Qt.SmoothTransformation)
        
        self.video_display.setPixmap(pixmap)
    
    def clear_roi(self):
        """Clear the region of interest selection."""
        self.video_display.roi = None
        self.video_display.roi_start = None
        self.video_display.roi_end = None
        self.video_display.update()
        self.roi_info_label.setText("Drag on the video to select a region of interest")
    
    def update_progress(self, message):
        """Update processing progress message."""
        self.progress_label.setText(message)
        self.status_bar.showMessage(message)
        QApplication.processEvents()
    
    def processing_finished(self, results):
        """Handle completion of video processing."""
        self.processor, frames, heart_rate, freqs, power = results
        
        # Update UI with results
        self.heart_rate_label.setText(f"Heart Rate: {heart_rate:.1f} BPM")
        
        # Plot results
        time = np.arange(len(self.processor.raw_signal)) / self.processor.fps
        self.canvas.plot_data(time, self.processor.raw_signal, freqs, power, heart_rate)
        
        # Enable results panel and save button
        self.results_group.setEnabled(True)
        self.save_button.setEnabled(True)
        
        # Hide progress indicators
        self.progress_bar.hide()
        self.progress_label.hide()
        
        # Restart video preview
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.timer.start(30)
        
        # Update status
        self.status_bar.showMessage(f"Analysis complete. Heart Rate: {heart_rate:.1f} BPM")
        
        # Re-enable UI elements
        self.process_button.setEnabled(True)
        self.load_button.setEnabled(True)
        
        # Clean up thread
        self.processing_thread = None
        gc.collect()
    
    def processing_error(self, error_message):
        """Handle processing errors."""
        QMessageBox.critical(self, "Error", f"Error processing video: {error_message}")
        self.status_bar.showMessage("Error processing video")
        
        # Hide progress indicators
        self.progress_bar.hide()
        self.progress_label.hide()
        
        # Re-enable UI elements
        self.process_button.setEnabled(True)
        self.load_button.setEnabled(True)
        
        # Clean up thread
        self.processing_thread = None
        gc.collect()
    
    def process_video(self):
        """Process the video using Eulerian magnification."""
        if self.video_path is None:
            QMessageBox.warning(self, "Warning", "Please load a video first")
            return
        
        # Stop the preview timer
        self.timer.stop()
        
        # Update UI
        self.process_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.progress_bar.show()
        self.progress_label.show()
        self.update_progress("Preparing to process video...")
        
        try:
            # Get parameters
            low_cutoff = self.lower_freq_spin.value() / 60.0  # Convert BPM to Hz
            high_cutoff = self.upper_freq_spin.value() / 60.0  # Convert BPM to Hz
            amplification = self.amplification_spin.value()
            pyramid_levels = self.pyramid_spin.value()
            
            # Scale ROI if selected
            roi = None
            if self.video_display.roi is not None:
                # Get current ROI from display
                display_roi = self.video_display.roi
                
                # Scale ROI to actual video size
                pixmap_size = self.video_display.pixmap().size()
                
                # Get video dimensions
                video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                scale_x = video_width / pixmap_size.width()
                scale_y = video_height / pixmap_size.height()
                
                x = int(display_roi[0] * scale_x)
                y = int(display_roi[1] * scale_y)
                w = int(display_roi[2] * scale_x)
                h = int(display_roi[3] * scale_y)
                
                roi = (x, y, w, h)
                self.roi_info_label.setText(f"ROI: (x={x}, y={y}, w={w}, h={h})")
            
            # Create and start the processing thread
            self.processing_thread = ProcessingThread(
                video_path=self.video_path,
                roi=roi,
                low_cutoff=low_cutoff,
                high_cutoff=high_cutoff,
                amplification=amplification,
                pyramid_levels=pyramid_levels
            )
            
            # Connect signals
            self.processing_thread.progress_update.connect(self.update_progress)
            self.processing_thread.processing_complete.connect(self.processing_finished)
            self.processing_thread.processing_error.connect(self.processing_error)
            
            # Start processing in the background
            self.processing_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error setting up processing: {str(e)}")
            self.status_bar.showMessage("Error processing video")
            
            # Hide progress indicators
            self.progress_bar.hide()
            self.progress_label.hide()
            
            # Re-enable UI elements
            self.process_button.setEnabled(True)
            self.load_button.setEnabled(True)
    
    def save_results(self):
        """Save the processed video and results."""
        if self.processor is None:
            return
        
        try:
            # Open save dialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Results", "", "Video Files (*.avi);;All Files (*)"
            )
            
            if file_path:
                # Ensure file has .avi extension
                if not file_path.endswith('.avi'):
                    file_path += '.avi'
                
                # Show progress
                self.progress_bar.show()
                self.progress_label.show()
                self.update_progress("Processing video for saving... This may take a while.")
                
                # Create a separate thread for saving
                class SaveThread(QThread):
                    save_complete = pyqtSignal(str)
                    save_error = pyqtSignal(str)
                    
                    def __init__(self, processor, file_path, parent=None):
                        super(SaveThread, self).__init__(parent)
                        self.processor = processor
                        self.file_path = file_path
                    
                    def run(self):
                        try:
                            # Process the video again to generate frames
                            frames, _ = self.processor.process_video(display=False)
                            
                            # Save results
                            self.processor.save_results(self.file_path, frames)
                            
                            # Save heart rate plot
                            plot_path = self.file_path.rsplit('.', 1)[0] + '_plot.png'
                            self.save_complete.emit(self.file_path)
                        except Exception as e:
                            self.save_error.emit(str(e))
                
                # Create and start the save thread
                save_thread = SaveThread(self.processor, file_path)
                
                def on_save_complete(path):
                    self.progress_bar.hide()
                    self.progress_label.hide()
                    self.status_bar.showMessage(f"Results saved to {path}")
                    # Save plot separately
                    plot_path = path.rsplit('.', 1)[0] + '_plot.png'
                    self.canvas.fig.savefig(plot_path, dpi=150)
                
                def on_save_error(error):
                    self.progress_bar.hide()
                    self.progress_label.hide()
                    QMessageBox.critical(self, "Error", f"Error saving results: {error}")
                    self.status_bar.showMessage("Error saving results")
                
                save_thread.save_complete.connect(on_save_complete)
                save_thread.save_error.connect(on_save_error)
                save_thread.start()
                
        except Exception as e:
            self.progress_bar.hide()
            self.progress_label.hide()
            QMessageBox.critical(self, "Error", f"Error saving results: {str(e)}")
            self.status_bar.showMessage("Error saving results")
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Clean up
        if self.cap is not None:
            self.cap.release()
        
        # Stop timer
        self.timer.stop()
        
        # Stop any running threads
        if self.processing_thread is not None and self.processing_thread.isRunning():
            self.processing_thread.terminate()
            self.processing_thread.wait()
        
        # Accept the close event
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BabyHeartRateMonitor()
    window.show()
    sys.exit(app.exec_()) 