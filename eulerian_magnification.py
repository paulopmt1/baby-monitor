import numpy as np
import cv2
from scipy.fftpack import fft, ifft
from scipy import signal
import matplotlib.pyplot as plt
from skimage.util import img_as_float, img_as_ubyte
import gc

class EulerianMagnification:
    def __init__(self, 
                 video_path, 
                 low_cutoff=0.83,     # 50 BPM
                 high_cutoff=2.5,     # 150 BPM  
                 amplification=50,    # Amplification factor
                 pyramid_levels=4):   # Levels in Gaussian pyramid
        """
        Initialize the Eulerian Video Magnification processor.
        
        Parameters:
        - video_path: Path to the video file to process
        - low_cutoff: Lower bound of frequency range to amplify (in Hz)
        - high_cutoff: Upper bound of frequency range to amplify (in Hz)
        - amplification: Amplification factor
        - pyramid_levels: Number of levels in the Laplacian pyramid
        """
        self.video_path = video_path
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.amplification = amplification
        self.pyramid_levels = pyramid_levels
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize temporal filter
        self._init_filters()
        
        # Store the region of interest
        self.roi = None
        
        # Store raw signal for heart rate calculation
        self.raw_signal = []
    
    def _init_filters(self):
        """Initialize bandpass filter based on cutoff frequencies."""
        nyquist = self.fps / 2.0
        self.b, self.a = signal.butter(2, 
                                      [self.low_cutoff / nyquist, 
                                       self.high_cutoff / nyquist], 
                                      btype='bandpass')
    
    def set_roi(self, x, y, width, height):
        """Set region of interest for processing."""
        self.roi = (x, y, width, height)
    
    def build_gaussian_pyramid(self, frame, levels):
        """Build a Gaussian pyramid with the specified number of levels."""
        pyramid = [frame]
        current_frame = frame.copy()
        
        for i in range(levels):
            # Check if the current frame is too small to downsample further
            if current_frame.shape[0] < 2 or current_frame.shape[1] < 2:
                break
                
            current_frame = cv2.pyrDown(current_frame)
            pyramid.append(current_frame)
        
        return pyramid
    
    def build_laplacian_pyramid(self, frame, levels):
        """Build a Laplacian pyramid with the specified number of levels."""
        # First build Gaussian pyramid
        gaussian_pyramid = self.build_gaussian_pyramid(frame, levels)
        
        # The actual number of levels might be less than requested
        actual_levels = len(gaussian_pyramid) - 1
        
        laplacian_pyramid = []
        
        for i in range(actual_levels):
            # Get Gaussian level
            gaussian_i = gaussian_pyramid[i]
            
            # Get next level and upscale it
            next_gaussian = gaussian_pyramid[i + 1]
            up_next_gaussian = cv2.pyrUp(next_gaussian, dstsize=(gaussian_i.shape[1], gaussian_i.shape[0]))
            
            # Compute Laplacian level
            laplacian = cv2.subtract(gaussian_i, up_next_gaussian)
            laplacian_pyramid.append(laplacian)
        
        # Add the smallest Gaussian as the last Laplacian level
        laplacian_pyramid.append(gaussian_pyramid[-1])
        
        return laplacian_pyramid
    
    def collapse_laplacian_pyramid(self, pyramid):
        """Collapse a Laplacian pyramid back to a full-resolution image."""
        reconstructed = pyramid[-1]
        for i in range(len(pyramid) - 2, -1, -1):
            # Upscale the current reconstructed image
            upscaled = cv2.pyrUp(reconstructed, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
            # Add the current level of the Laplacian pyramid
            reconstructed = cv2.add(upscaled, pyramid[i])
        
        return reconstructed
    
    def apply_temporal_filter(self, input_signal):
        """Apply bandpass filter to the signal."""
        return signal.filtfilt(self.b, self.a, input_signal, axis=0, padtype='odd')
    
    def determine_pyramid_levels(self, frame_shape):
        """Determine the appropriate number of pyramid levels based on frame size."""
        min_dimension = min(frame_shape[0], frame_shape[1])
        
        # Calculate how many times we can divide by 2 until we reach a minimum size (e.g., 16 pixels)
        max_levels = 0
        size = min_dimension
        while size >= 16:  # Minimum size threshold
            size = size // 2
            max_levels += 1
        
        # Use the smaller of the requested levels or calculated max levels
        return min(self.pyramid_levels, max_levels)
    
    def process_video(self, roi=None, display=False):
        """
        Process the video using Eulerian Video Magnification.
        
        Parameters:
        - roi: Region of interest (x, y, width, height) or None for full frame
        - display: Whether to display the processed video
        
        Returns:
        - Processed frames and raw signal
        """
        if roi:
            self.roi = roi
        
        # Reset to the beginning of the video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Read the first frame
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to read the first frame")
        
        if self.roi:
            x, y, w, h = self.roi
            frame_roi = frame[y:y+h, x:x+w]
        else:
            frame_roi = frame
        
        # Convert to floating point for better precision
        frame_roi = img_as_float(frame_roi)
        
        # Determine appropriate pyramid levels based on the frame size
        pyramid_levels = self.determine_pyramid_levels(frame_roi.shape)
        print(f"Using {pyramid_levels} pyramid levels based on frame size")
        
        # Build the first Laplacian pyramid
        first_pyramid = self.build_laplacian_pyramid(frame_roi, pyramid_levels)
        
        # We process the video in chunks to reduce memory usage
        chunk_size = min(100, self.frame_count)  # Process 100 frames at a time
        num_chunks = (self.frame_count + chunk_size - 1) // chunk_size
        
        # Create storage for the raw signal
        self.raw_signal = []
        
        # Storage for processed frames
        reconstructed_frames = []
        
        # Process video in chunks
        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * chunk_size
            end_frame = min(start_frame + chunk_size, self.frame_count)
            current_chunk_size = end_frame - start_frame
            
            print(f"Processing chunk {chunk_idx+1}/{num_chunks} (frames {start_frame+1}-{end_frame})")
            
            # Set video position to start of chunk
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Create arrays to store the pyramid and filtered pyramid for each level for this chunk
            pyramid_history = [np.zeros((current_chunk_size, *level.shape)) for level in first_pyramid]
            
            # Read and process frames for this chunk
            for frame_idx in range(current_chunk_size):
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                if self.roi:
                    x, y, w, h = self.roi
                    frame_roi = frame[y:y+h, x:x+w]
                else:
                    frame_roi = frame
                
                # Convert to floating point
                frame_roi = img_as_float(frame_roi)
                
                # Build Laplacian pyramid for this frame
                pyramid = self.build_laplacian_pyramid(frame_roi, pyramid_levels)
                
                # Store pyramid in history
                for i, level in enumerate(pyramid):
                    pyramid_history[i][frame_idx] = level
            
            # Apply temporal filtering to each pyramid level in this chunk
            filtered_pyramids = []
            for level_idx in range(len(first_pyramid)):
                # Get all frames for this level
                level_data = pyramid_history[level_idx]
                
                # Reshape to 2D array for filtering
                original_shape = level_data.shape
                reshaped_data = level_data.reshape(original_shape[0], -1)
                
                # Apply temporal filter
                filtered_data = self.apply_temporal_filter(reshaped_data)
                
                # Reshape back to original shape
                filtered_data = filtered_data.reshape(original_shape)
                
                # Store filtered data
                filtered_pyramids.append(filtered_data)
            
            # Reset video capture for reconstruction of this chunk
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Reconstruct each frame in the chunk
            for i in range(current_chunk_size):
                # Read original frame
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Extract ROI if specified
                if self.roi:
                    x, y, w, h = self.roi
                    orig_roi = frame[y:y+h, x:x+w].copy()
                else:
                    orig_roi = frame.copy()
                
                # Build amplified pyramid for this frame
                amplified_pyramid = []
                for level_idx in range(len(first_pyramid)):
                    # Get filtered data for this level and frame
                    filtered_level = filtered_pyramids[level_idx][i]
                    
                    # Amplify
                    amplified_level = filtered_level * self.amplification
                    
                    # Add to original pyramid
                    original_level = pyramid_history[level_idx][i]
                    combined_level = original_level + amplified_level
                    
                    amplified_pyramid.append(combined_level)
                
                # Collapse pyramid to get amplified frame
                amplified_frame = self.collapse_laplacian_pyramid(amplified_pyramid)
                
                # Clip values to valid range [0, 1]
                amplified_frame = np.clip(amplified_frame, 0, 1)
                
                # Convert back to 8-bit
                amplified_frame = img_as_ubyte(amplified_frame)
                
                # Calculate and store the average pixel value for heart rate detection
                self.raw_signal.append(np.mean(amplified_frame))
                
                # Combine original and amplified frames
                if self.roi:
                    frame_copy = frame.copy()
                    frame_copy[y:y+h, x:x+w] = amplified_frame
                    reconstructed_frames.append(frame_copy)
                    
                    # Draw ROI rectangle
                    cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    reconstructed_frames.append(amplified_frame)
                
                # Display if requested
                if display:
                    if self.roi:
                        cv2.imshow('Original ROI', orig_roi)
                        cv2.imshow('Amplified ROI', amplified_frame)
                        cv2.imshow('Full Frame', frame_copy)
                    else:
                        cv2.imshow('Original', frame)
                        cv2.imshow('Amplified', amplified_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            # Free memory after processing the chunk
            del pyramid_history
            del filtered_pyramids
            gc.collect()
        
        # Release video capture
        self.cap.release()
        cv2.destroyAllWindows()
        
        return reconstructed_frames, np.array(self.raw_signal)
    
    def calculate_heart_rate(self):
        """
        Calculate heart rate from the raw signal.
        
        Returns:
        - heart_rate: Heart rate in beats per minute
        - freqs: Frequency array for plotting
        - power: Power spectrum for plotting
        """
        if len(self.raw_signal) == 0:
            raise ValueError("No signal data available. Run process_video first.")
        
        # Detrend the signal to remove any DC component
        detrended = signal.detrend(self.raw_signal)
        
        # Apply Hanning window to reduce spectral leakage
        windowed = detrended * signal.windows.hann(len(detrended))
        
        # Compute FFT
        fft_result = fft(windowed)
        
        # Get the power spectrum (magnitude squared)
        power = np.abs(fft_result)**2
        
        # Get the corresponding frequencies
        freqs = np.fft.fftfreq(len(power), 1.0/self.fps)
        
        # Only keep the positive frequencies within our range of interest
        idx = np.where((freqs > self.low_cutoff) & (freqs < self.high_cutoff))
        freqs = freqs[idx]
        power = power[idx]
        
        # Find the peak frequency
        if len(power) > 0:
            peak_idx = np.argmax(power)
            peak_freq = freqs[peak_idx]
            
            # Convert frequency to BPM
            heart_rate = peak_freq * 60
            return heart_rate, freqs, power
        else:
            return 0, freqs, power
    
    def plot_signal(self):
        """Plot the raw signal and its power spectrum."""
        if len(self.raw_signal) == 0:
            raise ValueError("No signal data available. Run process_video first.")
        
        heart_rate, freqs, power = self.calculate_heart_rate()
        
        plt.figure(figsize=(12, 6))
        
        # Plot the raw signal
        plt.subplot(2, 1, 1)
        time = np.arange(len(self.raw_signal)) / self.fps
        plt.plot(time, self.raw_signal)
        plt.title('Raw Signal Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Plot the power spectrum
        plt.subplot(2, 1, 2)
        plt.plot(freqs, power)
        plt.axvline(x=heart_rate/60, color='r', linestyle='--')
        plt.title(f'Power Spectrum (Heart Rate: {heart_rate:.1f} BPM)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, output_path, frames=None):
        """
        Save the processed video and heart rate information.
        
        Parameters:
        - output_path: Path to save the processed video
        - frames: Processed frames (if None, use the last processed frames)
        """
        if frames is None:
            raise ValueError("No frames provided to save.")
        
        # Get video properties
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = self.fps
        frame_size = (frames[0].shape[1], frames[0].shape[0])
        
        # Create video writer
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        # Write frames
        for frame in frames:
            out.write(frame)
        
        # Release writer
        out.release()
        
        print(f"Processed video saved to {output_path}")
        
        # Calculate and save heart rate information
        heart_rate, _, _ = self.calculate_heart_rate()
        
        # Save heart rate to text file
        txt_path = output_path.rsplit('.', 1)[0] + '_heartrate.txt'
        with open(txt_path, 'w') as f:
            f.write(f"Video: {self.video_path}\n")
            f.write(f"Heart Rate: {heart_rate:.1f} BPM\n")
            f.write(f"Analysis Parameters:\n")
            f.write(f"- Low Cutoff: {self.low_cutoff} Hz\n")
            f.write(f"- High Cutoff: {self.high_cutoff} Hz\n")
            f.write(f"- Amplification: {self.amplification}\n")
            f.write(f"- FPS: {self.fps}\n")
            if self.roi:
                f.write(f"- ROI: {self.roi}\n")
        
        print(f"Heart rate information saved to {txt_path}")

# Example usage
if __name__ == "__main__":
    # Example usage of the class
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python eulerian_magnification.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    try:
        # Create magnification processor
        processor = EulerianMagnification(
            video_path=video_path,
            low_cutoff=0.83,  # 50 BPM
            high_cutoff=2.5,  # 150 BPM
            amplification=50,
            pyramid_levels=4
        )
        
        # Allow user to select ROI interactively
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            roi = cv2.selectROI("Select ROI", frame, False)
            cv2.destroyWindow("Select ROI")
            
            if roi[2] > 0 and roi[3] > 0:
                processor.set_roi(*roi)
                
                # Process the video
                frames, _ = processor.process_video(display=True)
                
                # Calculate and display heart rate
                heart_rate, _, _ = processor.calculate_heart_rate()
                print(f"Estimated Heart Rate: {heart_rate:.1f} BPM")
                
                # Plot the signal
                processor.plot_signal()
                
                # Save results
                processor.save_results("output_video.avi", frames)
            else:
                print("No ROI selected.")
        
        cap.release()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 