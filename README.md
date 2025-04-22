# Baby Heart Rate Monitor

This application uses Eulerian Video Magnification to detect a baby's heart rate from a video without any physical contact.

## How It Works

The software analyzes subtle color changes in the baby's skin that correspond to blood flow. These changes are typically invisible to the naked eye, but can be captured by a camera and amplified using signal processing techniques.

## Features

- Load and process videos of babies
- Select region of interest for heart rate monitoring
- Visualize the amplified signal
- Calculate and display heart rate in beats per minute (BPM)
- Save results to a log file
- Memory-optimized processing for larger videos
- Background processing to keep the UI responsive

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- SciPy
- Matplotlib
- scikit-image
- PyQt5

## Installation

1. Clone this repository
2. Run the installation script:
   ```
   chmod +x install.sh
   ./install.sh
   ```
   
   Or manually install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the GUI application with:

```
python run_gui.py
```

Or use the command-line version:

```
python detect_heart_rate.py <video_path>
```

## Step-by-Step Guide

1. **Load a video file** through the interface
2. **Select a region of interest (ROI)** on the baby's skin (face or exposed area)
   - This is important for both accuracy and memory optimization
   - Try to select a small area with good blood flow visibility
3. **Start the analysis** by clicking "Analyze Video"
4. **View the calculated heart rate** and signal visualization
5. **Save the results** if desired

## Memory Optimization

This application processes video frames using a multi-level image pyramid, which can be memory-intensive for large videos. For best performance:

1. Use shorter video clips (under 30 seconds)
2. Select a small region of interest (ROI)
3. Adjust the pyramid levels if needed (3 is the default, use lower for smaller ROIs)
4. Close other memory-intensive applications

See the [Troubleshooting Guide](TROUBLESHOOTING.md) for more detailed advice.

## References

Based on the Eulerian Video Magnification technique:
- https://medium.com/augmented-startups/heart-rate-detection-using-eulerian-video-magnification-yolor-49818dd1b2f5
- https://github.com/SkySingh04/Motion-Amplification-Video

## License

MIT License

## Graphical User Interface (GUI)

### Respiration Rate Monitor GUI

A graphical user interface is available to make the respiration rate detection more user-friendly. The GUI allows you to:

- Load video files for analysis
- Select a specific region of interest (ROI) in the video
- Adjust analysis parameters like frequency ranges and filter settings
- View results including respiration rate and breathing pattern graph
- Save analysis results

To run the GUI application:

```bash
python run_respiration_gui.py
```

#### Instructions for using the Respiration Monitor GUI:

1. Click "Load Video" to select a video file for analysis
2. Select a region of interest (ROI) by clicking and dragging on the displayed frame
3. Adjust any analysis parameters as needed
4. Click "Start Analysis" to begin processing
5. View the results in the application window
6. Optionally save the results by clicking "Save Results"

The GUI displays real-time progress during analysis and shows the final respiratory rate in breaths per minute, along with a visualization of the breathing pattern.

## Baby Position Monitor

The Baby Position Monitor is a safety feature that analyzes video to detect if a baby is sleeping in an unsafe position. For safety, babies should always sleep on their backs (supine position) to prevent Sudden Infant Death Syndrome (SIDS).

### Features

- Real-time monitoring of baby sleeping position
- Alerts when baby is detected in unsafe positions (on side or stomach)
- Uses computer vision and optional pose detection to analyze baby position
- Saves screenshots of detected unsafe positions with timestamps
- Works with webcams or video files

### Usage

Start the baby position monitoring with:

```bash
python run_position_gui.py
```

Or use the command line version:

```bash
python detect_baby_position.py --source <webcam_number_or_video_path>
```

### How to Use

1. Select a video source (webcam or video file)
2. Click "Start Monitoring"
3. The system will analyze the video in real-time
4. If an unsafe position is detected, an alert will be shown and saved
5. Screenshots are saved to the "position_alerts" folder by default

For more accurate detection, enable MediaPipe (requires the mediapipe package). 