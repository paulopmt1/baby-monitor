#!/bin/bash
# Installation script for Baby Heart Rate Monitor

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 is required but not installed. Please install Python 3 and try again."
    exit 1
fi

# Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Display memory advice
echo ""
echo "===== Memory Usage Advice ====="
echo "This application processes video files which can be memory-intensive."
echo "For best performance, consider the following tips:"
echo ""
echo "1. Use smaller videos (under 30 seconds) when possible"
echo "2. Select a small region of interest (ROI) on the baby's skin"
echo "3. Use the default pyramid levels (3) or lower if you experience memory issues"
echo "4. Close other memory-intensive applications while running this software"
echo "5. If processing fails, try reducing the video resolution"
echo ""

echo "Installation complete!"
echo ""
echo "To use the application:"
echo "1. Activate the virtual environment with: source venv/bin/activate"
echo "2. Run the GUI application with: python run_gui.py"
echo "3. Or use the command line version: python detect_heart_rate.py <video_path>"
echo ""
echo "Thank you for installing Baby Heart Rate Monitor!" 