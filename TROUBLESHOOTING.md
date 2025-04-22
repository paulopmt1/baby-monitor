# Troubleshooting Guide

This document provides solutions for common issues you might encounter when using the Baby Heart Rate Monitor application.

## Memory Issues

### Symptoms:
- Application freezes or crashes
- Error message about index out of bounds
- Memory error messages
- Computer becomes very slow or unresponsive

### Solutions:
1. **Select a smaller Region of Interest (ROI)** - This is the most effective way to reduce memory usage
2. **Reduce the Pyramid Levels** - Try using 2 or 3 pyramid levels instead of the default 4
3. **Process shorter videos** - Try using clips under 30 seconds
4. **Close other applications** - Free up memory by closing other programs
5. **Use a lower resolution video** - Downscale the video before processing
6. **Try the command-line version** - The command-line tool is more memory-efficient
7. **Restart the application** - Sometimes simply restarting helps

## Processing Errors

### Error: "index X is out of bounds for axis 0 with size Y"
This error occurs when the video or ROI is too small for the requested number of pyramid levels.

**Solution:** Reduce the pyramid levels in the Analysis Parameters panel or select a larger ROI.

### Error when processing large videos
This is usually a memory-related issue.

**Solution:** Follow the memory optimization tips above, particularly using a smaller ROI.

### No heart rate detected or unrealistic heart rate values

**Solutions:**
1. **Select a better ROI** - Focus on areas with good skin visibility and blood flow (face, wrists)
2. **Check video quality** - Ensure the video has good lighting and is stable
3. **Adjust the heart rate range** - Set a narrower range based on expected values (e.g., 80-160 BPM for babies)
4. **Increase amplification** - Try a higher amplification value (e.g., 75-100)

## Interface Issues

### Interface becomes unresponsive during processing
This is normal for larger videos as processing is resource-intensive.

**Solution:** The latest version includes a background processing feature that keeps the UI responsive. If you're still experiencing issues, try processing smaller videos or selecting a smaller ROI.

### Can't select a ROI
Make sure you drag from the top-left to the bottom-right of the area you want to select.

**Solution:** Click and hold the left mouse button at the starting corner, drag to the opposite corner, then release.

## Video File Issues

### Error: "Could not open video file"
The application cannot read the video file format.

**Solutions:**
1. **Check file format** - Ensure the video is in a common format (MP4, AVI, MOV)
2. **Convert the video** - Use a tool like FFmpeg to convert to a compatible format
3. **Check file permissions** - Ensure you have read permissions for the file

### Video plays too fast or slow
This can happen if the video's frame rate is detected incorrectly.

**Solution:** Try converting the video to a standard frame rate (e.g., 30fps) using video editing software.

## Installation Issues

### Missing dependencies

**Solution:** Run the installation script again or manually install the requirements:
```
pip install -r requirements.txt
```

### Permission denied when running install.sh

**Solution:** 
```
chmod +x install.sh
```
Then run the script again.

## Need More Help?

If you're still experiencing issues, please:

1. Check the README.md file for basic usage instructions
2. Make sure your computer meets the minimum requirements
3. Try using the simpler command-line version with a small video
4. If you're a developer, check the logs for more detailed error information

Remember that Eulerian Video Magnification is a computationally intensive technique. For best results, use good quality videos with clear visibility of the baby's skin under stable lighting conditions. 