#!/usr/bin/env python3
"""
Run the Baby Heart Rate Monitor application with GUI.
"""

import sys
from PyQt5.QtWidgets import QApplication
from baby_heart_rate_monitor import BabyHeartRateMonitor

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style and dark theme
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = BabyHeartRateMonitor()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec_()) 