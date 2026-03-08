# MotionClipper

A Python application for detecting motion in video files and creating clipped versions of video projects.

## Issues Fixed

The original code had several critical issues that have been addressed:

### 1. Missing Dependencies
The code requires several external libraries that weren't installed:
- `opencv-python` - for computer vision operations
- `imutils` - for image processing utilities  
- `PyQt5` - for GUI components
- `numpy` - for numerical operations
- `tqdm` - for progress bars

**Solution**: Install dependencies using:
```bash
pip install -r requirements.txt
```

### 2. XML Element Access Issues
The code was accessing XML elements without checking if they exist, causing `AttributeError` when elements were `None`.

**Fixed Issues**:
- Added null checks before accessing XML element properties
- Safe element access with proper error handling
- Graceful handling of missing XML elements

### 3. OpenCV API Issues
The code used an incorrect OpenCV API signature for `findContours()`.

**Fixed**: Updated from `(_, cnts, _)` to `(cnts, _)` to match current OpenCV version.

### 4. Deprecated ElementTree Method
The code used the deprecated `_setroot()` method.

**Fixed**: Replaced with proper `ET.ElementTree()` constructor.

### 5. Type Safety Issues
The code didn't handle cases where functions returned `None` or empty lists.

**Fixed**: Added proper null checks and empty list handling throughout the code.

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python motion_clipper.py -f your_project_file.xml
```

## Features

- Motion detection in video files
- Support for both Premiere Pro (.xml) and Final Cut Pro (.fcpxml) project files
- Configurable motion detection parameters
- Progress tracking and GUI support
- Automatic creation of clipped project files

## Parameters

- `min_area`: Minimum area size for motion detection
- `alpha`: Learning rate for background model
- `threshold`: Motion detection threshold
- `width`: Resize width for processing
- `minMotionFrames`: Minimum frames to consider as motion
- `minNonMotionFrames`: Minimum frames to consider as still
- `nonMotionBeforeStart`: Frames to include before motion starts
- `nonMotionAfter`: Frames to include after motion ends
- `minFramesToKeep`: Minimum frames to keep in clips 