# Face Recognition Attendance System
This project implements a real-time face recognition attendance system using Python. The system captures video from a webcam, detects faces, matches them with pre-stored face encodings, and updates attendance records in a CSV file.


##  Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Example Output](#example-output)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)


## Features

- **Real-time Face Detection and Recognition**: Captures video frames, detects faces, and matches them with known encodings.
- **Automated Attendance Marking**: Updates attendance status in a CSV file based on face recognition.
- **Performance Metrics**: Tracks and analyzes the system's accuracy and efficiency over time.
- **Visualization**: Generates heatmaps of face encodings and distance metrics.
- **User Interface**: Displays real-time video feed with marked faces and IDs.


## Technologies Used

- Python
- OpenCV
- face_recognition
- pandas
- matplotlib
- seaborn
- pickle
- datetime


## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/face-recognition-attendance-system.git
    cd face-recognition-attendance-system
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```


## Usage

1. **Encode Faces**: Run `encode_generator.py` to generate and save face encodings.
    ```bash
    python encode_generator.py
    ```

2. **Capture Attendance**: Run `main.py` to start the attendance system.
    ```bash
    python main.py
    ```


## File Structure

- `encode_generator.py`: Script to generate and save face encodings.
- `main.py`: Script to capture attendance using face recognition.
- `requirements.txt`: List of required Python packages.


## Example Output

The system updates the `attendance.csv` file with attendance records and generates various visualizations saved as images (e.g., `accuracy_over_time.png`, `Stored_encoding_heatmap.png`, `Detected_encoding_heatmap.png`, `Distances_heatmap.png`).


## Contributing

Contributions are welcome! Please fork the repository and create a pull request.

## Acknowledgments

- [face_recognition](https://github.com/ageitgey/face_recognition)
- [OpenCV](https://opencv.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
