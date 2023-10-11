# Simple Face Detection Model

**Welcome to the Simple Face Detection Model!** This Python script enables you to detect faces within images and real-time video streams using OpenCV. The model offers a straightforward way to identify and highlight faces within your media.

## Features

- Detects faces in both images and video streams.
- Real-time face detection using your computer's webcam.
- Image processing enhancements like median blur and edge detection.
- Draws bounding rectangles around detected faces and labels them.
- Saves processed video frames to an output video file.

## Quick Start

1. **Installation**:

   To get started, make sure to install the required Python libraries:

   ```bash
   pip install opencv-python numpy
   ```

2. **Run the Model**:

   Run the Simple Face Detection Model using the following command:

   ```bash
   python opencv_video.py
   ```

   You can replace `face_detection.py` with your Python script's filename if it differs.

3. **Configuration**:

   - By default, the model captures video from your computer's webcam (camera index 0). Modify the `cv2.VideoCapture` line in the script to specify a video file if desired.

   - The processed video frames are saved to an output file named "output.avi" using the XVID codec. Feel free to change the filename or codec to suit your needs.

4. **Exiting the Model**:

   While the model is running, press the 'q' key to exit and close the application.

## Enjoy Exploring!

Start exploring the Simple Face Detection Model and use it to detect faces in your images and videos. Feel free to customize and expand upon this model for your own projects!

Happy coding!
