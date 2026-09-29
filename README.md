# SAM 2.1 Automatic Object Segmentation with YOLO

Real-time webcam-based object segmentation using **Ultralytics SAM 2.1** with **YOLOv8** for automatic object detection.

This project combines YOLO object detection with SAM 2.1 segmentation to detect objects in a webcam stream and generate pixel-level segmentation masks.

The system also provides an **Everything Mode**, where SAM 2.1 attempts to segment objects directly without using YOLO detection.

---

## Features

- Real-time webcam segmentation
- SAM 2.1 for pixel-level object segmentation
- YOLOv8 for automatic object detection
- Two segmentation modes:
  - **AUTO Mode** — YOLO detection + SAM 2.1 segmentation
  - **EVERYTHING Mode** — SAM 2.1 segmentation without YOLO
- Adjustable YOLO confidence threshold
- Colored segmentation overlays
- Object contours
- Bounding boxes in AUTO mode
- Center-point prompts for SAM 2.1
- FPS display
- Frame saving
- Temporary-file management
- Frame skipping for improved real-time performance

---

## System Architecture

### AUTO Mode

The default mode uses YOLOv8 to detect objects and SAM 2.1 to generate segmentation masks.

```text
Webcam Frame
     |
     v
 YOLOv8 Detection
     |
     v
Bounding Boxes
     |
     v
Center Point of Each Box
     |
     v
 SAM 2.1
     |
     v
Segmentation Masks
     |
     v
Colored Overlay + Contours
     |
     v
Display
EVERYTHING Mode
Webcam Frame
     |
     v
    SAM 2.1
     |
     v
Multiple Segmentation Masks
     |
     v
Colored Overlay
     |
     v
Contours
     |
     v
Display
Requirements
Hardware

Recommended:

NVIDIA GPU with CUDA support
Webcam
8 GB+ system RAM
6 GB+ GPU VRAM recommended for comfortable inference

The program can also run on CPU, but segmentation performance will generally be significantly slower.

Software
Python 3.9+
OpenCV
NumPy
Ultralytics
PyTorch
CUDA-enabled PyTorch (recommended for NVIDIA GPUs)
Installation
1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
2. Create a virtual environment

Using Python venv:

python3 -m venv sam2_env

Activate it:

source sam2_env/bin/activate

For Windows:

sam2_env\Scripts\activate
3. Install dependencies
pip install -r requirements.txt

For NVIDIA GPU users, install a CUDA-compatible version of PyTorch appropriate for your system before running the project.

You can verify the PyTorch CUDA installation with:

python -c "import torch; print(torch.cuda.is_available())"

If the output is:

True

PyTorch can access the NVIDIA GPU.

Model Files

This project uses two models:

SAM 2.1

The code uses:

sam2_t.pt

This is the SAM 2 Tiny model.

The code can also be configured to use other SAM 2 model variants:

sam2_t.pt   # Tiny
sam2_s.pt   # Small
sam2_b.pt   # Base
sam2_l.pt   # Large

The model is loaded using:

self.sam = SAM(sam_model)

Ultralytics can automatically handle supported model downloads when the model name is passed to the API. Alternatively, place the downloaded .pt checkpoint in the project directory.

YOLOv8

The default detection model is:

yolov8n.pt

This is the YOLOv8 Nano model.

Other YOLOv8 variants can be used:

yolov8n.pt   # Nano
yolov8s.pt   # Small
yolov8m.pt   # Medium

The model is loaded using:

self.yolo = YOLO(yolo_model)

If the model is not available locally, Ultralytics can download the supported model automatically.

Project Structure

Recommended project structure:

sam2-segmentation/
│
├── sam2_segmentation.py
├── requirements.txt
├── README.md
│
├── sam2_t.pt
└── yolov8n.pt

The .pt model files can be kept locally or downloaded automatically by Ultralytics.

Running the Program

Activate the virtual environment:

source sam2_env/bin/activate

Run:

python sam2_segmentation.py

The program will attempt to open webcam 0.

Segmentation Modes
1. AUTO Mode

AUTO mode is the default.

It uses:

YOLOv8 → Object Detection → SAM 2.1 → Segmentation

YOLO detects objects using bounding boxes.

For every detected object, the center of its bounding box is calculated:

center_x = int((x1 + x2) / 2)
center_y = int((y1 + y2) / 2)

The center point is then supplied to SAM 2.1 as a positive point prompt:

self.sam(
    temp_frame_path,
    points=[[center_x, center_y]],
    labels=[1]
)

SAM 2.1 generates a segmentation mask for the detected object.

2. EVERYTHING Mode

Press:

E

to switch to EVERYTHING mode.

In this mode, YOLO detection is bypassed.

SAM 2.1 processes the frame and generates multiple segmentation masks.

The masks are combined with the original frame to create a colored segmentation overlay.

Keyboard Controls
Key	Function
A	Switch to AUTO mode
E	Switch to EVERYTHING mode
+ / =	Increase confidence threshold
- / _	Decrease confidence threshold
S	Save the current segmented frame
Q	Quit
Confidence Threshold

The default YOLO confidence threshold is:

self.confidence_threshold = 0.5

This controls which YOLO detections are accepted.

For example:

Confidence = 0.50

means detections with confidence below 0.50 are ignored.

The value can be changed while the program is running.

Press:

+

to increase the threshold.

Press:

-

to decrease the threshold.

The allowed range in the program is:

0.10 – 0.90
Frame Processing Optimization

SAM 2.1 segmentation is computationally expensive.

To improve webcam responsiveness, the program processes only every third frame:

process_every_n_frames = 3

Therefore:

Frame 1 → Display previous result
Frame 2 → Display previous result
Frame 3 → Run segmentation
Frame 4 → Display previous result
Frame 5 → Display previous result
Frame 6 → Run segmentation
...

This reduces the number of expensive SAM 2.1 inference operations.

Output

The webcam window displays:

FPS: XX.X
Mode: AUTO / EVERYTHING
Objects: X
Confidence: X.XX

In AUTO mode, detected objects additionally receive:

Bounding boxes
Center points
Segmentation masks
Colored overlays
Contours

In EVERYTHING mode, SAM 2.1-generated masks are shown with:

Different colors
Segmentation overlay
Object contours
Saving Segmented Frames

Press:

S

to save the currently displayed frame.

The program generates a filename similar to:

segmented_123456789.jpg

The image is saved in the current working directory.

Code Overview

The main class is:

AutomaticSAM2Segmenter
Initialization
AutomaticSAM2Segmenter(
    sam_model="sam2_t.pt",
    yolo_model="yolov8n.pt"
)

It initializes:

SAM 2.1
YOLOv8
Temporary directory
Webcam resolution
Confidence threshold
Segmentation mode
segment_everything()
segment_everything(frame)

Runs SAM 2.1 directly on the input frame.

Returns:

result_frame, num_masks
segment_detected_objects()
segment_detected_objects(frame)

Runs the complete YOLO + SAM pipeline.

Process:

Run YOLOv8 detection.
Extract bounding boxes.
Calculate the center point of each bounding box.
Send the center point to SAM 2.1.
Generate segmentation masks.
Apply colored overlays.
Draw bounding boxes.
Draw contours.
Return the segmented frame.

Returns:

result_frame, num_objects
cleanup()

Removes the temporary directory and temporary frame files created during processing.

run()

Starts the webcam and continuously:

Captures frames.
Runs segmentation.
Calculates FPS.
Displays the segmentation result.
Handles keyboard commands.
Saves frames when requested.
Releases the webcam on exit.
Temporary Files

The program creates a temporary directory using:

tempfile.mkdtemp()

Frames are temporarily written as:

temp_frame.jpg

These files are removed after processing.

The temporary directory is also deleted when the application exits.

GPU Acceleration

For NVIDIA GPUs, CUDA acceleration is recommended.

Check GPU availability:

nvidia-smi

Check PyTorch CUDA:

python -c "import torch; print(torch.cuda.is_available())"

Check the detected GPU:

python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

Expected example:

True
NVIDIA GeForce RTX ...
Performance Considerations

SAM 2.1 is significantly more computationally expensive than conventional object detection.

Performance depends on:

GPU model
GPU VRAM
SAM 2 model size
YOLO model size
Webcam resolution
Number of detected objects
Number of segmentation masks
Frame-processing interval

For lower-end GPUs, consider:

sam2_t.pt
yolov8n.pt

For higher-performance systems, larger models can be tested.

Troubleshooting
Webcam not opening

If you see:

Error: Could not open webcam

check:

ls /dev/video*

On Linux, verify that your user has permission to access the webcam.

You can also test the camera using:

v4l2-ctl --list-devices
CUDA not available

Check:

nvidia-smi

Then:

python -c "import torch; print(torch.cuda.is_available())"

If it returns:

False

check that the installed PyTorch build supports your CUDA environment.

Model not found

Make sure the required model names are available:

sam2_t.pt
yolov8n.pt

You can also allow Ultralytics to download supported model weights automatically.

Low FPS

Try:

Using sam2_t.pt
Using yolov8n.pt
Reducing webcam resolution
Increasing process_every_n_frames
Using a CUDA-enabled PyTorch installation
Using a more powerful GPU

For example:

process_every_n_frames = 5

will reduce the frequency of segmentation processing.

High GPU memory usage

Use the smaller SAM 2 model:

sam2_t.pt

and the smaller YOLO model:

yolov8n.pt

Also consider reducing the webcam resolution.

Limitations
SAM 2 segmentation is computationally intensive.
AUTO mode depends on YOLO successfully detecting an object.
The center-point prompt is used to guide SAM 2.1.
Processing every frame can significantly reduce FPS.
The current implementation uses temporary JPEG files for SAM inference.
Segmentation quality depends on the selected SAM 2 model and input image.
YOLO confidence threshold affects which objects are passed to SAM 2.1.
The webcam index is currently fixed to 0.
Future Improvements

Possible improvements include:

Direct image-array inference without temporary files
GPU/device configuration from command-line arguments
Object tracking between frames
Persistent object IDs
Segmentation mask export
Video input support
Image input support
FPS optimization
CUDA stream optimization
Multi-camera support
Adjustable webcam resolution
Class-specific segmentation
Instance segmentation tracking
GUI controls for model and confidence selection
Technologies Used
Python
OpenCV
NumPy
PyTorch
Ultralytics
SAM 2.1
YOLOv8
CUDA (optional, recommended for NVIDIA GPUs)
Applications

This project can serve as a foundation for:

Robotic perception
Computer vision
Object-aware robotics
Autonomous systems
Scene understanding
Vision-based manipulation
Human-robot interaction
Object tracking pipelines
Semantic/instance segmentation experiments
Physical AI research
Author

Arun M

Robotics & Automation Engineer
Computer Vision | ROS 2 | Autonomous Systems | Physical AI
