# Detection pipeline
- run these step on all the frames of the clip
	- maybe run on just 10-15 fps to save time and resource
- run the character detection on the first frame
	- uses YOLO to find every person in the scene
	- uses face finder `yolov8n-face.pt` (and equivalent community ports for YOLO11/YOLO26)
		- or a dedicated face finder
	- uses Facial Recognition model (like FaceNet, DeepFace, or an embedding model) to identify the character
		- just need 1 clear image of the character face, although a few with left, right angle, different lighting would work better
- label the bounding box, hand it over to Object Tracker, this will track the box coordinate on the remaining frame, or i could just run the first step on each frame, since we only have 6 frames per clip
	- i may change this to solve the [[#The Faceless Character Problem]]
	- [[#when to run the face recognition]]
- run these step on just the frames output of [[#Dynamic frame extraction]]
- use openCV to draw the bounding boxes onto the frames
	- draw boxes with labels `[1]`, `[2]`, `[3]`.
- then pass the frames along with text information, which should looks like this
	- Frame 1 Entities: 
		- `[1] Person (Identified as John Doe): [0.15, 0.05, 0.20, 0.10]` 
		-  `[2] Person (Identified as Jane Smith): [0.60, 0.20, 0.70, 0.50]` 
		- `[3] Object (Car): [0.50, 0.40, 0.90, 0.80]`
- to the VLM, along side story chapter summary, world building overview, the previous clip caption.
I may combine this with [[Character_detection_for_animated_movie]]

# when to run the face recognition
Use BoT-SORT as tracker
Trackers act as state managers. Every time the tracker sees a bounding box, it assigns an integer ID to it.
To know when a face is new, you simply maintain a "registry" (a Python dictionary or list) in your code.
**The Logic:**
- Frame 1: Tracker outputs `[Box: 150, 50, 200, 100, ID: 1]`.
	- Your Code: "Is `ID 1` in my registry? No. This is a new appearance."
- Frame 2: Tracker outputs `[Box: 152, 52, 202, 102, ID: 1]`.
	- Your Code: "Is `ID 1` in my registry? Yes. Ignore it and move to the next frame."

**Condition A: A New Tracker ID is Created** whenever registry encounters an ID it hasn't seen before, you crop that specific bounding box and send it to the Identity Matcher. Once the matcher returns a name (e.g., "John Doe"), you map `ID: 1` to "John Doe" in your registry. For every subsequent frame where `ID: 1` appears, you instantly know it's John Doe without running the matcher again.

**Condition B: The Tracker Recovers a "Lost" Track** If John Doe walks behind a pillar at Frame 30 and emerges at Frame 40, the tracker will likely lose `ID: 1` and assign him a brand new ID, like `ID: 2`, when he emerges. Because `ID: 2` is not in your registry, Condition A triggers automatically. You run the matcher, identify the face as John Doe, and update your registry to show that _both_ `ID: 1` and `ID: 2` belong to John Doe.


# Dynamic frame extraction
use **PySceneDetect**.
- Run a video through PySceneDetect, it returns a list of "Scenes" with their exact start and end frames: 
	- `[(Scene 1 Start, Scene 1 End), (Scene 2 Start, Scene 2 End)...]`
- If a scene is longer than 1 second divide it into 1 second segments

for each scene:
- grab the 1/4 (6 frames out of 24) total frames of the scene in the middle
- run it through OpenCV function called the **Variance of the Laplacian**.1
	- A crisp, highly detailed image has sharp edges (high variance).
	- A smeared, motion-blurred image has soft edges (low variance).
- get the frame with the highest variance






# Some info i need to know
**1. Input Preprocessing & Formatting**
- **Resolution Scaling:** YOLO models expect square inputs (e.g., 640x640). If you feed it a standard 16:9 movie frame, the model or library will resize it. You need to know how it's doing this (usually a technique called "letterboxing" which adds padding) so the aspect ratio isn't crushed, which would distort the objects.
- **Color Spaces:** OpenCV reads images in BGR format, but YOLO expects RGB. Forgetting to convert this is a classic mistake that ruins accuracy.
**2. Output Parsing & Filtering**
- **Confidence Thresholds:** YOLO will output a detection for almost everything, even if it's only 5% sure. You need to set a confidence threshold (e.g., `conf=0.5`) so your downstream knowledge graph isn't flooded with false positives.
- **Non-Maximum Suppression (NMS):** YOLO often draws multiple overlapping bounding boxes around the same object. NMS is the algorithm that filters these down to the single best box. You'll need to tweak the NMS threshold to control how aggressively it merges overlapping boxes.
- **Coordinates:** You must understand the output format of the bounding boxes. Are they `[x_center, y_center, width, height]` or `[x1, y1, x2, y2]`?
**3. Hardware & Batching (The Performance Bottleneck)** When extracting 1-3 frames per second across 5-6 second video segments, you are dealing with a massive volume of images. Pushing these frames through the model sequentially (one by one) will severely bottleneck a local GPU.
- **Batch Inference:** You need to understand how to stack those 15-18 frames into a single tensor batch. Sending batches to the GPU maximizes its parallel processing capabilities and drastically cuts down total processing time.



# some other stuff

## The Faceless Character Problem
Upgrade from **Face Detection** to **Multi-Object Tracking (MOT)**.
Instead of just detecting faces frame-by-frame, use a tracking algorithm like **ByteTrack** or **DeepSORT** combined with a YOLO model.
- **How it works:** In Frame 1, YOLO detects the "person" (the whole body, not just the face), and your face-detector identifies them as John. DeepSORT assigns John an ID (e.g., `Track_ID_1`).
- In Frame 2, John turns around. The face detector fails. However, DeepSORT remembers `Track_ID_1` based on the velocity of the bounding box and the visual features of John's clothing.
- Your Python script maintains the link: `Track_ID_1 = John_Doe`. You continue to pass John's body coordinates to the VLM even though his face isn't visible.

https://ieeexplore.ieee.org/document/8604143
https://www.ijert.org/robust-identification-of-movie-character-using-face-name-graph-matching